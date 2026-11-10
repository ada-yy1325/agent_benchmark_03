#!/usr/bin/env python3
"""
PD Proxy for vLLM-Ascend Mooncake PD disaggregation (1P1D).
Fixed architecture:
  Client → Proxy(:8002)
    Stage 1: → P(:8000) prefill (do_remote_decode, max_tokens=1)
    Stage 2: → P(:8998)/query → get remote_engine_id
    Stage 3: → D(:8001) decode (do_remote_prefill, same transfer_id,
               remote_engine_id from 8998, remote_bootstrap_addr)

Official vLLM behavior: P ALWAYS returns kv_transfer_params=null in HTTP response.
The engine_id comes from the Mooncake bootstrap server (:8998/query), not from P's response.
The connector pushes KV internally — the proxy's job is to get engine_id from bootstrap
and construct the correct D-side payload.

Usage:
  python3 pd_proxy.py --prefill-url http://127.0.0.1:8000 --decode-url http://127.0.0.1:8001 --port 8002
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PD proxy (1P1D)")
parser.add_argument("--prefill-url", default="http://127.0.0.1:8000")
parser.add_argument("--decode-url", default="http://127.0.0.1:8001")
parser.add_argument("--port", type=int, default=8002)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--timeout", type=float, default=600)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [proxy] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pd_proxy")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="PD-Proxy-1P1D")

# Shared httpx client (connection pooling)
client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(args.timeout))
    logger.info(f"Proxy started: P={args.prefill_url}  D={args.decode_url}")


@app.on_event("shutdown")
async def shutdown():
    global client
    if client:
        await client.aclose()
        client = None


# ---------------------------------------------------------------------------
# Two-stage routing
# ---------------------------------------------------------------------------

# P bootstrap address (derived from prefill-url)
# vLLM Mooncake bootstrap listens on port 8998 on the P host
BOOTSTRAP_URL = "http://127.0.0.1:8998"


def _extract_host(url: str) -> str:
    """Extract host from http://host:port."""
    return url.split("://")[1].split(":")[0]


async def _get_engine_id(prefill_host: str, request_id: str) -> Optional[str]:
    """Query Mooncake bootstrap server to get P's engine_id."""
    try:
        resp = await client.get(
            f"http://{prefill_host}:8998/query",
            timeout=httpx.Timeout(5.0),
        )
        resp.raise_for_status()
        data = resp.json()
        # Response: {"0": {"engine_id": "...", "worker_addr": {...}}}
        if "0" in data and "engine_id" in data["0"]:
            engine_id = data["0"]["engine_id"]
            logger.info(f"[{request_id}] Got engine_id from bootstrap: {engine_id[:40]}...")
            return engine_id
        logger.warning(f"[{request_id}] Bootstrap response unexpected: {json.dumps(data)[:200]}")
        return None
    except Exception as e:
        logger.error(f"[{request_id}] Failed to query bootstrap: {e}")
        return None


def _build_prefill_payload(original: dict, transfer_id: str) -> dict:
    """Build payload for prefill-only request to P."""
    payload = original.copy()
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": transfer_id,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["min_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    return payload


async def pd_route(api_path: str, req_data: dict, request: Request) -> StreamingResponse:
    """Three-stage PD routing: prefill P → get engine_id → decode D."""
    transfer_id = f"pd-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()
    stream_flag = bool(req_data.get("stream", False))
    prefill_host = _extract_host(args.prefill_url)
    prompt_len = len(req_data.get("prompt", ""))

    # ── Stage 1: Prefill on P ──
    prefill_url = f"{args.prefill_url}{api_path}"
    prefill_payload = _build_prefill_payload(req_data, transfer_id)
    logger.info(f"[{transfer_id}] Stage1 → P prefill ({prompt_len} chars, max_tokens=1, do_remote_decode=True)")

    try:
        prefill_resp = await client.post(
            prefill_url, json=prefill_payload,
            timeout=httpx.Timeout(args.timeout),
        )
        prefill_resp.raise_for_status()
    except Exception as e:
        logger.error(f"[{transfer_id}] Stage1 P prefill failed: {e}")
        raise

    # ── Stage 2: Get engine_id from bootstrap ──
    await asyncio.sleep(0.1)  # Brief delay for P to register
    engine_id = await _get_engine_id(prefill_host, transfer_id)
    if not engine_id:
        logger.error(f"[{transfer_id}] Cannot get engine_id — aborting KV transfer, falling back to direct D")

    # ── Stage 3: Decode on D with kv_transfer_params ──
    decode_url = f"{args.decode_url}{api_path}"
    req_data["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "transfer_id": transfer_id,
        "remote_engine_id": engine_id,
        "remote_bootstrap_addr": f"http://{prefill_host}:8998",
    }
    logger.info(f"[{transfer_id}] Stage3 → D decode (kv={'yes' if engine_id else 'no'}, engine_id={'set' if engine_id else 'MISSING'})")

    decode_req = client.build_request(
        "POST", decode_url, json=req_data,
        timeout=httpx.Timeout(args.timeout),
    )
    decode_resp = await client.send(decode_req, stream=True)
    decode_resp.raise_for_status()

    elapsed = time.perf_counter() - t0
    ct = decode_resp.headers.get("content-type",
        "text/event-stream" if stream_flag else "application/json")

    async def stream_response():
        byte_count = 0
        try:
            async for chunk in decode_resp.aiter_bytes():
                byte_count += len(chunk)
                yield chunk
        finally:
            await decode_resp.aclose()
            logger.info(f"[{transfer_id}] Done {elapsed:.1f}s {byte_count}B kv={'yes' if engine_id else 'no'}")

    return StreamingResponse(
        stream_response(),
        media_type=ct,
        headers={"X-Request-ID": transfer_id, "X-PD-Proxy": "v2"},
    )


@app.post("/v1/completions")
async def completions(request: Request):
    req_data = await request.json()
    return await pd_route("/v1/completions", req_data, request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_data = await request.json()
    return await pd_route("/v1/chat/completions", req_data, request)


@app.get("/health")
async def health():
    return {"status": "ok", "prefill": args.prefill_url, "decode": args.decode_url}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")