#!/usr/bin/env python3
"""
Minimal PD Proxy for vLLM-Ascend Mooncake PD disaggregation (1P1D).

Architecture:
  Client → Proxy(:8002) → P(:8000) prefill (inject kv_transfer_params, max_tokens=1)
                        → P returns kv_transfer_params
                        → Proxy forwards to D(:8001) decode with kv_transfer_params
                        → streaming response back to client

Based on official pd_proxy_official.py build_prefill_request pattern:
  Stage 1: Send to P with kv_transfer_params={do_remote_decode: True}, max_tokens=1
  Stage 2: P responds with kv_transfer_params → inject into request → send to D

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

def _build_prefill_payload(original: dict, request_id: str) -> dict:
    """Build payload for prefill-only request to P.

    Per official vLLM Mooncake PD protocol:
    - kv_transfer_params tells P: "do remote decode, save KV to Mooncake transport"
    - transfer_id is REQUIRED — connector skips _reqs_need_send without it (line 648)
    - max_tokens=1: P only outputs 1 token (stop token), saving full KV cache
    - stream=False: prefill stage is not streaming
    """
    payload = original.copy()
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": request_id,
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
    """Two-stage PD routing: prefill P → decode D."""
    request_id = f"pd-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()
    stream_flag = bool(req_data.get("stream", False))

    # ── Stage 1: Prefill on P (inject kv_transfer_params, max_tokens=1) ──
    prefill_url = f"{args.prefill_url}{api_path}"
    prefill_payload = _build_prefill_payload(req_data, request_id)
    prompt_len = len(req_data.get("prompt", ""))
    logger.info(f"[{request_id}] Stage1 → P prefill ({prompt_len} chars, "
                f"max_tokens=1, do_remote_decode=True, transfer_id={request_id})")

    prefill_resp = await client.post(
        prefill_url, json=prefill_payload,
        timeout=httpx.Timeout(args.timeout),
    )
    prefill_resp.raise_for_status()
    prefill_body = prefill_resp.json()

    kv_params = prefill_body.get("kv_transfer_params")
    if kv_params:
        logger.info(f"[{request_id}] ✓ Got kv_transfer_params from P: "
                    f"{json.dumps(kv_params, default=str)[:300]}")
        # Construct D-side kv_transfer_params: do_remote_prefill + transfer_id + engine info
        req_data["kv_transfer_params"] = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "transfer_id": request_id,
            "remote_engine_id": kv_params.get("remote_engine_id"),
            "remote_block_ids": kv_params.get("remote_block_ids"),
            "remote_host": kv_params.get("remote_host"),
            "remote_port": kv_params.get("remote_port"),
        }
    else:
        logger.warning(f"[{request_id}] ✗ P returned null kv_transfer_params "
                       f"— KV transfer will NOT happen. Check P bootstrap (8998).")

    # ── Stage 2: Decode on D (original max_tokens + kv_transfer_params) ──
    decode_url = f"{args.decode_url}{api_path}"
    logger.info(f"[{request_id}] Stage2 → D decode "
                f"(kv={'yes' if kv_params else 'no'}, "
                f"max_tokens={req_data.get('max_tokens','?')})")

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
            logger.info(f"[{request_id}] Done {elapsed:.1f}s {byte_count}B "
                        f"kv={'yes' if kv_params else 'no'}")

    return StreamingResponse(
        stream_response(),
        media_type=ct,
        headers={"X-Request-ID": request_id, "X-PD-Proxy": "v1"},
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