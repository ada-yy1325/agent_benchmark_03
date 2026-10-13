import httpx, asyncio
async def test():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000/v1", timeout=10) as c:
        r = c.build_request("POST", "/v1/completions")
        print("built URL:", r.url)
        resp = await c.post("/v1/completions", json={"model":"dsv4","prompt":"test","max_tokens":1})
        print("status:", resp.status_code)
        print("actual URL:", resp.request.url)
asyncio.run(test())
