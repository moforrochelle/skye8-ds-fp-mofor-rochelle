import os
import json
import time
import collections
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

MODEL = "gemini-3.5-flash-lite"
KEY = os.environ.get("GEMINI_KEY", "")
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       + MODEL + ":streamGenerateContent?alt=sse")

SYSTEM = ("You are a voice assistant. Keep answers short (one or two sentences) "
          "unless the user asks for a long or detailed answer.")

app = FastAPI()
hits = collections.defaultdict(list)
all_hits = []


def too_many(ip):
    now = time.time()
    recent = [t for t in hits[ip] if now - t < 60]
    if len(recent) >= 20:
        hits[ip] = recent
        return True
    recent.append(now)
    hits[ip] = recent
    return False


def over_global_cap():
    now = time.time()
    all_hits[:] = [t for t in all_hits if now - t < 60]
    if len(all_hits) >= 120:
        return True
    all_hits.append(now)
    return False


def clean(raw):
    data = json.loads(raw)
    out = []
    for c in data.get("contents", [])[-7:]:
        role = "model" if c.get("role") == "model" else "user"
        text = "".join(str(p.get("text", "")) for p in c.get("parts", []))[:1500]
        out.append({"role": role, "parts": [{"text": text}]})
    if not out:
        raise ValueError("empty")
    return {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": out,
            "generationConfig": {"maxOutputTokens": 400}}


@app.post("/api/chat")
async def chat(request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    if too_many(ip) or over_global_cap():
        return Response("Too many requests", status_code=429)
    body = await request.body()
    if len(body) > 20000:
        return Response("Request too large", status_code=413)
    if not KEY:
        return Response("Server has no API key set", status_code=500)
    try:
        payload = clean(body)
    except Exception:
        return Response("Bad request", status_code=400)

    client = httpx.AsyncClient(timeout=60)
    try:
        req = client.build_request("POST", URL, json=payload,
                                   headers={"x-goog-api-key": KEY})
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        return Response("Upstream unavailable", status_code=502)

    async def pipe():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        pipe(), status_code=upstream.status_code, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse("voice.html")


app.mount("/results", StaticFiles(directory="results"), name="results")