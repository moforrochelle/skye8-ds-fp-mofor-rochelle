import os
import re
import json
import time
import collections
import httpx
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

MODEL = "gemini-3.5-flash-lite"
KEY = os.environ.get("GEMINI_KEY", "")
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       + MODEL + ":streamGenerateContent?alt=sse")

SYSTEM = ("You are a voice assistant. Keep answers short (one or two sentences) "
          "unless the user asks for a long or detailed answer.")

CAMEROON = (" Cameroon has 10 regions (capital in brackets): Adamawa (Ngaoundere),"
            " Centre (Yaounde), East (Bertoua), Far North (Maroua), Littoral (Douala),"
            " North (Garoua), Northwest (Bamenda), South (Ebolowa), Southwest (Buea)"
            " and West (Bafoussam). Yaounde is the political capital and Douala is the"
            " largest city and main port. If you are asked about a small town or village"
            " and you are not sure, say you are not sure instead of guessing.")

app = FastAPI()
hits = collections.defaultdict(list)
all_hits = []
places = {}


def valid_coords(lat, lon):
    try:
        lat = round(float(lat), 2)
        lon = round(float(lon), 2)
    except Exception:
        return None
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return (lat, lon)
    return None


async def lookup_place(lat, lon):
    key = (lat, lon)
    if key in places:
        return places[key]
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format": "jsonv2", "lat": lat, "lon": lon,
                        "zoom": 14, "accept-language": "en"},
                headers={"User-Agent": "live-voice-assistant-demo/1.0"})
        a = r.json().get("address", {})
    except Exception:
        return ""
    parts = []
    for k in ("neighbourhood", "suburb", "village", "town", "city",
              "county", "state", "country"):
        v = a.get(k)
        if v and v not in parts:
            parts.append(v)
    place = re.sub(r"[^\w ,'\-\.]", "", ", ".join(parts[:5]))[:120]
    if len(places) > 500:
        places.clear()
    places[key] = place
    return place


def system_text(coords=None):
    now = datetime.now(timezone(timedelta(hours=1)))
    if coords:
        place = places.get(coords, "")
        loc = (" The user has shared their approximate location"
               + (" (" + place + ")" if place else "")
               + ", coordinates " + str(coords[0]) + ", " + str(coords[1])
               + ". Use it when they ask where they are or about nearby places,"
               + " and say it is approximate.")
    else:
        loc = (" You do not know their exact city or town, so if it matters,"
               " ask them.")
    return (SYSTEM
            + " The user is in Cameroon. Treat Cameroon as their country for any"
            + " question about where they are or about local matters."
            + loc
            + CAMEROON
            + " The current date and time in Cameroon is "
            + now.strftime("%A %d %B %Y, %H:%M") + ".")


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
    coords = valid_coords(data.get("lat"), data.get("lon"))
    return {"systemInstruction": {"parts": [{"text": system_text(coords)}]},
            "contents": out,
            "generationConfig": {"maxOutputTokens": 400}}


@app.get("/api/locate")
async def locate(request: Request, lat: float, lon: float):
    ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    if too_many(ip):
        return Response("Too many requests", status_code=429)
    coords = valid_coords(lat, lon)
    if not coords:
        return Response("Bad coordinates", status_code=400)
    return {"place": await lookup_place(coords[0], coords[1])}


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


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse("voice.html")


app.mount("/results", StaticFiles(directory="results"), name="results")