---
title: Live Voice Conversation System
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Live Voice Conversation System

An English voice assistant you can talk to like a phone call: press Start, speak, hear a spoken reply with captions, and (optionally) interrupt it. Its core is an end-of-turn (endpointing) model trained on the AMI Meeting Corpus that decides when the speaker has finished, instead of a fixed silence timer.

Live demo: PASTE-THE-SPACE-URL-HERE (needs internet and Chrome; Chrome on Android works best)

## How it works
1. The browser listens and transcribes speech live (browser speech recognition).
2. A gradient-boosting model, trained by me on AMI and exported to JSON, runs in the page every 100 ms and estimates whether the speaker has finished, using pause length, the last word, and how much has been said.
3. When it decides the turn is over, the text goes to a small FastAPI server, which adds the secret API key and relays it to the Gemini API.
4. The reply streams back and is spoken sentence by sentence, with captions.
5. Optional interruption: Silero VAD detects the user speaking while the assistant talks and stops the speech.

## Run it yourself
Requirements: Python 3.10+, a free Gemini API key.

    pip install -r requirements.txt
    export GEMINI_KEY="your-key"
    uvicorn app:app --port 8001

Open http://localhost:8001 in Chrome.

## Reproduce the model
The AMI data is not in this repo. Download the manual annotations and a few Edinburgh scenario meetings (ES2008a, ES2008b) from the AMI corpus download page, put them under `data/amicorpus/`, then:

    pip install scikit-learn matplotlib numpy
    python3 build-example.py
    python3 train_endpoint.py
    python3 export_model.py

`train_endpoint.py` trains on ES2008a, tests on ES2008b, and writes the trade-off plot to `results/tradeoff.png`.

## Results
See the technical report. Key numbers are in `results/` and `stage-a-results.md`.

## Limitations
- The model sees pause length and words, not pitch or energy.
- Trained and tested on only two meetings.
- On laptop speakers the assistant can hear itself; use headphones for interruption.
- Speech recognition needs Chrome; iPhone browsers are unreliable.

## Data and keys
No data files or credentials are stored in this repository. The API key is set as an environment variable (a secret on the host).