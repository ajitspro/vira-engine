import os
import re
import time
import uuid
import json
import subprocess
from pathlib import Path
from typing import Optional
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from google import genai

app = FastAPI(title="Vira AI Cloud Brain")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()
SUPADATA_KEY = os.environ.get("SUPADATA_API_KEY", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
BUCKET = os.environ.get("SUPABASE_BUCKET", "vira-clips").strip()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "ajitspro/vira-engine").strip()

TMP_DIR = Path("/tmp/vira")
TMP_DIR.mkdir(parents=True, exist_ok=True)


class YTAnalyzeReq(BaseModel):
    url: str
    max_clips: int = 20
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class RenderReq(BaseModel):
    job_id: str
    clip_index: int
    start: str
    end: str
    caption_style: str = "bold_yellow"


def supa_upload(path: str, data: bytes, content_type: str) -> str:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(500, "Supabase config missing on server")
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    r = requests.post(url, headers=headers, data=data, timeout=180)
    if not r.ok:
        raise HTTPException(500, f"Storage upload failed: {r.text[:200]}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"


def supa_get_json(path: str):
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}?t={int(time.time())}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Cache-Control": "no-cache",
    }
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 404 or r.status_code == 400:
        return None
    r.raise_for_status()
    return r.json()


def extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|live/|shorts/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise HTTPException(400, "Invalid YouTube URL")
    return m.group(1)


def sec_to_hms(s: float) -> str:
    s = max(0, int(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def hms_to_sec(t: str) -> float:
    parts = [float(x) for x in t.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def _parse_supadata_content(content):
    if isinstance(content, list):
        entries = []
        for item in content:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            if "offset" in item:
                start_sec = float(item["offset"]) / 1000.0
            else:
                start_sec = float(item.get("start", 0))
            entries.append({"start": start_sec, "text": text})
        if entries:
            return entries
    elif isinstance(content, str) and content.strip():
        return [{"start": 0.0, "text": content.strip()}]
    return []


def fetch_yt_transcript(vid: str):
    if SUPADATA_KEY:
        headers = {"x-api-key": SUPADATA_KEY}
        yt_url = f"https://www.youtube.com/watch?v={vid}"
        r = requests.get(
            "https://api.supadata.ai/v1/youtube/transcript",
            headers=headers,
            params={"url": yt_url, "text": "false"},
            timeout=45,
        )
        if not r.ok:
            raise RuntimeError(f"Supadata error ({r.status_code}): {r.text[:200]}")

        data = r.json()
        if r.status_code == 202 and data.get("jobId"):
            job_id = data["jobId"]
            for _ in range(15):
                time.sleep(2)
                pr = requests.get(
                    f"https://api.supadata.ai/v1/transcript/{job_id}",
                    headers=headers,
                    timeout=30,
                )
                if pr.ok:
                    pdata = pr.json()
                    if pdata.get("status") == "completed":
                        data = pdata
                        break
                    if pdata.get("status") == "failed":
                        raise RuntimeError(f"Supadata transcript failed: {pdata}")

        entries = _parse_supadata_content(data.get("content"))
        if entries:
            return entries
        raise RuntimeError("No captions available for this video via Supadata")

    ytt = YouTubeTranscriptApi()
    try:
        return ytt.fetch(vid, languages=["hi", "en", "en-IN", "en-US"]).to_raw_data()
    except Exception:
        tlist = ytt.list(vid)
        for tr in tlist:
            return tr.fetch().to_raw_data()
        raise RuntimeError("No captions available for this video")


def pick_viral_clips(transcript_text: str, max_clips: int):
    if not GEMINI_KEY:
        raise HTTPException(500, "GEMINI_API_KEY missing on server")
    client = genai.Client(api_key=GEMINI_KEY)
    prompt = f"""You are a viral short-form video editor. Analyze this transcript and extract up to {max_clips} high-retention vertical clips (each 30 to 60 seconds long).
Return ONLY a valid JSON array of objects with exact keys:
- "title": catchy hook title
- "viral_score": integer 75 to 99
- "start": "HH:MM:SS"
- "end": "HH:MM:SS"
- "script": short engaging summary/hook of the moment

Transcript:
{transcript_text[:120000]}"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    clips = json.loads(resp.text)
    if isinstance(clips, dict) and "clips" in clips:
        clips = clips["clips"]
    return clips


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "vira-cloud-1080p",
        "supadata": bool(SUPADATA_KEY),
    }


@app.post("/api/analyze-youtube")
def analyze_youtube(req: YTAnalyzeReq):
    vid = extract_video_id(req.url)
    try:
        entries = fetch_yt_transcript(vid)
    except Exception as e:
        raise HTTPException(400, f"YouTube transcript error: {e}")

    s_lim = hms_to_sec(req.start_time) if req.start_time else 0
    e_lim = hms_to_sec(req.end_time) if req.end_time else 999999
    lines = [
        f"[{sec_to_hms(item['start'])}] {item['text']}"
        for item in entries
        if s_lim <= item["start"] <= e_lim
    ]
    if not lines:
        raise HTTPException(400, "Selected time range mein transcript nahi mila.")

    clips = pick_viral_clips("\n".join(lines), req.max_clips)
    job_id = uuid.uuid4().hex[:10]
    job_data = {"type": "yt", "url": req.url}
    supa_upload(f"jobs/{job_id}.json", json.dumps(job_data).encode(), "application/json")
    return {"job_id": job_id, "clips": clips}


@app.post("/api/analyze-upload")
async def analyze_upload(file: UploadFile = File(...), max_clips: int = Form(10)):
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY missing on server")
    job_id = uuid.uuid4().hex[:10]
    vid_path = TMP_DIR / f"{job_id}.mp4"
    aud_path = TMP_DIR / f"{job_id}.mp3"

    try:
        raw_bytes = await file.read()
        vid_path.write_bytes(raw_bytes)
        public_src = supa_upload(f"uploads/{job_id}.mp4", raw_bytes, "video/mp4")

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(vid_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(aud_path)],
            check=True,
        )
        gclient = Groq(api_key=GROQ_KEY)
        with open(aud_path, "rb") as af:
            tr = gclient.audio.transcriptions.create(
                file=(aud_path.name, af.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        segs = getattr(tr, "segments", None) or []
        lines = [
            f"[{sec_to_hms(s['start'] if isinstance(s, dict) else s.start)}] "
            f"{s['text'] if isinstance(s, dict) else s.text}"
            for s in segs
        ]
        clips = pick_viral_clips("\n".join(lines) or tr.text, max_clips)
        job_data = {"type": "file", "url": public_src}
        supa_upload(f"jobs/{job_id}.json", json.dumps(job_data).encode(), "application/json")
        return {"job_id": job_id, "clips": clips}
    finally:
        vid_path.unlink(missing_ok=True)
        aud_path.unlink(missing_ok=True)


@app.post("/api/render")
def start_render(req: RenderReq):
    if not GITHUB_TOKEN:
        raise HTTPException(500, "GITHUB_TOKEN missing on server")

    job = supa_get_json(f"jobs/{req.job_id}.json")
    if not job:
        raise HTTPException(404, "Job expired or not found, dubara Analyze karo.")

    render_id = f"{req.job_id}_{req.clip_index}_{uuid.uuid4().hex[:6]}"
    init_status = {
        "render_id": render_id,
        "status": "queued",
        "message": "1080p Cloud Renderer start ho raha hai (45-75 sec)...",
        "download_url": "",
    }
    supa_upload(f"status/{render_id}.json", json.dumps(init_status).encode(), "application/json")

    gh_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/render-1080p.yml/dispatches"
    r = requests.post(
        gh_url,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "ref": "main",
            "inputs": {
                "render_id": render_id,
                "source_type": job["type"],
                "source_url": job["url"],
                "start_time": req.start,
                "end_time": req.end,
                "caption_style": req.caption_style,
            },
        },
        timeout=30,
    )
    if r.status_code not in (200, 201, 204):
        raise HTTPException(500, f"GitHub Actions trigger failed: {r.text[:250]}")

    return {"render_id": render_id, "status": "queued"}


@app.get("/api/render-status/{render_id}")
def render_status(render_id: str):
    data = supa_get_json(f"status/{render_id}.json")
    if not data:
        return {"render_id": render_id, "status": "queued", "message": "Preparing renderer..."}
    return data
    
