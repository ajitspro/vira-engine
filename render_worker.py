import os
import sys
import json
import glob
import subprocess
import requests
from pathlib import Path
from groq import Groq

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET = os.environ.get("SUPABASE_BUCKET", "vira-clips")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
YT_COOKIES = os.environ.get("YT_COOKIES", "").strip()

RENDER_ID = os.environ["RENDER_ID"]
SOURCE_TYPE = os.environ.get("SOURCE_TYPE", "yt")
SOURCE_URL = os.environ["SOURCE_URL"]
START_TIME = os.environ["START_TIME"]
END_TIME = os.environ["END_TIME"]
CAPTION_STYLE = os.environ.get("CAPTION_STYLE", "bold_yellow")


def upload_bytes(path: str, data: bytes, content_type: str):
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    r = requests.post(url, headers=headers, data=data, timeout=180)
    r.raise_for_status()
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"


def set_status(status: str, message: str = "", download_url: str = ""):
    payload = {
        "render_id": RENDER_ID,
        "status": status,
        "message": message,
        "download_url": download_url,
    }
    upload_bytes(
        f"status/{RENDER_ID}.json",
        json.dumps(payload).encode("utf-8"),
        "application/json",
    )


def hms_to_sec(t: str) -> float:
    parts = [float(x) for x in t.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass(words, style_name: str, out_path: Path):
    styles = {
        "bold_yellow": ("&H0000E5FF", "&H00000000", 72, 6),
        "neon_glow": ("&H00B672F4", "&H00200820", 70, 6),
        "clean_white": ("&H00FFFFFF", "&H00000000", 64, 4),
    }
    primary, outline, fsize, owidth = styles.get(style_name, styles["bold_yellow"])

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{fsize},{primary},&H00FFFFFF,{outline},&H96000000,-1,0,0,0,100,100,0,0,1,{owidth},3,2,80,80,360,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    chunk = []
    for w in words:
        chunk.append(w)
        if len(chunk) >= 3 or (chunk[-1]["end"] - chunk[0]["start"] >= 1.2):
            st = ass_time(chunk[0]["start"])
            en = ass_time(max(chunk[-1]["end"], chunk[0]["start"] + 0.25))
            txt = " ".join(x["word"].strip() for x in chunk).upper().replace("{", "").replace("}", "")
            if txt:
                events.append(f"Dialogue: 0,{st},{en},Default,,0,0,0,,{txt}")
            chunk = []
    if chunk:
        st = ass_time(chunk[0]["start"])
        en = ass_time(max(chunk[-1]["end"], chunk[0]["start"] + 0.25))
        txt = " ".join(x["word"].strip() for x in chunk).upper().replace("{", "").replace("}", "")
        if txt:
            events.append(f"Dialogue: 0,{st},{en},Default,,0,0,0,,{txt}")

    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def generate_captions(video_path: str, ass_path: Path) -> bool:
    if not GROQ_API_KEY:
        return False
    try:
        audio_path = Path("clip_audio.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(audio_path)],
            check=True,
        )
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as af:
            tr = client.audio.transcriptions.create(
                file=(audio_path.name, af.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        words = getattr(tr, "words", None) or []
        if not words:
            return False
        norm = [
            {"word": w["word"], "start": w["start"], "end": w["end"]}
            if isinstance(w, dict)
            else {"word": w.word, "start": w.start, "end": w.end}
            for w in words
        ]
        build_ass(norm, CAPTION_STYLE, ass_path)
        return True
    except Exception as e:
        print(f"Caption warning (continuing without captions): {e}")
        return False


def main():
    try:
        set_status("rendering", "Fetching 1080p clip segment...")
        raw_base = "raw_clip"

        if SOURCE_TYPE == "yt":
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
                "--download-sections", f"*{START_TIME}-{END_TIME}",
                "--force-keyframes-at-cuts",
                "--merge-output-format", "mp4",
                "-o", f"{raw_base}.%(ext)s",
            ]
            if YT_COOKIES:
                Path("cookies.txt").write_text(YT_COOKIES, encoding="utf-8")
                cmd.extend(["--cookies", "cookies.txt"])
            cmd.append(SOURCE_URL)
            subprocess.run(cmd, check=True)
            matches = glob.glob(f"{raw_base}.*")
            if not matches:
                raise RuntimeError("yt-dlp did not produce a clip file.")
            raw_file = matches[0]
        else:
            full_src = "uploaded_src.mp4"
            with requests.get(SOURCE_URL, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(full_src, "wb") as f:
                    for ch in r.iter_content(chunk_size=1024 * 1024):
                        f.write(ch)
            dur = max(5.0, hms_to_sec(END_TIME) - hms_to_sec(START_TIME))
            raw_file = f"{raw_base}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", START_TIME, "-i", full_src, "-t", str(dur), "-c", "copy", raw_file],
                check=True,
            )

        set_status("rendering", "Generating auto-captions & rendering 1080p 9:16...")
        ass_path = Path("captions.ass")
        has_caps = generate_captions(raw_file, ass_path)

        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
        if has_caps:
            vf += ",ass=captions.ass"

        out_mp4 = Path(f"{RENDER_ID}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_file,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                str(out_mp4),
            ],
            check=True,
        )

        set_status("rendering", "Uploading 1080p MP4 to cloud...")
        video_url = upload_bytes(f"clips/{RENDER_ID}.mp4", out_mp4.read_bytes(), "video/mp4")
        set_status("done", "1080p clip ready!", video_url)
        print("SUCCESS:", video_url)

    except Exception as e:
        err_msg = str(e)[:400]
        print("ERROR:", err_msg, file=sys.stderr)
        try:
            set_status("error", err_msg)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
  
