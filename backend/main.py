from __future__ import annotations

import base64
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
MAX_FRAMES = int(os.getenv("MAX_INSPECTION_FRAMES", "8"))
GEMINI_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
REQUESTED_PROVIDER = os.getenv("AI_PROVIDER", "auto").strip().lower()
if REQUESTED_PROVIDER not in {"auto", "gemini", "openai"}:
    raise RuntimeError("AI_PROVIDER must be 'auto', 'gemini', or 'openai'.")
if REQUESTED_PROVIDER == "auto":
    ACTIVE_PROVIDER = "gemini" if GEMINI_API_KEY else ("openai" if OPENAI_API_KEY else None)
else:
    ACTIVE_PROVIDER = REQUESTED_PROVIDER
client = OpenAI(api_key=OPENAI_API_KEY) if ACTIVE_PROVIDER == "openai" and OPENAI_API_KEY else None
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

app = FastAPI(title="RenovateSense AI", version="0.1.0")
app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")


class Defect(BaseModel):
    category: str = Field(min_length=2, max_length=80)
    severity: str = Field(min_length=2, max_length=30)
    confidence: float = Field(ge=0, le=1)
    location: str = Field(min_length=2, max_length=160)
    evidence: str = Field(min_length=2, max_length=500)
    estimated_area_sqm: float | None = Field(default=None, ge=0, le=10000)
    next_step: str = Field(min_length=2, max_length=300)
    professional_review: bool = True


class ScopePhase(BaseModel):
    phase: str = Field(min_length=2, max_length=100)
    tasks: list[str] = Field(min_length=1, max_length=8)
    depends_on: list[str] = Field(default_factory=list, max_length=8)


class VisualReview(BaseModel):
    summary: str = Field(min_length=2, max_length=700)
    risk_level: str = Field(min_length=2, max_length=30)
    visual_limitations: list[str] = Field(default_factory=list, max_length=8)
    defects: list[Defect] = Field(default_factory=list, max_length=20)
    scope: list[ScopePhase] = Field(default_factory=list, max_length=12)


def extract_keyframes(video_bytes: bytes, suffix: str) -> tuple[list[dict[str, Any]], float]:
    """Sample evenly spaced JPEG keyframes from an uploaded video."""
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise HTTPException(status_code=422, detail="This video could not be opened. Try an MP4 or MOV file.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
        if duration > 180:
            raise HTTPException(status_code=413, detail="Please use a walkthrough video under 3 minutes.")

        target_count = max(2, min(MAX_FRAMES, 8))
        frames: list[dict[str, Any]] = []
        if frame_count > 0:
            indexes = sorted({round(i * (frame_count - 1) / (target_count - 1)) for i in range(target_count)})
            for index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    continue
                frames.append(_encode_frame(frame, index / fps if fps > 0 else 0, index))
        else:
            interval = max(1, round((fps or 1) * 3))
            index = 0
            while len(frames) < target_count:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % interval == 0:
                    frames.append(_encode_frame(frame, index / fps if fps > 0 else float(index), index))
                index += 1
        capture.release()
        if not frames:
            raise HTTPException(status_code=422, detail="No readable frames were found in this video.")
        return frames, duration
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _encode_frame(frame: np.ndarray, seconds: float, index: int) -> dict[str, Any]:
    height, width = frame.shape[:2]
    max_width = 1280
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ok:
        raise HTTPException(status_code=422, detail="A video frame could not be encoded.")
    return {
        "timestamp_seconds": round(seconds, 1),
        "frame_index": index,
        "width": width,
        "height": height,
        "data_url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def review_frames(frames: list[dict[str, Any]], property_type: str, area_sqm: float, location: str) -> VisualReview:
    if ACTIVE_PROVIDER is None or (ACTIVE_PROVIDER == "gemini" and not GEMINI_API_KEY) or (ACTIVE_PROVIDER == "openai" and client is None):
        key_name = "GEMINI_API_KEY" if REQUESTED_PROVIDER == "gemini" or (REQUESTED_PROVIDER == "auto" and GEMINI_API_KEY) else "OPENAI_API_KEY"
        raise HTTPException(
            status_code=503,
            detail=f"Keyframes were sampled, but visual review is not configured yet. Add {key_name} to the project .env file and restart the API.",
        )

    prompt = (
        "Review these sampled frames from a pre-renovation walkthrough. Treat them only as visual evidence. "
        "Do not claim to determine structural safety, hidden conditions, mold species, or electrical compliance. "
        "Use cautious language and mark uncertain observations as uncertain. Avoid duplicate findings across frames. "
        "Only report visible issues among dampness/mold-like staining, cracks, peeling plaster/paint, uneven flooring, "
        "and outdated or visibly damaged electrical panels; include other clearly visible surface defects if useful. "
        "A crack cannot be called structural from a video alone: recommend an on-site qualified professional review "
        "when crack width/pattern or context is concerning. Estimate visible affected area in square metres only when "
        "there is enough visual context; otherwise use null. Give a concise, dependency-ordered repair scope. "
        "Return only a JSON object with keys: summary, risk_level (low/moderate/high/urgent/uncertain), "
        "visual_limitations (array of strings), defects (array of objects with category, severity (cosmetic/moderate/serious/uncertain), "
        "confidence (0..1), location, evidence, estimated_area_sqm (number or null), next_step, professional_review (boolean)), "
        "scope (array of objects with phase, tasks (array of strings), depends_on (array of phase names)). "
        f"Property type: {property_type}. Approximate floor area: {area_sqm:g} m². Location context: {location or 'not provided'}."
    )
    try:
        if ACTIVE_PROVIDER == "gemini":
            raw = _review_with_gemini(prompt, frames)
        else:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for frame in frames:
                content.append({"type": "image_url", "image_url": {"url": frame["data_url"], "detail": "low"}})
            completion = client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a cautious renovation visual triage assistant. A video review is not an engineering inspection."},
                    {"role": "user", "content": content},
                ],
                max_tokens=2200,
            )
            raw = completion.choices[0].message.content or "{}"
        return VisualReview.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=502, detail="The visual review returned an incomplete result. Please retry with a clearer walkthrough.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
        if ACTIVE_PROVIDER == "openai" and code in {"credit_balance_exhausted", "insufficient_quota"}:
            detail = "OpenAI API credits are exhausted for this account. Add API credits or use eligible account credits, then retry. The API key itself is valid."
        elif ACTIVE_PROVIDER == "openai" and status == 429:
            detail = "The OpenAI API request limit was reached. Check the project rate limits and retry shortly."
        elif ACTIVE_PROVIDER == "openai" and status == 401:
            detail = "OpenAI rejected the API key. Check that the full secret key is saved in the project .env file."
        elif ACTIVE_PROVIDER == "openai" and status == 403:
            detail = "This API project does not have permission to use the requested model. Check the project's model access."
        elif ACTIVE_PROVIDER == "openai" and status == 404:
            detail = "The requested model is unavailable to this API project. Check the configured model name and access."
        elif ACTIVE_PROVIDER == "gemini":
            detail = "Google Gemini visual review failed. Check the key, model access, and AI Studio quota, then retry."
        else:
            detail = "The visual review request failed. Check the selected API provider's key, model access, and usage limits, then retry."
        raise HTTPException(status_code=502, detail=detail) from exc


def _review_with_gemini(prompt: str, frames: list[dict[str, Any]]) -> str:
    """Send sampled keyframes to Gemini's generateContent REST endpoint."""
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for frame in frames:
        encoded = frame["data_url"].partition(",")[2]
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": encoded}})

    payload = {
        "systemInstruction": {
            "parts": [{"text": "You are a cautious renovation visual triage assistant. A video review is not an engineering inspection."}]
        },
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 2200,
            "temperature": 0.2,
        },
    }
    model = urllib.parse.quote(GEMINI_MODEL, safe="")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            error_body = json.loads(exc.read().decode("utf-8"))
            google_error = error_body.get("error", {})
        except (json.JSONDecodeError, UnicodeDecodeError):
            google_error = {}
        status = google_error.get("status", "")
        message = google_error.get("message", "")
        if exc.code == 429 or status == "RESOURCE_EXHAUSTED":
            detail = "The Gemini free-tier usage limit was reached. Check your AI Studio rate limits or try again later."
        elif exc.code in {401, 403} or "API_KEY_INVALID" in status or "API key not valid" in message:
            detail = "Google Gemini rejected the key. Check that GEMINI_API_KEY contains the complete API key from AI Studio."
        elif exc.code == 404:
            detail = "The configured Gemini model is unavailable. Check GEMINI_VISION_MODEL and your AI Studio model access."
        else:
            detail = "Google Gemini rejected the visual review request. Check the key, model access, and AI Studio project settings."
        raise HTTPException(status_code=502, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to Google Gemini. Check your internet connection and retry.") from exc

    candidates = result.get("candidates", [])
    if not candidates:
        raise HTTPException(status_code=502, detail="Google Gemini returned no visual review. Try a clearer or shorter walkthrough.")
    response_parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in response_parts if isinstance(part, dict)) or "{}"


def estimate_plan(review: VisualReview, property_area: float, currency: str) -> dict[str, Any]:
    """Apply transparent starter rates; replace these with local contractor data."""
    rates = {
        "dampness": (4, 9, 160, 520, 2),
        "mold-like staining": (4, 9, 160, 520, 2),
        "crack": (2, 6, 90, 320, 0),
        "plaster": (3, 7, 120, 360, 1),
        "peeling": (2, 5, 90, 260, 0),
        "floor": (4, 8, 220, 720, 0),
        "electrical": (6, 16, 450, 1500, 0),
        "default": (3, 8, 120, 420, 0),
    }
    factor = {"cosmetic": 0.8, "moderate": 1.0, "serious": 1.45, "uncertain": 1.15}
    labor_low = labor_high = cost_low = cost_high = buffer_days = 0
    for defect in review.defects:
        label = defect.category.lower()
        key = next((name for name in rates if name in label), "default")
        low_h, high_h, low_cost, high_cost, buffer = rates[key]
        affected_area = defect.estimated_area_sqm
        if key == "electrical":
            scale = 1
        else:
            scale = min(30, max(1, affected_area if affected_area is not None else property_area * 0.025))
        severity_scale = factor.get(defect.severity.lower(), 1.0)
        labor_low += round(low_h * scale * severity_scale)
        labor_high += round(high_h * scale * severity_scale)
        cost_low += round(low_cost * scale * severity_scale)
        cost_high += round(high_cost * scale * severity_scale)
        buffer_days += buffer
    return {
        "currency": currency,
        "labor_hours": {"low": labor_low, "high": labor_high},
        "material_cost": {"low": cost_low, "high": cost_high},
        "duration_days": {"low": max(0, (labor_low + 7) // 8 + buffer_days), "high": max(0, (labor_high + 7) // 8 + buffer_days + 2)},
        "basis": "Illustrative starter rates multiplied by visually estimated affected area; excludes access, permits, hidden damage, and local tax.",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    configured = (ACTIVE_PROVIDER == "gemini" and bool(GEMINI_API_KEY)) or (ACTIVE_PROVIDER == "openai" and client is not None)
    return {"status": "ok", "visual_review_configured": configured, "visual_review_provider": ACTIVE_PROVIDER}


@app.post("/api/inspect")
async def inspect_video(
    video: UploadFile = File(...),
    property_type: str = Form("Apartment"),
    area_sqm: float = Form(90, gt=5, le=10000),
    location: str = Form(""),
) -> dict[str, Any]:
    suffix = Path(video.filename or "walkthrough.mp4").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, M4V, AVI, MKV, or WEBM walkthrough.")
    if area_sqm <= 0:
        raise HTTPException(status_code=422, detail="Floor area must be greater than zero.")
    video_bytes = await video.read(MAX_UPLOAD_BYTES + 1)
    if not video_bytes:
        raise HTTPException(status_code=422, detail="The selected video is empty.")
    if len(video_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"The video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")

    frames, duration = extract_keyframes(video_bytes, suffix)
    review = review_frames(frames, property_type[:80], area_sqm, location[:120])
    estimate = estimate_plan(review, area_sqm, "AED")
    return {
        "filename": video.filename or "walkthrough",
        "frame_count": len(frames),
        "duration_seconds": round(duration, 1),
        "frames": [{key: value for key, value in frame.items() if key != "data_url"} | {"preview": frame["data_url"]} for frame in frames],
        "review": review.model_dump(),
        "estimate": estimate,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")
