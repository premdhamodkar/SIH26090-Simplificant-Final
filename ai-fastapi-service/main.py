"""
Visual Forge — AI Image Enhancement Microservice
--------------------------------------------------
FastAPI service that:
  1. Strips background via rembg
  2. Normalizes lighting/contrast via OpenCV and composites onto a white background
  3. Uploads the final image to Cloudinary
  4. Returns a clean JSON payload with the hosted URL

Phase Two adds:
  5. POST /api/catalog-audio — Dynamic Pricing Assistant. Accepts a seller's
     multilingual (Hindi/Marathi/English) voice note AND a Cloudinary product
     image URL, processes both simultaneously with the google-genai SDK
     (Gemini 3.6 Flash) to produce structured JSON catalog data with a
     visually + acoustically informed price recommendation.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2

Env vars required:
    CLOUDINARY_CLOUD_NAME
    CLOUDINARY_API_KEY
    CLOUDINARY_API_SECRET
    GEMINI_API_KEY
"""

import asyncio
import base64
import functools
import io
import logging
import os
import secrets
import tempfile
import time
import uuid
from typing import Any, List, Optional, Tuple, Type, Union

import cv2
import httpx
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from rembg import remove, new_session
import cloudinary
import cloudinary.uploader
from cloudinary.exceptions import Error as CloudinaryError
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError as GeminiAPIError
import groq
from groq import AsyncGroq

load_dotenv()

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visual_forge")

# --------------------------------------------------------------------------
# Config — Image pipeline (unchanged)
# --------------------------------------------------------------------------
MAX_UPLOAD_SIZE_MB = 12
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
CLOUDINARY_UPLOAD_FOLDER = "visual-forge/enhanced"

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True,
)

# rembg session is created once at startup and reused across requests —
# creating it per-request reloads the ONNX model every time, which is slow.
#
# Model choice matters here: "u2net" (the rembg default) is a solid general
# segmenter but is biased toward a single dominant salient subject. On
# product photos where a person holds a large secondary object (e.g. an
# artisan holding a product), u2net frequently keeps the person and
# misclassifies the product as background — exactly the failure mode this
# service exists to prevent. "birefnet-general" is rembg's current
# highest-quality general-purpose model and handles multi-object /
# competing-foreground compositions far more reliably. Tradeoff: it's a much
# larger model (~928MB vs ~168MB) with higher memory and per-inference
# latency, so it's worth confirming your deployment has the headroom.
# Overridable via env var so this can be tuned without a code change.
REMBG_MODEL_NAME = os.environ.get("REMBG_MODEL_NAME", "birefnet-general")
_REMBG_SESSION = None

# Secondary segmentation model used as a safety net when the primary model
# produces a degenerate mask (Order C). "isnet-general-use" is architecturally
# distinct from the primary — when one model misses or over-keeps the subject,
# the other frequently succeeds. Overridable via env var.
FALLBACK_REMBG_MODEL_NAME = os.environ.get("FALLBACK_REMBG_MODEL_NAME", "isnet-general-use")
_REMBG_FALLBACK_SESSION = None

# Degenerate-mask detection: a segmentation that keeps almost nothing of the
# frame (< 3% foreground) almost certainly missed the product entirely; one
# that keeps almost everything (> 98%) never separated subject from
# background. Both are red flags that trigger the fallback model. Healthy
# segmentations sit near a ~30% foreground share — used to pick the lesser of
# two evils when both models fail.
DEGENERATE_MIN_FOREGROUND_FRACTION = 0.03
DEGENERATE_MAX_FOREGROUND_FRACTION = 0.98
HEALTHY_MID_FOREGROUND_FRACTION = 0.3

# Fixed lighting constants — documented fallback used ONLY when the per-photo
# adaptive computation (compute_adaptive_brightness_contrast) fails. The old
# one-size-fits-all nudge was replaced by per-photo histogram stretching but
# must remain available so a lighting-computation failure can never crash the
# whole enhance pipeline (Order B).
DEFAULT_CONTRAST_ALPHA = 1.12
DEFAULT_BRIGHTNESS_BETA = 15

# CLAHE defaults — local luminance contrast enhancement (isolated helper,
# Phase 1; pipeline integration is a separate phase).
# clipLimit=2.0 is a deliberately conservative contrast cap (higher values
# amplify JPEG/phone-photo noise), and tileGridSize=(8, 8) is OpenCV's own
# default — both conform to the house "no aggressive processing" style.
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)


# Low-resolution upscale (e.g. the 296x435 benchmark case): rembg and its
# alpha matting are far more stable — and fine hair strands resolve far better —
# when the model sees a larger frame, and the final gigabytes of output blur
# disappear too. Lanczos-4 is the highest-quality resampler that does NOT
# invent detail (unlike CNN/super-resolution upsamplers), matching the "no
# obvious artificial detail" requirement. Applied only when the short edge is
# clearly too small.
UPSCALE_TARGET_MIN_EDGE = 900
UPSCALE_MAX_FACTOR = 2.0

# Pre-segmentation downscale (Order P1): alpha_matting=True runs pymatting's
# closed-form solver over the FULL frame — its cost scales with pixel count,
# not with anything about the product photo itself, which is the near-certain
# cause of the 1-15 minute timing variance across artisan uploads. Capping the
# longer dimension here makes processing time fast AND consistent. 1800 is
# chosen because Order D (standardize_canvas) already targets a ~2000px final
# canvas — matting at a higher resolution than the output will be resized to
# anyway is genuinely wasted computation, not a quality/speed tradeoff. Keep
# this tunable via the constant, not buried inline.
DOWNSCALE_MAX_PROCESSING_DIMENSION = 1800

# Segmentation-only downscale (Order Q1): decouples the MASK resolution from
# the COMPOSITE resolution. The alpha mask rembg derives doesn't need 1800px
# to be accurate — only the final composited output does. So strip_background()
# feeds rembg a further-bounded frame (this constant), then cheaply upscales
# just the resulting alpha channel back to the working image's dimensions. The
# AI/matting step is the wall-clock hog; bounding it here is pure speed with no
# resolution lost downstream. Keep tunable via the constant, not inline.
SEGMENTATION_MAX_DIMENSION = 1200

# Matting refinement for the halo/gray-remnant benchmark cases:
#   - kill translucent low-saturation white FOG (gray/dirty patches) outright,
#   - drop tiny detached foreground islands (compression speckles),
#   - then defringe the semi-transparent edge band by INPAINTING its colors
#     from fully-opaque subject pixels only — removing white/gray halos without
#     cutting into the object (the critical pot/hair requirement).
MASK_FOG_ALPHA_MAX = 150
MASK_FOG_MAX_SATURATION = 20
MASK_FOG_MIN_BRIGHTNESS = 200
MASK_DROP_COMPONENT_AREA_FRACTION = 0.001
DEFRINGE_INPAINT_RADIUS = 3

# --------------------------------------------------------------------------
# Config — Audio cataloger pipeline (new)
# --------------------------------------------------------------------------
MAX_AUDIO_SIZE_MB = 25
ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
    "video/mp4",  # mobile voice recordings are frequently containerized as video/mp4
}

# Client uploaders (browsers, Swagger UI, some mobile OSes) infer Content-Type
# from file extension only — they never inspect the actual stream. A WhatsApp
# voice note exported as "recording.mp4" is genuinely audio-only (AAC, no
# video track) but arrives labeled "video/mp4". If we forward that label to
# Gemini verbatim, its backend attempts VIDEO processing on a file with no
# visual stream and the upload lands in FAILED state. Since we already treat
# "video/mp4" as an accepted *alias* for an audio voice note at the
# validation layer above, we must also correct the label before it reaches
# Gemini — the client's mistaken MIME type should never leak into the AI call.
GEMINI_AUDIO_MIME_OVERRIDE = {
    "video/mp4": "audio/mp4",
}

GEMINI_MODEL_NAME = "gemini-3.6-flash"

# File state polling — client.files.upload() returns before Google's backend
# has finished processing the file into ACTIVE state; querying too soon
# raises 400 FAILED_PRECONDITION on generate_content().
FILE_ACTIVE_POLL_INTERVAL_SECONDS = 0.5
FILE_ACTIVE_POLL_TIMEOUT_SECONDS = 60.0

# Image acquired via URL (not upload) for the pricing pipeline.
MAX_IMAGE_DOWNLOAD_SIZE_MB = 15
ALLOWED_IMAGE_DOWNLOAD_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 15.0

_genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --------------------------------------------------------------------------
# Config — AI provider selection (Gemini or Groq)
# --------------------------------------------------------------------------
# AI_PROVIDER toggles the /api/catalog-audio backend:
#   "gemini" -> google-genai SDK (Gemini 3.6 Flash): transient Files API
#               upload/ACTIVE-poll, native multi-modal generate_content with
#               response_schema for schema-enforced JSON.
#   "groq"   -> Groq SDK: Whisper transcription + Qwen 3.6 27B vision chat
#               (single-shot request/response — no Files API lifecycle).
# Default is "groq": the free-tier Gemini quota (20 req/day) is too small for
# active development, while Groq's free tier is far more generous; a cheap,
# long-lived provider is the sensible default for a cataloging endpoint that
# consumes 2 LLM calls per request. The external API contract is byte-for-byte
# identical regardless of provider — callers never see which backend produced
# the answer.
AI_PROVIDER = os.environ.get("AI_PROVIDER", "groq").lower().strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Get a free key at console.groq.com — used for the 'groq' AI_PROVIDER option.
# NOTE: GROQ_VISION_MODEL must be a model currently on Groq's live model list
# that accepts image input (console.groq.com/docs/vision), and model ID strings
# on Groq change as models move out of Preview/deprecate. As of Sep 2026 the
# vision-capable models are the Qwen 3.6/3.8 27B pair (the former Llama-4
# Scout/Maverick IDs were deprecated and return 404) — verified live against
# the API below. Treat this string as the current best value, not a guarantee.
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

# Downscale images before inlining them into the Groq vision message. Groq's
# vision models cap per-request image payload size (Qwen: ~20 MB, and a base64
# inline payload is ~1.37x the raw file size), so a large Cloudinary photo can
# blow past the cap and be rejected with a 400. Downscaling to 1200px on the
# longest side keeps that payload small while preserving the finish/condition
# detail the pricing stage actually needs, and trims token cost (Groq charges
# 2048 tokens per image regardless of size).
# FREE-TIER FRIENDLY OUTPUT CAP: Groq's free tier enforces an output-tokens-
# per-minute (OTPM) limit (currently 1000) and rejects requests whose expected
# output exceeds it before generation even starts. The Qwen vision models run
# in thinking mode by default, which massively inflates the output-token
# estimate, so groq_structured_chat disables reasoning and caps completion
# tokens well under the free-tier budget. A truncated JSON response still
# degrades gracefully via the existing schema-validation 502 path.
GROQ_MAX_OUTPUT_TOKENS = 800
GROQ_IMAGE_MAX_DIMENSION = 1200

_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --------------------------------------------------------------------------
# Config — Security & Gemini call resilience
# --------------------------------------------------------------------------
# Routes are protected by an X-API-Key header mapped to this env var. When the
# variable is unset the protected routes fail closed with 503 (explicit
# misconfiguration) rather than silently opening the API. /health stays open
# for infrastructure probes. Loaded ONCE at startup; 'secrets.compare_digest'
# keeps the comparison constant-time.
X_API_KEY = os.environ.get("X_API_KEY", "")

# Transient-failure retry for Gemini calls (Files API upload + generate_content).
# Google's public API occasionally returns 429/5xx and raw transport blips; a
# handful of bounded, backoff'd retries absorbs those WITHOUT rebooting the
# request. Permanent errors (401 bad key, 400 FAILED_PRECONDITION, 404) are
# NEVER retried — they will not heal.
API_RETRY_ATTEMPTS = 3
API_RETRY_BASE_DELAY_SECONDS = 1.0
API_RETRY_MAX_DELAY_SECONDS = 8.0

app = FastAPI(title="Visual Forge", version="1.0.0")


# --------------------------------------------------------------------------
# Security — X-API-Key guard
# --------------------------------------------------------------------------
def verify_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Route dependency enforcing the X-API-Key header mapped to .env.

    Fails closed: if the server key is unset the route returns 503 (explicit
    misconfiguration, actionable) instead of opening access. A missing or
    wrong header returns 401. Comparison is constant-time via
    secrets.compare_digest, so header values can't be timing-attacked.
    """
    if not X_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server API key not configured (set X_API_KEY in .env).",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, X_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# --------------------------------------------------------------------------
# Gemini call resilience — bounded retry on transient failures only
# --------------------------------------------------------------------------
def _is_transient_gemini_error(exc: Exception) -> bool:
    """
    Classifies an exception as retryable (transient) or not.

    Retryable: HTTP 429 (rate limit) and 5xx from Gemini, plus transport
    blips (timeouts, connection resets, protocol errors) — these can heal on
    their own. NOT retryable: 401 (invalid/disabled key), 400
    FAILED_PRECONDITION (semantic — e.g. a file not yet ACTIVE), 404 and any
    other application error — retrying them is wasted latency.
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, GeminiAPIError):
        code = getattr(exc, "code", None)
        try:
            code = int(code)
        except (TypeError, ValueError):
            return False
        return code == 429 or code >= 500
    return False


async def run_google_call(operation, *, label: str):
    """
    Runs a (synchronous) Gemini SDK operation with bounded exponential-backoff
    retries on transient failures only.

    WHY this exists: Google's public API returns transient 429/5xx and inline
    transport errors under load; without a retry a single blip turns a
    legitimate catalog into a failed request. The operation runs via
    asyncio.to_thread so neither the call nor the backoff sleeps block the
    event loop (matching the asyncio.sleep discipline used by the file-state
    poller). Permanent errors are re-raised immediately on their first attempt.
    """
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            return await asyncio.to_thread(operation)
        except Exception as exc:
            if not _is_transient_gemini_error(exc) or attempt >= API_RETRY_ATTEMPTS:
                raise
            delay = min(
                API_RETRY_MAX_DELAY_SECONDS,
                API_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
            )
            logger.warning(
                "Gemini %s attempt %d/%d failed (%s); retrying in %.1fs",
                label,
                attempt,
                API_RETRY_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


async def run_google_call_async(operation, *, label: str):
    """
    Async variant of run_google_call for coroutines produced by
    _genai_client.aio.* methods. Identical retry semantics — bounded
    exponential backoff on transient errors, immediate raise on permanent
    ones. Used by the catalog-audio route to avoid blocking the event loop
    with synchronous SDK calls while preserving the same resilience guarantees.
    """
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            return await operation()
        except Exception as exc:
            if not _is_transient_gemini_error(exc) or attempt >= API_RETRY_ATTEMPTS:
                raise
            delay = min(
                API_RETRY_MAX_DELAY_SECONDS,
                API_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
            )
            logger.warning(
                "Gemini %s attempt %d/%d failed (%s); retrying in %.1fs",
                label,
                attempt,
                API_RETRY_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


# --------------------------------------------------------------------------
# Groq call resilience — bounded retry on transient failures only
# --------------------------------------------------------------------------
def _is_transient_groq_error(exc: Exception) -> bool:
    """
    Classifies a Groq SDK exception as retryable (transient) or not.

    Retryable: HTTP 429 (rate limit), 5xx from Groq, and transport blips
    (connection errors, request timeouts) — these can heal on their own. NOT
    retryable: 401 (invalid/disabled key), 400 (bad request — e.g. a
    rejected response_format or unsupported parameter), 404 and any other
    application error. status_code is an INSTANCE attribute on the OpenAI-style
    exceptions, so it is read defensively with getattr rather than assumed.
    """
    if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
        return True
    if isinstance(exc, groq.RateLimitError):
        return True
    if isinstance(exc, groq.APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code is not None and status_code >= 500
    return False


async def run_groq_call_async(operation, *, label: str):
    """
    Runs a Groq SDK coroutine with the same bounded exponential-backoff retry
    semantics as run_google_call_async (Order 2). A single 429/5xx/transport
    blip therefore cannot turn a legitimate catalog request into a failure,
    while permanent errors (bad key, 400) surface immediately on attempt 1.
    Different SDK families throw different exception shapes, so the retryable
    classification comes from _is_transient_groq_error (Groq's OpenAI-style
    hierarchy) rather than the Gemini/httpx-specific one.
    """
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            return await operation()
        except Exception as exc:
            if not _is_transient_groq_error(exc) or attempt >= API_RETRY_ATTEMPTS:
                raise
            delay = min(
                API_RETRY_MAX_DELAY_SECONDS,
                API_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
            )
            logger.warning(
                "Groq %s attempt %d/%d failed (%s); retrying in %.1fs",
                label,
                attempt,
                API_RETRY_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


def _parse_structured(
    response: Any,
    model: Type[BaseModel],
    label: str,
) -> Any:
    """
    Structured-parse a Gemini generate_content response, with a manual fallback
    when the SDK could not coerce .parsed (schema drift must never fail
    silently). Prefer response.parsed (SDK-validated Pydantic), otherwise
    model_validate_json(response.text); on failure log the payload and raise
    HTTP 502. Returns the validated Pydantic instance.
    """
    parsed = response.parsed
    if parsed is not None:
        return parsed
    try:
        return model.model_validate_json(response.text)
    except Exception as parse_exc:
        logger.error("%s response failed schema validation: %s", label, response.text)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model returned malformed {label} data: {parse_exc}",
        )


# --------------------------------------------------------------------------
# Groq helpers — Whisper transcription + vision structured chat
# --------------------------------------------------------------------------
def _image_to_base64_data_uri(local_image_path: str, content_type: str) -> str:
    """
    Reads the downloaded local image and encodes it as a base64 data URI for
    Groq's OpenAI-compatible vision API.

    Groq has no upload/Files lifecycle (unlike Gemini), so the image bytes are
    embedded inline in the chat message. Before encoding, the image is
    downscaled to at most GROQ_IMAGE_MAX_DIMENSION on its longest side:
    WITHOUT that, a large Cloudinary photo (up to the 15 MB download cap) would
    base64-encode (1.37x) to an inline payload exceeding Groq's per-request
    image size limit and be rejected with a 400. 1200px preserves the
    finish/condition detail the pricing stage needs while cutting token cost
    (Groq charges a flat 2048 tokens per image).
    """
    output_format = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }.get(content_type, "PNG")
    with Image.open(local_image_path) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if max(img.size) > GROQ_IMAGE_MAX_DIMENSION:
            scale = GROQ_IMAGE_MAX_DIMENSION / max(img.size)
            new_size = (
                max(1, round(img.width * scale)),
                max(1, round(img.height * scale)),
            )
            img = img.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format=output_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


async def transcribe_audio_groq(local_audio_path: str) -> str:
    """
    Transcribes the artisan's voice note via Groq-hosted Whisper.

    Returns the raw transcript text (response_format="text"). Language is
    deliberately left UNSET (auto-detect): the target audio is often
    code-mixed Hindi/Marathi/English, and force-setting a single ISO-639-1
    language can hurt accuracy on code-switched speech more than it helps. The
    file is embedded in the request as (basename, bytes) — Groq does not need
    a persistent upload.
    """
    with open(local_audio_path, "rb") as audio_file:
        transcription = await run_groq_call_async(
            lambda: _groq_client.audio.transcriptions.create(
                model=GROQ_WHISPER_MODEL,
                file=(os.path.basename(local_audio_path), audio_file.read()),
                response_format="text",
            ),
            label="whisper transcription",
        )
    # response_format="text" returns a bare string; a ChatCompletion-style
    # wrapper is returned for other formats — normalize both defensively.
    return transcription if isinstance(transcription, str) else transcription.text


async def groq_structured_chat(
    prompt_text: str,
    image_base64_data_uri: str,
    response_model: Type[BaseModel],
    label: str,
) -> BaseModel:
    """
    Sends a text+image message to Groq's vision model and gets validated JSON.

    Groq's OpenAI-compatible API does NOT offer Gemini's schema-constrained
    .parsed shortcut — response_format={"type": "json_object"} only guarantees
    valid JSON syntax, not schema conformance. So the PROMPT itself carries the
    exact field names/types (see GROQ_VERIFICATION_PROMPT /
    GROQ_PRICING_PROMPT) and the raw text is validated with Pydantic afterward
    via model_validate_json. On validation failure the raw payload is logged
    and HTTP 502 is raised — the identical failure-handling philosophy to the
    Gemini `_parse_structured` helper, so both providers surface malformed
    model output the same way.

    Output shaping: reasoning_effort="none" disables the Qwen vision model's
    default thinking mode for this deterministic JSON task (a fixed schema needs
    no chain-of-thought, and thinking massively inflates output tokens against
    Groq's free-tier OTPM budget), and max_completion_tokens caps the response.
    """
    response = await run_groq_call_async(
        lambda: _groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_base64_data_uri},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            reasoning_effort="none",
            max_completion_tokens=GROQ_MAX_OUTPUT_TOKENS,
        ),
        label=label,
    )
    raw_text = response.choices[0].message.content
    try:
        return response_model.model_validate_json(raw_text)
    except Exception as parse_exc:
        logger.error("%s response failed schema validation: %s", label, raw_text)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model returned malformed {label} data: {parse_exc}",
        )


def _safe_log_timing(route_name: str, **elapsed: float) -> None:
    """
    Emit one INFO summary line of per-stage timing for a request (Order P2).

    Purely additive observability: durations come from time.monotonic() (which
    never jumps with system clock changes). A formatting/user error here must
    NEVER fail a request, so the build of the line is guarded and degrades to
    a DEBUG note instead of raising.
    """
    try:
        parts = " ".join(f"{stage}={duration:.2f}s" for stage, duration in elapsed.items())
        logger.info("%s timing: %s", route_name, parts)
    except Exception:
        logger.debug("Timing summary skipped for %s", route_name, exc_info=True)


@app.on_event("startup")
def load_rembg_model() -> None:
    global _REMBG_SESSION, _REMBG_FALLBACK_SESSION
    logger.info("Loading rembg model (%s)...", REMBG_MODEL_NAME)
    _REMBG_SESSION = new_session(REMBG_MODEL_NAME)
    logger.info("rembg model loaded.")
    try:
        logger.info("Loading fallback rembg model (%s)...", FALLBACK_REMBG_MODEL_NAME)
        _REMBG_FALLBACK_SESSION = new_session(FALLBACK_REMBG_MODEL_NAME)
        logger.info("rembg fallback model loaded.")
    except Exception as exc:
        # A fallback that refuses to load must not take down the whole service —
        # strip_background() degrades gracefully when the fallback session is None.
        logger.warning(
            "Failed to load fallback rembg model '%s'; degenerate-mask fallback disabled: %s",
            FALLBACK_REMBG_MODEL_NAME,
            exc,
        )
_REMBG_FALLBACK_SESSION = None


@app.on_event("startup")
def check_groq_configuration() -> None:
    """
    Startup check for the strength provider selection.

    Mirrors the existing fail-closed-at-request-time philosophy (X_API_KEY /
    GEMINI_API_KEY): absence of GROQ_API_KEY does NOT crash startup, but when
    the active provider is Groq the catalog-audio endpoint cannot work — log a
    clear warning instead of letting operators discover it in a 500.
    """
    if AI_PROVIDER == "groq" and not GROQ_API_KEY:
        logger.warning(
            "AI_PROVIDER is 'groq' but GROQ_API_KEY is unset — /api/catalog-audio "
            "will fail for every request until it is configured in .env. "
            "Get a free key at console.groq.com."
        )


# --------------------------------------------------------------------------
# Image processing helpers
# --------------------------------------------------------------------------
def auto_crop_letterbox(
    raw_image: np.ndarray,
    uniformity_std_threshold: int = 8,
    dark_mean_threshold: int = 25,
    light_mean_threshold: int = 230,
) -> np.ndarray:
    """
    Order A — Scan the raw frame's outer edges for uniform letterbox/pillarbox
    bands and crop them off BEFORE segmentation ever runs.

    WHY this exists: real-world artisan photos routinely arrive wrapped in
    uniform bars — black/white letterboxing added by WhatsApp forwards, screen
    recordings, or camera apps that letterbox to a fixed aspect ratio. rembg
    has no concept of these bars and treats them as real pixels, so a black
    bar survives segmentation and materializes into the white-background
    composite as a dark band cutting across the product. Cropping the bars up
    front removes the artifact at its source, cheaply (a few edge row/column
    stats) instead of making the expensive AI step fight it.

    A row or column counts as "letterbox" ONLY if it is nearly uniform (std
    below uniformity_std_threshold) AND its mean is clearly dark (below
    dark_mean_threshold) or clearly light (above light_mean_threshold). Both
    conditions are required so a legitimate uniform mid-tone backdrop — or an
    earthy/grey product sitting near the edge — is never skimmed off. Each
    edge is capped at 15% of its dimension so a genuine dark/light product
    near the frame edge can never be over-cropped.

    Returns the cropped frame; if no edge qualifies, returns the input
    unchanged.
    """
    h, w = raw_image.shape[:2]
    if h < 2 or w < 2:
        return raw_image

    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    max_rows = int(h * 0.15)
    max_cols = int(w * 0.15)

    def is_letterbox_line(line: np.ndarray) -> bool:
        line_mean = float(line.mean())
        return (
            float(line.std()) < uniformity_std_threshold
            and (line_mean < dark_mean_threshold or line_mean > light_mean_threshold)
        )

    top = 0
    for offset in range(max_rows):
        if is_letterbox_line(gray[offset, :]):
            top = offset + 1
        else:
            break

    bottom = 0
    for offset in range(max_rows):
        if is_letterbox_line(gray[h - 1 - offset, :]):
            bottom = offset + 1
        else:
            break

    left = 0
    for offset in range(max_cols):
        if is_letterbox_line(gray[:, offset]):
            left = offset + 1
        else:
            break

    right = 0
    for offset in range(max_cols):
        if is_letterbox_line(gray[:, w - 1 - offset]):
            right = offset + 1
        else:
            break

    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return raw_image

    return raw_image[top : h - bottom, left : w - right]


def foreground_fraction(rgba_image: Image.Image) -> float:
    """Share of pixels the segmentation retained as opaque subject, in [0.0, 1.0]."""
    alpha_channel = np.array(rgba_image)[:, :, 3]
    return float((alpha_channel > 128).mean())


def auto_upscale_low_resolution(bgr_image: np.ndarray) -> np.ndarray:
    """
    Low-resolution upscale — run BEFORE segmentation so rembg and its alpha
    matting see a larger, more stable frame.

    WHY this exists: at 296x435 (a common artisan WhatsApp export), hair
    strands are a couple of pixels wide and the pot silhouette is pixelated;
    both segmentation stability and final output clarity suffer. This raises
    the short edge toward UPSCALE_TARGET_MIN_EDGE using Lanczos-4 — the best
    classical resampler that sharpens-looking without inventing grain (no
    hallucinated detail, unlike ML super-resolution). Images already at a
    healthy size pass through untouched.
    """
    h, w = bgr_image.shape[:2]
    min_edge = min(h, w)
    if min_edge >= UPSCALE_TARGET_MIN_EDGE:
        return bgr_image

    scale = min(UPSCALE_TARGET_MIN_EDGE / min_edge, UPSCALE_MAX_FACTOR)
    if scale <= 1.0:
        return bgr_image
    return cv2.resize(
        bgr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
    )


def downscale_for_processing(
    image: np.ndarray,
    max_dimension: int = DOWNSCALE_MAX_PROCESSING_DIMENSION,
) -> np.ndarray:
    """
    Pre-segmentation downscale (Order P1) — cap the longer dimension BEFORE
    rembg + alpha matting run, so the pymatting closed-form solver always
    operates on a bounded-size frame.

    WHY this exists: matting cost scales with pixel count (not with the
    photo's content), so uploads of wildly different resolutions produced
    wildly different (1-15 minute) timings. Downscaling here makes
    segmentation cost fast AND consistent for every artisan photo.

    If the longer dimension (height or width) exceeds max_dimension, the
    image is resized down with cv2.INTER_AREA (quality-preserving) so the
    longer dimension equals max_dimension, preserving aspect ratio exactly.
    Images already at or below max_dimension pass through unchanged — this
    NEVER upscales. max_dimension=1800 is not a quality tradeoff: Order D
    (standardize_canvas) already targets a ~2000px final canvas, so matting
    above that resolution is computation thrown away by the resize anyway.
    """
    h, w = image.shape[:2]
    longer = max(h, w)
    if longer <= max_dimension:
        return image
    scale = max_dimension / longer
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def clean_segmentation_mask(rgba_image: Image.Image) -> Image.Image:
    """
    Alpha-matte cleanup — removes the two background artifacts rembg leaves in
    low-quality phone photos WITHOUT touching the subject.

    WHY this exists: on a mostly-white canvas with gray/dirty patches around
    the subject, rembg's matte often keeps (a) translucent near-white FOG
    (alpha in the 30-150 range but essentially white/gray color) and
    (b) tiny detached foreground islands that are just JPEG compression
    speckles. Both re-appear as smudges after compositing.
      - FOG is killed by a color-conditioned alpha clamp: low saturation AND
        high brightness AND low-confidence alpha -> pure background. A white
        blouse or pale pot surface is safely opaque (alpha ~255) so it is
        never touched.
      - Islands are removed by connected-component analysis: any foreground
        blob smaller than MASK_DROP_COMPONENT_AREA_FRACTION of the frame is
        flipped to background.
    Hair strands and large objects are untouched because their alpha is
    graded, not speckle-sized.
    """
    rgba_np = np.array(rgba_image)  # H x W x 4
    rgb = rgba_np[:, :, :3]
    alpha = rgba_np[:, :, 3].astype(np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    near_white_fog = (
        (alpha < MASK_FOG_ALPHA_MAX)
        & (hsv[:, :, 1] < MASK_FOG_MAX_SATURATION)
        & (hsv[:, :, 2] >= MASK_FOG_MIN_BRIGHTNESS)
    )
    alpha[near_white_fog] = 0

    hard_mask = (alpha > 128).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hard_mask, connectivity=8
    )
    min_island_area = MASK_DROP_COMPONENT_AREA_FRACTION * hard_mask.size
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_island_area:
            alpha[labels == label] = 0

    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def defringe_halo_edges(rgba_image: Image.Image) -> Image.Image:
    """
    Halo defringing — rebuild the semi-transparent edge band's COLOR from
    fully-opaque subject pixels only, eliminating white/gray edge
    contamination.

    WHY this exists: the "light/gray outline around pot and clothing" comes
    from the partially-blended background baked into the matte band
    (0 < alpha < 255, e.g. hair edges, pot neck/handles). Alpha compositing
    can mathematically reveal the halo but cannot repaint it — the RGB in the
    band genuinely contains old background color. Here the band is declared
    "unknown" and inpainted (telea/FMM, small radius) using ONLY opaque
    subject pixels as the source, so every edge pixel is repainted with the
    local true subject color. The original alpha is kept untouched, so fine
    hair strands keep their gradation and the pot silhouette is not cut —
    halo removed without shrinking the object.
    """
    rgba_np = np.array(rgba_image)  # H x W x 4
    rgb = rgba_np[:, :, :3]
    alpha = rgba_np[:, :, 3]

    semi_band = (alpha > 0) & (alpha < 255)
    if not bool(semi_band.any()):
        return rgba_image

    unknown = np.uint8(semi_band | (alpha == 0)) * 255
    clean_rgb = cv2.inpaint(
        rgb, unknown, DEFRINGE_INPAINT_RADIUS, cv2.INPAINT_TELEA
    )

    repaired_rgb = rgb.copy()
    repaired_rgb[semi_band] = clean_rgb[semi_band]
    return Image.fromarray(np.dstack([repaired_rgb, alpha]), "RGBA")


def refine_matting(rgba_image: Image.Image) -> Image.Image:
    """
    Composite matting refinement — mask cleanup then halo defringe, each
    independently guarded so a failure degrades to the previous behaviour
    instead of crashing segmentation (resilience never becomes a single point
    of failure).
    """
    refined = rgba_image
    try:
        refined = clean_segmentation_mask(refined)
    except Exception as exc:
        logger.warning("Segmentation mask cleanup skipped: %s", exc)
    try:
        refined = defringe_halo_edges(refined)
    except Exception as exc:
        logger.warning("Halo defringe skipped: %s", exc)
    return refined


def _run_rembg_selection(image_bytes: bytes) -> Image.Image:
    """
    Run rembg.remove() with the existing primary model, fallback model, and
    alpha-matting parameters, then refine the resulting matte. Operates at
    whatever resolution `image_bytes` is encoded at, and returns a PIL RGBA at
    that same resolution. Model choice, matting parameters, and the
    degenerate-mask fallback logic are UNCHANGED (Campaign Order C) — this
    helper only isolates "run the AI on these bytes".

    WHY the fallback: no single segmentation model is perfect on every photo.
    A degenerate mask — keeping almost nothing (fraction < 3%, product missed)
    or almost everything (fraction > 98%, background never separated) — is a
    red flag, not a valid result. On either condition this re-runs the photo
    through the fallback model as a safety net; if BOTH models degenerate, the
    lesser of the two evils (whose foreground share is closest to the healthy
    ~30% midpoint) is used rather than failing the whole request — exactly how
    camera apps silently fall back when depth/subject detection misfires.
    """
    try:
        result_bytes = remove(
            image_bytes,
            session=_REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
        primary_image = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        primary_fraction = foreground_fraction(primary_image)

        if _REMBG_FALLBACK_SESSION is None or (
            DEGENERATE_MIN_FOREGROUND_FRACTION
            < primary_fraction
            < DEGENERATE_MAX_FOREGROUND_FRACTION
        ):
            selected_image = primary_image
        else:
            if primary_fraction <= DEGENERATE_MIN_FOREGROUND_FRACTION:
                trigger = "foreground_fraction < 0.03 (product likely missed entirely)"
            else:
                trigger = "foreground_fraction > 0.98 (background likely never separated)"
            logger.warning(
                "Degenerate primary segmentation (%s, fraction=%.4f). Retrying with "
                "fallback model '%s'.",
                trigger,
                primary_fraction,
                FALLBACK_REMBG_MODEL_NAME,
            )

            fallback_bytes = remove(
                image_bytes,
                session=_REMBG_FALLBACK_SESSION,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
            )
            fallback_image = Image.open(io.BytesIO(fallback_bytes)).convert("RGBA")
            fallback_fraction = foreground_fraction(fallback_image)

            if (
                DEGENERATE_MIN_FOREGROUND_FRACTION
                < fallback_fraction
                < DEGENERATE_MAX_FOREGROUND_FRACTION
            ):
                logger.warning(
                    "Fallback segmentation recovered the subject (fraction=%.4f).",
                    fallback_fraction,
                )
                selected_image = fallback_image
            else:
                # Both models degenerated — keep whichever is closest to a sane
                # foreground share rather than failing the request.
                logger.warning(
                    "Fallback segmentation also degenerate (fraction=%.4f). "
                    "Using the less extreme result.",
                    fallback_fraction,
                )
                if abs(fallback_fraction - HEALTHY_MID_FOREGROUND_FRACTION) < abs(
                    primary_fraction - HEALTHY_MID_FOREGROUND_FRACTION
                ):
                    selected_image = fallback_image
                else:
                    selected_image = primary_image

        # Matting refinement — fog/island cleanup + halo defringe. Runs on
        # every selected segmentation (primary or fallback) before the frame
        # reaches the lighting/compositing step.
        return refine_matting(selected_image)
    except Exception as exc:
        logger.exception("Background removal failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Background removal failed: {exc}",
        )


def strip_background(working_image: np.ndarray) -> Image.Image:
    """
    Step A — Remove background, return an RGBA PIL Image with alpha matte.

    Order Q1 — mask resolution is decoupled from composite resolution. The
    expensive rembg segmentation + alpha matting step (which is where the
    wall-clock time lives) runs on a BOUNDED frame, never larger than
    SEGMENTATION_MAX_DIMENSION. The AI mask does not need 1800px to be
    accurate — only the final composite needs high resolution. So this thin
    wrapper:
      1. downsizes `working_image` (already Order-P1-capped at 1800px) to the
         segmentation budget via the existing downscale_for_processing() —
         no duplicated resize logic,
      2. runs the UNCHANGED rembg/fallback/matting path on that small frame,
      3. upscales ONLY the resulting alpha mask back to `working_image`'s
         original H x W (INTER_LINEAR — the matte is a smooth probability
         field; nearest-neighbor would stair-step the edges),
      4. merges it with `working_image`'s own full-resolution RGB channels.
    Every downstream step (defringe, lighting, shadow, sanitize, sharpen)
    keeps operating at the full working resolution exactly as before — no
    change to their behavior or the output shape.
    """
    seg_input = downscale_for_processing(
        working_image, max_dimension=SEGMENTATION_MAX_DIMENSION
    )
    ok, encoded_seg = cv2.imencode(".png", seg_input)
    if not ok:
        raise ValueError("cv2.imencode failed to re-encode the segmentation input.")
    seg_bytes = encoded_seg.tobytes()

    # rembg + model selection + matte refinement — all at the bounded
    # segmentation resolution (this is the expensive, wall-clock part).
    seg_rgba = _run_rembg_selection(seg_bytes)

    # Upscale ONLY the alpha mask back to working resolution and merge it with
    # the working image's original RGB (BGR -> RGB to match PIL RGBA ordering,
    # which the rest of the pipeline expects).
    seg_np = np.array(seg_rgba, dtype=np.uint8)
    working_h, working_w = working_image.shape[:2]
    upscaled_mask = cv2.resize(
        seg_np[:, :, 3],
        (working_w, working_h),
        interpolation=cv2.INTER_LINEAR,
    )
    working_rgb = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.dstack([working_rgb, upscaled_mask]), "RGBA")


def compute_adaptive_brightness_contrast(
    bgr_image: np.ndarray,
    alpha_mask: np.ndarray,
    clip_percent: float = 1.0,
) -> Tuple[float, int]:
    """
    Order B — Compute a per-photo brightness/contrast pair from the subject's
    own brightness distribution, replacing the old fixed nudge.

    WHY this exists: fixed constants (+15 brightness, x1.12 contrast) are
    one-size-fits-all — a photo shot in bright sunlight and one shot in a dim
    workshop need opposite corrections, so a blanket nudge helps one and hurts
    the other. This measures the ACTUAL subject (only pixels the alpha mask
    keeps as foreground), takes the clip_percent and (100-clip_percent)
    intensity percentiles of its grey histogram, and stretches those two
    anchors to ~5 and ~250 (standard automatic contrast stretching with a
    little headroom — clipping the extreme outlier percentiles, never the true
    min/max). Both alpha (contrast) and beta (brightness) are clamped so a
    pathological photo can never produce a broken, extreme adjustment.

    Returns (contrast_alpha, brightness_beta) as OpenCV's new_pixel value.
    """
    subject_mask = alpha_mask > 128
    if int(subject_mask.sum()) < 100:
        return DEFAULT_CONTRAST_ALPHA, DEFAULT_BRIGHTNESS_BETA

    subject_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)[subject_mask]
    p_low = float(np.percentile(subject_gray, clip_percent))
    p_high = float(np.percentile(subject_gray, 100.0 - clip_percent))

    if p_high - p_low < 1.0:
        # Flat histogram — no dynamic range to stretch; the fixed nudge is fine.
        return DEFAULT_CONTRAST_ALPHA, DEFAULT_BRIGHTNESS_BETA

    target_low, target_high = 5.0, 250.0
    contrast_alpha = (target_high - target_low) / (p_high - p_low)
    brightness_beta = target_low - contrast_alpha * p_low

    contrast_alpha = float(np.clip(contrast_alpha, 0.8, 1.6))
    brightness_beta = int(round(float(np.clip(brightness_beta, -20.0, 40.0))))
    return contrast_alpha, brightness_beta





def standardize_canvas(
    subject_rgba: np.ndarray,
    canvas_size: int = 2000,
    margin_fraction: float = 0.08,
) -> np.ndarray:
    """
    Order D — Standardized subject framing for e-commerce marketplaces.

    WHY this exists (the "no standardized framing" defect): Flipkart, Amazon
    and Etsy all enforce listable product shots — the subject centered,
    occupying a consistent ~80-85% of a square canvas, at a minimum
    resolution. An artisan's phone photo drops the product wherever it
    happened to fall, which fails listing QA. This crops tightly to the real
    subject bounding box (alpha > 10, so soft matting edges are kept), resizes
    it preserving aspect ratio so the LONG dimension fills
    canvas_size*(1 - 2*margin_fraction) = 84% of the frame, and pastes it dead
    center on a canvas_size x canvas_size transparent canvas. Every subject —
    pot, textile, idol — now ships at the same 84%/centered/square standard.

    Everything downstream (lighting, shadow, sharpen) operates on this
    standardized frame, so the output is repeatable across arbitrary inputs.
    """
    h, w = subject_rgba.shape[:2]
    if h < 1 or w < 1:
        return np.zeros((canvas_size, canvas_size, 4), np.uint8)

    alpha = subject_rgba[:, :, 3]
    ys, xs = np.nonzero(alpha > 10)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((canvas_size, canvas_size, 4), np.uint8)

    # Tight bounding box with a small 5px buffer each side.
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0, y0 = max(0, x0 - 5), max(0, y0 - 5)
    x1, y1 = min(w - 1, x1 + 5), min(h - 1, y1 + 5)
    cropped = subject_rgba[y0 : y1 + 1, x0 : x1 + 1]

    target_size = int(canvas_size * (1 - 2 * margin_fraction))
    ch, cw = cropped.shape[:2]
    scale = target_size / max(ch, cw)
    # Downscale also possible (huge originals) — pick the right interpolator.
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = cv2.resize(cropped, (nw, nh), interpolation=interp)

    canvas = np.zeros((canvas_size, canvas_size, 4), np.uint8)
    ox = (canvas_size - nw) // 2
    oy = (canvas_size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def defringe_alpha_edge(
    rgba_image: np.ndarray,
    edge_erode_px: int = 1,
    edge_blur_sigma: float = 0.8,
) -> np.ndarray:
    """
    Order E — Alpha-edge defringe / anti-alias.

    WHY this exists (the "edge color fringing" defect): semi-transparent edge
    pixels left over from matting carry a faint tint of the original
    background color, which survives compositing as a colored fringe around
    fine-textured edges (woven textile fringe, hair). The erosion removes the
    outermost contaminated ring of alpha pixels; the blur then softens the
    now-harder edge back into a smooth anti-aliased transition. This is the
    standard defringe technique: erosion removes contaminated pixels, the
    blur prevents the edge from looking newly jagged after erosion. RGB
    channels are untouched — only the matte's confidence gradient is cleaned.
    """
    out = rgba_image.copy()
    alpha = out[:, :, 3]

    if edge_erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_erode_px + 1, 2 * edge_erode_px + 1)
        )
        alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=edge_blur_sigma)

    out[:, :, 3] = alpha
    return out


def build_shadow_layer(
    alpha_channel: np.ndarray,
    canvas_shape: tuple,
    shadow_opacity: int = 50,
    blur_radius: int = 35,
) -> np.ndarray:
    """
    Order F — Ground-contact shadow synthesis.

    WHY this exists (the "floating product" defect): a pure white cutout with
    no ground shadow looks like an amateur collage — the eye reads "floating
    in a void" instantly. Even minimal white-studio product photography has a
    soft shadow where the object meets the surface. This builds an elliptical
    shadow under the subject's contact row (the lowest alpha>128 pixel):
    width 70% of the subject bbox, height 12%, Gaussian-blurred with a kernel
    sized from blur_radius for a soft falloff, then normalized so its peak
    darkens the background by shadow_opacity/255. The caller must darken the
    background client WITH this mask BEFORE the subject is alpha-blended on
    top, so the shadow sits under the product, never on it.

    Returns an H x W single-channel mask (0 = untouched, 50 = darkest).
    """
    h, w = canvas_shape
    mask = np.zeros((h, w), np.float32)

    ys, xs = np.nonzero(alpha_channel > 128)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((h, w), np.uint8)

    bbox_w = int(xs.max()) - int(xs.min()) + 1
    bbox_h = int(ys.max()) - int(ys.min()) + 1
    ellipse_w = int(0.70 * bbox_w)
    ellipse_h = int(0.12 * bbox_h)

    if ellipse_w < 2 or ellipse_h < 2:
        return np.zeros((h, w), np.uint8)

    center_x = (int(xs.min()) + int(xs.max())) // 2
    contact_row = int(ys.max())

    cv2.ellipse(
        mask,
        (center_x, contact_row),
        (ellipse_w // 2, ellipse_h // 2),
        0,
        0,
        360,
        (1.0,),
        -1,
    )

    ksize = 2 * blur_radius + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=blur_radius / 3.0)
    peak = float(mask.max())
    if peak <= 0:
        return np.zeros((h, w), np.uint8)

    mask = mask * (shadow_opacity / peak)
    return np.clip(mask, 0, 255).astype(np.uint8)


def sanitize_corners_to_white(
    bgr_image: np.ndarray, near_white_tolerance: int = 8
) -> np.ndarray:
    """
    Strike 3 — Corner flood-fill sanitization of the final composite.

    WHY this exists (the "unclean frame edges" defect): after compositing,
    small near-white gradients — residual JPEG blocks, soft gray haze at
    frame borders, anti-aliased canvas seams — can survive into the corners
    of the output. This seeds a flood fill from each of the four corners and
    repaints every connected region with a pixel value within
    near_white_tolerance of pure white, leaving only clean pure-white corner
    regions. The standardized canvas (Order D) frames the subject with a >=8%
    white margin, and the ground shadow (Order F) is a local, blurred falloff
    that never reaches the corners — so a corner flood fill cannot eat it as
    long as margin_fraction stays >= 0.08 (do not lower it without
    re-validating this assumption).
    """
    out = bgr_image.copy()
    h, w = out.shape[:2]
    if h < 3 or w < 3:
        return out

    mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        cv2.floodFill(
            out,
            mask,
            seed,
            (255, 255, 255),
            near_white_tolerance,
            near_white_tolerance,
            flags=cv2.FLOODFILL_FIXED_RANGE | 8,  # 8 = 8-way connectivity
        )
    return out


def sharpen_product_detail(
    bgr_image: np.ndarray, sharpen_strength: float = 0.3
) -> np.ndarray:
    """
    Order G — Texture detail sharpening (deliberately mild unsharp mask).

    WHY this exists (the "soft/flat handicraft texture" defect): for textiles,
    weaves, carvings and embroidery, the texture IS the product. Flat
    brightness/contrast alone leaves the cutout looking slightly soft compared
    to real studio photography. This is a standard unsharp mask — sharpened =
    original + strength*(original - gaussian(original)) — tuned conservative:
    sharpen_strength 0.3 with a sigmaX=1.0 blur lifts weave/carving relief
    without creating visible white halos around edges. Do not raise the
    default above 0.3 without visual review: oversharpening produces halo
    fringes that read as worse than no sharpening at all.
    """
    blurred = cv2.GaussianBlur(bgr_image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(
        bgr_image, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the
    image's luminance only, preserving chromatic and alpha information.

    WHY this exists: the existing global brightness/contrast stage (Order B)
    stretches the whole frame with one affine gain/offset pair, which cannot
    lift LOCAL contrast — a well-exposed product sitting in shadow next to a
    bright workshop window stays flat inside its own region. CLAHE equalizes
    small tiles of the L (lightness) channel independently with a hard
    contrast cap (clip_limit), so local texture and detail emerge without the
    global wash-out of plain histogram equalization.

    Color/alpha safety rules (must hold for the Phase 2 insertion point):
      - Only the LAB L channel is modified; A/B chroma pass through untouched,
        so hue/saturation cannot drift. RGB channels are NEVER equalized
        independently — that produces colour casts.
      - The alpha channel (RGBA only) is detached before processing and
        restored bit-for-bit afterwards; CLAHE never sees it.
      - The caller's array is never mutated; a fresh uint8 array is returned
        with identical shape/dtype (H,W,3 or H,W,4) and, for RGBA, an
        identical alpha channel.

    Supported input: NumPy uint8 RGB (H,W,3) or RGBA (H,W,4). Grayscale and
    other dtypes/shapes raise ValueError — the project's image contract is
    uint8 RGB/RGBA only, so anything else fails loudly instead of silently
    producing corrupted output.

    Defaults: clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE
    (locked at 1.5 / (8, 8) by the Phase 4 controlled parameter study — the
    module constants are the single source of truth and are never duplicated
    here). Deterministic for fixed parameters. Pure local transformation —
    no shadow, segmentation, canvas, denoising or brightness/contrast logic
    lives here.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"apply_clahe expects a numpy.ndarray, got {type(image).__name__}."
        )
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            f"apply_clahe expects an RGB (H,W,3) or RGBA (H,W,4) uint8 array, "
            f"got shape {image.shape}."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"apply_clahe expects uint8 input, got dtype {image.dtype}."
        )
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be > 0, got {clip_limit}.")
    if (
        len(tile_grid_size) != 2
        or tile_grid_size[0] <= 0
        or tile_grid_size[1] <= 0
    ):
        raise ValueError(
            f"tile_grid_size must be a pair of positive ints, got {tile_grid_size}."
        )

    # Alpha (RGBA only) is preserved bit-for-bit and never CLAHE-processed.
    alpha_channel = (image[:, :, 3].copy() if image.shape[2] == 4 else None)

    # Work on a private copy so the caller's array is never mutated. RGB
    # ordering is the convention this helper honours (matches PIL/RGBA arrays
    # used elsewhere); the Phase 2 caller is responsible for any BGR<->RGB
    # conversion at the integration boundary.
    rgb = np.ascontiguousarray(image[:, :, :3].copy())

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    if alpha_channel is None:
        return enhanced_rgb
    return np.dstack([enhanced_rgb, alpha_channel])


def protect_product_tonal_range(
    bgr_source: np.ndarray,
    bgr_adjusted: np.ndarray,
    alpha_mask: np.ndarray,
) -> np.ndarray:
    """
    Protect the product's upper tonal structure from enhancement-induced
    crushing / overexposure (the pale-pink-vase / white-pottery defect).

    WHY this exists: the Order-B adaptive stretch
    (compute_adaptive_brightness_contrast -> cv2.convertScaleAbs) maps the
    subject's grey 1/99-percentile anchors toward 5/250 with the contrast
    clamped as high as 1.6. That is the right shape for a dark or mid-tone
    product, but for an ALREADY-bright subject (pale pottery, cream/white
    clay, light textiles, white ceramics) it pushes the whole upper tonal
    distribution into the representable top and crushes fine tonal structure
    into a featureless white mass. CLAHE cannot repair that afterwards —
    clipped information is gone.

    This helper runs AFTER the stretch and BEFORE CLAHE, and only intervenes
    when the processed product distribution measurably damaged the SOURCE
    product's tonal structure (Cases C/D/E of the correction brief). It
    compares the product-region luminance distribution of bgr_source vs
    bgr_adjusted and pulls the processed upper band back toward the source
    product's OWN quantile positions:
      - soft, monotone, luminance-only correction (no hard clip, no banding,
        no tonal seam — a smooth blend engages over the [product q50, q90]
        span of the processed frame),
      - nothing below the product median is touched, so legitimate
        shadow/mid improvements survive,
      - essentially a no-op for dark products (no clipping, no band collapse),
        for products whose highlights were already near/at the representable
        top in the source (no invented detail), and for healthy images,
      - never darkens the product below what the source product actually had
        (all anchors come from the source product's own distribution).

    Contract:
      - Inputs: uint8 BGR (H,W,3) bgr_source (pre-stretch per-photo subject),
        uint8 BGR (H,W,3) bgr_adjusted (post-stretch, identical geometry),
        uint8 (H,W) alpha_mask (product alpha, 0 = background).
      - Output: uint8 BGR (H,W,3); deterministic; caller arrays are never
        mutated; input shape/dtype contracts are enforced by failing loudly.
      - Color safety: ONLY the LAB L channel is changed; A/B chroma come
        untouched from the adjusted product (its enhanced color identity).
      - Mask semantics: the correction weight at each pixel is alpha/255, so
        fully transparent/background pixels are never modified and soft
        matting edges blend smoothly (no halo). The white presentation
        background is composed later and is never treated as product.
      - No-op conditions: invalid/degenerate mask, fewer than 100 product
        pixels, a processed distribution that is not measurably damaged, or a
        source product whose own upper band is flat/crushed.
      - No image-specific constants: every anchor derives from the product's
        own distributions; the only universal is the representable-range top
        (255) used to count clipping, plus dimensionless ratios.
    """
    if (
        not isinstance(bgr_source, np.ndarray)
        or not isinstance(bgr_adjusted, np.ndarray)
        or not isinstance(alpha_mask, np.ndarray)
    ):
        raise ValueError("protect_product_tonal_range expects numpy.ndarray inputs.")
    if bgr_source.ndim != 3 or bgr_source.shape[2] != 3:
        raise ValueError(
            f"bgr_source must be BGR (H,W,3), got shape {bgr_source.shape}."
        )
    if bgr_source.shape != bgr_adjusted.shape:
        raise ValueError("bgr_source and bgr_adjusted must share the same geometry.")
    if alpha_mask.shape != bgr_source.shape[:2]:
        raise ValueError("alpha_mask must match the frame geometry.")
    if (
        bgr_source.dtype != np.uint8
        or bgr_adjusted.dtype != np.uint8
        or alpha_mask.dtype != np.uint8
    ):
        raise ValueError("protect_product_tonal_range expects uint8 arrays.")

    product = alpha_mask > 128
    if int(product.sum()) < 100:
        # Same degenerate-mask guard as the adaptive lighting stage: too little
        # subject to reason about — leave the adjusted frame untouched.
        return bgr_adjusted

    src_lab = cv2.cvtColor(bgr_source, cv2.COLOR_BGR2LAB)
    adj_lab = cv2.cvtColor(bgr_adjusted, cv2.COLOR_BGR2LAB)
    src_l = src_lab[..., 0][product].astype(np.float32)
    adj_l = adj_lab[..., 0][product].astype(np.float32)

    pct = (10, 25, 50, 90, 95, 99)
    src_q = np.percentile(src_l, pct)  # source product's own tonal anchors
    adj_q = np.percentile(adj_l, pct)

    src_band = float(src_q[4] - src_q[2])  # source's own upper spread (q95-q50)
    adj_band = float(adj_q[4] - adj_q[2])

    # Damage measures (dimensionless, image-derived):
    #  1) upper-band collapse — how much of the source product's upper dynamic
    #     range survived the enhancement (1.0 = fully preserved).
    s_band = float(np.clip(1.0 - adj_band / max(src_band, 1e-3), 0.0, 1.0))
    #  2) clipping inflation — how much MORE of the product got pushed to the
    #     representable top than the source had there naturally.
    s_clip = float(
        np.clip(float((adj_l >= 254.0).mean()) - float((src_l >= 254.0).mean()), 0.0, 1.0)
    )

    strength = float(np.clip(max(s_band, s_clip), 0.0, 1.0))
    if strength < 1e-3:
        return bgr_adjusted

    # --- monotone quantile pull-back map on the L channel -------------------
    # The map passes through (0,0) (no invented shadows), through the source
    # product's OWN quantile positions taken at the adjusted product's
    # percentiles (band re-anchoring — restores the median/upper band the
    # stretch displaced), and ends with a top anchor placed far above the
    # representable range so a crushed stack is pulled back to the source's
    # own bright structure instead of re-clipping at 255. s_q is increasing
    # and xp is forced strictly increasing, so the map is monotone,
    # continuous and seam-free — equivalent to a smooth photographic tone
    # curve; blending it with the adjusted frame preserves softness.
    xp = [0.0]
    prev = 0.0
    for qv in adj_q:
        qv = max(float(qv), prev + 1.0)
        prev = qv
        xp.append(qv)
    xp.append(prev + 8.0)  # unreachable top anchor (defines the tail slope)
    fp = [0.0] + [float(q) for q in src_q] + [float(src_q[-1])]
    xp = np.asarray(xp, np.float32)
    fp = np.asarray(fp, np.float32)

    adj_l_full = adj_lab[..., 0].astype(np.float32)
    # L is uint8 -> the monotone pull-back map is a pure function of L and can
    # be collapsed to a 256-entry LUT (deterministic, byte-equivalent to the
    # full-frame interpolation, and far cheaper at canvas sizes).
    lut = np.interp(
        np.arange(256, dtype=np.float32), xp, fp
    ).astype(np.float32)
    mapped = lut[adj_lab[..., 0]]

    # Uniform strength, weighted spatially by alpha so soft matting edges
    # blend smoothly and fully transparent pixels stay byte-exact.
    alpha_w = (alpha_mask.astype(np.float32) / 255.0)
    corrected_l = adj_l_full + (mapped - adj_l_full) * strength * alpha_w
    corrected_l = np.clip(corrected_l, 0.0, 255.0).astype(np.uint8)

    out_lab = cv2.merge((corrected_l, adj_lab[..., 1].astype(np.uint8), adj_lab[..., 2].astype(np.uint8)))
    corrected_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    # Background/transparent pixels are left byte-for-byte identical to the
    # adjusted frame (the presentation background is composed later anyway).
    return np.where((alpha_w > 0.0)[..., None], corrected_bgr, bgr_adjusted)


def normalize_lighting_and_flatten(
    rgba_image: Union[Image.Image, np.ndarray],
    brightness_beta: int = DEFAULT_BRIGHTNESS_BETA,
    contrast_alpha: float = DEFAULT_CONTRAST_ALPHA,
    timings: Optional[dict] = None,
) -> np.ndarray:
    """
    Step B — Convert RGBA -> OpenCV BGR, then run FINAL PIPELINE steps 5-8 in
    their mandated order:

      5. per-photo adaptive brightness/contrast on the subject only,
      6. composite over a white + ground-shadow background (Order F),
      7. corner flood-fill sanitization to clean pure-white corners (Strike 3),
      8. mild texture unsharp-mask detail sharpening (Order G).

    The alpha (contrast) / beta (brightness) pair is computed per photo from
    the subject's own brightness histogram (Order B). The fixed constants in
    the signature exist purely as the documented fallback the adaptive path
    returns when it cannot produce a value — a failed lighting computation
    must degrade, never crash the request. Accepts either a PIL RGBA image or
    an already-standardized RGBA ndarray (Order D output) — np.array() yields
    the same HxWx4 array for both.

    Steps 6-8 are each wrapped in their own try/except: a robustness step must
    degrade to "skip this step" (log + continue), never kill the request.
    """
    try:
        rgba_np = np.array(rgba_image)  # H x W x 4, RGBA (PIL or ndarray)
        rgb = rgba_np[:, :, :3]
        alpha_channel = rgba_np[:, :, 3]

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Step 5 — Adaptive lighting: stretch the subject's brightness
        # histogram into a clean target range for THIS photo. Any failure (or
        # a degenerate subject mask that cannot support statistics) logs a
        # warning and falls back to the fixed constants, so a robustness
        # addition can never turn into a new single point of failure.
        _step_start = time.monotonic()
        try:
            contrast_alpha, brightness_beta = compute_adaptive_brightness_contrast(
                bgr, alpha_channel
            )
        except Exception as exc:
            logger.warning(
                "Adaptive lighting computation failed; falling back to fixed "
                "constants (alpha=%.2f, beta=%d): %s",
                DEFAULT_CONTRAST_ALPHA,
                DEFAULT_BRIGHTNESS_BETA,
                exc,
            )
            contrast_alpha, brightness_beta = (
                DEFAULT_CONTRAST_ALPHA,
                DEFAULT_BRIGHTNESS_BETA,
            )

        # No bilateral denoise here — it smoothed away the fine weave/carving/
        # clay-grain texture that craft product photography depends on.

        # Brightness / contrast normalization for consistent e-commerce look
        adjusted_bgr = cv2.convertScaleAbs(bgr, alpha=contrast_alpha, beta=brightness_beta)
        if timings is not None:
            timings["lighting"] = time.monotonic() - _step_start

        # Step 5a — product tonal-range protection. The Order-B stretch above
        # can crush an ALREADY-bright subject (pale pottery, cream/white clay)
        # into the representable top; CLAHE afterwards cannot recover clipped
        # structure. This image-adaptive step pulls the product's upper band
        # back toward the source product's own distribution (no-op for dark or
        # healthy images). Guarded: a failure degrades to skipping the step.
        _step_start = time.monotonic()
        try:
            adjusted_bgr = protect_product_tonal_range(bgr, adjusted_bgr, alpha_channel)
        except Exception as exc:
            logger.warning("Product tonal-range protection skipped: %s", exc)
        if timings is not None:
            timings["tonal_protect"] = time.monotonic() - _step_start

        # Step 5b — CLAHE local luminance contrast, AFTER the Order-B global
        # stretch (so local detail emerges from the already well-exposed
        # subject) and BEFORE synthetic shadow compositing (so the shadow
        # darkens the corrected frame). apply_clahe() owns the LAB transform
        # and expects RGB (H,W,3), so the boundary converts BGR -> RGB and
        # back; the separate alpha_channel is never exposed to CLAHE and
        # rides unchanged into the composite below. Fixed defaults:
        # CLAHE_CLIP_LIMIT / CLAHE_TILE_GRID_SIZE.
        _step_start = time.monotonic()
        adjusted_rgb = cv2.cvtColor(adjusted_bgr, cv2.COLOR_BGR2RGB)
        adjusted_rgb = apply_clahe(adjusted_rgb)
        adjusted_bgr = cv2.cvtColor(adjusted_rgb, cv2.COLOR_RGB2BGR)
        if timings is not None:
            timings["clahe"] = time.monotonic() - _step_start

        # Step 6 — Build the WHITE + ground-shadow background FIRST, then
        # alpha-blend the subject on top so the shadow sits under the product
        # (Order F). On failure, fall back to a plain white background.
        _step_start = time.monotonic()
        h, w = alpha_channel.shape[:2]
        white_bg = np.full((h, w, 3), 255, dtype=np.uint8)
        try:
            shadow_mask = build_shadow_layer(alpha_channel, (h, w))
            if shadow_mask is not None and int(shadow_mask.max()) > 0:
                white_bg = np.clip(
                    white_bg.astype(np.int16)
                    - np.stack([shadow_mask] * 3, axis=2).astype(np.int16),
                    0,
                    255,
                ).astype(np.uint8)
        except Exception as exc:
            logger.warning(
                "Ground-shadow synthesis skipped; using plain white background: %s", exc
            )

        alpha_norm = (alpha_channel.astype(np.float32) / 255.0)[:, :, None]
        composited = (adjusted_bgr.astype(np.float32) * alpha_norm) + (
            white_bg.astype(np.float32) * (1 - alpha_norm)
        )
        composited = np.clip(composited, 0, 255).astype(np.uint8)
        if timings is not None:
            timings["shadow_composite"] = time.monotonic() - _step_start

        # Step 7 — Corner flood-fill sanitization (Strike 3).
        _step_start = time.monotonic()
        try:
            composited = sanitize_corners_to_white(composited)
        except Exception as exc:
            logger.warning("Corner sanitization skipped: %s", exc)
        if timings is not None:
            timings["corner_sanitize"] = time.monotonic() - _step_start

        # Step 8 — Mild texture detail sharpening (Order G).
        _step_start = time.monotonic()
        try:
            composited = sharpen_product_detail(composited)
        except Exception as exc:
            logger.warning("Detail sharpening skipped: %s", exc)
        if timings is not None:
            timings["sharpen"] = time.monotonic() - _step_start

        return composited  # BGR, ready for encoding
    except Exception as exc:
        logger.exception("Lighting normalization / compositing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image processing failed: {exc}",
        )


def encode_to_jpeg_bytes(bgr_image: np.ndarray, quality: int = 95) -> bytes:
    success, buffer = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode processed image.",
        )
    return buffer.tobytes()


# --------------------------------------------------------------------------
# Cloudinary upload
# --------------------------------------------------------------------------
def upload_to_cloudinary(image_bytes: bytes, public_id: str) -> str:
    """Step C — Upload processed bytes to Cloudinary, return the secure URL."""
    try:
        upload_result = cloudinary.uploader.upload(
            io.BytesIO(image_bytes),
            folder=CLOUDINARY_UPLOAD_FOLDER,
            public_id=public_id,
            resource_type="image",
            overwrite=True,
            format="jpg",
        )
        secure_url = upload_result.get("secure_url")
        if not secure_url:
            raise ValueError("Cloudinary response missing 'secure_url'.")
        return secure_url
    except CloudinaryError as exc:
        logger.exception("Cloudinary upload failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cloudinary upload failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during Cloudinary upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected upload error: {exc}",
        )


# --------------------------------------------------------------------------
# Validation — image
# --------------------------------------------------------------------------
async def validate_and_read_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{file.content_type}'. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed is {MAX_UPLOAD_SIZE_MB} MB.",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file upload.",
        )
    return contents


# --------------------------------------------------------------------------
# Endpoint — image enhancement
# --------------------------------------------------------------------------
@app.post("/api/enhance-image", dependencies=[Depends(verify_api_key)])
async def enhance_image(file: UploadFile = File(...)):
    """
    Visual Forge pipeline (FINAL 9-step sequence):
        1. letterbox/border crop     (Order A)
        2. background strip w/ fallback segmentation
        3. standardized canvas       (Order D)
        4. alpha-edge defringe       (Order E)
        5. adaptive lighting + white-background composite (Order B)
        6. ground-shadow composite   (Order F)
        7. corner sanitize           (Strike 3)
        8. texture detail sharpen    (Order G)
        9. JPEG encode @ q95         (Order H)
        -> Cloudinary upload -> JSON response
    """
    request_start = time.monotonic()
    timings: dict = {}

    raw_bytes = await validate_and_read_upload(file)

    # Order A — auto letterbox/border crop on the RAW frame, before rembg ever
    # sees it, so a black/white bar can't survive into the composite. Any
    # failure here (e.g. an undecodable frame) logs a warning and falls back
    # to the un-cropped upload rather than blocking the request — resilience
    # steps never become a single point of failure. strip_background() now
    # takes the working image directly as an ndarray (Order Q1/Q2).
    cropped_np: Optional[np.ndarray] = None
    try:
        _stage_start = time.monotonic()
        raw_np = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw_np is None:
            raise ValueError("cv2.imdecode failed to parse the uploaded image.")
        cropped_np = auto_crop_letterbox(raw_np)

        # Low-resolution upscale BEFORE segmentation — rembg and its alpha
        # matting resolve hair strands and the pot silhouette far better on a
        # larger frame, and the output no longer pixelates. Individually
        # guarded: if upscaling fails we still proceed on the cropped frame.
        try:
            cropped_np = auto_upscale_low_resolution(cropped_np)
        except Exception as exc:
            logger.warning("Low-resolution upscale skipped; using cropped frame: %s", exc)
        timings["letterbox_crop"] = time.monotonic() - _stage_start

        # Order P1 — pre-segmentation downscale: rembg + alpha matting cost
        # scales with pixel count (not with the photo's content), so capping
        # the longer dimension HERE makes segmentation fast AND consistent for
        # artisan uploads of wildly different resolutions. Runs after the
        # letterbox crop and before strip_background. Individually guarded: if
        # downscaling fails we still proceed on the cropped frame.
        _stage_start = time.monotonic()
        try:
            cropped_np = downscale_for_processing(cropped_np)
        except Exception as exc:
            logger.warning("Pre-segmentation downscale skipped; using cropped frame: %s", exc)
        timings["downscale"] = time.monotonic() - _stage_start
    except Exception as exc:
        logger.warning("Letterbox auto-crop skipped; using raw upload as-is: %s", exc)
        # Recover a plain frame from the raw upload so segmentation still gets
        # an ndarray, unless the upload itself is undecodable (handled below).
        if cropped_np is None:
            raw_np = cv2.imdecode(
                np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if raw_np is not None:
                cropped_np = raw_np

    # Step A — strip_background() is CPU-bound (rembg + pymatting alpha
    # matting), so it runs on a worker thread via asyncio.to_thread: the event
    # loop stays responsive for every OTHER concurrent request while this one
    # computes (Order Q2). Exceptions raised inside the thread propagate
    # normally into this coroutine, so a failed segmentation still surfaces as
    # the same 422 HTTPException to the client.
    _stage_start = time.monotonic()
    if cropped_np is not None:
        rgba_image = await asyncio.to_thread(strip_background, cropped_np)
    else:
        rgba_image = await asyncio.to_thread(_run_rembg_selection, raw_bytes)
    timings["segmentation"] = time.monotonic() - _stage_start

    # Order D — standardized framing: subject tight-bbox, aspect-preserving
    # resize to 84% of a square canvas, centered. Independently guarded: if
    # framing fails, fall back to the raw RGBA matte.
    rgba_frame = np.array(rgba_image)
    try:
        rgba_frame = standardize_canvas(rgba_frame)
    except Exception as exc:
        logger.warning("Canvas standardization skipped; using raw matte: %s", exc)
    rgba_frame = np.ascontiguousarray(rgba_frame)

    # Order E — alpha-edge defringe: erode the outermost semi-transparent ring
    # (removes background-color contamination) then blur the matte's hard edge.
    # Independently guarded: if defringing fails, keep the edge as-is.
    _stage_start = time.monotonic()
    try:
        rgba_frame = defringe_alpha_edge(rgba_frame)
    except Exception as exc:
        logger.warning("Alpha-edge defringe skipped: %s", exc)
    timings["defringe"] = time.monotonic() - _stage_start

    # Steps 5-8 (adaptive lighting, shadow composite, corner sanitize, sharpen)
    # normalize_lighting_and_flatten records per-step durations into timings[].
    processed_bgr = normalize_lighting_and_flatten(rgba_frame, timings=timings)
    jpeg_bytes = encode_to_jpeg_bytes(processed_bgr)

    # Step C
    _stage_start = time.monotonic()
    public_id = f"forge_{uuid.uuid4().hex}"
    clean_image_url = upload_to_cloudinary(jpeg_bytes, public_id)
    timings["cloudinary_upload"] = time.monotonic() - _stage_start

    timings["total"] = time.monotonic() - request_start
    _safe_log_timing("enhance_image", **timings)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "success", "clean_image_url": clean_image_url},
    )


# ==========================================================================
# PHASE TWO — Multilingual Audio-to-JSON Cataloger
# ==========================================================================

# --------------------------------------------------------------------------
# Output contracts — two-stage trust-verified pricing
# --------------------------------------------------------------------------
class ProductVerification(BaseModel):
    image_detected_category: str
    image_detected_materials: List[str]
    image_visual_quality_tier: str  # one of: "basic", "standard", "premium"
    audio_described_product: str
    audio_mentioned_price_inr: Optional[int] = None
    audio_mentioned_cost_context: Optional[str] = None  # materials/hours/effort, if any
    consistency_match: bool
    consistency_confidence: str  # one of: "high", "medium", "low"
    consistency_notes: str  # brief explanation, especially if consistency_match is False


class CatalogData(BaseModel):
    seo_title: str
    title_hindi: str
    description_english: str
    description_hindi: str
    tags: List[str]
    suggested_price_inr: int
    price_confidence: str  # one of: "high", "medium", "low"
    pricing_reasoning: str


VERIFICATION_PROMPT = """You are the verification stage of a trust-first pricing pipeline for an
Indian D2C handicraft marketplace. Your ONLY job is to check whether the IMAGE
and the ARTISAN INPUT (Audio and/or Typed Notes) describe the same product, and to extract raw facts about what
you find. You must NOT produce a price, and you must NOT judge whether the
seller's claims are fair.

Independently analyze each input — never let one input inform the other:
- From the IMAGE alone: identify the product category, the materials you can
  actually see, and assign a visual quality tier:
    "basic"     = simple, mass-market finish, visible flaws or wear
    "standard"  = solid craftsmanship, clean finish, no obvious defects
    "premium"   = exceptional detail/craftsmanship, fine materials, presentation-grade
- From the ARTISAN INPUT alone: identify what product the seller describes, in their
  own words. Decipher it accurately regardless of language or accent (Hindi,
  Marathi, English, or code-mixed) and translate naturally. The input may include
  audio, typed text notes, or both. Treat them collectively as the artisan's description.
- Extract any price the seller mentions into audio_mentioned_price_inr (a
  plain integer, INR) and any cost/effort context (materials cost, hours of
  work, labor) into audio_mentioned_cost_context. Extract these as RAW DATA only - do not annotate whether they are fair, low, or high. THESE DETAILS ARE STRICTLY OPTIONAL. Do not lower confidence just because they are missing.
- Explicitly COMPARE the image-detected product against the artisan-described
  product. Set consistency_match=false if they are clearly different products
  (e.g. the input describes a saree but the image shows a clay pot). If the
  input is ambiguous, unclear, or too vague to confidently confirm a match
  either way (even when combining audio and notes), set consistency_confidence="low".
  "low" is treated as a mismatch downstream, so only use it if the COMBINED info
  is truly insufficient. A short but accurate description (e.g., "handmade ceramic vase") is sufficient and MUST yield "high" confidence if the image matches, even if price, costs, or long typed notes are missing.
- consistency_notes: one brief sentence. When consistency_match=false, state
  precisely what the image shows vs what the artisan input describes.

Respond strictly according to the provided JSON schema. Do not include any
commentary, markdown, or text outside the JSON object."""


PRICING_PROMPT = """You are the pricing stage of a trust-first pipeline for an Indian D2C
handicraft marketplace. A verification gate has ALREADY confirmed that the
product image and the seller's audio describe the same product, and the useful
content of the audio has already been extracted. You receive:
1. A text summary of that verification result (category, materials, quality tier,
   and optionally an unverified seller-stated price).
2. The product IMAGE itself, so you can judge finish and condition in detail.

Estimate a fair market price INDEPENDENTLY:
- Base the price ONLY on image_visual_quality_tier, image_detected_category,
  and image_detected_materials from the verification result, plus general
  knowledge of typical Indian handicraft/textile market rates for that category
  and quality tier.
- If the verification summary includes audio_mentioned_price_inr, that number
  is UNVERIFIED SELLER INPUT — a raw claim, not a target and not an anchor. Do
  NOT restate it, placate it, or stay close to it. Treat it as one weak signal
  subordinate to the visual quality assessment, and if you deliberately depart
  from it, explain the departure in pricing_reasoning.
- suggested_price_inr: a realistic integer in Indian Rupees.
- price_confidence: "low" if the visual quality tier was ambiguous or the
  category is one you have limited pricing knowledge of; otherwise "high" or
  "medium".
- seo_title: a concise, keyword-rich product title in English.
- title_hindi: a natural Hindi marketplace title (Devanagari script), not a word-for-word machine translation. Do not translate brand/proper names unnecessarily.
- description_english: a persuasive professional description (2-4 sentences).
- description_hindi: the same description localized naturally into Hindi (Devanagari script), not a literal translation.
- tags: 5-8 lowercase search tags.

Respond strictly according to the provided JSON schema. Do not include any
commentary, markdown, or text outside the JSON object."""


# Groq variant of the verification prompt. Groq's json_object response_format
# enforces JSON syntax but NOT schema — the exact field names/types must be
# spelled out in the prompt itself, and the transcript of the voice note is
# embedded inline (the audio cannot be passed natively as with Gemini, which
# turns the Files API file into first-class multimodal content). Behavior is
# otherwise identical to VERIFICATION_PROMPT.
GROQ_VERIFICATION_PROMPT = """You are the verification stage of a trust-first pricing pipeline for an
Indian D2C handicraft marketplace. Your ONLY job is to check whether the IMAGE
and the seller's input describe the same product, and to extract raw facts
about what you find. You must NOT produce a price, and you must NOT judge
whether the seller's claims are fair.

Below is the artisan's input (an automatic transcript of their voice note, and/or typed notes):
"{transcript}"

Independently analyze each input — never let one input inform the other:
- From the IMAGE alone: identify the product category, the materials you can
  actually see, and assign a visual quality tier:
    "basic"     = simple, mass-market finish, visible flaws or wear
    "standard"  = solid craftsmanship, clean finish, no obvious defects
    "premium"   = exceptional detail/craftsmanship, fine materials, presentation-grade
- From the ARTISAN INPUT alone: identify what product the seller describes, in
  their own words. Decipher it accurately regardless of language or accent
  (Hindi, Marathi, English, or code-mixed) and translate naturally.
- Extract any price the seller mentions into audio_mentioned_price_inr (a
  plain integer, INR) and any cost/effort context (materials cost, hours of
  work, labor) into audio_mentioned_cost_context. Extract these as RAW DATA only - do not annotate whether they are fair, low, or high. THESE DETAILS ARE STRICTLY OPTIONAL. Do not lower confidence just because they are missing.
- Explicitly COMPARE the image-detected product against the input-described
  product. Set consistency_match=false if they are clearly different products
  (e.g. the input describes a saree but the image shows a clay pot). If
  the input is ambiguous, unclear, or too vague to confidently confirm a
  match either way, set consistency_confidence="low" — "low" is treated as a
  mismatch downstream, so only use it if the COMBINED info is truly insufficient. A short but accurate description (e.g., "handmade ceramic vase") is sufficient and MUST yield "high" confidence if the image matches, even if price, costs, or long typed notes are missing.
- consistency_notes: one brief sentence. When consistency_match=false, state
  precisely what the image shows vs what the input describes.

Respond with ONLY a valid JSON object with exactly these fields, no markdown,
no code fences, no commentary:
- image_detected_category: string
- image_detected_materials: array of strings
- image_visual_quality_tier: one of "basic", "standard", "premium"
- audio_described_product: string
- audio_mentioned_price_inr: integer or null
- audio_mentioned_cost_context: string or null
- consistency_match: boolean
- consistency_confidence: one of "high", "medium", "low"
- consistency_notes: string"""


# Groq variant of the pricing prompt. Same philosophy as PRICING_PROMPT but
# the verification context AND the raw transcript are embedded inline as text,
# the field schema is spelled out in the prompt (Groq json_object does not
# carry a schema), and the image arrives as a base64 data URI in the message.
GROQ_PRICING_PROMPT = """You are the pricing stage of a trust-first pipeline for an Indian D2C
handicraft marketplace. A verification gate has ALREADY confirmed that the
product image and the seller's voice note describe the same product. You
receive a text summary of that verification result and the product IMAGE.

{verification_context}

Below is the seller's voice-note transcript for reference. Treat any price
claimed within it as UNVERIFIED SELLER INPUT — a raw claim, not a target and
not an anchor:
"{transcript}"

Estimate a fair market price INDEPENDENTLY:
- Base the price ONLY on the verification context above (category, materials,
  visual quality tier), plus general knowledge of typical Indian handicraft/
  textile market rates for that category and tier.
- If the verification summary includes audio_mentioned_price_inr, that number
  is UNVERIFIED SELLER INPUT — a raw claim, not a target and not an anchor. Do
  NOT restate it, placate it, or stay close to it. Treat it as one weak signal
  subordinate to the visual quality assessment, and if you deliberately depart
  from it, explain the departure in pricing_reasoning.
- suggested_price_inr: a realistic integer in Indian Rupees.
- price_confidence: "low" if the visual quality tier was ambiguous or the
  category is one you have limited pricing knowledge of; otherwise "high" or
  "medium".
- seo_title: a concise, keyword-rich product title in English.
- title_hindi: a natural Hindi marketplace title (Devanagari script), not a word-for-word machine translation. Do not translate brand/proper names unnecessarily.
- description_english: a persuasive professional description (2-4 sentences).
- description_hindi: the same description localized naturally into Hindi (Devanagari script), not a literal translation.
- tags: 5-8 lowercase search tags.

Respond with ONLY a valid JSON object with exactly these fields, no markdown,
no code fences, no commentary:
- seo_title: string
- title_hindi: string
- description_english: string
- description_hindi: string
- tags: array of 5-8 lowercase strings
- suggested_price_inr: integer
- price_confidence: one of "high", "medium", "low"
- pricing_reasoning: string"""


# STARTER PLACEHOLDERS — NOT production-tuned. These bounds must be reviewed
# by someone with real Indian handicraft/textile market knowledge before the
# pricing feature is trusted in production. They exist purely as a
# deterministic safety net: an LLM-suggested price can never ship outside a
# plausible range for its own detected category, no matter what reasoning the
# model gave.
CATEGORY_PRICE_BOUNDS_INR = {
    "pottery": (150, 5000),
    "handloom_textile": (400, 20000),
    "bamboo_cane": (100, 3000),
    "wood_carving": (300, 15000),
    "embroidery_textile": (300, 10000),
    "jewelry_imitation": (150, 5000),
    "leather_goods": (300, 8000),
    "metal_craft": (200, 10000),
    "painting_folk_art": (200, 12000),
    "default": (100, 25000),
}


def validate_price_within_bounds(category: str, price: int) -> Tuple[int, bool]:
    """
    Deterministic Python-side price floor/ceiling per category (Order M).

    WHY this exists: an LLM can produce a confidently-reasoned price that is
    wrong by an order of magnitude (e.g. Rs 2,00,000 for a clay pot). No amount
    of prompt instruction makes the final check trustworthy, because the check
    lives in the same system that produced the number. So this clamp runs in
    deterministic Python AFTER the model call, in code that cannot be anchored
    or confused: any price outside its category bounds is forced back to the
    nearest bound and the clamped flag is set, which the caller must use to
    force price_confidence to "low" — an out-of-bounds answer is evidence the
    upstream model output is unreliable, not a number to respect.

    Category lookup is case-insensitive and falls back to "default" when no key
    matches; a substring match keeps free-form model category strings (e.g.
    "clay pottery" -> "pottery") usable without a brittle enumerated enum.
    """
    normalized = (category or "").strip().lower()
    if normalized in CATEGORY_PRICE_BOUNDS_INR:
        bounds = CATEGORY_PRICE_BOUNDS_INR[normalized]
    elif not normalized:
        # Empty/whitespace category: ``"" in key`` is vacuously True for every
        # key, so substring matching must short-circuit to the default here.
        bounds = CATEGORY_PRICE_BOUNDS_INR["default"]
    else:
        matches = [
            key
            for key in CATEGORY_PRICE_BOUNDS_INR
            if key in normalized or normalized in key
        ]
        bounds = CATEGORY_PRICE_BOUNDS_INR[matches[0] if matches else "default"]

    low, high = bounds
    clamped = max(low, min(high, price))
    return clamped, clamped != price


# --------------------------------------------------------------------------
# Shared catalog-audio pipeline helpers (provider-agnostic)
# --------------------------------------------------------------------------
# These functions exist once and serve BOTH providers: Gemini and Groq must
# produce a byte-for-byte identical external contract, so every piece of
# deterministic post-processing — the hard consistency gate, the pricing
# context text, the category-bound clamp, the deviation flag, and the final
# response envelope — lives in ONE shared place instead of being duplicated
# per-provider (where a drift would silently diverge their behavior).
def _build_verification_context(
    verification: "ProductVerification",
    additional_notes: Optional[str] = None
) -> str:
    """
    Builds the plain-text context block passed to the pricing stage. The
    verification's useful fields are handed over as text (never re-sent as the
    raw audio), with any seller-stated price explicitly labeled as UNVERIFIED
    SELLER INPUT so the pricing model does not anchor on it.
    """
    context = (
        "Verification result (already confirmed: image and artisan input match):\n"
        f"- image_detected_category: {verification.image_detected_category}\n"
        f"- image_detected_materials: {', '.join(verification.image_detected_materials)}\n"
        f"- image_visual_quality_tier: {verification.image_visual_quality_tier}\n"
        f"- audio_described_product: {verification.audio_described_product}\n"
    )
    if verification.audio_mentioned_price_inr is not None:
        context += (
            f"- audio_mentioned_price_inr (UNVERIFIED SELLER INPUT — one weak "
            f"signal, NOT an anchor): {verification.audio_mentioned_price_inr}\n"
        )
    if verification.audio_mentioned_cost_context:
        context += (
            f"- audio_mentioned_cost_context: {verification.audio_mentioned_cost_context}\n"
        )
    if additional_notes:
        context += f"\nADDITIONAL ARTISAN TYPED NOTES:\n{additional_notes}\n"
    return context


def _build_needs_review_response(verification: "ProductVerification", transcript: str = "") -> Optional[JSONResponse]:
    """
    ORDER J — the HARD GATE, shared by both providers. Never proceed to pricing
    for a pair the system itself flagged as inconsistent, nor when the audio is
    too ambiguous to confirm a match ("low" confidence is treated the same as a
    mismatch: asking the artisan to redo the upload beats guessing). Returns
    the 422 JSONResponse when the gate trips, else None to proceed.
    """
    if verification.consistency_match and verification.consistency_confidence != "low":
        return None
    logger.warning(
        "Consistency gate failed for catalog-audio request — refusing to price. "
        "match=%s confidence=%s notes=%r image=%r audio=%r",
        verification.consistency_match,
        verification.consistency_confidence,
        verification.consistency_notes,
        verification.image_detected_category,
        verification.audio_described_product,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "needs_review", "transcript": transcript,
            "reason": verification.consistency_notes,
            "verification": verification.model_dump(),
        },
    )


def _clamp_price_to_category_bounds(
    verification: "ProductVerification",
    catalog_data: "CatalogData",
) -> bool:
    """
    ORDER M — deterministic category price bounds (Python-side safety net),
    shared by both providers. If the model suggested a price outside its own
    detected category's plausible range, clamp to the nearest bound and force
    confidence to "low" — an out-of-bounds number is evidence of an upstream
    problem, not a value to trust, regardless of the reasoning the model
    provided. Returns whether a clamp actually occurred.
    """
    clamped_price, price_bounds_clamped = validate_price_within_bounds(
        verification.image_detected_category,
        catalog_data.suggested_price_inr,
    )
    if price_bounds_clamped:
        logger.warning(
            "Suggested price %d for category '%s' fell outside plausible bounds; "
            "clamped to %d and price_confidence forced to 'low'.",
            catalog_data.suggested_price_inr,
            verification.image_detected_category,
            clamped_price,
        )
        catalog_data.suggested_price_inr = clamped_price
        catalog_data.price_confidence = "low"
    return price_bounds_clamped


def _compute_price_deviation_flag(
    verification: "ProductVerification",
    catalog_data: "CatalogData",
) -> bool:
    """
    ORDER L — deterministic deviation flag, computed in Python (never via LLM
    self-report), shared by both providers. If the seller stated a price and
    the independent estimate is >40% apart from it, flag for human review
    rather than silently picking either number.
    """
    if verification.audio_mentioned_price_inr is not None:
        seller_price = verification.audio_mentioned_price_inr
        model_price = catalog_data.suggested_price_inr
        deviation_ratio = abs(model_price - seller_price) / max(seller_price, 1)
        return deviation_ratio > 0.40
    return False


def _build_catalog_success_response(transcript: str,
    catalog_data: "CatalogData",
    verification: "ProductVerification",
    price_deviation_flag: bool,
    price_bounds_clamped: bool,
) -> JSONResponse:
    """
    ORDER N — final response contract, shared by both providers. The FULL
    verification block ships on every success intentionally: for a
    government-backed scheme an auditor must see exactly what the AI detected
    and compared, not just the final number.
    """
    catalog_payload = catalog_data.model_dump()
    catalog_payload.update(
        {
            "seller_stated_price_inr": verification.audio_mentioned_price_inr,
            "price_deviation_flag": price_deviation_flag,
            "price_bounds_clamped": price_bounds_clamped,
        }
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "success", "transcript": transcript,
            "verification": verification.model_dump(),
            "catalog": catalog_payload,
        },
    )


# --------------------------------------------------------------------------
# Validation — audio
# --------------------------------------------------------------------------
def _discard_tempfile(path: str) -> None:
    """Best-effort removal of an orphaned temp file; never raises."""
    try:
        os.remove(path)
    except OSError:
        logger.warning("Failed to delete orphaned temp file '%s'", path, exc_info=True)


async def validate_and_save_audio_upload(file: UploadFile) -> str:
    """
    Validates the incoming audio upload and persists it to a temp file on disk.
    Gemini's upload_file() requires a filesystem path (or file-like object with
    a name), so the in-memory bytes are written out before the SDK call.
    """
    raw_content_type = (file.content_type or "").lower().strip()
    base_content_type = raw_content_type.split(";", 1)[0].strip()

    # TEMPORARY LOGGING
    logger.info(f"Incoming audio upload - filename: {file.filename}, raw content_type: {file.content_type}, base: {base_content_type}")

    if base_content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{file.content_type}'. "
            f"Allowed: {', '.join(ALLOWED_AUDIO_CONTENT_TYPES)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_AUDIO_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed is {MAX_AUDIO_SIZE_MB} MB.",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file upload.",
        )

    mime_to_ext = {
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "video/mp4": ".mp4",
    }
    suffix = mime_to_ext.get(base_content_type) or os.path.splitext(file.filename or "")[1] or ".audio"
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp_file.write(contents)
    finally:
        tmp_file.close()

    return tmp_file.name


# --------------------------------------------------------------------------
# Acquisition — remote product image
# --------------------------------------------------------------------------
async def download_image_to_tempfile(image_url: str) -> Tuple[str, str]:
    """
    Asynchronously streams the product image from `image_url` (e.g. a Cloudinary
    URL) to a local temp file. Streaming (rather than a single .read()) lets us
    abort mid-download the moment the size cap is exceeded, instead of buffering
    an arbitrarily large response into memory first.

    Returns (local_path, content_type).
    """
    try:
        async with httpx.AsyncClient(timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as http_client:
            async with http_client.stream("GET", image_url) as remote_response:
                if remote_response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Could not fetch image_url (status {remote_response.status_code}).",
                    )

                content_type = remote_response.headers.get("content-type", "").split(";")[0].strip()
                if content_type not in ALLOWED_IMAGE_DOWNLOAD_CONTENT_TYPES:
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail=f"Unsupported image content type '{content_type}' at image_url. "
                        f"Allowed: {', '.join(ALLOWED_IMAGE_DOWNLOAD_CONTENT_TYPES)}",
                    )

                suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[
                    content_type
                ]
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                downloaded_bytes = 0
                max_bytes = MAX_IMAGE_DOWNLOAD_SIZE_MB * 1024 * 1024

                try:
                    async for chunk in remote_response.aiter_bytes(chunk_size=65536):
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > max_bytes:
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"Image at image_url exceeds {MAX_IMAGE_DOWNLOAD_SIZE_MB} MB limit.",
                            )
                        tmp_file.write(chunk)
                finally:
                    tmp_file.close()

                if downloaded_bytes == 0:
                    os.remove(tmp_file.name)
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="image_url returned an empty response body.",
                    )

                return tmp_file.name, content_type

    except HTTPException:
        raise
    except httpx.RequestError as exc:
        logger.exception("Image download failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to download image from image_url: {exc}",
        )


# --------------------------------------------------------------------------
# File state synchronization — wait for Gemini's transient upload to
# finish backend processing before it can be referenced in generate_content
# --------------------------------------------------------------------------
async def wait_for_file_active(uploaded_file, label: str = "file"):
    """
    Polls client.files.get(name=...) until the file's state is ACTIVE.

    Google's Files API returns immediately from upload() while the file is
    still being processed server-side (state=PROCESSING) — larger media
    containers like video/mp4-wrapped voice recordings take longer than
    plain audio codecs. Calling generate_content() before the file reaches
    ACTIVE raises 400 FAILED_PRECONDITION. This polls with a short async
    sleep between checks (asyncio.sleep, not time.sleep — a blocking sleep
    here would stall the entire event loop and every other in-flight
    request on this worker, not just this one) and enforces a hard timeout
    so a stuck file can never hang the request indefinitely.

    Both a backend-reported FAILED state and a timeout collapse to the same
    clean HTTP 500 per architectural directive, rather than distinguishing
    upstream failure from upstream slowness at the status-code level.
    """
    elapsed = 0.0
    current_file = uploaded_file

    while True:
        state_name = getattr(getattr(current_file, "state", None), "name", None)

        if state_name == "ACTIVE":
            return current_file

        if state_name == "FAILED":
            # Google's File object carries an `error` field (FileStatus) with
            # the actual reason on FAILED — code/message/details. Surfacing
            # this in the server log (not the client response, which stays a
            # clean 500) turns "black box failure" into an actionable log line
            # for the next edge case, instead of requiring a screenshot to diagnose.
            error_detail = getattr(current_file, "error", None)
            error_message = getattr(error_detail, "message", None) if error_detail else None
            logger.error(
                "Gemini file '%s' (%s) entered FAILED state. Reason: %s",
                uploaded_file.name,
                label,
                error_message or "no error detail returned by Gemini",
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gemini failed to process the uploaded {label}.",
            )

        if elapsed >= FILE_ACTIVE_POLL_TIMEOUT_SECONDS:
            logger.error(
                "Timed out waiting for Gemini file '%s' (%s) to become ACTIVE (last state: %s).",
                uploaded_file.name,
                label,
                state_name,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Timed out waiting for the uploaded {label} to become ready.",
            )

        # asyncio.sleep, not time.sleep — see docstring.
        await asyncio.sleep(FILE_ACTIVE_POLL_INTERVAL_SECONDS)
        elapsed += FILE_ACTIVE_POLL_INTERVAL_SECONDS
        current_file = await _genai_client.aio.files.get(name=uploaded_file.name)


# --------------------------------------------------------------------------
# Endpoint — audio cataloger
# --------------------------------------------------------------------------
@app.post("/api/catalog-audio", dependencies=[Depends(verify_api_key)])
async def catalog_audio(
    file: UploadFile = File(...),
    image_url: str = Form(...),
    additional_notes: Optional[str] = Form(None),
):
    """
    Dynamic Pricing Assistant pipeline (two-stage trust-verified pricing):
        raw audio upload          -> local temp file
        image_url (Cloudinary)    -> async download -> local temp file
        backend selected by AI_PROVIDER (default "groq"):
          gemini -> transient Files API upload + ACTIVE-poll; audio and image
                    passed natively with response_schema (schema-enforced JSON)
          groq   -> Whisper transcription + Qwen 3.6 27B vision chat with the
                    image as an inline base64 data URI; JSON validated by
                    Pydantic in code (Groq json_object enforces syntax only)
        VERIFICATION stage -> independently compare image vs audio(-transcript)
        HARD GATE (Order J): mismatch OR low-confidence audio -> HTTP 422
                             "needs_review"; NO price is ever generated.
        shared verification context + image + pricing prompt -> CatalogData
        price deviation flag (Order L) + category bounds clamp (Order M), both
        computed in deterministic Python — never by the LLM self-reporting.
        -> CatalogData JSON + full verification block + seller-stated price +
           deviation/bound flags, so an auditor can see what was compared.
        The external JSON contract is byte-for-byte identical for both
        backends — callers never see which provider produced the answer.
        -> guaranteed cleanup: 2 provider files (gemini only) + 2 local files,
           success or failure
    """
    request_start = time.monotonic()
    timings: dict = {}

    local_audio_path: Optional[str] = None
    local_image_path: Optional[str] = None
    transcript: str = ''
    remote_audio_file = None
    remote_image_file = None

    # Order Q3 — audio save and image download are independent (neither needs
    # the other), so acquire them CONCURRENTLY instead of serially. Both local
    # temp files must still be guaranteed deletion on EVERY path, exactly as
    # before:
    #   - both succeed    -> assigned below; the request's finally cleans both
    #   - audio rejected  -> remove whatever temp image the download already
    #                        landed, then raise the audio error
    #   - image download  -> remove the already-saved audio temp file, then
    #     failed             raise the download error
    _stage_start = time.monotonic()
    audio_result, image_result = await asyncio.gather(
        validate_and_save_audio_upload(file),
        download_image_to_tempfile(image_url),
        return_exceptions=True,
    )
    timings["image_download"] = time.monotonic() - _stage_start
    if isinstance(audio_result, BaseException):
        if isinstance(image_result, tuple):
            _discard_tempfile(image_result[0])
        raise audio_result
    if isinstance(image_result, BaseException):
        _discard_tempfile(audio_result)
        raise image_result
    local_audio_path = audio_result
    local_image_path = image_result[0]
    image_content_type = image_result[1]

    try:
        # Multimodal upload — both files pushed to Gemini's transient storage.
        # mime_type is passed explicitly for both, sourced from already-validated
        # values rather than inferred from (untrustworthy) filenames/extensions.
        # The audio MIME is remapped through GEMINI_AUDIO_MIME_OVERRIDE so a
        # client-mislabeled "video/mp4" voice note is correctly declared as
        # audio to Gemini's backend, not passed through as-is.
        gemini_audio_mime_type = GEMINI_AUDIO_MIME_OVERRIDE.get(file.content_type, file.content_type)

        async def _upload_and_poll(file_path, mime_type, label, timing_key):
            """Upload a file to Gemini and wait for it to become ACTIVE, recording per-file upload/poll durations."""
            _stage_start = time.monotonic()
            remote = await run_google_call_async(
                functools.partial(
                    _genai_client.aio.files.upload,
                    file=file_path,
                    config=types.UploadFileConfig(mime_type=mime_type),
                ),
                label=f"{label} upload",
            )
            timings[f"{timing_key}_upload"] = time.monotonic() - _stage_start
            _stage_start = time.monotonic()
            active = await wait_for_file_active(remote, label=label)
            timings[f"{timing_key}_poll"] = time.monotonic() - _stage_start
            return active

        # Both uploads are independent — run them concurrently so the total
        # wall time equals the slower of the two (roughly halving the
        # upload+poll phase versus a sequential pair).
        if AI_PROVIDER == "gemini":
            remote_audio_file, remote_image_file = await asyncio.gather(
                _upload_and_poll(local_audio_path, gemini_audio_mime_type, "audio file", "audio"),
                _upload_and_poll(local_image_path, image_content_type, "image file", "image"),
            )

            # ORDER I — Verification call: independently extract image identity
            # and audio identity, then compare them. Reuses the SAME
            # uploaded-and-ACTIVE file objects from the polling logic above —
            # we must never upload again for this stage. Wrapped in the same
            # transient-retry helper so a rate-limit or 5xx blip is absorbed;
            # permanent errors surface immediately without wasted retries.
            _stage_start = time.monotonic()
            gemini_contents = [VERIFICATION_PROMPT, remote_image_file, remote_audio_file]
            if additional_notes:
                gemini_contents.append(f"ADDITIONAL ARTISAN TYPED NOTES:\n{additional_notes}")

            verification_response = await run_google_call_async(
                functools.partial(
                    _genai_client.aio.models.generate_content,
                    model=GEMINI_MODEL_NAME,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ProductVerification,
                    ),
                ),
                label="verification generate_content",
            )
            timings["verification_call"] = time.monotonic() - _stage_start

            # response.parsed is the SDK-validated Pydantic instance when
            # response_schema is set; fall back to manual validation if the SDK
            # couldn't coerce it, so a schema drift doesn't fail silently.
            verification = _parse_structured(
                verification_response, ProductVerification, label="verification"
            )
        elif AI_PROVIDER == "groq":
            # Full Groq path — single-shot request/response, NO Files API:
            #   1. Whisper transcribes the voice note to text (language is
            #      auto-detected; the audio is often code-mixed Hindi/Marathi/
            #      English where a forced language tag hurts accuracy).
            #   2. The vision model sees the transcript plus the image (as a
            #      downscaled base64 data URI) and returns JSON — validated by
            #      Pydantic in code, since Groq's json_object mode does not
            #      carry a schema.
            # The audio is embedded in the whisper call as raw bytes and never
            # persisted anywhere but the local temp file that the finally block
            # still guarantees to delete.
            _stage_start = time.monotonic()
            transcript = await transcribe_audio_groq(local_audio_path)
            timings["transcription"] = time.monotonic() - _stage_start

            image_base64_data_uri = _image_to_base64_data_uri(
                local_image_path, image_content_type
            )

            _stage_start = time.monotonic()
            if additional_notes:
                transcript += f"\n\nADDITIONAL TYPED NOTES:\n{additional_notes}"

            verification = await groq_structured_chat(
                GROQ_VERIFICATION_PROMPT.format(transcript=transcript),
                image_base64_data_uri,
                ProductVerification,
                label="groq verification",
            )
            timings["verification_call"] = time.monotonic() - _stage_start
        else:
            # Unknown provider — fail closed with a loud 500 rather than
            # silently routing to a default backend.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unknown AI_PROVIDER '{AI_PROVIDER}' (expected 'gemini' or 'groq').",
            )

        # ORDER J — the HARD GATE, shared by both providers. Never proceed to
        # pricing for a pair the system itself flagged as inconsistent, nor when
        # the audio is too ambiguous to confirm a match ("low" confidence is
        # treated the same as a mismatch: asking the artisan to redo the upload
        # beats guessing). Short-circuits the entire pricing step before any
        # price exists.
        gate_response = _build_needs_review_response(verification, transcript)
        if gate_response is not None:
            return gate_response

        # Shared text context used by the pricing stage of BOTH providers. The
        # audio's useful content was already extracted by the verification stage
        # and lives in the verification result — the pricing stage receives only
        # this text plus the image (never the raw audio), and the prompt
        # explicitly instructs that any seller-stated price is UNVERIFIED INPUT,
        # not an anchor.
        verification_context = _build_verification_context(verification, additional_notes)

        # ORDER K — Independent pricing call (provider branch). Both variants
        # consume the identical shared context text above, so the model sees the
        # same inputs regardless of backend.
        if AI_PROVIDER == "gemini":
            _stage_start = time.monotonic()
            pricing_response = await run_google_call_async(
                functools.partial(
                    _genai_client.aio.models.generate_content,
                    model=GEMINI_MODEL_NAME,
                    contents=[PRICING_PROMPT, verification_context, remote_image_file],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CatalogData,
                    ),
                ),
                label="pricing generate_content",
            )
            timings["pricing_call"] = time.monotonic() - _stage_start

            catalog_data = _parse_structured(pricing_response, CatalogData, label="catalog")
        elif AI_PROVIDER == "groq":
            _stage_start = time.monotonic()
            catalog_data = await groq_structured_chat(
                GROQ_PRICING_PROMPT.format(
                    verification_context=verification_context,
                    transcript=transcript,
                ),
                image_base64_data_uri,
                CatalogData,
                label="groq pricing",
            )
            timings["pricing_call"] = time.monotonic() - _stage_start
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unknown AI_PROVIDER '{AI_PROVIDER}' (expected 'gemini' or 'groq').",
            )

        # ORDER M — deterministic category price bounds (Python-side safety
        # net), shared by both providers: an out-of-bounds model price is
        # clamped and confidence forced to "low" in code that cannot be
        # anchored or confused by prompt drift.
        price_bounds_clamped = _clamp_price_to_category_bounds(verification, catalog_data)

        # ORDER L — deterministic deviation flag, computed in Python (never via
        # LLM self-report), shared by both providers.
        price_deviation_flag = _compute_price_deviation_flag(verification, catalog_data)

        # ORDER N — final response contract, shared by both providers. The FULL
        # verification block ships on every success intentionally: for a
        # government-backed scheme an auditor must see exactly what the AI
        # detected and compared, not just the final number.
        return _build_catalog_success_response(transcript,
            catalog_data,
            verification,
            price_deviation_flag,
            price_bounds_clamped,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Audio+image cataloging/pricing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio+image cataloging/pricing failed: {exc}",
        )
    finally:
        timings["total"] = time.monotonic() - request_start
        _safe_log_timing("catalog_audio", **timings)

        # Absolute zero leakage: every acquired resource — 2 remote Gemini
        # files, 2 local temp files — is torn down independently. Each
        # cleanup step is individually guarded so a failure in one (e.g.
        # the image was never downloaded because validation failed first)
        # never prevents the others from running.
        # Remote Files-API deletion only ever applies to the gemini path: the
        # groq path performs a single-shot request/response and never touches
        # the Files API, so remote_* stay None there and these guards already
        # skip them — the extra AI_PROVIDER check makes that intent explicit.
        if AI_PROVIDER == "gemini" and remote_audio_file is not None:
            try:
                await _genai_client.aio.files.delete(name=remote_audio_file.name)
            except Exception:
                logger.warning(
                    "Failed to delete remote Gemini audio file '%s'", remote_audio_file.name, exc_info=True
                )
        if AI_PROVIDER == "gemini" and remote_image_file is not None:
            try:
                await _genai_client.aio.files.delete(name=remote_image_file.name)
            except Exception:
                logger.warning(
                    "Failed to delete remote Gemini image file '%s'", remote_image_file.name, exc_info=True
                )
        if local_audio_path is not None:
            try:
                os.remove(local_audio_path)
            except OSError:
                logger.warning("Failed to delete local audio temp file '%s'", local_audio_path, exc_info=True)
        if local_image_path is not None:
            try:
                os.remove(local_image_path)
            except OSError:
                logger.warning("Failed to delete local image temp file '%s'", local_image_path, exc_info=True)


@app.get("/health")
def health_check():
    return {"status": "ok"}

