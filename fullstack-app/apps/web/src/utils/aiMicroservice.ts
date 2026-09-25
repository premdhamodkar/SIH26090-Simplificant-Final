/**
 * aiMicroservice.ts
 *
 * Typed client for the Simplificant image-enhancer microservice (FastAPI), the
 * one hosted at ~/simplificant_image_enhancer (Visual Forge + Dynamic Pricing
 * Assistant). All requests use SAME-ORIGIN relative URLs (e.g.
 * `/api/ai/enhance-image`), proxied by the Vite dev server to
 * `http://localhost:8000/api/*`. The web app runs inside the Android WebView
 * via the cloudflared tunnel, so same-origin relative fetch calls reach the
 * laptop-hosted microservice without CORS or "localhost-on-phone" problems.
 *
 * Endpoints (proxy base `/api/ai` -> service `/api`):
 *   POST /api/ai/enhance-image   multipart `file`        -> EnhanceResult
 *   POST /api/ai/catalog-audio   multipart `file` + form `image_url`
 *                              (requires `X-API-Key`)    -> CatalogAudioResult
 *
 * The service also does the independent price estimation itself
 * (`suggested_price_inr` inside the catalog), so there is no separate
 * /predict-fair-price call.
 */

import type { CraftCategory, LanguageCode } from "../types"

// ─── Proxy base path (Vite rewrites `/api/ai` -> service `/api`) ────────────
export const AI_PROXY_BASE = "/api/ai"

/* eslint-disable @typescript-eslint/no-explicit-any */

// Thrown when the enhancer service's trust gate rejects the pairing of the
// uploaded image and voice note (they describe different products).
export class NeedsReviewError extends Error {
  public transcript?: string
  constructor(message: string, transcript?: string) {
    super(message)
    this.name = "NeedsReviewError"
    this.transcript = transcript
  }
}
// ─── Response contracts (mirror ~/simplificant_image_enhancer/main.py) ──────

export interface EnhanceResult {
  status: string
  /** Studio-clean JPEG hosted on Cloudinary. */
  clean_image_url: string
}

export interface CatalogData {
  seo_title: string
  title_hindi?: string
  description_english: string
  description_hindi: string
  tags: string[]
  suggested_price_inr: number
  price_confidence: "high" | "medium" | "low"
  pricing_reasoning: string
  seller_stated_price_inr?: number | null
  price_deviation_flag?: boolean
  price_bounds_clamped?: boolean
}

export interface CatalogVerification {
  image_detected_category?: string | null
  image_detected_materials?: string[] | null
  image_visual_quality_tier?: string | null
  audio_described_product?: string | null
  audio_mentioned_price_inr?: number | null
  consistency_match?: boolean
  consistency_confidence?: string
  consistency_notes?: string | null
}

export interface CatalogAudioResult {
  status: string
  transcript?: string
  catalog: CatalogData
  verification: CatalogVerification
}

// ─── Language / category mapping helpers ────────────────────────────────────

/**
 * Map a product category from the service's verification block
 * (image_detected_category) to a frontend CraftCategory. Falls back to
 * Handicrafts when unknown.
 */
export function mapServiceCategoryToCraft(category: string): CraftCategory {
  const c = (category || "").toUpperCase()
  if (/POTTERY|CLAY|TERRACOTTA/.test(c)) return "Pottery"
  if (/TEXTILE|HANDLOOM|SARI|FABRIC|WEAVE/.test(c)) return "Textile"
  if (/WOOD|LACQUER|CARVING/.test(c)) return "Woodwork"
  if (/METAL|BRASS|DHOKRA|BRONZE/.test(c)) return "Metalwork"
  if (/JEWELL|JEWEL/.test(c)) return "Jewelry"
  if (/BAMBOO|CANE|FIBER|FIBRE/.test(c)) return "Bamboo & Cane"
  if (/STONE|MARBLE|INLAY/.test(c)) return "Stone Craft"
  if (/GLASS|PAPER|MACHE/.test(c)) return "Glass & Paper"
  if (/PAINT|MADHUBANI|WARLI|FOLK/.test(c)) return "Folk & Tribal Art"
  return "Handicrafts"
}

/** Frontend language codes that must always be present in Product.description. */
export const FRONTEND_LANGUAGE_CODES: readonly LanguageCode[] = [
  "en", "hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa", "or",
  "es", "fr", "de", "ja", "ar",
]

/**
 * Build a full frontend-language description map from the service's
 * description_english / description_hindi fields. Every non-hindi language
 * falls back to the English copy (the enhancer service only produces en+hi).
 */
export function buildLocalizedDescription(
  en: string,
  hi: string,
  enFallback: string,
): Record<LanguageCode, string> {
  const enText = en?.trim() || enFallback || ""
  const hiText = hi?.trim() || enText
  const out = {} as Record<LanguageCode, string>
  for (const code of FRONTEND_LANGUAGE_CODES) {
    out[code] = code === "hi" ? hiText : enText
  }
  return out
}

// ─── File / data-URL helpers ────────────────────────────────────────────────

const IMAGE_EXT_BY_MIME: Record<string, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
  "image/avif": "avif",
  "image/heic": "heic",
  "image/heif": "heif",
}

const AUDIO_EXT_BY_MIME: Record<string, string> = {
  "audio/mp4": "m4a",
  "audio/mpeg": "mp3",
  "audio/wav": "wav",
  "audio/ogg": "ogg",
  "audio/aac": "aac",
  "audio/flac": "flac",
  "audio/webm": "webm",
  "audio/webm;codecs=opus": "webm",
}

function mimeFromDataUrl(dataUrl: string): string {
  const m = /^data:([^;,]+)/.exec(dataUrl)
  return m ? m[1] : ""
}

async function dataUrlToBlob(dataUrl: string): Promise<Blob | null> {
  try {
    const res = await fetch(dataUrl)
    if (!res.ok) return null
    return await res.blob()
  } catch (err) {
    console.warn("dataUrlToBlob failed", err)
    return null
  }
}

/** Convert a data URL (native camera/gallery/voice record) into a File. */
export async function dataUrlToFile(
  dataUrl: string,
  opts?: { mimeType?: string; kind?: "image" | "audio"; fallbackName?: string },
): Promise<File | null> {
  const blob = await dataUrlToBlob(dataUrl)
  if (!blob) return null
  const mime = opts?.mimeType || blob.type || mimeFromDataUrl(dataUrl)
  const extMap = opts?.kind === "audio" ? AUDIO_EXT_BY_MIME : IMAGE_EXT_BY_MIME
  const ext = extMap[mime] || (opts?.kind === "audio" ? "m4a" : "jpg")
  const name = opts?.fallbackName || (opts?.kind === "audio" ? "voice.m4a" : `workshop.${ext}`)
  try {
    return new File([blob], name, { type: mime || "application/octet-stream" })
  } catch (err) {
    console.warn("File construction failed", err)
    return null
  }
}

// ─── X-API-Key header ───────────────────────────────────────────────────────

/**
 * Your service enforces an X-API-Key header on every route. When running in
 * the same repo workspace the key is injected by the Vite proxy (vite.config.ts
 * reads it from the service .env); the header below simply mirrors it so the
 * browser also sends one when the proxy does not.
 */
function apiKeyHeaders(): Record<string, string> {
  const key =
    // Vite exposes non-VITE_ env vars to the server config only; the app reads
    // it via import.meta.env when the build pipeline defines it.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (import.meta.env?.VITE_AI_API_KEY as string | undefined) || ""
  return key ? { "X-API-Key": key } : {}
}

// ─── Endpoint calls ─────────────────────────────────────────────────────────

/**
 * POST /api/ai/enhance-image — Visual Forge 9-step pipeline (rembg+CV).
 * @param imageFile Image file (jpg/jpeg/png/webp/heic/heif).
 */
export async function enhanceImage(
  imageFile: File,
  timeoutMs = 120_000,
): Promise<EnhanceResult | null> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const form = new FormData()
    form.append("file", imageFile)
    const res = await fetch(`${AI_PROXY_BASE}/enhance-image`, {
      method: "POST",
      headers: apiKeyHeaders(),
      body: form,
      signal: controller.signal,
    })
    if (!res.ok) {
      console.warn("enhance-image failed", res.status, await safeText(res))
      return null
    }
    const data = (await res.json()) as EnhanceResult
    return data?.status === "success" && data?.clean_image_url ? data : null
  } catch (err) {
    console.warn("enhance-image unavailable:", err)
    return null
  } finally {
    clearTimeout(timer)
  }
}

/**
 * POST /api/ai/catalog-audio — whisper transcription + vision pricing into a
 * full CatalogData (title, en/hi descriptions, tags, suggested price and
 * reasoning). Fails with status 422 `needs_review` when the image and audio
 * do not describe the same product — return null so the UI can show a message.
 * @param audioFile Native voice recording (m4a/wav/mp3/ogg/aac/flac).
 * @param imageUrl http(s) Cloudinary studio-clean image URL.
 */
export async function catalogFromAudio(
  audioFile: File,
  imageUrl: string,
  additionalNotes?: string,
  timeoutMs = 180_000,
): Promise<CatalogAudioResult | null> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const form = new FormData()
    form.append("file", audioFile)
    form.append("image_url", imageUrl)
    if (additionalNotes) {
      form.append("additional_notes", additionalNotes)
    }
    const res = await fetch(`${AI_PROXY_BASE}/catalog-audio`, {
      method: "POST",
      headers: apiKeyHeaders(),
      body: form,
      signal: controller.signal,
    })
    if (!res.ok) {
      // Your service's trust gate: image vs audio mismatch -> HTTP 422 with
      // {"status":"needs_review","reason":...}. Surface the reason so the
      // artisan knows to re-record / re-shoot instead of a generic failure.
      const bodyText = await safeText(res)
      try {
        const body = JSON.parse(bodyText) as {
          status?: string
          reason?: string
        }
        if (body?.status === "needs_review") {
          throw new NeedsReviewError(
            body.reason ||
              "The photo and voice don't describe the same product - please re-shoot and re-record.", body.transcript
          )
        }
      } catch (err) {
        if (err instanceof NeedsReviewError) throw err
        // not JSON / not needs_review — fall through to generic failure
      }
      console.warn("catalog-audio failed", res.status, bodyText)
      return null
    }
    const data = (await res.json()) as CatalogAudioResult
    return data?.status === "success" && data?.catalog ? data : null
  } catch (err) {
    console.warn("catalog-audio unavailable:", err)
    return null
  } finally {
    clearTimeout(timer)
  }
}

/** Build a friendly one-line rationale from the service's pricing fields. */
export function pricingRationaleText(
  catalog: Pick<
    CatalogData,
    "price_confidence" | "pricing_reasoning" | "price_deviation_flag" | "price_bounds_clamped"
  >,
  fallback: string,
): string {
  if (catalog?.pricing_reasoning?.trim()) {
    const prefix =
      catalog.price_confidence === "high"
        ? "High confidence"
        : catalog.price_confidence === "low"
          ? "Low confidence"
          : "Medium confidence"
    let out = `${prefix} · ${catalog.pricing_reasoning.trim()}`
    if (catalog.price_bounds_clamped) out += " · price clamped to category bounds"
    if (catalog.price_deviation_flag) out += " · seller price deviated >40%, flagged for review"
    return out
  }
  return fallback
}
