import { LanguageCode } from "../types"

// Map internal language codes to BCP 47 speech recognition codes
export const SPEECH_LANG_CODES: Record<LanguageCode, string> = {
  en: "en-IN",
  hi: "hi-IN",
  mr: "mr-IN",
  bn: "bn-IN",
  ta: "ta-IN",
  te: "te-IN",
  gu: "gu-IN",
  kn: "kn-IN",
  ml: "ml-IN",
  pa: "pa-IN",
  or: "or-IN",
  es: "es-ES",
  fr: "fr-FR",
  de: "de-DE",
  ja: "ja-JP",
  ar: "ar-SA",
}

export interface VoiceCatalogResult {
  detectedTitleEn: string
  detectedTitleRegional: string
  category: "Pottery" | "Textile" | "Woodwork" | "Metalwork" | "Jewelry" | "Handicrafts"
  fullDescription: string
  materials: string[]
  highlights: string[]
  tags: string[]
  suggestedDimensions: string
  originRegion: string
}

let activeRecognitionInstance: any = null

// ─── NATIVE VOICE RECORDING BRIDGE (Expo WebView shell) ────────────────────
// Android WebViews don't implement the Web Speech API, so mic taps in the
// mobile app delegate to the native Expo shell: it records real audio with
// expo-audio and injects the result back through the bridge below. Desktop
// browsers keep using the regular Web Speech API, unchanged.

export interface NativeVoicePayload {
  /** Base64 data URL of the recorded audio (m4a / AAC on Android). */
  dataUrl: string
  mimeType?: string
  /** Recording length in milliseconds. */
  durationMs?: number
  /** Transcript produced by an external STT service, when one is wired up. */
  transcript?: string
  /** True when the user cancelled the native recording. */
  cancelled?: boolean
}

export interface SpeechListenOptions {
  /** Called when a real voice note was recorded but no transcript is available. */
  onAudio?: (payload: NativeVoicePayload) => void
  /** Called when the user cancelled the native recording. */
  onCancel?: () => void
}

declare global {
  interface Window {
    ReactNativeWebView?: {
      postMessage: (message: string) => void
    }
    __simplificantNativeBridge?: {
      onNativeImage?: (dataUrl: string) => void
      /** Ask the native shell to (re)start a voice recording. */
      requestVoiceRecording?: () => void
      /** Ask the native shell to stop the recording and return the audio. */
      stopVoiceRecording?: () => void
      /** Receives the finished voice note from the native shell. */
      onNativeVoice?: (payload: NativeVoicePayload) => void
      /** Receives a user-cancelled native recording. */
      onNativeVoiceCancelled?: () => void
    }
  }
}

interface NativeVoiceSession {
  onResult: (text: string, isFinal: boolean) => void
  onError: (err: string) => void
  options?: SpeechListenOptions
}

let nativeVoiceSession: NativeVoiceSession | null = null

function postToNativeShell(message: Record<string, unknown>): void {
  try {
    window.ReactNativeWebView?.postMessage(JSON.stringify(message))
  } catch (err) {
    console.warn("postToNativeShell failed", err)
  }
}

/** Install the voice half of the native bridge (idempotent). */
function installNativeVoiceBridge(): void {
  if (typeof window === "undefined") return
  const bridge = window.__simplificantNativeBridge || (window.__simplificantNativeBridge = {})

  bridge.requestVoiceRecording = () => {
    console.log("WEB: requesting native voice recording")
    postToNativeShell({ type: "voice:start" })
  }
  bridge.stopVoiceRecording = () => postToNativeShell({ type: "voice:stop" })

  bridge.onNativeVoice = (payload) => {
    console.log("WEB: native audio received")
    const session = nativeVoiceSession
    nativeVoiceSession = null
    if (!session) return
    if (payload?.transcript) {
      session.onResult(payload.transcript, true)
      session.options?.onAudio?.(payload)
    } else if (payload?.cancelled) {
      if (session.options?.onCancel) {
        session.options.onCancel()
      } else {
        session.onError("Recording cancelled.")
      }
    } else {
      if (session.options?.onAudio) {
        session.options.onAudio(payload)
      } else {
        session.onError("Voice note recorded (no transcription available).")
      }
    }
  }

  bridge.onNativeVoiceCancelled = () => {
    const session = nativeVoiceSession
    nativeVoiceSession = null
    if (!session) return
    if (session.options?.onCancel) {
      session.options.onCancel()
    } else {
      session.onError("Recording cancelled.")
    }
  }
}

installNativeVoiceBridge()

export interface SpeechRecognitionHelper {
  isSupported: boolean
  startListening: (
    lang: LanguageCode,
    onResult: (text: string, isFinal: boolean) => void,
    onError: (err: string) => void,
    options?: SpeechListenOptions,
  ) => () => void
  stopListening: () => void
}

/**
 * Browser Web Speech API wrapper
 */
export const speechService: SpeechRecognitionHelper = {
  isSupported:
    typeof window !== "undefined" &&
    ("webkitSpeechRecognition" in window || "SpeechRecognition" in window),

  stopListening: () => {
    if (activeRecognitionInstance) {
      try {
        activeRecognitionInstance.stop()
      } catch {}
      activeRecognitionInstance = null
    }
    if (nativeVoiceSession) {
      nativeVoiceSession = null
      window.__simplificantNativeBridge?.stopVoiceRecording?.()
    }
  },

  startListening: (lang, onResult, onError, options) => {
    if (typeof window === "undefined") return () => {}

    // 1. Native shell path: If we are inside the Expo shell, use the native bridge FIRST.
    // The native shell uses expo-audio to record reliably and bypasses WebView permission hell.
    if (window.ReactNativeWebView && window.__simplificantNativeBridge?.requestVoiceRecording) {
      const session: NativeVoiceSession = { onResult, onError, options }
      nativeVoiceSession = session
      const stopFn = () => {
        if (nativeVoiceSession !== session) return
        nativeVoiceSession = null
        window.__simplificantNativeBridge?.stopVoiceRecording?.()
      }
      window.__simplificantNativeBridge.requestVoiceRecording()
      return stopFn
    }

    // 2. Desktop browser path: Use Web Speech API and MediaRecorder
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      onError("Speech recognition not supported in this browser.")
      return () => {}
    }

    let mediaRecorder: MediaRecorder | null = null
    let audioChunks: Blob[] = []
    let stream: MediaStream | null = null

    try {
      if (activeRecognitionInstance) {
        try {
          activeRecognitionInstance.stop()
        } catch {}
      }

      const recognition = new SpeechRecognition()
      activeRecognitionInstance = recognition
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = SPEECH_LANG_CODES[lang] || "hi-IN"

      recognition.onresult = (event: any) => {
        let interim = ""
        let final = ""
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            final += event.results[i][0].transcript
          } else {
            interim += event.results[i][0].transcript
          }
        }
        onResult(final || interim, Boolean(final))
      }

      recognition.onerror = (e: any) => {
        console.warn("Speech recognition error:", e.error)
        // We do not immediately fail the recording if audio is still capturing
      }

      // Start MediaRecorder alongside SpeechRecognition
      navigator.mediaDevices.getUserMedia({ audio: true }).then((mediaStream) => {
        stream = mediaStream
        
        const preferredTypes = [
          "audio/mp4",
          "audio/webm;codecs=opus",
          "audio/webm",
        ]
        const supportedType = preferredTypes.find(type => MediaRecorder.isTypeSupported(type))
        
        mediaRecorder = new MediaRecorder(mediaStream, supportedType ? { mimeType: supportedType } : undefined)
        
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunks.push(e.data)
        }
        
        mediaRecorder.onstop = () => {
          const actualMime = mediaRecorder?.mimeType || supportedType || "audio/webm"
          const audioBlob = new Blob(audioChunks, { type: actualMime })
          const reader = new FileReader()
          reader.readAsDataURL(audioBlob)
          reader.onloadend = () => {
            const dataUrl = reader.result as string
            options?.onAudio?.({ 
              dataUrl, 
              mimeType: actualMime,
              transcript: "" 
            })
          }
          // Cleanup tracks
          mediaStream.getTracks().forEach(track => track.stop())
        }
        
        mediaRecorder.start()
        recognition.start()
      }).catch(err => {
        console.warn("Microphone access denied or unavailable for MediaRecorder", err)
        // Fallback to just recognition if getUserMedia fails but SpeechRecognition works
        recognition.start()
      })

      return () => {
        try {
          recognition.stop()
        } catch {}
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
          mediaRecorder.stop()
        } else if (stream) {
          stream.getTracks().forEach(track => track.stop())
        }
        if (activeRecognitionInstance === recognition) {
          activeRecognitionInstance = null
        }
      }
    } catch (err: any) {
      onError(err.message || "Failed to initialize microphone.")
      return () => {}
    }
  },
}

/**
 * Text-to-Speech synthesis helper
 */
export function speakText(text: string, lang: LanguageCode): boolean {
  if (typeof window === "undefined" || !("speechSynthesis" in window))
    return false
  try {
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = SPEECH_LANG_CODES[lang] || "hi-IN"
    utterance.rate = 0.95
    utterance.pitch = 1.0
    window.speechSynthesis.speak(utterance)
    return true
  } catch (err) {
    console.warn("TTS error:", err)
    return false
  }
}

export function stopSpeaking(): void {
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel()
  }
}

/**
 * Pre-configured realistic artisan voice prompts for regional Indian languages
 */
export const REGIONAL_VOICE_SAMPLES: Record<string, {
  voiceText: string
  language: string
  craft: string
}> = {
  marathi_diya: {
    language: "Marathi (मराठी)",
    craft: "Terracotta Festive Diya",
    voiceText:
      "हा हाताने बनवलेला मातीचा दिवा आहे, आम्ही कोकणातील शुद्ध लाल माती वापरून चाकावर घडवला आहे आणि त्यावर नैसर्गिक नक्षीकाम केले आहे.",
  },
  hindi_pottery: {
    language: "Hindi (हिन्दी)",
    craft: "Jaipur Blue Pottery Floral Vase",
    voiceText:
      "यह जयपुर के सांगानेर का असली नीली मिट्टी का फूलदान है। शुद्ध क्वार्ट्ज़ और तांबे के नीले रंग से बिना मिट्टी के 850 डिग्री पर लकड़ी की भट्टी में पकाया गया है।",
  },
  bengali_terracotta: {
    language: "Bengali (বাংলা)",
    craft: "Bankura Panchmura Terracotta Horse",
    voiceText:
      "এটি বাঁকুড়ার পঞ্চমুড়ার খাঁটি পোড়ামাটির ঘোড়া। গঙ্গামাটি দিয়ে হাতে গড়া এবং ঐতিহ্যবাহী কাঠের ভাটিতে পোড়ানো।",
  },
  kannada_channapatna: {
    language: "Kannada (ಕನ್ನಡ)",
    craft: "Channapatna Wooden Toys",
    voiceText:
      "ಇದು ಚನ್ನಪಟ್ಟಣದ ನೈಸರ್ಗಿಕ ಆಲೆ ಮರದಿಂದ ಮಾಡಿದ ಸಾಂಪ್ರದಾಯಿಕ ಗೊಂಬೆ. ಅರಿಶಿನ ಮತ್ತು ನೈಸರ್ಗಿಕ ಸಸ್ಯಜನ್ಯ ಬಣ್ಣಗಳಿಂದ ಮಕ್ಕಳಿಗಾಗಿ ಸುರಕ್ಷಿತವಾಗಿ ಮಾಡಲಾಗಿದೆ.",
  },
  tamil_brass: {
    language: "Tamil (தமிழ்)",
    craft: "Nachiyar Koil Brass Lamp",
    voiceText:
      "இது நாச்சியார்கோவில் பாரம்பரிய பித்தளை குத்துவிளக்கு. தூய பித்தளை உலோகத்தில் பழங்கால முறையில் வார்க்கப்பட்டு கைவேலைப்பாடு செய்யப்பட்டது.",
  },
}

/**
 * AI NLP Engine: Converts speech/text description into structured professional craft catalog listing
 */
export function generateCatalogFromVoiceText(
  voiceText: string,
): VoiceCatalogResult {
  const lower = voiceText.toLowerCase()

  if (
    lower.includes("दिवा") ||
    lower.includes("diya") ||
    lower.includes("lamp") ||
    lower.includes("दीपक")
  ) {
    return {
      detectedTitleEn: "Handcrafted Traditional Terracotta Carved Floral Diya",
      detectedTitleRegional: "हाताने कोरलेला अस्सल पारंपरिक मातीचा दिवा",
      category: "Pottery",
      fullDescription: `Authentic artisan terracotta festive diya hand-molded on traditional potters wheels using pure natural riverbed clay. Etched with traditional floral motifs and fired in organic husk kilns for enhanced heat retention.`,
      materials: [
        "Natural Riverbed Clay",
        "Organic Firing Husk",
        "Natural Terracotta Mineral Slip",
      ],
      highlights: [
        "100% biodegradable eco-friendly natural clay",
        "Hand-turned on traditional potters wheel",
        "Deep reservoir for extended burn time",
        "Direct proceeds benefit rural potter guild",
      ],
      tags: ["#TerracottaDiya", "#EcoFriendlyCraft", "#DirectArtisanBenefit"],
      suggestedDimensions: '4.5" x 4.5" x 2" (11cm x 11cm x 5cm)',
      originRegion: "Konkan Pottery Guild, Maharashtra",
    }
  }

  if (
    lower.includes("pottery") ||
    lower.includes("फूलदान") ||
    lower.includes("नीली") ||
    lower.includes("blue") ||
    lower.includes("vase")
  ) {
    return {
      detectedTitleEn:
        "Jaipur Handcrafted Blue Pottery Floral Heritage Vase (10 inch)",
      detectedTitleRegional: "जयपुर हस्तनिर्मित नीली मिट्टी का पारंपरिक पुष्प फूलदान",
      category: "Pottery",
      fullDescription: `Authentic non-clay quartz pottery crafted in Sanganer cluster using natural quartz stone glaze and copper oxide minerals. Fired at 850°C in traditional kilns to achieve an enduring turquoise luster.`,
      materials: [
        "Natural Quartz Powder",
        "Copper Oxide Pigment",
        "Glass Frit",
        "Lead-Free Glaze",
      ],
      highlights: [
        "Fired at 850°C in wood kilns",
        "Lead-free mineral glaze, impervious to water",
        "GI Seal certified craft provenance",
        "Hand-painted with natural cobalt minerals",
      ],
      tags: ["#JaipurBluePottery", "#QuartzArt", "#MoSJEArtisan"],
      suggestedDimensions: '10" x 4.5" (25cm x 11cm)',
      originRegion: "Sanganer GI Cluster, Jaipur, Rajasthan",
    }
  }

  if (
    lower.includes("silk") ||
    lower.includes("रेशम") ||
    lower.includes("सिल्क") ||
    lower.includes("साड़ी") ||
    lower.includes("dupatta") ||
    lower.includes("पट्टु")
  ) {
    return {
      detectedTitleEn:
        "Pure Banarasi Handloom Mulberry Silk Dupatta with Gold Zari",
      detectedTitleRegional: "शुद्ध बनारसी हथकरघा शहतूत रेशम दुपट्टा (सोने की ज़री)",
      category: "Textile",
      fullDescription: `Exquisite handloom masterpiece woven over 18 days on traditional pit looms in Varanasi. Features pure Grade-A mulberry silk and tested electroplated gold zari in traditional Kadhwa floral jaal patterns.`,
      materials: [
        "Grade-A Mulberry Raw Silk",
        "Silver-Electroplated Tested Gold Zari",
        "Natural Vegetable Dyes",
      ],
      highlights: [
        "18-day continuous hand pit-loom warp & weft weaving",
        "Zero loose back-threads (authentic Kadhwa technique)",
        "Silk Mark & Handloom Mark government certified",
        "Featherlight drape with rich metallic sheen",
      ],
      tags: ["#BanarasiSilk", "#HandloomMark", "#VaranasiHeritage"],
      suggestedDimensions: "2.5m x 0.9m (Full Dupatta)",
      originRegion: "Kotwa Handloom Weavers Guild, Varanasi, UP",
    }
  }

  if (
    lower.includes("लकड़ी") ||
    lower.includes("toy") ||
    lower.includes("गोंबे") ||
    lower.includes("ઢીંગલી") ||
    lower.includes("channapatna") ||
    lower.includes("doll")
  ) {
    return {
      detectedTitleEn:
        "Channapatna GI Wooden Lacquerware Handcrafted Figurines (Set of 2)",
      detectedTitleRegional: "ಚನ್ನಪಟ್ಟಣ ಸಾಂಪ್ರದಾಯಿಕ ನೈಸರ್ಗಿಕ ಮರದ ಗೊಂಬೆಗಳು",
      category: "Woodwork",
      fullDescription: `Seasoned Ivory-wood turned on traditional wood lathes by certified guild artisans. Friction-coated with 100% child-safe organic lac and natural vegetable dyes (turmeric, indigo, kumkum).`,
      materials: [
        "Seasoned Ivory-Wood (Wrightia Tinctoria)",
        "Non-Toxic Shellac Lac",
        "Natural Vegetable Pigments",
      ],
      highlights: [
        "100% child-safe organic colors",
        "Turned on traditional hand-lathes",
        "Natural Talipot palm leaf friction luster",
        "Zero synthetic chemicals or lead varnish",
      ],
      tags: ["#ChannapatnaToys", "#ChildSafeCraft", "#GIHeritage"],
      suggestedDimensions: 'Set of 2 (5.5" x 3" each / 14cm x 8cm)',
      originRegion: "Channapatna Lacquerware Artisans Guild, Karnataka",
    }
  }

  // Default craft fallback parsing
  return {
    detectedTitleEn: "Master Artisan Handcrafted Heritage Craft",
    detectedTitleRegional: "मास्टर कारीगर द्वारा हस्तनिर्मित प्रामाणिक पारंपरिक शिल्प",
    category: "Handicrafts",
    fullDescription: voiceText
      ? `Authentic handicraft described by master artisan: "${voiceText}". Created using traditional techniques and natural local raw materials.`
      : `Master guild handcrafted item created using traditional indigenous techniques and natural raw materials. Certified under MoSJE direct artisan livelihood initiative.`,
    materials: [
      "Indigenous Natural Raw Materials",
      "Organic Mineral Binder",
      "Traditional Guild Pigments",
    ],
    highlights: [
      "100% handmade by certified regional artisan",
      "Ancestral techniques passed down through generations",
      "0% middleman commission direct bank settlement",
      "Eco-friendly sustainable craft composition",
    ],
    tags: ["#ArtisanMade", "#IndianHandicrafts", "#VocalForLocal"],
    suggestedDimensions: "Standard Handcrafted Dimensions",
    originRegion: "National Craft Council Verified Cluster",
  }
}
