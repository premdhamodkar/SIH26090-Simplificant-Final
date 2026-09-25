import { useState, useEffect, useRef } from "react"
import {
  Product,
  CraftCategory,
  LanguageCode,
  AIValuationResult,
} from "../types"
import LiveCameraViewfinder from "./LiveCameraViewfinder"
import { speechService } from "../utils/speechService"
import {
  enhanceImage,
  catalogFromAudio,
  dataUrlToFile,
  mapServiceCategoryToCraft,
  NeedsReviewError,
  type CatalogAudioResult,
} from "../utils/aiMicroservice"

interface SellerCameraUploadModalProps {
  isOpen: boolean
  onClose: () => void
  selectedLanguage: LanguageCode
  prefillValuation?: AIValuationResult | null
  prefillImageUrl?: string | null
  nativeIncomingImage?: string | null
  consumeNativeIncomingImage?: () => void
  onProductPublished: (newProduct: Product) => void
  showToast: (msg: string) => void
}

type ModalPhase = "INPUT" | "PROCESSING" | "REVIEW"

export default function SellerCameraUploadModal({
  isOpen,
  onClose,
  selectedLanguage,
  prefillValuation = null,
  prefillImageUrl = null,
  nativeIncomingImage = null,
  consumeNativeIncomingImage,
  onProductPublished,
  showToast,
}: SellerCameraUploadModalProps) {
  const [phase, setPhase] = useState<ModalPhase>("INPUT")
  
  // Camera Modal State
  const [showCamera, setShowCamera] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // INPUT STATE
  const [originalImageDataUrl, setOriginalImageDataUrl] = useState<string | null>(prefillImageUrl || null)
  const [originalImageFile, setOriginalImageFile] = useState<File | null>(null)
  
  const [isRecording, setIsRecording] = useState(false)
  const [voiceTranscript, setVoiceTranscript] = useState("")
  const [rawDescription, setRawDescription] = useState("")
  const [recordedAudioDataUrl, setRecordedAudioDataUrl] = useState<string | null>(null)
  const [recordedAudioMimeType, setRecordedAudioMimeType] = useState<string | undefined>(undefined)
  const [stopListeningFn, setStopListeningFn] = useState<(() => void) | null>(null)

  // PROCESSING STATE
  const [enhancedImageUrl, setEnhancedImageUrl] = useState<string | null>(null)
  const [processingStep, setProcessingStep] = useState<0 | 1 | 2 | 3>(0)
  const [processingError, setProcessingError] = useState<string | null>(null)

  // REVIEW FORM STATE
  const [reviewLang, setReviewLang] = useState<"en" | "hi">("en")
  const [productTitle, setProductTitle] = useState("")
  const [productTitleHi, setProductTitleHi] = useState("")
  const [productCategory, setProductCategory] = useState<CraftCategory>("Pottery")
  const [descriptionEn, setDescriptionEn] = useState("")
  const [descriptionHi, setDescriptionHi] = useState("")
  const [craftTags, setCraftTags] = useState<string[]>([])
  const [sellingPrice, setSellingPrice] = useState<number>(0)
  const [stockQuantity, setStockQuantity] = useState<number>(1)
  const [isPublishing, setIsPublishing] = useState(false)
  const [materialsList, setMaterialsList] = useState<string[]>([])
  const [highlightsList, setHighlightsList] = useState<string[]>([])
  
  // Defaults for product payload
  const originCluster = "Sanganer GI Cluster, Jaipur, Rajasthan"
  const artisanName = "Mohan Lal Kumhar"

  // ─── INITIALIZATION / NATIVE BRIDGE ───
  useEffect(() => {
    if (prefillImageUrl) {
      dataUrlToFile(prefillImageUrl, { kind: "image", mimeType: "image/jpeg", fallbackName: "photo.jpg" })
        .then(f => { if (f) setOriginalImageFile(f) })
    }
  }, [prefillImageUrl])

  useEffect(() => {
    if (isOpen && nativeIncomingImage) {
      setOriginalImageDataUrl(nativeIncomingImage)
      dataUrlToFile(nativeIncomingImage, { kind: "image", mimeType: "image/jpeg", fallbackName: "photo.jpg" })
        .then(f => { if (f) setOriginalImageFile(f) })
      consumeNativeIncomingImage?.()
    }
  }, [isOpen, nativeIncomingImage])

  useEffect(() => {
    return () => {
      if (stopListeningFn) stopListeningFn()
    }
  }, [stopListeningFn])

  if (!isOpen) return null

  // ─── INPUT ACTIONS ───
  const handlePhotoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setOriginalImageFile(file)
    const reader = new FileReader()
    reader.onload = ev => {
      if (ev.target?.result) setOriginalImageDataUrl(ev.target.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleCapturePhoto = async (capturedDataUrl: string) => {
    setShowCamera(false)
    setOriginalImageDataUrl(capturedDataUrl)
    const file = await dataUrlToFile(capturedDataUrl, { kind: "image", mimeType: "image/jpeg", fallbackName: "photo.jpg" })
    if (file) setOriginalImageFile(file)
  }

  const handleToggleMic = () => {
    if (isRecording) {
      if (stopListeningFn) {
        stopListeningFn()
        setStopListeningFn(null)
      }
      setIsRecording(false)
    } else {
      setIsRecording(true)
      const stop = speechService.startListening(
        selectedLanguage,
        (text) => {
          setVoiceTranscript(text)
          setRawDescription(text)
        },
        (err) => {
          console.warn("Speech error:", err)
          setIsRecording(false)
          showToast("Microphone error. Please try again.")
        },
        {
          onAudio: (payload) => {
            setIsRecording(false)
            if (payload?.dataUrl) {
              setRecordedAudioDataUrl(payload.dataUrl)
              setRecordedAudioMimeType(payload.mimeType)
            }
          },
          onCancel: () => {
            setIsRecording(false)
          },
        }
      )
      setStopListeningFn(() => stop)
    }
  }

  // ─── GENERATE PIPELINE ───
  const handleGenerate = async () => {
    if (!originalImageFile) { showToast("Please select a product photo."); return }
    if (!recordedAudioDataUrl) { showToast("Please record a voice description."); return }

    setPhase("PROCESSING")
    setProcessingError(null)
    setProcessingStep(1)

    let cleanUrl = enhancedImageUrl

    try {
      // STEP A: Enhance Image (only if we haven't already succeeded in a prior attempt)
      if (!cleanUrl) {
        const enhanceRes = await enhanceImage(originalImageFile)
        if (!enhanceRes || !enhanceRes.clean_image_url) {
          throw new Error("Photo enhancement failed.")
        }
        cleanUrl = enhanceRes.clean_image_url
        setEnhancedImageUrl(cleanUrl)
      }
      
      setProcessingStep(2)

      // STEP B: Catalog from Audio + Enhanced Image URL
      const audioFile = await dataUrlToFile(recordedAudioDataUrl, {
        kind: "audio",
        mimeType: recordedAudioMimeType,
        fallbackName: "voice.m4a",
      })
      if (!audioFile) throw new Error("Could not process recorded audio.")

      const catalogRes = await catalogFromAudio(audioFile, cleanUrl, rawDescription)
      if (!catalogRes || !catalogRes.catalog) throw new Error("Catalog generation failed.")

      // Populate Review form
      populateReviewForm(catalogRes)
      
      setProcessingStep(3)
      setPhase("REVIEW")
    } catch (err: any) {
      console.warn("AI processing error:", err)
      if (err instanceof NeedsReviewError) {
        if (err.transcript && err.transcript.trim()) {
          const t = err.transcript.trim()
          setRawDescription(prev => {
            const p = prev.trim()
            if (!p) return t
            if (p.includes(t) || t.includes(p)) return p.length > t.length ? p : t
            return `${t}\n\n${p}`
          })
        }
        setProcessingError(err.message + " Please describe your product in a little more detail.")
      } else {
        setProcessingError("AI processing failed. Please describe your product in a little more detail.")
      }
    }
  }

  const populateReviewForm = (result: CatalogAudioResult) => {
    const catalog = result.catalog
    const en = catalog.description_english?.trim() || catalog.seo_title
    const hi = catalog.description_hindi?.trim() || en
    
    setProductTitle(catalog.seo_title)
    setProductTitleHi(catalog.title_hindi || catalog.seo_title)
    setDescriptionEn(en)
    setDescriptionHi(hi)

    // Merge backend transcript into the typed notes area
    if (result.transcript && result.transcript.trim()) {
      const t = result.transcript.trim()
      setRawDescription(prev => {
        const p = prev.trim()
        if (!p) return t
        if (p.includes(t) || t.includes(p)) return p.length > t.length ? p : t
        return `${t}\n\n${p}`
      })
    }
    
    if (result.verification?.image_detected_category) {
      setProductCategory(mapServiceCategoryToCraft(result.verification.image_detected_category))
    }
    if (catalog.tags?.length) setCraftTags(catalog.tags)
    if (catalog.suggested_price_inr > 0) setSellingPrice(catalog.suggested_price_inr)
    
    if (result.verification) {
      setHighlightsList((result.verification as any)?.image_detected_materials?.length ? (result.verification as any).image_detected_materials : [])
    }
  }

  // ─── PUBLISH ───
  const handlePublishProduct = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (isPublishing) return // prevent double-clicks
    if (!productTitle.trim() || sellingPrice <= 0) {
      showToast("Please enter a valid title and price.")
      return
    }
    if (!enhancedImageUrl) {
      showToast("Enhanced image is missing. Cannot publish.")
      return
    }
    if (stockQuantity < 1) {
      showToast("Please enter a valid quantity (at least 1).")
      return
    }

    const FRONTEND_TO_BACKEND_CATEGORY: Record<string, string> = {
      "Pottery": "POTTERY",
      "Textile": "TEXTILE",
      "Woodwork": "WOODWORK",
      "Metalwork": "METALWARE",
      "Jewelry": "JEWELLERY",
      "Handicrafts": "OTHER",
      "Folk & Tribal Art": "PAINTING",
      "Bamboo & Cane": "CANE_BAMBOO",
      "Stone Craft": "STONEWORK",
      "Glass & Paper": "OTHER",
      "Craft Materials": "OTHER",
      "Painting": "PAINTING",
      "Leather Craft": "LEATHERWORK",
    }

    const safeTags = Array.isArray(craftTags)
      ? craftTags
      : String(craftTags || "")
          .split(",")
          .map(t => t.trim())
          .filter(Boolean)

    const payload = {
      title: productTitle.trim(),
      titleHi: productTitleHi?.trim() || productTitle.trim(),
      description: {
        en: descriptionEn.trim(),
        hi: descriptionHi?.trim() || descriptionEn.trim()
      },
      price: Number(sellingPrice),
      category: FRONTEND_TO_BACKEND_CATEGORY[productCategory] || "OTHER",
      cleanImageUrl: enhancedImageUrl,
      tags: safeTags.map(t => t.trim()).filter(Boolean),
      stockQuantity: Math.max(1, Math.min(9999, Math.floor(stockQuantity))),
      materials: highlightsList.length > 0 ? highlightsList : ["Not specified"]
    }

    setIsPublishing(true)

    try {
      console.log("Publishing product", payload)
      const token = localStorage.getItem("simplificant_access_token")
      const res = await fetch("/api/v1/products", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { "Authorization": `Bearer ${token}` } : {})
        },
        body: JSON.stringify(payload)
      })
      
      console.log("Publish status", res.status)

      if (!res.ok) {
        const errorBody = await res.json().catch(() => null)
        console.error("Publish error body:", errorBody)
        throw new Error(errorBody?.error?.message || errorBody?.message || `Publish failed (${res.status})`)
      }

      const json = await res.json()
      if (!json.success) {
        throw new Error(json.error?.message || "Server Error")
      }

      const backendProduct = json.data
      
      const BACKEND_TO_FRONTEND_CATEGORY: Record<string, CraftCategory> = {
        POTTERY: "Pottery",
        TEXTILE: "Textile",
        WOODWORK: "Woodwork",
        METALWARE: "Metalwork",
        JEWELLERY: "Jewelry",
        CANE_BAMBOO: "Bamboo & Cane",
        STONEWORK: "Stone Craft",
        PAINTING: "Folk & Tribal Art",
        LEATHERWORK: "Leather Craft",
        OTHER: "Handicrafts",
      }

      const mappedProduct: Product = {
        id: backendProduct.id,
        artisan_id: backendProduct.artisanId,
        name: { en: backendProduct.title, hi: backendProduct.titleHi || backendProduct.title, mr: backendProduct.title, bn: backendProduct.title, ta: backendProduct.title, te: backendProduct.title, gu: backendProduct.title, kn: backendProduct.title, ml: backendProduct.title, pa: backendProduct.title, or: backendProduct.title, es: backendProduct.title, fr: backendProduct.title, de: backendProduct.title, ja: backendProduct.title, ar: backendProduct.title },
        artisan: backendProduct.artisanName,
        location: { en: backendProduct.artisanGiCluster || originCluster, hi: backendProduct.artisanGiCluster || originCluster, mr: originCluster, bn: originCluster, ta: originCluster, te: originCluster, gu: originCluster, kn: originCluster, ml: originCluster, pa: originCluster, or: originCluster, es: originCluster, fr: originCluster, de: originCluster, ja: originCluster, ar: originCluster },
        price: Number(backendProduct.price),
        rating: 5.0,
        reviews: 0,
        category: BACKEND_TO_FRONTEND_CATEGORY[backendProduct.category] || "Handicrafts",
        image: backendProduct.cleanImageUrl,
        description: backendProduct.description as Record<LanguageCode, string>,
        tags: backendProduct.tags,
        gi_tagged: backendProduct.giTagged || true,
        materials: backendProduct.materials,
        stockQuantity: backendProduct.stockQuantity
      }

      onProductPublished(mappedProduct)
      showToast("Product published successfully! 🎉")
      onClose()
    } catch (err: any) {
      console.error("Publish error:", err)
      showToast(err.message || "Unable to publish product. Please try again.")
    } finally {
      setIsPublishing(false)
    }
  }

  // ─── RENDERERS ───
  if (showCamera) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#140F0B]/85 backdrop-blur-md">
        <div className="relative w-full max-w-3xl bg-[#FAF7F2] rounded-3xl overflow-hidden shadow-2xl">
           <LiveCameraViewfinder onCapturePhoto={handleCapturePhoto} onClose={() => setShowCamera(false)} showToast={showToast} />
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-[#140F0B]/85 backdrop-blur-md animate-in fade-in">
      <div className="relative w-full max-w-5xl bg-[#FAF7F2] rounded-3xl border border-[#E0D5C1] shadow-2xl p-5 sm:p-7 max-h-[94vh] overflow-y-auto flex flex-col">
        
        {/* HEADER */}
        <div className="flex items-center justify-between border-b border-[#E4DAC8] pb-4 mb-6 shrink-0">
          <div className="flex items-center gap-3">
            <button onClick={onClose} className="text-[#8C7E6D] hover:text-[#241C15] font-bold text-sm bg-white border border-[#E4DAC8] px-3 py-1.5 rounded-full cursor-pointer transition-colors shadow-xs">
              ← Close
            </button>
            <h2 className="text-xl sm:text-2xl font-bold font-serif text-[#241C15]">
              {phase === "REVIEW" ? "AI Product Listing Ready ✓" : "Create Product Listing"}
            </h2>
          </div>
          {phase === "INPUT" && (
             <p className="hidden sm:block text-xs font-bold text-[#6B6255] uppercase tracking-wide">
               Add photo & voice. AI does the rest.
             </p>
          )}
        </div>

        {/* --- PHASE: INPUT --- */}
        {phase === "INPUT" && (
          <div className="flex-1 flex flex-col space-y-6">
            <div className="flex flex-col md:flex-row gap-6 flex-1">
              
              {/* LEFT CARD: PHOTO */}
              <div className="flex-1 bg-white p-5 sm:p-6 rounded-2xl border border-[#E4DAC8] shadow-sm flex flex-col">
                <h3 className="text-sm font-bold uppercase tracking-wider text-[#9C9182] mb-4">Product Photo</h3>
                {originalImageDataUrl ? (
                  <div className="flex-1 flex flex-col space-y-4">
                    <div className="flex-1 min-h-[200px] flex items-center justify-center bg-gray-50 rounded-xl border border-gray-100 p-2">
                      <img src={originalImageDataUrl} alt="Product preview" className="w-full h-full max-h-[300px] object-contain rounded-lg" />
                    </div>
                    <div className="flex gap-2 mt-auto">
                      <button onClick={() => setShowCamera(true)} className="flex-1 py-2.5 bg-[#FAF7F2] border border-[#E4DAC8] rounded-xl font-bold text-xs text-[#241C15] hover:bg-[#F0EBE0] cursor-pointer transition-colors">
                        📷 Change (Camera)
                      </button>
                      <button onClick={() => fileInputRef.current?.click()} className="flex-1 py-2.5 bg-[#FAF7F2] border border-[#E4DAC8] rounded-xl font-bold text-xs text-[#241C15] hover:bg-[#F0EBE0] cursor-pointer transition-colors">
                        🖼 Change (Gallery)
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col justify-center gap-3">
                    <button onClick={() => setShowCamera(true)} className="py-6 bg-[#FAF7F2] border-2 border-dashed border-[#E4DAC8] hover:border-[#C9922E] rounded-xl font-bold text-[#241C15] hover:bg-[#F0EBE0] cursor-pointer transition-all flex flex-col items-center gap-2">
                      <span className="text-2xl">📷</span> Open Camera
                    </button>
                    <button onClick={() => fileInputRef.current?.click()} className="py-6 bg-[#FAF7F2] border-2 border-dashed border-[#E4DAC8] hover:border-[#C9922E] rounded-xl font-bold text-[#241C15] hover:bg-[#F0EBE0] cursor-pointer transition-all flex flex-col items-center gap-2">
                      <span className="text-2xl">🖼</span> Choose Gallery
                    </button>
                  </div>
                )}
                <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handlePhotoFileChange} className="hidden" />
              </div>

              {/* RIGHT CARD: VOICE */}
              <div className="flex-1 bg-white p-5 sm:p-6 rounded-2xl border border-[#E4DAC8] shadow-sm flex flex-col">
                <h3 className="text-sm font-bold uppercase tracking-wider text-[#9C9182] mb-4">Voice Description</h3>
                
                <div className="flex flex-col items-center justify-center flex-1 space-y-5">
                  <button
                    onClick={handleToggleMic}
                    className={`w-32 h-32 rounded-full flex flex-col items-center justify-center transition-all cursor-pointer shadow-lg border-4 ${isRecording ? "bg-red-600 text-white animate-pulse border-red-200" : "bg-[#C9922E] hover:bg-[#DCA33C] text-[#241C15] border-[#F2D79E] hover:scale-105"}`}
                  >
                    <span className="text-4xl">{isRecording ? "⏹" : "🎙️"}</span>
                    <span className="text-xs font-bold mt-2 uppercase tracking-wide">{isRecording ? "Stop" : "Start"}</span>
                  </button>
                  
                  <div className="h-6 flex items-center justify-center">
                    {recordedAudioDataUrl && !isRecording && (
                      <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200">✓ Voice recorded</span>
                    )}
                    {isRecording && <span className="text-xs font-bold text-red-600">Recording audio...</span>}
                  </div>

                  <div className="w-full mt-auto">
                    <p className="text-[10px] font-bold text-[#6B6255] uppercase mb-1">Optional: Add or edit details</p>
                    <textarea
                      rows={3}
                      value={rawDescription}
                      onChange={(e) => setRawDescription(e.target.value)}
                      placeholder="Your voice transcript will appear here. You can optionally type more details."
                      className="w-full p-3 rounded-xl bg-[#FAF7F2] border border-[#E4DAC8] text-sm outline-none focus:border-[#C9922E] resize-none"
                    />
                  </div>
                </div>
              </div>

            </div>

            {/* BOTTOM: GENERATE BUTTON */}
            <div className="pt-4 border-t border-[#E4DAC8] shrink-0">
              <button
                onClick={handleGenerate}
                disabled={!originalImageFile || !recordedAudioDataUrl}
                className="w-full py-5 rounded-2xl bg-[#241C15] text-[#F7F2E9] font-bold text-xl disabled:opacity-40 hover:bg-[#3A2C20] cursor-pointer shadow-xl transition-all hover:-translate-y-0.5 disabled:hover:translate-y-0"
              >
                ✨ Generate Product Listing
              </button>
              {(!originalImageFile || !recordedAudioDataUrl) && (
                <p className="text-xs text-center text-[#8C7E6D] mt-3 font-semibold tracking-wide">
                  Please add a photo and record a voice description to generate your AI listing.
                </p>
              )}
            </div>
          </div>
        )}

        {/* --- PHASE: PROCESSING --- */}
        {phase === "PROCESSING" && (
          <div className="flex-1 flex flex-col items-center justify-center py-12 space-y-8 animate-in fade-in zoom-in-95">
            <h2 className="text-2xl font-bold font-serif text-[#241C15]">Creating your product listing...</h2>
            
            <div className="w-full max-w-md bg-white p-6 rounded-2xl border border-[#E4DAC8] shadow-sm space-y-5">
              <div className="flex items-center gap-4">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full font-bold ${processingStep > 1 ? "bg-emerald-100 text-emerald-700" : (processingStep === 1 ? "bg-[#241C15] text-white animate-pulse" : "bg-gray-100 text-gray-400")}`}>
                  {processingStep > 1 ? "✓" : "1"}
                </div>
                <span className={`text-sm font-bold ${processingStep >= 1 ? "text-[#241C15]" : "text-gray-400"}`}>
                  Enhancing product photo...
                </span>
              </div>
              
              <div className="flex items-center gap-4">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full font-bold ${processingStep > 2 ? "bg-emerald-100 text-emerald-700" : (processingStep === 2 ? "bg-[#241C15] text-white animate-pulse" : "bg-gray-100 text-gray-400")}`}>
                  {processingStep > 2 ? "✓" : "2"}
                </div>
                <span className={`text-sm font-bold ${processingStep >= 2 ? "text-[#241C15]" : "text-gray-400"}`}>
                  Understanding description & calculating price...
                </span>
              </div>
            </div>

            {originalImageDataUrl && (
              <div className="mt-4 opacity-50">
                 <img src={originalImageDataUrl} alt="Preview" className="w-24 h-24 object-cover rounded-xl border border-[#E4DAC8] grayscale" />
              </div>
            )}

            {processingError && (
              <div className="w-full max-w-md bg-red-50 p-5 rounded-2xl border border-red-200 text-center space-y-4 animate-in fade-in mt-4">
                <p className="text-sm font-bold text-red-700">{processingError}</p>
                <div className="flex gap-3 justify-center">
                  <button onClick={() => setPhase("INPUT")} className="px-4 py-2 bg-white border border-red-200 rounded-lg text-sm font-bold text-red-700 hover:bg-red-50 cursor-pointer">
                    ← Edit Inputs
                  </button>
                  <button onClick={handleGenerate} className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-bold hover:bg-red-700 cursor-pointer">
                    Try Again ⟳
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* --- PHASE: REVIEW --- */}
        {phase === "REVIEW" && (
          <form onSubmit={handlePublishProduct} className="flex-1 flex flex-col space-y-6 animate-in slide-in-from-right-8">
            
            <div className="flex-1 flex flex-col md:flex-row gap-6">
              {/* LEFT: ENHANCED IMAGE */}
              <div className="flex-1 bg-white p-3 rounded-2xl border border-[#E4DAC8] shadow-sm flex flex-col">
                <h3 className="text-[10px] font-bold uppercase tracking-wider text-[#9C9182] mb-2 px-2">AI Enhanced Photo</h3>
                <div className="w-full flex-1 min-h-[300px] bg-gray-50 rounded-xl flex items-center justify-center overflow-hidden border border-gray-100 relative">
                  {enhancedImageUrl && <img src={enhancedImageUrl} alt="Enhanced Product" className="w-full h-full object-contain" />}
                </div>
              </div>

              {/* RIGHT: EDITABLE FIELDS */}
              <div className="flex-1 bg-white p-5 sm:p-6 rounded-2xl border border-[#E4DAC8] shadow-sm space-y-5 overflow-y-auto">
                <div className="flex justify-between items-center mb-2 border-b border-[#E4DAC8] pb-2">
                  <h3 className="text-[10px] font-bold uppercase tracking-wider text-[#9C9182]">AI Generated Details</h3>
                  <div className="flex bg-[#FAF7F2] border border-[#E4DAC8] rounded-lg p-0.5">
                    <button 
                      type="button"
                      onClick={() => setReviewLang("en")}
                      className={`px-3 py-1 rounded-md text-xs font-bold transition-colors ${reviewLang === "en" ? "bg-[#C9922E] text-white shadow-sm" : "text-[#8C7E6D] hover:text-[#241C15]"}`}
                    >
                      English
                    </button>
                    <button 
                      type="button"
                      onClick={() => setReviewLang("hi")}
                      className={`px-3 py-1 rounded-md text-xs font-bold transition-colors ${reviewLang === "hi" ? "bg-[#C9922E] text-white shadow-sm" : "text-[#8C7E6D] hover:text-[#241C15]"}`}
                    >
                      हिन्दी
                    </button>
                  </div>
                </div>
                
                {reviewLang === "en" ? (
                  <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-left-2">
                    <div>
                      <label className="text-xs font-bold text-[#6B6255] uppercase">Product Title</label>
                      <input type="text" value={productTitle} onChange={e => setProductTitle(e.target.value)} className="w-full mt-1 p-2.5 border border-[#E4DAC8] rounded-lg text-base font-bold text-[#241C15] outline-none focus:border-[#C9922E]" />
                    </div>
                    <div>
                      <label className="text-xs font-bold text-[#6B6255] uppercase">English Description</label>
                      <textarea rows={3} value={descriptionEn} onChange={e => setDescriptionEn(e.target.value)} className="w-full mt-1 p-2.5 border border-[#E4DAC8] rounded-lg text-sm outline-none focus:border-[#C9922E] resize-none" />
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-right-2">
                    <div>
                      <label className="text-xs font-bold text-[#6B6255] uppercase">Product Title (Hindi)</label>
                      <input type="text" value={productTitleHi} onChange={e => setProductTitleHi(e.target.value)} className="w-full mt-1 p-2.5 border border-[#E4DAC8] rounded-lg text-base font-bold text-[#241C15] outline-none focus:border-[#C9922E]" />
                    </div>
                    <div>
                      <label className="text-xs font-bold text-[#6B6255] uppercase">Hindi Description</label>
                      <textarea rows={3} value={descriptionHi} onChange={e => setDescriptionHi(e.target.value)} className="w-full mt-1 p-2.5 border border-[#E4DAC8] rounded-lg text-sm outline-none focus:border-[#C9922E] resize-none" />
                    </div>
                  </div>
                )}


                <div className="flex flex-col sm:flex-row gap-5 pt-2">
                  <div className="flex-1 bg-[#FAF7F2] p-4 rounded-xl border border-[#E4DAC8]">
                    <label className="text-xs font-bold text-[#6B6255] uppercase">Suggested Price (₹)</label>
                    <div className="flex items-center mt-1">
                      <span className="text-2xl font-bold text-emerald-800 mr-2">₹</span>
                      <input type="number" value={sellingPrice} onChange={e => setSellingPrice(Number(e.target.value))} className="w-full bg-transparent text-2xl font-bold text-emerald-800 outline-none border-b border-emerald-800/30 focus:border-emerald-800" />
                    </div>
                  </div>
                  
                  <div className="flex-1 flex flex-col justify-between">
                    <div>
                      <label className="text-xs font-bold text-[#6B6255] uppercase">Category</label>
                      <input type="text" value={productCategory} readOnly className="w-full mt-1 p-2 border-b border-[#E4DAC8] text-sm outline-none bg-transparent" />
                    </div>
                    <div>
                      <label className="text-xs font-bold text-[#6B6255] uppercase">Tags</label>
                      <input type="text" value={craftTags.join(", ")} onChange={e => setCraftTags(e.target.value.split(",").map(t=>t.trim()))} className="w-full mt-1 p-2 border-b border-[#E4DAC8] text-sm outline-none focus:border-[#C9922E] bg-transparent" />
                    </div>
                  </div>
                </div>

                {/* QUANTITY AVAILABLE */}
                <div className="bg-[#FAF7F2] p-4 rounded-xl border border-[#E4DAC8]">
                  <label className="text-xs font-bold text-[#6B6255] uppercase">Quantity Available</label>
                  <div className="flex items-center gap-3 mt-2">
                    <button
                      type="button"
                      onClick={() => setStockQuantity(q => Math.max(1, q - 1))}
                      className="w-10 h-10 rounded-lg bg-white border border-[#E4DAC8] text-lg font-bold text-[#241C15] hover:bg-[#FAF7F2] cursor-pointer transition-colors flex items-center justify-center shadow-sm"
                    >
                      −
                    </button>
                    <input
                      type="number"
                      min={1}
                      max={9999}
                      value={stockQuantity}
                      onChange={e => {
                        const v = parseInt(e.target.value, 10)
                        if (!isNaN(v) && v >= 1 && v <= 9999) setStockQuantity(v)
                        else if (e.target.value === "") setStockQuantity(1)
                      }}
                      className="w-20 text-center text-xl font-bold text-[#241C15] border border-[#E4DAC8] rounded-lg p-2 outline-none focus:border-[#C9922E] bg-white"
                    />
                    <button
                      type="button"
                      onClick={() => setStockQuantity(q => Math.min(9999, q + 1))}
                      className="w-10 h-10 rounded-lg bg-white border border-[#E4DAC8] text-lg font-bold text-[#241C15] hover:bg-[#FAF7F2] cursor-pointer transition-colors flex items-center justify-center shadow-sm"
                    >
                      +
                    </button>
                    <span className="text-xs text-[#9C9182] ml-2">pieces</span>
                  </div>
                </div>
              </div>
            </div>

            {/* ACTION BUTTONS */}
            <div className="flex flex-col-reverse sm:flex-row gap-4 pt-4 shrink-0 border-t border-[#E4DAC8] mt-4">
              <button type="button" onClick={() => setPhase("INPUT")} disabled={isPublishing} className="py-4 px-6 rounded-xl bg-white border-2 border-[#E4DAC8] text-[#241C15] font-bold text-lg hover:bg-[#FAF7F2] cursor-pointer transition-colors text-center shadow-sm hover:shadow-md disabled:opacity-40">
                ← Edit Photo / Voice
              </button>
              <button type="submit" disabled={isPublishing} className="flex-1 py-4 px-6 rounded-xl bg-[#C9922E] text-[#241C15] font-bold text-xl hover:bg-[#DCA33C] cursor-pointer shadow-lg transition-transform hover:-translate-y-0.5 text-center disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:translate-y-0">
                {isPublishing ? "Publishing..." : "Publish Product"}
              </button>
            </div>
          </form>
        )}

      </div>
    </div>
  )
}

