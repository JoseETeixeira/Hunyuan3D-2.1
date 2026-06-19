"use client"

import { Check, Download, Eraser, Hand, Loader2, RotateCcw, Trash2, Upload } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"

const MAX_ZOOM = 6

// Hand-paint touch-up surface: the backdrop is a render of the face AS IT CURRENTLY LOOKS on the mesh,
// and the user paints strokes on top with a palette pulled from the reference image. "Apply" exports an
// RGBA overlay (transparent except the strokes) that the backend bakes straight onto that face. The
// user can also download that backdrop, upload an externally-edited image (baked immediately via the
// same overlay path), and zoom/pan the surface for fine detail.
export function HandPaintCanvas({
  backdropUrl,
  refUrl,
  onApply,
  busy,
  downloadName = "handpaint",
}: {
  backdropUrl: string | null
  refUrl: string | null
  onApply: (overlay: Blob) => void
  busy: boolean
  downloadName?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const uploadInput = useRef<HTMLInputElement>(null)
  const drawing = useRef(false)
  const last = useRef<{ x: number; y: number } | null>(null)
  const panning = useRef(false)
  const panStart = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null)
  const [palette, setPalette] = useState<string[]>([])
  const [color, setColor] = useState("#ffffff")
  const [size, setSize] = useState(14)
  const [erase, setErase] = useState(false)
  const [dim, setDim] = useState(640)
  const [dirty, setDirty] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [panMode, setPanMode] = useState(false)

  // Latest zoom/pan for the (mount-once) wheel listener so it never re-attaches mid-drag.
  const zoomRef = useRef(zoom)
  zoomRef.current = zoom
  const panRef = useRef(pan)
  panRef.current = pan

  // Match the drawing buffer to the (square) backdrop render so strokes bake exactly where painted.
  useEffect(() => {
    if (!backdropUrl) return
    const img = new Image()
    img.onload = () => setDim(Math.max(384, Math.min(1024, img.naturalWidth || 640)))
    img.src = backdropUrl
    // A fresh backdrop resets the view (zoom/pan back to fit).
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [backdropUrl])

  // Pull a palette from the reference image so the user paints in the model's own colors.
  useEffect(() => {
    if (!refUrl) return
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => {
      const p = extractPalette(img)
      setPalette(p)
      if (p.length) setColor(p[0])
    }
    img.onerror = () => setPalette([])
    img.src = refUrl
  }, [refUrl])

  // Mount-once, non-passive wheel zoom centered on the cursor. (React's onWheel can be passive, which
  // blocks preventDefault, so attach natively.)
  useEffect(() => {
    const vp = viewportRef.current
    if (!vp) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = vp.getBoundingClientRect()
      const cx = e.clientX - r.left
      const cy = e.clientY - r.top
      const z = zoomRef.current
      const p = panRef.current
      const z2 = Math.min(MAX_ZOOM, Math.max(1, z * Math.exp(-e.deltaY * 0.0015)))
      if (z2 === z) return
      // Keep the world point under the cursor fixed: world = (cursor - pan) / zoom.
      const wx = (cx - p.x) / z
      const wy = (cy - p.y) / z
      const next = z2 === 1 ? { x: 0, y: 0 } : clampPan({ x: cx - wx * z2, y: cy - wy * z2 }, z2, r.width, r.height)
      setZoom(z2)
      setPan(next)
    }
    vp.addEventListener("wheel", onWheel, { passive: false })
    return () => vp.removeEventListener("wheel", onWheel)
  }, [])

  function ptr(e: React.PointerEvent<HTMLCanvasElement>) {
    const c = canvasRef.current!
    const r = c.getBoundingClientRect()
    // r reflects the on-screen (transformed) rect, so this maps to canvas pixels at any zoom/pan.
    return { x: ((e.clientX - r.left) / r.width) * c.width, y: ((e.clientY - r.top) / r.height) * c.height }
  }
  function stroke(a: { x: number; y: number }, b: { x: number; y: number }) {
    const ctx = canvasRef.current!.getContext("2d")!
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over"
    ctx.strokeStyle = color
    ctx.lineWidth = size
    ctx.lineCap = "round"
    ctx.lineJoin = "round"
    ctx.beginPath()
    ctx.moveTo(a.x, a.y)
    ctx.lineTo(b.x, b.y)
    ctx.stroke()
  }
  function down(e: React.PointerEvent<HTMLCanvasElement>) {
    if (busy || !backdropUrl) return
    e.currentTarget.setPointerCapture(e.pointerId)
    // Middle-mouse or the Pan toggle pans; plain left-drag paints.
    if (e.button === 1 || panMode) {
      e.preventDefault()
      panning.current = true
      panStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y }
      return
    }
    drawing.current = true
    last.current = ptr(e)
    stroke(last.current, last.current)
    setDirty(true)
  }
  function move(e: React.PointerEvent<HTMLCanvasElement>) {
    if (panning.current && panStart.current) {
      const vp = viewportRef.current
      if (!vp) return
      const r = vp.getBoundingClientRect()
      const dx = e.clientX - panStart.current.x
      const dy = e.clientY - panStart.current.y
      setPan(clampPan({ x: panStart.current.panX + dx, y: panStart.current.panY + dy }, zoom, r.width, r.height))
      return
    }
    if (!drawing.current) return
    const p = ptr(e)
    stroke(last.current!, p)
    last.current = p
  }
  function up() {
    drawing.current = false
    last.current = null
    panning.current = false
    panStart.current = null
  }
  function clear() {
    const c = canvasRef.current
    if (!c) return
    c.getContext("2d")!.clearRect(0, 0, c.width, c.height)
    setDirty(false)
  }
  function apply() {
    canvasRef.current?.toBlob((blob) => blob && onApply(blob), "image/png")
  }
  function resetView() {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }

  // Download the current face render so it can be edited externally and re-uploaded.
  async function downloadBackdrop() {
    if (!backdropUrl) return
    try {
      const res = await fetch(backdropUrl)
      const blob = await res.blob()
      const a = document.createElement("a")
      a.href = URL.createObjectURL(blob)
      a.download = `${downloadName}.png`
      a.click()
      URL.revokeObjectURL(a.href)
    } catch {
      /* ignore — download is best-effort */
    }
  }

  // Upload an image and bake it straight onto the face: contain-fit into the square overlay buffer
  // (no distortion; a downloaded backdrop re-registers 1:1) and reuse the existing overlay bake.
  function onUploadChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file || !file.type.startsWith("image/")) return
    const img = new Image()
    img.onload = () => {
      const off = document.createElement("canvas")
      off.width = dim
      off.height = dim
      const ctx = off.getContext("2d")
      if (ctx) {
        const s = Math.min(dim / img.naturalWidth, dim / img.naturalHeight)
        const w = img.naturalWidth * s
        const h = img.naturalHeight * s
        ctx.drawImage(img, (dim - w) / 2, (dim - h) / 2, w, h)
        off.toBlob((b) => b && onApply(b), "image/png")
      }
      URL.revokeObjectURL(img.src)
    }
    img.onerror = () => URL.revokeObjectURL(img.src)
    img.src = URL.createObjectURL(file)
  }

  const swatches = ["#ffffff", "#000000", ...palette]
  const cursor = !backdropUrl ? "default" : panMode ? "grab" : "crosshair"

  return (
    <div className="flex w-full flex-col items-center gap-3">
      <div
        ref={viewportRef}
        className="relative aspect-square w-full max-w-[560px] overflow-hidden rounded-lg border border-border bg-background"
      >
        {backdropUrl ? (
          <div
            className="absolute inset-0"
            style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: "0 0" }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={backdropUrl} alt="current face" className="pointer-events-none absolute inset-0 size-full object-contain" />
            <canvas
              ref={canvasRef}
              width={dim}
              height={dim}
              onPointerDown={down}
              onPointerMove={move}
              onPointerUp={up}
              onPointerLeave={up}
              className="absolute inset-0 size-full touch-none"
              style={{ cursor }}
            />
          </div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted-foreground">
            <Loader2 className="size-6 animate-spin" />
            <span className="text-xs">Rendering the current face…</span>
          </div>
        )}
      </div>

      <div className="flex w-full max-w-[560px] flex-col gap-3">
        {/* View + I/O controls */}
        <div className="flex flex-wrap items-center gap-1.5">
          <Button size="xs" variant="secondary" onClick={() => uploadInput.current?.click()} disabled={busy || !backdropUrl} title="Upload an image and bake it onto this face">
            <Upload className="size-3" />
            Upload
          </Button>
          <Button size="xs" variant="secondary" onClick={downloadBackdrop} disabled={busy || !backdropUrl} title="Download the current face image">
            <Download className="size-3" />
            Download
          </Button>
          <Button
            size="xs"
            variant={panMode ? "default" : "secondary"}
            onClick={() => setPanMode((v) => !v)}
            disabled={busy || !backdropUrl}
            className="ml-auto"
            title="Pan mode (or hold the middle mouse button). Off = paint."
          >
            <Hand className="size-3" />
            Pan
          </Button>
          <Button size="xs" variant="secondary" onClick={resetView} disabled={busy || zoom === 1} title="Reset zoom to fit">
            <RotateCcw className="size-3" />
            {Math.round(zoom * 100)}%
          </Button>
          <input ref={uploadInput} type="file" accept="image/*" onChange={onUploadChange} className="sr-only" />
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {swatches.map((c, i) => (
            <button
              key={`${c}-${i}`}
              type="button"
              onClick={() => {
                setColor(c)
                setErase(false)
              }}
              aria-label={`Use color ${c}`}
              className={`size-6 rounded-md border ${
                color === c && !erase ? "ring-2 ring-ring ring-offset-1 ring-offset-background" : "border-border"
              }`}
              style={{ backgroundColor: c }}
            />
          ))}
          <label className="ml-1 flex size-6 cursor-pointer items-center justify-center rounded-md border border-dashed border-border text-[10px] text-muted-foreground" title="Custom color">
            +
            <input
              type="color"
              value={color}
              onChange={(e) => {
                setColor(e.target.value)
                setErase(false)
              }}
              className="sr-only"
            />
          </label>
          <Button
            size="xs"
            variant={erase ? "default" : "secondary"}
            onClick={() => setErase((v) => !v)}
            className="ml-auto"
            title="Eraser (remove strokes)"
          >
            <Eraser className="size-3" />
            Erase
          </Button>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-[11px] text-muted-foreground">Brush</span>
          <Slider value={[size]} min={2} max={64} step={1} onValueChange={(v) => setSize(Array.isArray(v) ? v[0] : v)} aria-label="Brush size" className="flex-1" />
          <span className="w-7 font-mono text-xs text-primary">{size}</span>
        </div>

        <div className="flex gap-2">
          <Button variant="secondary" onClick={clear} disabled={busy || !dirty} className="flex-1">
            <Trash2 className="size-4" />
            Clear
          </Button>
          <Button onClick={apply} disabled={busy || !dirty || !backdropUrl} className="flex-1">
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
            {busy ? "Baking…" : "Apply paint"}
          </Button>
        </div>
        <p className="text-[10px] text-muted-foreground">
          Painting on the current face. Strokes bake exactly where you draw — palette is sampled from the reference.
          Download the face to edit it elsewhere, or Upload an image to bake it straight on. Scroll to zoom; pan with
          the Pan toggle or the middle mouse button.
        </p>
      </div>
    </div>
  )
}

// Clamp the pan offset so the scaled content always covers the square viewport (transform-origin 0,0).
function clampPan(p: { x: number; y: number }, zoom: number, w: number, h: number): { x: number; y: number } {
  const minX = -(zoom - 1) * w
  const minY = -(zoom - 1) * h
  return { x: Math.min(0, Math.max(minX, p.x)), y: Math.min(0, Math.max(minY, p.y)) }
}

// Coarse palette: bin colors into a 5-bit-per-channel grid over a downsampled copy, drop the near-black
// background frame, and return the most common bins as averaged hex colors.
function extractPalette(img: HTMLImageElement): string[] {
  try {
    const n = 48
    const c = document.createElement("canvas")
    c.width = n
    c.height = n
    const ctx = c.getContext("2d", { willReadFrequently: true })
    if (!ctx) return []
    ctx.drawImage(img, 0, 0, n, n)
    const data = ctx.getImageData(0, 0, n, n).data
    const bins = new Map<string, { count: number; r: number; g: number; b: number }>()
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] < 200) continue
      const r = data[i]
      const g = data[i + 1]
      const b = data[i + 2]
      if (r < 16 && g < 16 && b < 16) continue // skip the dark background frame
      const key = `${r >> 5}-${g >> 5}-${b >> 5}`
      const e = bins.get(key) ?? { count: 0, r: 0, g: 0, b: 0 }
      e.count++
      e.r += r
      e.g += g
      e.b += b
      bins.set(key, e)
    }
    return [...bins.values()]
      .sort((a, b) => b.count - a.count)
      .slice(0, 12)
      .map((e) => rgbToHex(Math.round(e.r / e.count), Math.round(e.g / e.count), Math.round(e.b / e.count)))
  } catch {
    return []
  }
}

function rgbToHex(r: number, g: number, b: number): string {
  return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("")
}
