"use client"

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react"

export interface MaskCanvasHandle {
  // Returns a PNG blob where the brushed pixels are opaque (the region to edit), or null if nothing
  // was painted. The backend converts this into an OpenAI inpaint mask.
  getMaskBlob: () => Promise<Blob | null>
  clear: () => void
}

// An image with a transparent brush canvas overlaid exactly on top. Brush strokes are painted at full
// opacity (the canvas element is shown at 0.5 CSS opacity so the user sees a translucent highlight),
// so exporting the canvas yields a clean mask of the painted region.
export const MaskCanvas = forwardRef<
  MaskCanvasHandle,
  { imageUrl: string; brushSize: number; onPaintedChange?: (painted: boolean) => void }
>(function MaskCanvas({ imageUrl, brushSize, onPaintedChange }, ref) {
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawing = useRef(false)
  const last = useRef<{ x: number; y: number } | null>(null)
  const painted = useRef(false)
  const brush = useRef(brushSize)
  brush.current = brushSize

  function syncSize() {
    const img = imgRef.current
    const cv = canvasRef.current
    if (!img || !cv) return
    const w = Math.round(img.clientWidth)
    const h = Math.round(img.clientHeight)
    if (!w || !h) return
    if (cv.width !== w || cv.height !== h) {
      cv.width = w
      cv.height = h
      painted.current = false
      onPaintedChange?.(false)
    }
    cv.style.width = `${w}px`
    cv.style.height = `${h}px`
  }

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const ro = new ResizeObserver(syncSize)
    ro.observe(img)
    if (img.complete) syncSize()
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl])

  function ctx() {
    return canvasRef.current?.getContext("2d") ?? null
  }
  function point(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }
  function paintTo(x: number, y: number) {
    const c = ctx()
    if (!c) return
    c.strokeStyle = "rgba(255,64,64,1)"
    c.fillStyle = "rgba(255,64,64,1)"
    c.lineCap = "round"
    c.lineJoin = "round"
    c.lineWidth = brush.current
    if (last.current) {
      c.beginPath()
      c.moveTo(last.current.x, last.current.y)
      c.lineTo(x, y)
      c.stroke()
    }
    c.beginPath()
    c.arc(x, y, brush.current / 2, 0, Math.PI * 2)
    c.fill()
    last.current = { x, y }
    if (!painted.current) {
      painted.current = true
      onPaintedChange?.(true)
    }
  }

  useImperativeHandle(ref, () => ({
    getMaskBlob: () =>
      new Promise<Blob | null>((resolve) => {
        if (!painted.current || !canvasRef.current) return resolve(null)
        canvasRef.current.toBlob((b) => resolve(b), "image/png")
      }),
    clear: () => {
      const c = ctx()
      const cv = canvasRef.current
      if (c && cv) c.clearRect(0, 0, cv.width, cv.height)
      painted.current = false
      last.current = null
      onPaintedChange?.(false)
    },
  }))

  return (
    <div className="relative inline-block max-h-full max-w-full leading-none">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        src={imageUrl}
        alt="Reference"
        onLoad={syncSize}
        draggable={false}
        className="max-h-[80vh] max-w-full select-none object-contain"
      />
      <canvas
        ref={canvasRef}
        className="absolute left-0 top-0 cursor-crosshair touch-none"
        style={{ opacity: 0.5 }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          drawing.current = true
          last.current = null
          const p = point(e)
          paintTo(p.x, p.y)
        }}
        onPointerMove={(e) => {
          if (!drawing.current) return
          const p = point(e)
          paintTo(p.x, p.y)
        }}
        onPointerUp={(e) => {
          drawing.current = false
          last.current = null
          try {
            e.currentTarget.releasePointerCapture(e.pointerId)
          } catch {
            /* ignore */
          }
        }}
        onPointerLeave={() => {
          drawing.current = false
          last.current = null
        }}
      />
    </div>
  )
})
