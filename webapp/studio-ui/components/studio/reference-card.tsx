"use client"

import { Brush, Check, Eraser, Expand, ImageUp, Lock, Sparkles, Wand2 } from "lucide-react"
import { useRef, useState } from "react"
import { api } from "@/lib/api"
import type { Model, ReferenceImage, ViewId } from "@/lib/types"
import { VIEW_INPUTS, VIEW_LABELS } from "@/lib/views"
import { canGenerate } from "@/lib/workflow"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { ImageDialog } from "./image-dialog"
import { MaskCanvas, type MaskCanvasHandle } from "./mask-canvas"
import { useStudio } from "./studio-provider"

export function ReferenceCard({
  model,
  view,
  locked,
}: {
  model: Model
  view: ViewId
  locked: boolean
}) {
  const { runJob, updateModel } = useStudio()
  const ref: ReferenceImage = model.references[view]
  const [open, setOpen] = useState(false)
  const [prompt, setPrompt] = useState("")
  const fileInput = useRef<HTMLInputElement>(null)
  const maskRef = useRef<MaskCanvasHandle>(null)
  const [brush, setBrush] = useState(28)
  const [hasMask, setHasMask] = useState(false)

  const ready = canGenerate(model, view)
  const deps = VIEW_INPUTS[view]
  const isGenerating = ref.status === "generating"

  async function generate(editPrompt?: string) {
    await runJob(() => api.generateReference(model.id, view, editPrompt), `Generating ${VIEW_LABELS[view]}`)
    setPrompt("")
  }

  async function approve() {
    const m = await api.approveReference(model.id, view)
    updateModel(m)
  }

  async function upload(file: File) {
    const m = await api.uploadReference(model.id, view, file)
    updateModel(m)
    setOpen(false)
  }

  async function editMasked() {
    const blob = await maskRef.current?.getMaskBlob()
    if (!blob) return
    await runJob(() => api.editReferenceMasked(model.id, view, blob, prompt || undefined), `Editing ${VIEW_LABELS[view]}`)
    maskRef.current?.clear()
    setHasMask(false)
    setPrompt("")
  }

  const statusLabel =
    ref.status === "approved"
      ? ref.source === "uploaded"
        ? "Uploaded"
        : "Approved"
      : ref.status === "pending"
        ? "Awaiting review"
        : ref.status === "generating"
          ? "Generating"
          : "Not generated"

  return (
    <>
      <div
        className={`flex flex-col overflow-hidden rounded-xl border bg-card transition-colors ${
          ref.status === "approved" ? "border-primary/50" : "border-border"
        } ${locked ? "opacity-50" : ""}`}
      >
        <button
          type="button"
          onClick={() => setOpen(true)}
          disabled={locked}
          className="group relative aspect-square w-full bg-background outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed"
          aria-label={`Open ${VIEW_LABELS[view]} reference`}
        >
          {ref.url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={ref.url || "/placeholder.svg"} alt={`${VIEW_LABELS[view]} reference`} className="size-full object-cover" />
          ) : (
            <div className="flex size-full flex-col items-center justify-center gap-1 text-muted-foreground">
              {locked ? <Lock className="size-5" /> : <Sparkles className="size-5" />}
              <span className="text-[11px]">{locked ? "Locked" : "Not generated"}</span>
            </div>
          )}
          <span className="absolute left-2 top-2 rounded-md bg-background/80 px-1.5 py-0.5 text-[11px] font-medium backdrop-blur">
            {VIEW_LABELS[view]}
          </span>
          <StatusBadge status={ref.status} source={ref.source} />
          {ref.url && !isGenerating && (
            <span className="absolute inset-0 flex items-center justify-center bg-background/0 opacity-0 transition-opacity group-hover:bg-background/30 group-hover:opacity-100">
              <Expand className="size-5 text-foreground" />
            </span>
          )}
          {isGenerating && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/60 backdrop-blur-sm">
              <Sparkles className="size-5 animate-pulse text-primary" />
            </div>
          )}
        </button>

        <div className="flex flex-col gap-1.5 p-2">
          {!locked && deps.length > 0 && ref.status === "empty" && !ready && (
            <p className="text-[11px] leading-snug text-muted-foreground">
              Needs {deps.map((d) => VIEW_LABELS[d]).join(", ")} approved first
            </p>
          )}

          <div className="flex flex-wrap gap-1.5">
            {ref.status === "empty" && !locked && (
              <Button size="xs" onClick={() => generate()} disabled={!ready || isGenerating} className="flex-1">
                <Sparkles className="size-3" />
                Generate
              </Button>
            )}
            {ref.status === "pending" && (
              <>
                <Button size="xs" onClick={approve} className="flex-1">
                  <Check className="size-3" />
                  Approve
                </Button>
                <Button size="xs" variant="secondary" onClick={() => setOpen(true)}>
                  <Wand2 className="size-3" />
                  Edit
                </Button>
              </>
            )}
            {ref.status === "approved" && (
              <Button size="xs" variant="ghost" onClick={() => setOpen(true)} className="flex-1">
                <Wand2 className="size-3" />
                Edit
              </Button>
            )}
            {ref.status === "generating" && (
              <Button size="xs" variant="ghost" disabled className="flex-1">
                Generating…
              </Button>
            )}
          </div>
        </div>
      </div>

      <ImageDialog
        open={open}
        onOpenChange={setOpen}
        title={`${VIEW_LABELS[view]} view`}
        description={
          deps.length > 0
            ? `Generated from ${deps.map((d) => VIEW_LABELS[d]).join(", ")}.`
            : "Generated from the uploaded seed image."
        }
        imageUrl={ref.url}
        imageAlt={`${VIEW_LABELS[view]} reference`}
        imageSlot={
          ref.url && !locked ? (
            <MaskCanvas key={ref.url} ref={maskRef} imageUrl={ref.url} brushSize={brush} onPaintedChange={setHasMask} />
          ) : undefined
        }
        badge={
          <span className="rounded-md bg-secondary px-2 py-0.5 text-[11px] font-medium text-secondary-foreground">
            {statusLabel}
          </span>
        }
      >
        {locked ? (
          <p className="rounded-md bg-secondary px-2.5 py-2 text-xs text-muted-foreground">
            Locked — approve the previous stage&apos;s views to unlock this one.
          </p>
        ) : (
          <>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor={`prompt-${view}`} className="text-xs">
                Edit instruction
              </Label>
              <textarea
                id={`prompt-${view}`}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder={ref.url ? "Describe what to change…" : "Optional: guide the generation…"}
                rows={3}
                className="resize-none rounded-md border border-border bg-background px-2.5 py-2 text-sm outline-none focus-visible:border-ring"
              />
            </div>

            {ref.url && (
              <div className="flex flex-col gap-2 rounded-md border border-border p-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">Masked edit</span>
                  <Button
                    size="xs"
                    variant="ghost"
                    onClick={() => {
                      maskRef.current?.clear()
                      setHasMask(false)
                    }}
                  >
                    <Eraser className="size-3" />
                    Clear
                  </Button>
                </div>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Brush over the parts to change on the image; only those are edited, the rest stays identical.
                </p>
                <label className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  Brush
                  <input
                    type="range"
                    min={8}
                    max={80}
                    value={brush}
                    onChange={(e) => setBrush(Number(e.target.value))}
                    className="flex-1 accent-primary"
                  />
                </label>
                <Button variant="secondary" onClick={editMasked} disabled={!hasMask || isGenerating}>
                  <Brush className="size-4" />
                  Edit masked region
                </Button>
              </div>
            )}

            <Button onClick={() => generate(prompt || undefined)} disabled={!ready || isGenerating}>
              <Wand2 className="size-4" />
              {ref.url ? "Regenerate whole view" : "Generate"}
            </Button>
            {!ready && ref.status === "empty" && (
              <p className="text-[11px] text-muted-foreground">
                Approve {deps.map((d) => VIEW_LABELS[d]).join(", ")} first to generate.
              </p>
            )}

            {ref.status === "pending" && (
              <Button variant="secondary" onClick={approve}>
                <Check className="size-4" />
                Approve this view
              </Button>
            )}

            <div className="mt-auto flex flex-col gap-1.5 border-t border-border pt-3">
              <Button variant="ghost" onClick={() => fileInput.current?.click()}>
                <ImageUp className="size-4" />
                Upload custom image
              </Button>
              <p className="text-[11px] text-muted-foreground">Uploading replaces this view and marks it approved.</p>
            </div>
          </>
        )}
      </ImageDialog>

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) upload(f)
        }}
      />
    </>
  )
}

function StatusBadge({
  status,
  source,
}: {
  status: ReferenceImage["status"]
  source: ReferenceImage["source"]
}) {
  if (status === "empty") return null
  const map: Record<string, { label: string; cls: string }> = {
    generating: { label: "Generating", cls: "bg-secondary text-secondary-foreground" },
    pending: { label: "Review", cls: "bg-chart-2/20 text-foreground" },
    approved: { label: source === "uploaded" ? "Uploaded" : "Approved", cls: "bg-primary/20 text-primary" },
  }
  const item = map[status]
  if (!item) return null
  return (
    <span className={`absolute right-2 top-2 rounded-md px-1.5 py-0.5 text-[10px] font-medium ${item.cls}`}>
      {item.label}
    </span>
  )
}
