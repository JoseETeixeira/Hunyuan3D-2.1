"use client"

import { Brush, Check, Expand, History, Layers, Loader2, Paintbrush, Pencil, RotateCcw, Settings2, Trash2, Undo2, Wand2 } from "lucide-react"
import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { DEFAULT_MESH_CONFIG, type FaceMode, type MeshConfig, type Model, type ViewId } from "@/lib/types"
import { ALL_VIEWS, VIEW_LABELS } from "@/lib/views"
import { allReferencesApproved, nonFrontViews, refacedCount } from "@/lib/workflow"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { HandPaintCanvas } from "./hand-paint-canvas"
import { ImageDialog } from "./image-dialog"
import { useStudio } from "./studio-provider"

type FaceMethod = FaceMode | "handpaint"

export function TexturePanel({ model }: { model: Model }) {
  const { runJob, updateModel } = useStudio()
  const refsReady = allReferencesApproved(model)
  const stage = model.textureStage
  const [config, setConfig] = useState<MeshConfig>(DEFAULT_MESH_CONFIG)
  const [sourceView, setSourceView] = useState<ViewId>(model.meshSourceView ?? "front")
  const [histBusy, setHistBusy] = useState(false)

  // Restore/reset return the updated Model synchronously (no GPU job) — apply it directly.
  async function restoreTo(seq: number) {
    setHistBusy(true)
    try {
      updateModel(await api.restoreTexture(model.id, seq))
    } finally {
      setHistBusy(false)
    }
  }
  async function resetTexture() {
    setHistBusy(true)
    try {
      updateModel(await api.resetTexture(model.id))
    } finally {
      setHistBusy(false)
    }
  }

  async function generateMesh() {
    await runJob(() => api.generateMesh(model.id, sourceView, config), `Generating mesh from ${VIEW_LABELS[sourceView]}`)
  }
  async function buildTextures() {
    await runJob(() => api.textureBase(model.id, config), "Per-face AI paint")
  }

  async function refaceRemaining() {
    for (const v of nonFrontViews()) {
      if (model.faces[v].status === "done") continue
      // eslint-disable-next-line no-await-in-loop
      await runJob(() => api.refaceFace(model.id, v), `Refacing ${VIEW_LABELS[v]}`)
    }
  }

  async function refaceEveryFace() {
    for (const v of nonFrontViews()) {
      // eslint-disable-next-line no-await-in-loop
      await runJob(() => api.refaceFace(model.id, v), `Refacing ${VIEW_LABELS[v]}`)
    }
  }

  if (stage === "none") {
    const hasMesh = !!model.meshUrl
    return (
      <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
        <div className="flex items-start gap-3">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
            <Paintbrush className="size-5" />
          </span>
          <div className="flex-1">
            <h3 className="text-sm font-semibold">Build textured model</h3>
            <p className="text-xs text-muted-foreground">
              First generate the 3D <strong>mesh</strong> from a reference view (preview it, regenerate with another
              view if needed), then apply the base <strong>Per-face AI paint</strong> texture.
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="mesh-source" className="text-xs">
            Mesh source view
          </Label>
          <select
            id="mesh-source"
            value={sourceView}
            onChange={(e) => setSourceView(e.target.value as ViewId)}
            className="h-8 rounded-md border border-border bg-background px-2 text-sm outline-none focus-visible:border-ring"
          >
            {ALL_VIEWS.map((v) => (
              <option key={v} value={v}>
                {VIEW_LABELS[v]}
              </option>
            ))}
          </select>
          <span className="text-[10px] text-muted-foreground">
            Which reference drives the 3D shape. A 3/4 corner often gives Hunyuan more depth than a flat view.
          </span>
        </div>

        <MeshConfigForm config={config} onChange={setConfig} />

        {!refsReady && (
          <p className="rounded-md bg-secondary px-2.5 py-2 text-xs text-muted-foreground">
            Approve all ten reference views first, or upload custom references for each face.
          </p>
        )}

        {!hasMesh ? (
          <Button onClick={generateMesh} disabled={!refsReady}>
            <Wand2 className="size-4" />
            Generate mesh from {VIEW_LABELS[sourceView]}
          </Button>
        ) : (
          <>
            <p className="rounded-md bg-secondary px-2.5 py-2 text-xs text-muted-foreground">
              Mesh ready — preview it in the 3D viewer (Shape tab). Not happy? Pick another source view and regenerate.
            </p>
            <div className="flex gap-2">
              <Button variant="secondary" onClick={generateMesh} disabled={!refsReady} className="flex-1">
                <RotateCcw className="size-4" />
                Regenerate mesh
              </Button>
              <Button onClick={buildTextures} disabled={!refsReady} className="flex-1">
                <Paintbrush className="size-4" />
                Generate textures
              </Button>
            </div>
          </>
        )}

        <p className="text-[10px] text-muted-foreground">
          Generated meshes are hole-filled automatically (watertight). To use your own geometry, upload
          a .blend from the 3D viewer toolbar.
        </p>
      </div>
        <TextureHistory model={model} onRestore={restoreTo} busy={histBusy} />
      </div>
    )
  }

  const totalFaces = nonFrontViews().length + 1
  const done = refacedCount(model)

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between rounded-xl border border-border bg-card p-3">
        <div className="flex items-center gap-2">
          <Layers className="size-4 text-primary" />
          <div>
            <p className="text-sm font-medium">Face textures</p>
            <p className="text-xs text-muted-foreground">
              {done} / {totalFaces} faces textured
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {stage === "complete" ? (
            <Button size="sm" variant="secondary" onClick={refaceEveryFace}>
              <Brush className="size-3.5" />
              Reface all
            </Button>
          ) : (
            <Button size="sm" onClick={refaceRemaining}>
              <Brush className="size-3.5" />
              Reface remaining
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            onClick={resetTexture}
            disabled={histBusy}
            title="Delete the texture back to the untextured mesh (history kept)"
          >
            <Trash2 className="size-3.5" />
            Reset
          </Button>
        </div>
      </div>

      <p className="rounded-md bg-secondary px-2.5 py-2 text-xs text-muted-foreground">
        {stage === "complete"
          ? "All faces textured. Reface any face (including the front) to repaint it on the mesh, or open a face to repaint with per-face AI paint."
          : "Reface each face to paint its reference onto the mesh. You can reface any face — including the front base — at any time."}
      </p>

      <TextureHistory model={model} onRestore={restoreTo} busy={histBusy} />

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        <FaceTile model={model} view="front" base />
        {nonFrontViews().map((v) => (
          <FaceTile key={v} model={model} view={v} />
        ))}
      </div>

      <details className="rounded-xl border border-border bg-card p-3">
        <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium">
          <Wand2 className="size-4 text-primary" />
          Remesh (regenerate the 3D shape)
        </summary>
        <div className="mt-3 flex flex-col gap-2">
          <p className="rounded-md bg-secondary px-2.5 py-2 text-xs text-muted-foreground">
            Regenerating the mesh rebuilds the 3D shape from a reference view and <strong>resets the texture and
            its history</strong>. Use it if the shape itself is wrong.
          </p>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="remesh-source" className="text-xs">
              Mesh source view
            </Label>
            <select
              id="remesh-source"
              value={sourceView}
              onChange={(e) => setSourceView(e.target.value as ViewId)}
              className="h-8 rounded-md border border-border bg-background px-2 text-sm outline-none focus-visible:border-ring"
            >
              {ALL_VIEWS.map((v) => (
                <option key={v} value={v}>
                  {VIEW_LABELS[v]}
                </option>
              ))}
            </select>
          </div>
          <Button variant="secondary" onClick={generateMesh}>
            <RotateCcw className="size-4" />
            Remesh from {VIEW_LABELS[sourceView]}
          </Button>
        </div>
      </details>
    </div>
  )
}

// ── Texture undo timeline ────────────────────────────────────────────────────
function TextureHistory({
  model,
  onRestore,
  busy,
}: {
  model: Model
  onRestore: (seq: number) => void
  busy: boolean
}) {
  const hist = model.textureHistory ?? []
  if (!hist.length) return null
  return (
    <div className="rounded-xl border border-border bg-card p-3">
      <div className="mb-2 flex items-center gap-2">
        <History className="size-4 text-primary" />
        <p className="text-sm font-medium">Texture history</p>
        <span className="text-xs text-muted-foreground">
          {hist.length} step{hist.length > 1 ? "s" : ""}
        </span>
      </div>
      <ul className="flex max-h-52 flex-col gap-1 overflow-y-auto">
        {[...hist].reverse().map((e) => (
          <li
            key={e.seq}
            className="flex items-center justify-between gap-2 rounded-md bg-background/60 px-2 py-1.5"
          >
            <div className="flex min-w-0 items-center gap-2">
              <span className="font-mono text-[10px] text-muted-foreground">#{e.seq}</span>
              <span className="truncate text-xs">{e.label}</span>
            </div>
            <Button
              size="xs"
              variant="ghost"
              disabled={busy}
              onClick={() => onRestore(e.seq)}
              title={`Restore the texture to "${e.label}"`}
            >
              <Undo2 className="size-3" />
              Restore
            </Button>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[10px] text-muted-foreground">
        Restoring rolls the whole-mesh texture back to that step. Re-texture forward from there.
      </p>
    </div>
  )
}

// ── Mesh generation parameters ───────────────────────────────────────────────
function MeshConfigForm({ config, onChange }: { config: MeshConfig; onChange: (c: MeshConfig) => void }) {
  function set<K extends keyof MeshConfig>(key: K, value: MeshConfig[K]) {
    onChange({ ...config, [key]: value })
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-background/60 p-3">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-xs font-medium text-foreground">
          <Settings2 className="size-3.5 text-primary" />
          Generation settings
        </span>
        <button
          type="button"
          onClick={() => onChange(DEFAULT_MESH_CONFIG)}
          className="flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
        >
          <RotateCcw className="size-3" />
          Reset
        </button>
      </div>

      <SliderRow
        label="Inference steps"
        hint="Shape denoising steps"
        value={config.inferenceSteps}
        min={1}
        max={50}
        step={1}
        onChange={(v) => set("inferenceSteps", v)}
      />
      <SliderRow
        label="Guidance scale"
        hint="Prompt adherence (CFG)"
        value={config.guidanceScale}
        min={0}
        max={20}
        step={0.5}
        onChange={(v) => set("guidanceScale", v)}
      />
      <SliderRow
        label="Octree resolution"
        hint="Surface detail"
        value={config.octreeResolution}
        min={64}
        max={512}
        step={64}
        onChange={(v) => set("octreeResolution", v)}
      />
      <SliderRow
        label="Texture views"
        hint="Camera views baked into the texture"
        value={config.textureViews}
        min={1}
        max={12}
        step={1}
        onChange={(v) => set("textureViews", v)}
      />

      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="cfg-seed" className="text-[11px]">
            Seed
          </Label>
          <Input
            id="cfg-seed"
            type="number"
            value={config.seed}
            onChange={(e) => set("seed", Number(e.target.value) || 0)}
            className="h-8 text-xs"
          />
          <span className="text-[10px] text-muted-foreground">0 = random</span>
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="cfg-faces" className="text-[11px]">
            Mesh faces
          </Label>
          <Input
            id="cfg-faces"
            type="number"
            min={1000}
            step={1000}
            value={config.meshFaces}
            onChange={(e) => set("meshFaces", Number(e.target.value) || 0)}
            className="h-8 text-xs"
          />
          <span className="text-[10px] text-muted-foreground">Target triangle count</span>
        </div>
      </div>
    </div>
  )
}

function SliderRow({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  hint: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between">
        <Label className="text-[11px]">{label}</Label>
        <span className="font-mono text-xs text-primary">{value}</span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : v)}
        aria-label={label}
      />
      <span className="text-[10px] text-muted-foreground">{hint}</span>
    </div>
  )
}

// ── Individual face tile + edit modal ────────────────────────────────────────
function FaceTile({ model, view, base = false }: { model: Model; view: ViewId; base?: boolean }) {
  const { runJob } = useStudio()
  const face = model.faces[view]
  const ref = model.references[view]
  const complete = model.textureStage === "complete"
  const busy = face.status === "texturing"
  const [open, setOpen] = useState(false)
  const [method, setMethod] = useState<FaceMethod>("reface")
  const [prompt, setPrompt] = useState("")
  const [backdrop, setBackdrop] = useState<string | null>(null)
  const [rendering, setRendering] = useState(false)

  async function apply() {
    const label = method === "paint" ? "Painting" : "Refacing"
    await runJob(
      () =>
        method === "reface"
          ? api.refaceFace(model.id, view, prompt || undefined)
          : api.editFace(model.id, view, "paint", prompt || undefined),
      `${label} ${VIEW_LABELS[view]}`,
    )
    setOpen(false)
    setPrompt("")
  }

  // Hand paint: render the current face as a backdrop, let the user paint, then bake the overlay.
  async function prepareCanvas() {
    setRendering(true)
    try {
      await runJob(() => api.renderFace(model.id, view), `Rendering ${VIEW_LABELS[view]}`)
      setBackdrop(api.faceRenderUrl(model.id, view, Date.now()))
    } finally {
      setRendering(false)
    }
  }
  async function applyHandpaint(overlay: Blob) {
    await runJob(() => api.handpaintFace(model.id, view, overlay), `Hand painting ${VIEW_LABELS[view]}`)
    setOpen(false)
    setBackdrop(null)
  }
  useEffect(() => {
    if (open && method === "handpaint" && !backdrop && !rendering) prepareCanvas()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, method])

  // One-click reface straight from the tile — works on every face.
  async function quickReface() {
    await runJob(() => api.refaceFace(model.id, view), `Refacing ${VIEW_LABELS[view]}`)
  }

  // Revert just this face back to the base texture (drops its refaces/paints).
  const canClear = (model.textureHistory?.length ?? 0) > 0 && face.status === "done"
  async function clearFace() {
    await runJob(() => api.clearFace(model.id, view), `Clearing ${VIEW_LABELS[view]}`)
  }

  const statusLabel =
    face.status === "done" ? (face.mode === "paint" ? "Painted" : "Refaced") : base ? "Base" : "Pending"

  return (
    <>
      <div
        className={`flex flex-col overflow-hidden rounded-xl border bg-card ${
          face.status === "done" ? "border-primary/40" : "border-border"
        }`}
      >
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="group relative aspect-square w-full bg-background outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={`Open ${VIEW_LABELS[view]} face`}
        >
          {ref.url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={ref.url || "/placeholder.svg"} alt={`${VIEW_LABELS[view]} reference`} className="size-full object-cover" />
          ) : (
            <div className="flex size-full items-center justify-center text-[11px] text-muted-foreground">no ref</div>
          )}
          <span className="absolute left-2 top-2 rounded-md bg-background/80 px-1.5 py-0.5 text-[11px] font-medium backdrop-blur">
            {VIEW_LABELS[view]}
          </span>
          {busy ? (
            <div className="absolute inset-0 flex items-center justify-center bg-background/60 backdrop-blur-sm">
              <Loader2 className="size-5 animate-spin text-primary" />
            </div>
          ) : (
            ref.url && (
              <span className="absolute inset-0 flex items-center justify-center opacity-0 transition-opacity group-hover:bg-background/30 group-hover:opacity-100">
                <Expand className="size-5 text-foreground" />
              </span>
            )
          )}
          {face.status === "done" && (
            <span className="absolute right-2 top-2 flex items-center gap-1 rounded-md bg-primary/20 px-1.5 py-0.5 text-[10px] font-medium text-primary">
              <Check className="size-3" />
              {face.mode === "paint" ? "Paint" : base ? "Base" : "Reface"}
            </span>
          )}
        </button>

        <div className="flex items-center gap-1.5 p-2">
          <Button size="xs" onClick={quickReface} disabled={busy} className="flex-1">
            <Brush className="size-3" />
            {busy ? "Working…" : "Reface"}
          </Button>
          {canClear && (
            <Button
              size="xs"
              variant="ghost"
              onClick={clearFace}
              disabled={busy}
              aria-label={`Clear ${VIEW_LABELS[view]} back to base`}
              title="Revert this face to the base texture"
            >
              <Undo2 className="size-3" />
            </Button>
          )}
          <Button
            size="xs"
            variant="secondary"
            onClick={() => setOpen(true)}
            disabled={busy}
            aria-label={`More options for ${VIEW_LABELS[view]}`}
            title="More options"
          >
            <Settings2 className="size-3" />
          </Button>
        </div>
      </div>

      <ImageDialog
        open={open}
        onOpenChange={(o) => {
          setOpen(o)
          if (!o) setBackdrop(null)
        }}
        title={`${VIEW_LABELS[view]} face`}
        description="Reface from the reference, regenerate with AI paint, or hand-paint touch-ups on the current face."
        imageUrl={ref.url}
        imageAlt={`${VIEW_LABELS[view]} reference`}
        imageSlot={
          method === "handpaint" ? (
            <HandPaintCanvas backdropUrl={backdrop} refUrl={ref.url} onApply={applyHandpaint} busy={busy || rendering} downloadName={`handpaint-${view}`} />
          ) : undefined
        }
        badge={
          busy ? (
            <span className="rounded-md bg-secondary px-2 py-0.5 text-[11px] font-medium">Working…</span>
          ) : (
            <span
              className={`rounded-md px-2 py-0.5 text-[11px] font-medium ${
                face.status === "done" ? "bg-primary/20 text-primary" : "bg-secondary text-muted-foreground"
              }`}
            >
              {statusLabel}
            </span>
          )
        }
      >
        <div className="flex flex-col gap-1.5">
          <Label className="text-xs">Texture method</Label>
          <div className="grid grid-cols-3 gap-1 rounded-lg border border-border bg-background/60 p-1">
            <MethodOption
              active={method === "reface"}
              onClick={() => setMethod("reface")}
              icon={<Brush className="size-3.5" />}
              label="Reface"
            />
            <MethodOption
              active={method === "paint"}
              onClick={() => setMethod("paint")}
              icon={<Paintbrush className="size-3.5" />}
              label="AI paint"
            />
            <MethodOption
              active={method === "handpaint"}
              onClick={() => setMethod("handpaint")}
              icon={<Pencil className="size-3.5" />}
              label="Hand paint"
            />
          </div>
          <p className="text-[11px] text-muted-foreground">
            {method === "reface"
              ? `Re-projects the ${VIEW_LABELS[view]} reference onto the mesh with the reface model.`
              : method === "paint"
                ? "Regenerates this face from scratch with per-face AI paint."
                : "Hand-paint touch-ups on the current face, using a palette sampled from the reference."}
          </p>
        </div>

        {method === "handpaint" ? (
          <p className="rounded-md bg-secondary px-2.5 py-2 text-[11px] text-muted-foreground">
            Brush over small imperfections on the left, then Apply — strokes bake exactly where you paint. The
            reference (above) is your color guide.
          </p>
        ) : (
          <>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor={`face-prompt-${view}`} className="text-xs">
                Edit instruction <span className="text-muted-foreground">(optional)</span>
              </Label>
              <textarea
                id={`face-prompt-${view}`}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe a texture change to guide the model…"
                rows={3}
                className="resize-none rounded-md border border-border bg-background px-2.5 py-2 text-sm outline-none focus-visible:border-ring"
              />
            </div>

            <Button onClick={apply} disabled={busy} className="mt-auto">
              {method === "reface" ? <Brush className="size-4" /> : <Paintbrush className="size-4" />}
              {busy ? "Working…" : method === "reface" ? "Reface this face" : "Paint this face"}
            </Button>
          </>
        )}
      </ImageDialog>
    </>
  )
}

function MethodOption({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`flex items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
        active
          ? "bg-primary text-primary-foreground"
          : "text-muted-foreground hover:bg-secondary hover:text-foreground"
      }`}
    >
      {icon}
      {label}
    </button>
  )
}
