"use client"

import { Box, Crosshair, Download, Upload } from "lucide-react"
import { useRef, useState } from "react"
import { api, type CustomCam } from "@/lib/api"
import { MARKER_LABELS, type MarkerId, type Model } from "@/lib/types"
import type { ModelViewerElement } from "@/types/model-viewer"
import { buttonVariants } from "@/components/ui/button"
import { HandPaintCanvas } from "./hand-paint-canvas"
import { ImageDialog } from "./image-dialog"
import { useStudio } from "./studio-provider"

const FORMATS = ["glb", "fbx", "blend"] as const

export function Model3DViewer({ model }: { model: Model | null }) {
  const { runJob, updateModel, rigActive, rigJoint, setRigJoint, notifyMeshUploaded } = useStudio()
  const [tab, setTab] = useState<"shape" | "tex">("tex")
  const [fmt, setFmt] = useState<string>("glb")
  const mvRef = useRef<HTMLElement | null>(null)
  const blendInput = useRef<HTMLInputElement>(null)

  // Custom (free-camera) hand-paint state.
  const [customOpen, setCustomOpen] = useState(false)
  const [angle, setAngle] = useState<{ elev: number; azim: number; cam?: CustomCam } | null>(null)
  const [backdrop, setBackdrop] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const shapeUrl = model?.meshUrl ?? null
  const texUrl = model?.texturedUrl ?? null
  const riggedUrl = model?.rig?.riggedUrl ?? null
  const showTex = tab === "tex" && texUrl
  // While the Rigging step is open, show the rigged GLB so marker coords + click raycasts share one
  // coordinate space.
  const baseSrc = rigActive && riggedUrl ? riggedUrl : showTex ? texUrl : shapeUrl
  // Cache-bust on updatedAt so a bake (which bumps updatedAt) reloads the GLB in the viewer.
  const src = baseSrc ? `${baseSrc}${baseSrc.includes("?") ? "&" : "?"}v=${model?.updatedAt ?? 0}` : null
  const markers = model?.rig?.markers ?? {}
  const showMarkers = rigActive && !!riggedUrl

  // Capture the live orbit of the 3D viewer and open a hand-paint canvas for that exact camera.
  // model-viewer orbit is in radians: theta = azimuth (0 = front/+Z), phi = polar from +Y (PI/2 = equator).
  async function paintThisAngle() {
    if (!model || !texUrl) return
    const mv = mvRef.current as ModelViewerElement | null
    if (!mv?.getCameraOrbit) return
    const orbit = mv.getCameraOrbit()
    const thetaDeg = (orbit.theta * 180) / Math.PI
    const phiDeg = (orbit.phi * 180) / Math.PI
    const elev = Math.max(-90, Math.min(90, 90 - phiDeg))
    // model-viewer theta maps 1:1 to the render azimuth: it matches every built-in camera
    // (PROJECTION_CAMS front=0, right=90, back=180, left=270; corners fl=315/fr=45). Using
    // (360 - theta) would reflect right<->left (mirrored render), so use theta directly.
    const azim = ((thetaDeg % 360) + 360) % 360
    // Also capture the live PERSPECTIVE framing — vertical fov, orbit radius (zoom) and pan target —
    // so the backdrop matches what's on screen, not just the angle. Without these the backend falls
    // back to its orthographic full-object framing ("close but not exact"). The same cam is sent to
    // the render and the bake so strokes land where painted. Guarded so an older model-viewer that
    // lacks these getters degrades gracefully to the angle-only (orthographic) path.
    let cam: CustomCam | undefined
    if (typeof mv.getFieldOfView === "function" && typeof mv.getCameraTarget === "function") {
      const t = mv.getCameraTarget()
      // Capture the viewport aspect too: model-viewer's fov is VERTICAL, so a wide viewer shows a wide
      // (large horizontal-fov) view. Rendering the backdrop at this aspect reproduces it exactly
      // instead of a square centre-crop. Guard against a zero-height read.
      const aspect = mv.clientHeight > 0 ? mv.clientWidth / mv.clientHeight : 1
      cam = { fov: mv.getFieldOfView(), radius: orbit.radius, target: [t.x, t.y, t.z], aspect }
    }
    const a = { elev, azim, cam }
    setAngle(a)
    setBackdrop(null)
    setCustomOpen(true)
    setBusy(true)
    try {
      await runJob(() => api.renderCustomView(model.id, a.elev, a.azim, a.cam), "Rendering custom view")
      setBackdrop(api.customRenderUrl(model.id, Date.now()))
    } finally {
      setBusy(false)
    }
  }

  // Replace the mesh with an uploaded .blend (keeps references, resets texture so it can be re-built).
  async function uploadBlend(file: File) {
    if (!model) return
    setBusy(true)
    try {
      await runJob(() => api.uploadMesh(model.id, file), "Importing .blend")
      notifyMeshUploaded() // jump the workflow to the Mesh & textures step — the mesh is ready to texture
    } finally {
      setBusy(false)
    }
  }

  // Rig marker placement: click the model with a joint selected → raycast to the surface, send the
  // hit point + normal so the backend places the marker at the limb center.
  async function placeMarker(e: React.MouseEvent) {
    if (!rigActive || !rigJoint || !model) return
    const mv = mvRef.current as ModelViewerElement | null
    if (!mv?.positionAndNormalFromPoint) return
    const r = mv.getBoundingClientRect()
    const hit = mv.positionAndNormalFromPoint(e.clientX - r.left, e.clientY - r.top)
    if (!hit) return
    const point: [number, number, number] = [hit.position.x, hit.position.y, hit.position.z]
    const normal: [number, number, number] = [hit.normal.x, hit.normal.y, hit.normal.z]
    setBusy(true)
    try {
      const joint = rigJoint
      const m = await api.setRigMarker(model.id, joint, point, normal)
      updateModel(m)
    } finally {
      setBusy(false)
    }
  }

  // Bake the painted/uploaded overlay at the captured camera (same camera as the render → aligned).
  async function applyCustom(overlay: Blob) {
    if (!model || !angle) return
    setBusy(true)
    try {
      await runJob(() => api.handpaintCustomView(model.id, angle.elev, angle.azim, overlay, angle.cam), "Hand painting custom view")
      setCustomOpen(false)
      setBackdrop(null)
    } finally {
      setBusy(false)
    }
  }

  // AI fix the captured custom view with Gemini (keep style + base colours, fix only inconsistencies).
  async function applyGeminiFixCustom(image: Blob) {
    if (!model || !angle) return
    setBusy(true)
    try {
      await runJob(() => api.handpaintAiCustomView(model.id, angle.elev, angle.azim, image, angle.cam), "AI fixing custom view")
      setCustomOpen(false)
      setBackdrop(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex flex-col gap-2">
        <div className="flex w-fit items-center gap-1 rounded-lg border border-border bg-card p-1">
          <button
            type="button"
            onClick={() => setTab("shape")}
            disabled={!shapeUrl}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors disabled:opacity-40 ${
              tab === "shape" && shapeUrl ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Shape
          </button>
          <button
            type="button"
            onClick={() => setTab("tex")}
            disabled={!texUrl}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors disabled:opacity-40 ${
              tab === "tex" && texUrl ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Textured
          </button>
        </div>
        {model ? (
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => blendInput.current?.click()}
              disabled={busy}
              title="Upload a .blend to replace the mesh (keeps references, resets the texture)"
              className={buttonVariants({ variant: "secondary", size: "sm" })}
            >
              <Upload className="size-3.5" />
              Upload .blend
            </button>
            <input
              ref={blendInput}
              type="file"
              accept=".blend"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0]
                e.target.value = ""
                if (f) uploadBlend(f)
              }}
            />
            {shapeUrl || texUrl ? (
              <>
                {texUrl ? (
                  <button
                    type="button"
                    onClick={paintThisAngle}
                    disabled={busy}
                    title="Hand-paint the model from the angle you're currently viewing"
                    className={buttonVariants({ variant: "secondary", size: "sm" })}
                  >
                    <Crosshair className="size-3.5" />
                    Paint this angle
                  </button>
                ) : null}
                <select
                  value={fmt}
                  onChange={(e) => setFmt(e.target.value)}
                  className="h-8 rounded-md border border-border bg-card px-2 text-xs text-foreground outline-none focus-visible:border-ring"
                >
                  {FORMATS.map((f) => (
                    <option key={f} value={f}>
                      {f.toUpperCase()}
                    </option>
                  ))}
                </select>
                <a
                  href={api.downloadUrl(model.id, fmt)}
                  download
                  className={buttonVariants({ variant: "secondary", size: "sm" })}
                >
                  <Download className="size-3.5" />
                  Export
                </a>
              </>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className="relative flex-1 overflow-hidden rounded-xl border border-border bg-[radial-gradient(circle_at_50%_30%,oklch(0.24_0.03_240),oklch(0.15_0.02_250))]">
        {src ? (
          // Lock the field of view to a constant. model-viewer's default "auto" fov keeps re-framing
          // as you orbit (and its default zoom changes fov + radius under damping), so the fov captured
          // for "Paint this angle" was a moving, mid-animation value that didn't match the settled view
          // — making the backdrop more zoomed than the live camera. With fov fixed, zoom becomes a pure
          // radius dolly and the captured fov always equals what's on screen, so the backdrop matches.
          <model-viewer
            ref={mvRef}
            src={src}
            alt={`3D model: ${model?.name ?? ""}`}
            camera-controls
            field-of-view="30deg"
            min-field-of-view="30deg"
            max-field-of-view="30deg"
            shadow-intensity="1"
            exposure="1.1"
            environment-image="neutral"
            onClick={showMarkers && rigJoint ? placeMarker : undefined}
            style={{
              width: "100%",
              height: "100%",
              backgroundColor: "transparent",
              cursor: showMarkers && rigJoint ? "crosshair" : undefined,
            }}
          >
            {showMarkers
              ? Object.entries(markers).map(([key, pos]) =>
                  pos ? (
                    <button
                      key={key}
                      type="button"
                      slot={`hotspot-${key}`}
                      data-position={`${pos[0]}m ${pos[1]}m ${pos[2]}m`}
                      data-normal="0m 1m 0m"
                      onClick={(e) => {
                        e.stopPropagation()
                        setRigJoint(key as MarkerId)
                      }}
                      title={MARKER_LABELS[key as MarkerId]}
                      className={`flex size-3.5 items-center justify-center rounded-full border-2 text-[0px] shadow ${
                        rigJoint === key
                          ? "border-white bg-primary ring-2 ring-primary"
                          : "border-white bg-primary/70 hover:bg-primary"
                      }`}
                    >
                      {MARKER_LABELS[key as MarkerId]}
                    </button>
                  ) : null,
                )
              : null}
          </model-viewer>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-muted-foreground">
            <Box className="size-10 opacity-40" />
            <div>
              <p className="text-sm font-medium text-foreground">No mesh yet</p>
              <p className="text-xs">Approve references, then generate the textured model.</p>
            </div>
          </div>
        )}
        {model?.meshStats ? (
          <div className="pointer-events-none absolute bottom-2 left-2 rounded-md border border-border bg-background/80 px-2 py-1 font-mono text-[11px] text-muted-foreground backdrop-blur">
            {model.meshStats.faces.toLocaleString()} faces · {model.meshStats.vertices.toLocaleString()} verts
          </div>
        ) : null}
      </div>

      <ImageDialog
        open={customOpen}
        onOpenChange={(o) => {
          setCustomOpen(o)
          if (!o) setBackdrop(null)
        }}
        title="Custom view"
        description={
          angle
            ? `Free camera at elev ${Math.round(angle.elev)}° / azim ${Math.round(angle.azim)}°. Paint or upload, then Apply to bake it onto the mesh.`
            : "Rotate the model, then paint this angle."
        }
        imageUrl={null}
        imageAlt="custom view"
        imageSlot={
          <HandPaintCanvas
            backdropUrl={backdrop}
            refUrl={backdrop}
            onApply={applyCustom}
            onGeminiFix={applyGeminiFixCustom}
            busy={busy}
            downloadName="handpaint-custom"
          />
        }
        badge={
          busy ? <span className="rounded-md bg-secondary px-2 py-0.5 text-[11px] font-medium">Working…</span> : null
        }
      >
        <p className="rounded-md bg-secondary px-2.5 py-2 text-[11px] text-muted-foreground">
          This bakes at the exact camera you captured, so strokes land where you paint. It is a free-camera touch-up —
          it isn&apos;t tied to one of the ten faces and pushes its own texture-history snapshot.
        </p>
      </ImageDialog>
    </div>
  )
}
