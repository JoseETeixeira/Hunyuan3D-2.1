"use client"

import { Crosshair, Loader2, Wand2 } from "lucide-react"
import { useEffect } from "react"
import { api } from "@/lib/api"
import { MARKER_IDS, MARKER_LABELS, type Model } from "@/lib/types"
import { Button } from "@/components/ui/button"
import { useStudio } from "./studio-provider"

// Step 3: AI rig (UniRig). Run the rig, then select a joint and click the model in the 3D viewer to
// move that marker to the limb center; "Apply rig changes" re-skins on the edited skeleton.
export function RigPanel({ model }: { model: Model }) {
  const { runJob, rigJoint, setRigJoint, setRigActive } = useStudio()
  const rig = model.rig
  const stage = rig?.stage ?? "none"
  const hasMesh = !!(model.meshUrl || model.texturedUrl)
  const busy = stage === "rigging" || stage === "reskinning"
  const markers = rig?.markers ?? {}
  const rigged = !!rig?.riggedUrl

  // Mark the rig step active so the 3D viewer shows markers + accepts click-to-place; reset on leave.
  useEffect(() => {
    setRigActive(true)
    return () => {
      setRigActive(false)
      setRigJoint(null)
    }
  }, [setRigActive, setRigJoint])

  async function runRig() {
    await runJob(() => api.rigModel(model.id), "AI rigging")
  }
  async function applyRig() {
    await runJob(() => api.applyRig(model.id), "Re-skinning rig")
  }

  if (!hasMesh) {
    return (
      <p className="rounded-md bg-secondary px-2.5 py-2 text-xs text-muted-foreground">
        Generate or upload a mesh in step 2 first, then rig it here.
      </p>
    )
  }

  if (!rigged) {
    return (
      <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
        <div className="flex items-start gap-3">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
            <Wand2 className="size-5" />
          </span>
          <div className="flex-1">
            <h3 className="text-sm font-semibold">Rig with AI</h3>
            <p className="text-xs text-muted-foreground">
              UniRig predicts a skeleton and skin weights for the mesh, then surfaces the key joints as
              markers you can reposition. Export the rigged model as GLB, FBX, or .blend.
            </p>
          </div>
        </div>
        <Button onClick={runRig} disabled={busy}>
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
          {busy ? "Rigging…" : "Rig with AI"}
        </Button>
      </div>
    )
  }

  const placed = MARKER_IDS.filter((k) => markers[k]).length
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Joint markers</h3>
        <span className="font-mono text-xs text-muted-foreground">{placed}/12 placed</span>
      </div>
      <p className="rounded-md bg-secondary px-2.5 py-2 text-[11px] text-muted-foreground">
        <Crosshair className="mr-1 inline size-3" />
        Select a joint, then click the model in the 3D viewer — the marker snaps to the limb&apos;s
        center. Hit <strong>Apply rig changes</strong> to re-skin on the new joints.
      </p>

      <div className="grid grid-cols-2 gap-1.5">
        {MARKER_IDS.map((k) => {
          const isSel = rigJoint === k
          const isPlaced = !!markers[k]
          return (
            <button
              key={k}
              type="button"
              onClick={() => setRigJoint(isSel ? null : k)}
              disabled={busy}
              className={`flex items-center justify-between rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors disabled:opacity-50 ${
                isSel
                  ? "border-ring bg-primary/15 text-foreground ring-1 ring-ring"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              <span>{MARKER_LABELS[k]}</span>
              <span className={`size-2 rounded-full ${isPlaced ? "bg-primary" : "bg-muted-foreground/30"}`} />
            </button>
          )
        })}
      </div>

      <div className="flex gap-2">
        <Button variant="secondary" onClick={runRig} disabled={busy} className="flex-1">
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
          Re-rig
        </Button>
        <Button onClick={applyRig} disabled={busy} className="flex-1">
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Crosshair className="size-4" />}
          {busy ? "Working…" : "Apply rig changes"}
        </Button>
      </div>
      <p className="text-[10px] text-muted-foreground">
        Export the rigged model (GLB / FBX / .blend) from the 3D viewer toolbar.
      </p>
    </div>
  )
}
