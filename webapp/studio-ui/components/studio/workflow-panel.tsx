"use client"

import { useEffect, useState } from "react"
import { allReferencesApproved, approvedCount } from "@/lib/workflow"
import { ReferencesPanel } from "./references-panel"
import { useStudio } from "./studio-provider"
import { TexturePanel } from "./texture-panel"

type Step = "references" | "texture"

export function WorkflowPanel() {
  const { activeModel } = useStudio()
  const [step, setStep] = useState<Step>("references")

  const refsReady = activeModel ? allReferencesApproved(activeModel) : false

  // Auto-advance to texturing once everything is approved (first time only).
  useEffect(() => {
    if (refsReady && activeModel?.textureStage !== "none") setStep("texture")
  }, [refsReady, activeModel?.textureStage])

  if (!activeModel) return null

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <StepTab
          n={1}
          label="Reference views"
          sub={`${approvedCount(activeModel)}/10 approved`}
          active={step === "references"}
          onClick={() => setStep("references")}
        />
        <div className="h-px flex-1 bg-border" />
        <StepTab
          n={2}
          label="Mesh & textures"
          sub={textureSub(activeModel.textureStage)}
          active={step === "texture"}
          disabled={!refsReady && activeModel.textureStage === "none"}
          onClick={() => setStep("texture")}
        />
      </div>

      {step === "references" ? <ReferencesPanel model={activeModel} /> : <TexturePanel model={activeModel} />}
    </div>
  )
}

function textureSub(stage: string) {
  switch (stage) {
    case "none":
      return "not started"
    case "base-running":
      return "painting base"
    case "base-done":
      return "base ready"
    case "refacing":
      return "refacing faces"
    case "complete":
      return "complete"
    default:
      return ""
  }
}

function StepTab({
  n,
  label,
  sub,
  active,
  disabled,
  onClick,
}: {
  n: number
  label: string
  sub: string
  active: boolean
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
        active ? "border-primary/60 bg-primary/10" : "border-border bg-card hover:bg-secondary"
      }`}
    >
      <span
        className={`flex size-6 items-center justify-center rounded-full text-xs font-semibold ${
          active ? "bg-primary text-primary-foreground" : "bg-secondary text-foreground"
        }`}
      >
        {n}
      </span>
      <span className="leading-tight">
        <span className="block text-sm font-medium">{label}</span>
        <span className="block text-[11px] text-muted-foreground">{sub}</span>
      </span>
    </button>
  )
}
