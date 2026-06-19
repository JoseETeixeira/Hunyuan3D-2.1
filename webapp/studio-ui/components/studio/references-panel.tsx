"use client"

import { CheckCircle2, ChevronRight, ImageUp, Sparkles } from "lucide-react"
import { useRef } from "react"
import { api } from "@/lib/api"
import type { Model, ReferenceStage } from "@/lib/types"
import { STAGE_LABELS, STAGE_ORDER, STAGE_VIEWS } from "@/lib/views"
import { canGenerate, isStageApproved, isStageUnlocked } from "@/lib/workflow"
import { Button } from "@/components/ui/button"
import { ReferenceCard } from "./reference-card"
import { useStudio } from "./studio-provider"

export function ReferencesPanel({ model }: { model: Model }) {
  return (
    <div className="flex flex-col gap-4">
      <SeedRow model={model} />
      {STAGE_ORDER.map((stage) => (
        <StageBlock key={stage} model={model} stage={stage} />
      ))}
    </div>
  )
}

function SeedRow({ model }: { model: Model }) {
  const { updateModel } = useStudio()
  const fileInput = useRef<HTMLInputElement>(null)

  async function uploadSeed(file: File) {
    // Update the SEED image (the generation input) — not the front reference.
    const m = await api.updateSeed(model.id, file)
    updateModel(m)
  }

  return (
    <div className="flex items-center gap-3 rounded-xl border border-border bg-card p-3">
      <span className="size-14 shrink-0 overflow-hidden rounded-lg border border-border bg-background">
        {model.seedImageUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={model.seedImageUrl} alt="Seed" className="size-full object-cover" />
        ) : (
          <span className="flex size-full items-center justify-center text-[10px] text-muted-foreground">none</span>
        )}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium">Seed image</p>
        <p className="text-xs text-muted-foreground">
          Any view of the model. Front is generated from this, then the rest follow imperatively.
        </p>
      </div>
      <Button size="sm" variant="secondary" onClick={() => fileInput.current?.click()}>
        <ImageUp className="size-3.5" />
        {model.seedImageUrl ? "Replace" : "Upload"}
      </Button>
      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) uploadSeed(f)
        }}
      />
    </div>
  )
}

function StageBlock({ model, stage }: { model: Model; stage: ReferenceStage }) {
  const { runJob } = useStudio()
  const unlocked = isStageUnlocked(model, stage)
  const approved = isStageApproved(model, stage)
  const views = STAGE_VIEWS[stage]
  const generatable = views.filter((v) => model.references[v].status === "empty" && canGenerate(model, v))

  async function generateAll() {
    // Kick off sequentially so the mock/back-end ordering stays sane.
    for (const v of generatable) {
      // eslint-disable-next-line no-await-in-loop
      await runJob(() => api.generateReference(model.id, v), `Generating ${v}`)
    }
  }

  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span
          className={`flex size-5 items-center justify-center rounded-full text-[11px] font-semibold ${
            approved ? "bg-primary text-primary-foreground" : unlocked ? "bg-secondary text-foreground" : "bg-secondary/50 text-muted-foreground"
          }`}
        >
          {approved ? <CheckCircle2 className="size-3.5" /> : stage}
        </span>
        <h3 className="text-sm font-semibold">{STAGE_LABELS[stage]}</h3>
        {!unlocked && (
          <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <ChevronRight className="size-3" />
            approve previous stage first
          </span>
        )}
        {unlocked && generatable.length > 1 && (
          <Button size="xs" variant="secondary" className="ml-auto" onClick={generateAll}>
            <Sparkles className="size-3" />
            Generate all
          </Button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        {views.map((v) => (
          <ReferenceCard key={v} model={model} view={v} locked={!unlocked} />
        ))}
      </div>
    </section>
  )
}
