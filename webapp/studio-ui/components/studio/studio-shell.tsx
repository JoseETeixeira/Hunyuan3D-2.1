"use client"

import { Boxes } from "lucide-react"
import { JobBanner } from "./job-banner"
import { Model3DViewer } from "./model-3d-viewer"
import { ModelLibrary } from "./model-library"
import { ModelName } from "./model-name"
import { StudioProvider, useStudio } from "./studio-provider"
import { WorkflowPanel } from "./workflow-panel"

export function StudioShell() {
  return (
    <StudioProvider>
      <div className="flex h-dvh flex-col bg-background text-foreground">
        <Header />
        <div className="flex min-h-0 flex-1 flex-col gap-3 p-3 lg:flex-row">
          <aside className="w-full shrink-0 rounded-xl border border-border bg-sidebar p-3 lg:w-64">
            <ModelLibrary />
          </aside>
          <Workspace />
        </div>
      </div>
    </StudioProvider>
  )
}

function Header() {
  return (
    <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
      <div className="flex items-center gap-2.5">
        <span className="flex size-8 items-center justify-center rounded-lg bg-primary/15 text-primary">
          <Boxes className="size-5" />
        </span>
        <div>
          <h1 className="text-sm font-semibold leading-tight">Model Studio</h1>
          <p className="text-[11px] leading-tight text-muted-foreground">Image to 3D · per-model references &amp; textures</p>
        </div>
      </div>
      <ModelName />
    </header>
  )
}

function Workspace() {
  const { activeModel } = useStudio()

  if (!activeModel) {
    return (
      <main className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-border bg-card/40 p-8 text-center">
        <div className="max-w-sm">
          <Boxes className="mx-auto mb-3 size-10 text-muted-foreground" />
          <h2 className="text-base font-semibold">Select or create a model</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Each model keeps its reference views, mesh and textures so you can reuse them across runs without
            re-uploading.
          </p>
        </div>
      </main>
    )
  }

  return (
    <main className="flex min-h-0 flex-1 flex-col gap-3 lg:flex-row">
      <section className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto rounded-xl border border-border bg-card/40 p-4">
        <JobBanner />
        <WorkflowPanel />
      </section>
      <section className="h-72 shrink-0 rounded-xl border border-border bg-card/40 p-4 lg:h-auto lg:w-[40%] lg:max-w-xl">
        <Model3DViewer model={activeModel} />
      </section>
    </main>
  )
}
