"use client"

import { AlertTriangle, Loader2, X } from "lucide-react"
import { useStudio } from "./studio-provider"

export function JobBanner() {
  const { job, jobError, clearJobError } = useStudio()

  if (jobError) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
        <AlertTriangle className="size-4 shrink-0" />
        <span className="flex-1">{jobError}</span>
        <button type="button" onClick={clearJobError} aria-label="Dismiss">
          <X className="size-4" />
        </button>
      </div>
    )
  }

  if (!job) return null

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-card px-3 py-2">
      <div className="flex items-center gap-2 text-sm">
        <Loader2 className="size-4 shrink-0 animate-spin text-primary" />
        <span className="flex-1 text-foreground">{job.label}</span>
        <span className="font-mono text-xs text-muted-foreground">{Math.round(job.progress)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
        <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${job.progress}%` }} />
      </div>
    </div>
  )
}
