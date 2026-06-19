"use client"

import { Check, Layers, Plus, Trash2, X } from "lucide-react"
import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import type { ModelSummary } from "@/lib/types"
import { useStudio } from "./studio-provider"

export function ModelLibrary() {
  const { models, modelsLoading, activeId, selectModel, createModel, deleteModel } = useStudio()
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState("")
  const [busy, setBusy] = useState(false)
  const [seedFile, setSeedFile] = useState<File | null>(null)
  const [seedPreview, setSeedPreview] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  function clearSeed() {
    setSeedFile(null)
    setSeedPreview((p) => {
      if (p) URL.revokeObjectURL(p)
      return null
    })
  }

  async function handleCreate() {
    if (busy) return
    setBusy(true)
    try {
      await createModel(newName.trim() || "Untitled model", seedFile)
      setNewName("")
      clearSeed()
      setCreating(false)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          <Layers className="size-4 text-primary" />
          Models
        </h2>
        <Button size="icon-sm" variant="ghost" onClick={() => setCreating((v) => !v)} aria-label="New model">
          <Plus className="size-4" />
        </Button>
      </div>

      {creating && (
        <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
          <input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            placeholder="Model name"
            className="h-8 rounded-md border border-border bg-background px-2 text-sm outline-none focus-visible:border-ring"
          />
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0] ?? null
              setSeedFile(f)
              setSeedPreview((prev) => {
                if (prev) URL.revokeObjectURL(prev)
                return f ? URL.createObjectURL(f) : null
              })
            }}
          />
          {seedPreview ? (
            <div className="flex items-center gap-2 rounded-md border border-border bg-background p-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={seedPreview}
                alt="Seed preview"
                className="size-12 shrink-0 rounded border border-border object-cover"
              />
              <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">{seedFile?.name}</span>
              <button
                type="button"
                onClick={() => fileInput.current?.click()}
                className="text-xs text-primary hover:underline"
              >
                Change
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              className="rounded-md border border-dashed border-border px-2 py-1.5 text-xs text-muted-foreground hover:border-primary hover:text-foreground"
            >
              Add seed image (optional)
            </button>
          )}
          <div className="flex gap-2">
            <Button size="sm" onClick={handleCreate} disabled={busy} className="flex-1">
              <Check className="size-3.5" />
              Create
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setCreating(false)
                clearSeed()
              }}
            >
              <X className="size-3.5" />
            </Button>
          </div>
        </div>
      )}

      <div className="flex flex-1 flex-col gap-1.5 overflow-y-auto">
        {modelsLoading && <p className="px-1 text-xs text-muted-foreground">Loading…</p>}
        {!modelsLoading && models.length === 0 && (
          <p className="px-1 text-xs text-muted-foreground">No models yet. Create one to start.</p>
        )}
        {models.map((m) => (
          <LibraryRow
            key={m.id}
            model={m}
            active={m.id === activeId}
            onSelect={() => selectModel(m.id)}
            onDelete={() => deleteModel(m.id)}
          />
        ))}
      </div>
    </div>
  )
}

function LibraryRow({
  model,
  active,
  onSelect,
  onDelete,
}: {
  model: ModelSummary
  active: boolean
  onSelect: () => void
  onDelete: () => void
}) {
  const [confirm, setConfirm] = useState(false)
  return (
    <div
      className={`group flex items-center gap-2 rounded-lg border p-2 transition-colors ${
        active ? "border-primary/60 bg-primary/10" : "border-transparent hover:bg-secondary"
      }`}
    >
      <button type="button" onClick={onSelect} className="flex min-w-0 flex-1 items-center gap-2 text-left">
        <span className="size-9 shrink-0 overflow-hidden rounded-md border border-border bg-background">
          {model.previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={model.previewUrl} alt="" className="size-full object-cover" />
          ) : (
            <span className="flex size-full items-center justify-center text-[10px] text-muted-foreground">3D</span>
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{model.name}</span>
          <span className="block text-[11px] text-muted-foreground">{model.textured ? "Textured" : "Draft"}</span>
        </span>
      </button>
      {confirm ? (
        <span className="flex items-center gap-1">
          <Button size="icon-xs" variant="destructive" onClick={onDelete} aria-label="Confirm delete">
            <Check className="size-3" />
          </Button>
          <Button size="icon-xs" variant="ghost" onClick={() => setConfirm(false)} aria-label="Cancel">
            <X className="size-3" />
          </Button>
        </span>
      ) : (
        <Button
          size="icon-xs"
          variant="ghost"
          className="opacity-0 group-hover:opacity-100"
          onClick={() => setConfirm(true)}
          aria-label="Delete model"
        >
          <Trash2 className="size-3" />
        </Button>
      )}
    </div>
  )
}
