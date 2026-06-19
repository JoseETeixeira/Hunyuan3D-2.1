"use client"

import { Check, Pencil, X } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { useStudio } from "./studio-provider"

export function ModelName() {
  const { activeModel, renameActive } = useStudio()
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState("")

  if (!activeModel) return null

  function start() {
    setValue(activeModel!.name)
    setEditing(true)
  }

  async function save() {
    const next = value.trim()
    if (next && next !== activeModel!.name) await renameActive(next)
    setEditing(false)
  }

  if (editing) {
    return (
      <div className="flex items-center gap-1.5">
        <input
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") save()
            if (e.key === "Escape") setEditing(false)
          }}
          className="h-8 w-56 max-w-full rounded-md border border-border bg-background px-2 text-base font-semibold outline-none focus-visible:border-ring"
        />
        <Button size="icon-sm" variant="ghost" onClick={save} aria-label="Save name">
          <Check className="size-4" />
        </Button>
        <Button size="icon-sm" variant="ghost" onClick={() => setEditing(false)} aria-label="Cancel">
          <X className="size-4" />
        </Button>
      </div>
    )
  }

  return (
    <button
      type="button"
      onClick={start}
      className="group flex items-center gap-2 text-left"
      title="Rename model"
    >
      <span className="text-lg font-semibold text-balance">{activeModel.name}</span>
      <Pencil className="size-3.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
    </button>
  )
}
