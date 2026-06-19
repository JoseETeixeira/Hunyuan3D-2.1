"use client"

import type { ReactNode } from "react"
import { ImageOff } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

// Shared dialog that shows a large (real-size) image on the left and an
// arbitrary controls column on the right. Used for editing reference views
// and individual model faces, and for simply inspecting a view at full size.
export function ImageDialog({
  open,
  onOpenChange,
  title,
  description,
  imageUrl,
  imageAlt,
  imageSlot,
  badge,
  children,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description?: string
  imageUrl: string | null
  imageAlt: string
  imageSlot?: ReactNode
  badge?: ReactNode
  children?: ReactNode
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[96vw] max-w-[1000px] gap-0 overflow-hidden p-0 sm:max-w-[1000px]">
        <div className="grid max-h-[88vh] grid-cols-1 md:grid-cols-[1fr_320px]">
          <div className="flex min-h-[280px] items-center justify-center overflow-auto bg-background p-4 md:max-h-[88vh]">
            {imageSlot ? (
              imageSlot
            ) : imageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={imageUrl || "/placeholder.svg"}
                alt={imageAlt}
                className="max-h-full max-w-full object-contain"
              />
            ) : (
              <div className="flex flex-col items-center gap-2 text-muted-foreground">
                <ImageOff className="size-8" />
                <span className="text-sm">No image yet</span>
              </div>
            )}
          </div>

          <div className="flex flex-col gap-4 border-t border-border bg-card p-5 md:border-l md:border-t-0">
            <DialogHeader className="space-y-1 text-left">
              <div className="flex items-center justify-between gap-2">
                <DialogTitle className="text-base">{title}</DialogTitle>
                {badge}
              </div>
              {description ? (
                <DialogDescription className="text-xs leading-relaxed">{description}</DialogDescription>
              ) : null}
            </DialogHeader>
            {children ? <div className="flex flex-1 flex-col gap-3">{children}</div> : null}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
