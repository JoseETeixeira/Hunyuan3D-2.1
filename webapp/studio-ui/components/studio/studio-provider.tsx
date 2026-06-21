"use client"

import { createContext, useCallback, useContext, useMemo, useState } from "react"
import useSWR from "swr"
import { api } from "@/lib/api"
import type { MarkerId, Model, ModelSummary } from "@/lib/types"
import { useJobRunner } from "@/lib/use-job"

interface StudioContextValue {
  models: ModelSummary[]
  modelsLoading: boolean
  activeModel: Model | null
  activeId: string | null
  selecting: boolean
  selectModel: (id: string) => Promise<void>
  clearActive: () => void
  createModel: (name: string, seed: File | null) => Promise<void>
  renameActive: (name: string) => Promise<void>
  deleteModel: (id: string) => Promise<void>
  updateModel: (model: Model) => void
  refreshModels: () => void
  // job runner
  job: { id: string; label: string; progress: number } | null
  jobError: string | null
  clearJobError: () => void
  runJob: (start: () => Promise<import("@/lib/types").Job>, label: string) => Promise<unknown>
  // Step 3 rig: the joint marker currently selected for placement (shared by RigPanel + viewer).
  rigJoint: MarkerId | null
  setRigJoint: (joint: MarkerId | null) => void
  // True while the Rigging step is open, so the 3D viewer shows markers + enables click-to-place.
  rigActive: boolean
  setRigActive: (v: boolean) => void
  // Bumped after a .blend upload so the workflow jumps straight to the Mesh & textures step.
  meshUploadTick: number
  notifyMeshUploaded: () => void
}

const StudioContext = createContext<StudioContextValue | null>(null)

export function StudioProvider({ children }: { children: React.ReactNode }) {
  const { data: models, isLoading, mutate } = useSWR<ModelSummary[]>("models", api.listModels)
  const [activeModel, setActiveModel] = useState<Model | null>(null)
  const [selecting, setSelecting] = useState(false)
  const [rigJoint, setRigJoint] = useState<MarkerId | null>(null)
  const [rigActive, setRigActive] = useState(false)
  const [meshUploadTick, setMeshUploadTick] = useState(0)
  const notifyMeshUploaded = useCallback(() => setMeshUploadTick((t) => t + 1), [])

  const updateModel = useCallback(
    (model: Model) => {
      setActiveModel(model)
      mutate()
    },
    [mutate],
  )

  const { active: job, error: jobError, run: runJob, clearError } = useJobRunner(updateModel)

  const selectModel = useCallback(async (id: string) => {
    setSelecting(true)
    try {
      const m = await api.getModel(id)
      setActiveModel(m)
    } finally {
      setSelecting(false)
    }
  }, [])

  const createModel = useCallback(
    async (name: string, seed: File | null) => {
      const m = await api.createModel(name, seed)
      setActiveModel(m)
      mutate()
    },
    [mutate],
  )

  const renameActive = useCallback(
    async (name: string) => {
      if (!activeModel) return
      const m = await api.renameModel(activeModel.id, name)
      setActiveModel(m)
      mutate()
    },
    [activeModel, mutate],
  )

  const deleteModel = useCallback(
    async (id: string) => {
      await api.deleteModel(id)
      setActiveModel((cur) => (cur?.id === id ? null : cur))
      mutate()
    },
    [mutate],
  )

  const value = useMemo<StudioContextValue>(
    () => ({
      models: models ?? [],
      modelsLoading: isLoading,
      activeModel,
      activeId: activeModel?.id ?? null,
      selecting,
      selectModel,
      clearActive: () => setActiveModel(null),
      createModel,
      renameActive,
      deleteModel,
      updateModel,
      refreshModels: () => mutate(),
      job,
      jobError,
      clearJobError: clearError,
      runJob,
      rigJoint,
      setRigJoint,
      rigActive,
      setRigActive,
      meshUploadTick,
      notifyMeshUploaded,
    }),
    [
      models,
      isLoading,
      activeModel,
      selecting,
      selectModel,
      createModel,
      renameActive,
      deleteModel,
      updateModel,
      mutate,
      job,
      jobError,
      clearError,
      runJob,
      rigJoint,
      rigActive,
      meshUploadTick,
      notifyMeshUploaded,
    ],
  )

  return <StudioContext.Provider value={value}>{children}</StudioContext.Provider>
}

export function useStudio() {
  const ctx = useContext(StudioContext)
  if (!ctx) throw new Error("useStudio must be used within StudioProvider")
  return ctx
}
