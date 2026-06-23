import type { DetailedHTMLProps, HTMLAttributes } from "react"

// The <model-viewer> DOM element exposes the live camera orbit (theta/phi/radius in RADIANS).
// theta = azimuth around the model (0 = front/+Z), phi = polar angle from +Y (PI/2 = equator).
export interface ModelViewerVector3D {
  x: number
  y: number
  z: number
  toString(): string
}

export interface ModelViewerElement extends HTMLElement {
  getCameraOrbit(): { theta: number; phi: number; radius: number }
  // Realtime VERTICAL field of view in degrees, and the orbit pivot (camera target) in model meters.
  // Captured with the orbit to reproduce the exact perspective view for free-camera hand-paint.
  getFieldOfView(): number
  getCameraTarget(): ModelViewerVector3D
  // Raycast from a viewport pixel to the model surface; null when the ray misses.
  positionAndNormalFromPoint(
    pixelX: number,
    pixelY: number,
  ): { position: ModelViewerVector3D; normal: ModelViewerVector3D } | null
}

// Minimal JSX typing for the <model-viewer> web component.
declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      "model-viewer": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement> & {
        src?: string
        alt?: string
        poster?: string
        "auto-rotate"?: boolean
        "camera-controls"?: boolean
        "shadow-intensity"?: string | number
        exposure?: string | number
        "environment-image"?: string
        "rotation-per-second"?: string
        "interaction-prompt"?: string
        "disable-zoom"?: boolean
        "disable-pan"?: boolean
        "field-of-view"?: string
        "min-field-of-view"?: string
        "max-field-of-view"?: string
      }
    }
  }
}
