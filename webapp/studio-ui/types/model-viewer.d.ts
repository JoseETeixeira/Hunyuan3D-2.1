import type { DetailedHTMLProps, HTMLAttributes } from "react"

// The <model-viewer> DOM element exposes the live camera orbit (theta/phi/radius in RADIANS).
// theta = azimuth around the model (0 = front/+Z), phi = polar angle from +Y (PI/2 = equator).
export interface ModelViewerElement extends HTMLElement {
  getCameraOrbit(): { theta: number; phi: number; radius: number }
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
      }
    }
  }
}
