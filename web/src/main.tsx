import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { GeistSans, GeistMono } from './lib/fonts'
import { loadThree } from './lib/three-preload'
import './index.css'
import App from './App.tsx'

// Surface the lazy three loader on window so the DevTools console can warm
// the chunk manually. No static work happens at startup; the chunk only
// downloads when a caller invokes window.__loadThree().
if (typeof window !== 'undefined') {
  (window as unknown as { __loadThree?: typeof loadThree }).__loadThree =
    loadThree
}

// Apply Geist classes to <html> so the fonts take effect site-wide and so
// `--font-geist-sans` / `--font-geist-mono` are available to every descendant.
document.documentElement.classList.add(GeistSans.className)
document.documentElement.style.setProperty(GeistMono.variable, "var(--font-geist-mono)")

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
