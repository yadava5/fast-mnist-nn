import { lazy, Suspense } from 'react';
import { useReducedMotion } from '../hooks/useReducedMotion';

// Lazy-load the shader so the ~8 KB gz WebGL blob is code-split out of the
// main bundle. First paint shows a dark fallback while the chunk hydrates.
const MeshGradient = lazy(() =>
  import('@paper-design/shaders-react').then((mod) => ({ default: mod.MeshGradient })),
);

/**
 * Animated OKLCH mesh-gradient backdrop rendered behind the hero section.
 * The shader respects `prefers-reduced-motion` (speed -> 0) so it doesn't
 * animate for users who opt out.
 */
export function HeroBackdrop() {
  const reduced = useReducedMotion();
  return (
    <Suspense fallback={<div className="hero-backdrop hero-backdrop-fallback" aria-hidden />}>
      <MeshGradient
        className="hero-backdrop"
        colors={[
          'oklch(0.28 0.04 180)',
          'oklch(0.72 0.17 195)',
          'oklch(0.76 0.16 145)',
          'oklch(0.24 0.03 300)',
        ]}
        speed={reduced ? 0 : 0.25}
        distortion={0.7}
        swirl={0.4}
        aria-hidden
      />
    </Suspense>
  );
}
