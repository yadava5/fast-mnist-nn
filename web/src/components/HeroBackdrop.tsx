import { MeshGradient } from '@paper-design/shaders-react';
import { useReducedMotion } from '../hooks/useReducedMotion';

/**
 * Animated OKLCH mesh-gradient backdrop rendered behind the hero section.
 * The shader respects `prefers-reduced-motion` (speed -> 0) so it doesn't
 * animate for users who opt out.
 */
export function HeroBackdrop() {
  const reduced = useReducedMotion();
  return (
    <MeshGradient
      className="hero-backdrop"
      colors={[
        'oklch(0.25 0.04 260)',
        'oklch(0.5 0.22 280)',
        'oklch(0.45 0.15 220)',
        'oklch(0.2 0.02 260)',
      ]}
      speed={reduced ? 0 : 0.25}
      distortion={0.7}
      swirl={0.4}
      aria-hidden
    />
  );
}
