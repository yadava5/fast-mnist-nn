import { useReducedMotion as useMotionReducedMotion } from 'motion/react';

/**
 * Wrapper around Motion's useReducedMotion so every consumer has one
 * import path. Returns `true` when the user has requested reduced motion
 * at the OS level.
 */
export function useReducedMotion(): boolean {
  return !!useMotionReducedMotion();
}
