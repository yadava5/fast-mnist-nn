import { motion, useScroll, useSpring } from 'motion/react';
import { useReducedMotion } from '../hooks/useReducedMotion';

export function ScrollProgress() {
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, {
    stiffness: 110,
    damping: 28,
    mass: 0.65,
  });

  return (
    <motion.div
      className="scroll-progress"
      aria-hidden
      style={{ scaleX: reduced ? scrollYProgress : progress }}
    />
  );
}
