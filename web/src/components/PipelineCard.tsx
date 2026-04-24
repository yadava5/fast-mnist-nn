import { motion } from 'motion/react';
import { useReducedMotion } from '../hooks/useReducedMotion';
import { springs } from '../lib/springs';

export interface PipelineCardProps {
  step: string;
  title: string;
  copy: string;
}

/**
 * Single step in the sticky-parallax pipeline section. Fades + rises
 * into view on scroll using Motion's whileInView. Reduced-motion users
 * get the final state synchronously, no animation.
 */
export function PipelineCard({ step, title, copy }: PipelineCardProps) {
  const reduced = useReducedMotion();

  if (reduced) {
    return (
      <div className="pipeline-card">
        <span className="pipeline-step">{step}</span>
        <h3 className="pipeline-title">{title}</h3>
        <p className="pipeline-copy">{copy}</p>
      </div>
    );
  }

  return (
    <motion.div
      className="pipeline-card"
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: false, amount: 0.6 }}
      transition={springs.gentle}
    >
      <span className="pipeline-step">{step}</span>
      <h3 className="pipeline-title">{title}</h3>
      <p className="pipeline-copy">{copy}</p>
    </motion.div>
  );
}
