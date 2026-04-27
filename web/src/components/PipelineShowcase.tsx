import { useRef } from 'react';
import { Activity, Binary, BrainCircuit, Cpu, type LucideIcon } from 'lucide-react';
import { motion, useScroll, useSpring, useTransform, type MotionValue } from 'motion/react';
import { useReducedMotion } from '../hooks/useReducedMotion';
import { PipelineCard } from './PipelineCard';

const steps = [
  {
    step: '01',
    title: 'You draw.',
    copy: '28x28 canvas, pixel values in [0, 1].',
  },
  {
    step: '02',
    title: 'C++ classifies.',
    copy: 'SIMD kernels (AVX-512 / AVX2 / NEON) run the forward pass.',
  },
  {
    step: '03',
    title: 'You see the answer.',
    copy: '10 softmax probabilities, argmax wins.',
  },
];

const panels: Array<{
  label: string;
  title: string;
  icon: LucideIcon;
  className: string;
}> = [
  {
    label: 'Input',
    title: '28x28 raster',
    icon: Binary,
    className: 'input-panel',
  },
  {
    label: 'Hidden',
    title: '100 neurons',
    icon: BrainCircuit,
    className: 'hidden-panel',
  },
  {
    label: 'Output',
    title: 'Softmax 10',
    icon: Activity,
    className: 'output-panel',
  },
];

const inputCells = Array.from({ length: 49 }, (_, index) => {
  const x = index % 7;
  const y = Math.floor(index / 7);
  return x === 3 || y === 1 || (x > 1 && x < 5 && y > 3);
});

const hiddenBars = [64, 35, 82, 48, 72, 28, 56, 90];
const outputBars = [8, 12, 4, 18, 10, 92, 16, 6, 14, 20];

function PanelVisual({ type }: { type: string }) {
  if (type === 'input-panel') {
    return (
      <div className="mini-raster" aria-hidden>
        {inputCells.map((active, index) => (
          <span key={index} className={active ? 'active' : undefined} />
        ))}
      </div>
    );
  }

  if (type === 'hidden-panel') {
    return (
      <div className="mini-activations" aria-hidden>
        {hiddenBars.map((width, index) => (
          <span key={index}>
            <i style={{ width: `${width}%` }} />
          </span>
        ))}
      </div>
    );
  }

  return (
    <div className="mini-softmax" aria-hidden>
      {outputBars.map((width, index) => (
        <span key={index} className={index === 5 ? 'winner' : undefined}>
          <i style={{ height: `${width}%` }} />
        </span>
      ))}
    </div>
  );
}

function FloatingPanel({
  panel,
  index,
  progress,
  reduced,
}: {
  panel: (typeof panels)[number];
  index: number;
  progress: MotionValue<number>;
  reduced: boolean;
}) {
  const x = useTransform(progress, [0, 0.45, 1], [40 - index * 18, 0, -46 + index * 22]);
  const y = useTransform(progress, [0, 0.45, 1], [index * 28 - 28, 0, 34 - index * 18]);
  const rotateY = useTransform(progress, [0, 0.5, 1], [-24 + index * 12, 0, 18 - index * 7]);
  const rotateX = useTransform(progress, [0, 0.5, 1], [10 - index * 5, 0, -8 + index * 4]);
  const scale = useTransform(progress, [0, 0.5, 1], [0.92 + index * 0.03, 1, 0.96]);
  const opacity = useTransform(progress, [0, 0.12, 0.9, 1], [0.55, 1, 1, 0.72]);
  const Icon = panel.icon;

  return (
    <motion.div
      className={`pipeline-media-frame frame-${index}`}
      style={reduced ? undefined : { x, y, rotateX, rotateY, scale, opacity }}
    >
      <div className="pipeline-media-depth">
        <div className="media-frame-topline">
          <span className="media-chip">
            <Icon size={14} strokeWidth={1.9} aria-hidden />
            {panel.label}
          </span>
          <span className="media-dots" aria-hidden>
            <i />
            <i />
            <i />
          </span>
        </div>
        <h3>{panel.title}</h3>
        <PanelVisual type={panel.className} />
      </div>
    </motion.div>
  );
}

export function PipelineShowcase() {
  const sectionRef = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start end', 'end start'],
  });
  const progress = useSpring(scrollYProgress, {
    stiffness: reduced ? 1000 : 92,
    damping: reduced ? 100 : 24,
    mass: 0.7,
  });
  const stageRotate = useTransform(progress, [0, 1], [-5, 5]);
  const stageY = useTransform(progress, [0, 0.5, 1], [18, 0, -18]);

  return (
    <section id="pipeline" className="pipeline-section" ref={sectionRef}>
      <div className="pipeline-sticky">
        <motion.div
          className="pipeline-stage"
          style={reduced ? undefined : { rotateY: stageRotate, y: stageY }}
        >
          <div className="stage-header">
            <span className="stage-kicker">
              <Cpu size={14} strokeWidth={1.9} aria-hidden />
              Scroll the forward pass
            </span>
            <span className="stage-rule" aria-hidden />
          </div>
          <div className="pipeline-media-stack" aria-hidden>
            {panels.map((panel, index) => (
              <FloatingPanel
                key={panel.label}
                panel={panel}
                index={index}
                progress={progress}
                reduced={reduced}
              />
            ))}
          </div>
        </motion.div>

        <div className="pipeline-copy-stack">
          {steps.map((step) => (
            <PipelineCard key={step.step} {...step} />
          ))}
        </div>
      </div>
    </section>
  );
}
