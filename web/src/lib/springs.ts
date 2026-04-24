import type { Transition } from 'motion/react';

export const springs = {
  gentle: { type: 'spring', stiffness: 170, damping: 26, mass: 1 } as Transition,
  quick: { type: 'spring', stiffness: 300, damping: 30, mass: 0.8 } as Transition,
  bouncy: { type: 'spring', stiffness: 500, damping: 15, mass: 1 } as Transition,
};
