import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { cn } from '../lib/cn';

const Scene = lazy(() => import('./NeuralNetHero.Scene'));

interface Props {
  className?: string;
}

export function NeuralNetHero({ className }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [shouldLoad, setShouldLoad] = useState(false);

  useEffect(() => {
    if (shouldLoad) return;
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          const start = () => setShouldLoad(true);
          if ('requestIdleCallback' in window) {
            (window as unknown as { requestIdleCallback: (cb: () => void) => void }).requestIdleCallback(start);
          } else {
            setTimeout(start, 0);
          }
          io.disconnect();
        }
      },
      { rootMargin: '200px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [shouldLoad]);

  return (
    <div
      ref={ref}
      className={cn('relative aspect-square w-full max-w-lg', className)}
    >
      <img
        src="/hero-poster.svg"
        alt=""
        aria-hidden
        className="absolute inset-0 h-full w-full object-contain opacity-80"
      />
      {shouldLoad && (
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
      )}
    </div>
  );
}
