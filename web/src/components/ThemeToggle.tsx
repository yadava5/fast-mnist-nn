import { AnimatePresence, motion } from 'motion/react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';
import { useReducedMotion } from '../hooks/useReducedMotion';
import { springs } from '../lib/springs';
import { cn } from '../lib/cn';

interface Props {
  className?: string;
}

export function ThemeToggle({ className }: Props) {
  const { theme, toggleTheme } = useTheme();
  const reduced = useReducedMotion();

  return (
    <motion.button
      type="button"
      onClick={toggleTheme}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
      whileHover={reduced ? undefined : { y: -1 }}
      whileTap={reduced ? undefined : { scale: 0.98 }}
      transition={springs.quick}
      className={cn(
        'inline-flex h-10 w-10 items-center justify-center rounded-full',
        'border border-[color:var(--color-border)]',
        'bg-[color:var(--color-bg-elev)] text-[color:var(--color-fg)]',
        'transition-colors hover:bg-[color:color-mix(in_oklch,var(--color-bg-elev),var(--color-fg)_6%)]',
        className,
      )}
    >
      <AnimatePresence mode="wait" initial={false}>
        {theme === 'dark' ? (
          <motion.span
            key="moon"
            initial={reduced ? false : { rotate: -90, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            exit={reduced ? { opacity: 0 } : { rotate: 90, opacity: 0 }}
            transition={springs.quick}
            className="inline-flex"
          >
            <Moon size={18} strokeWidth={1.75} />
          </motion.span>
        ) : (
          <motion.span
            key="sun"
            initial={reduced ? false : { rotate: 90, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            exit={reduced ? { opacity: 0 } : { rotate: -90, opacity: 0 }}
            transition={springs.quick}
            className="inline-flex"
          >
            <Sun size={18} strokeWidth={1.75} />
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  );
}
