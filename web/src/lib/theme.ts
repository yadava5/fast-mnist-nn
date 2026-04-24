export type Theme = 'dark' | 'light';

export const THEME_KEY = 'theme';

export function applyTheme(next: Theme) {
  const doc = document.documentElement;
  const run = () => {
    doc.dataset.theme = next;
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // localStorage may be unavailable (e.g. privacy mode); theme still
      // applies visually, just not persisted.
    }
  };
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduced && 'startViewTransition' in document) {
    (document as unknown as { startViewTransition: (cb: () => void) => void })
      .startViewTransition(run);
  } else {
    run();
  }
}

export function readInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // ignored — fall through to system preference
  }
  return window.matchMedia('(prefers-color-scheme: light)').matches
    ? 'light'
    : 'dark';
}
