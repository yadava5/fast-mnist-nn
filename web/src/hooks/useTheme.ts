import { useState, useEffect, useCallback, useRef } from 'react';
import { applyTheme, readInitialTheme, type Theme } from '../lib/theme';

export type { Theme };

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => readInitialTheme());
  const didApplyInitialTheme = useRef(false);

  useEffect(() => {
    applyTheme(theme, { animate: didApplyInitialTheme.current });
    didApplyInitialTheme.current = true;
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  return { theme, setTheme, toggleTheme };
}
