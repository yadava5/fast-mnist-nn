import { Command } from 'cmdk';
import { useEffect, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTheme } from '../hooks/useTheme';
import { cn } from '../lib/cn';

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const { toggleTheme } = useTheme();

  useHotkeys('mod+k', (e) => {
    e.preventDefault();
    setOpen((o) => !o);
  });
  useHotkeys('escape', () => setOpen(false), { enableOnFormTags: true });

  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 backdrop-blur-sm pt-[20vh]"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
    >
      <Command
        className={cn(
          'w-full max-w-lg rounded-2xl border border-[color:var(--color-border)]',
          'bg-[color:var(--color-bg-elev)]/80 backdrop-blur-xl shadow-2xl',
          'overflow-hidden',
        )}
      >
        <Command.Input
          placeholder="Type a command..."
          className="w-full bg-transparent px-4 py-3 text-base outline-none border-b border-[color:var(--color-border)]"
        />
        <Command.List className="max-h-[300px] overflow-auto p-2">
          <Command.Empty className="px-3 py-4 text-sm text-[color:var(--color-fg-muted)]">
            No results.
          </Command.Empty>
          <Command.Group heading="Actions">
            <Command.Item
              onSelect={() => {
                toggleTheme();
                setOpen(false);
              }}
              className="cursor-pointer select-none rounded-md px-3 py-2 text-sm aria-selected:bg-[color:var(--color-bg)]"
            >
              Toggle theme
            </Command.Item>
            <Command.Item
              onSelect={() => {
                window.open('https://github.com/yadava5/fast-mnist-nn', '_blank', 'noopener');
                setOpen(false);
              }}
              className="cursor-pointer select-none rounded-md px-3 py-2 text-sm aria-selected:bg-[color:var(--color-bg)]"
            >
              Open repo on GitHub
            </Command.Item>
            <Command.Item
              onSelect={() => setOpen(false)}
              className="cursor-pointer select-none rounded-md px-3 py-2 text-sm aria-selected:bg-[color:var(--color-bg)]"
            >
              Close palette
            </Command.Item>
          </Command.Group>
        </Command.List>
      </Command>
    </div>
  );
}
