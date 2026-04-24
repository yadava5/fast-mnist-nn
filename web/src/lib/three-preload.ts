// Lazy loader for the three.js + react-three-fiber + drei chunk.
//
// The manualChunks config in vite.config.ts routes these modules into a
// dedicated `three-vendor` chunk so the initial bundle stays small. Nothing
// imports three statically yet — this helper exists so consumers (future
// 3D scene components) call `loadThree()` to warm the chunk ahead of mount,
// and so the build emits the chunk instead of dead-coding it away.
export async function loadThree() {
  const [three, fiber, drei] = await Promise.all([
    import('three'),
    import('@react-three/fiber'),
    import('@react-three/drei'),
  ]);
  return { three, fiber, drei };
}

export type LoadedThree = Awaited<ReturnType<typeof loadThree>>;
