import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { Group } from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { useReducedMotion } from '../hooks/useReducedMotion';
import { Edges, Layer } from './NeuralNetHero.Layers';
import { gridPositions } from '../lib/layout';
import { GlassBackdrop } from './NeuralNetHero.glass';

const EDGE_COLOR = 'oklch(0.5 0.05 260)';

/**
 * Auto-rotate the scene around Y when the user is not actively dragging.
 * Reads the OrbitControls azimuth to detect user interaction; when it has
 * been idle (no change) for a short window, spins the root group slowly.
 */
function AutoRotate({
  rootRef,
  controlsRef,
  disabled,
}: {
  rootRef: React.RefObject<Group | null>;
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
  disabled: boolean;
}) {
  const lastAzimuth = useRef<number | null>(null);
  const idleTime = useRef(0);

  useFrame((_, delta) => {
    if (disabled) return;
    const root = rootRef.current;
    const controls = controlsRef.current;
    if (!root || !controls) return;

    const az = controls.getAzimuthalAngle();
    if (lastAzimuth.current === null) lastAzimuth.current = az;
    const moved = Math.abs(az - lastAzimuth.current) > 1e-4;
    lastAzimuth.current = az;

    if (moved) {
      idleTime.current = 0;
      return;
    }
    // brief grace period so damping finishes before auto-rotate kicks in
    idleTime.current += delta;
    if (idleTime.current < 0.4) return;
    root.rotation.y += delta * 0.1;
  });
  return null;
}

export default function NeuralNetHeroScene() {
  const reduced = useReducedMotion();
  const rootRef = useRef<Group>(null);
  const controlsRef = useRef<OrbitControlsImpl>(null);

  // Layer geometry. Z spacing places the backdrop at -2.5, input at -1.2,
  // hidden at 0, output at 1.2.
  const { input, hidden, output } = useMemo(() => {
    return {
      input: gridPositions(14, 14, 2.8, 2.8, -1.2),
      hidden: gridPositions(10, 10, 2.2, 2.2, 0),
      output: gridPositions(1, 10, 0, 2.4, 1.2),
    };
  }, []);

  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [3, 1.5, 4], fov: 45 }}
      gl={{ powerPreference: 'high-performance', antialias: false }}
      frameloop={reduced ? 'demand' : 'always'}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={1.2} />

      <group ref={rootRef}>
        <GlassBackdrop />

        <Layer
          positions={input}
          radius={0.055}
          color="oklch(0.7 0.1 220)"
          emissive="oklch(0.5 0.15 220)"
          emissiveIntensity={0.25}
        />
        <Layer
          positions={hidden}
          radius={0.07}
          color="oklch(0.7 0.22 280)"
          emissive="oklch(0.5 0.2 280)"
          emissiveIntensity={0.25}
        />
        <Layer
          positions={output}
          radius={0.11}
          color="oklch(0.7 0.22 280)"
          emissive="oklch(0.5 0.2 280)"
          emissiveIntensity={0.4}
        />

        <Edges from={input} to={hidden} count={60} color={EDGE_COLOR} />
        <Edges from={hidden} to={output} count={30} color={EDGE_COLOR} />
      </group>

      <AutoRotate rootRef={rootRef} controlsRef={controlsRef} disabled={reduced} />

      <OrbitControls
        ref={controlsRef}
        enablePan={false}
        enableZoom={false}
        enableDamping={!reduced}
        dampingFactor={0.08}
      />
    </Canvas>
  );
}
