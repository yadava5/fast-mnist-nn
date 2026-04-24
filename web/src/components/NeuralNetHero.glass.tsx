import { MeshTransmissionMaterial } from '@react-three/drei';

/**
 * Frosted-glass backdrop panel behind the neural-net scene.
 * Sits at z = -2.5 and subtly refracts the nodes/edges in front of it.
 */
export function GlassBackdrop() {
  return (
    <mesh position={[0, 0, -2.5]}>
      <planeGeometry args={[6, 4]} />
      <MeshTransmissionMaterial
        thickness={0.3}
        roughness={0.1}
        transmission={1}
        ior={1.4}
        chromaticAberration={0.04}
        backside
        backsideResolution={256}
        samples={6}
        color="oklch(0.85 0.08 280)"
      />
    </mesh>
  );
}
