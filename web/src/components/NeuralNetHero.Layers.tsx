import { useMemo } from 'react';
import * as THREE from 'three';
import { Instance, Instances } from '@react-three/drei';

interface LayerProps {
  positions: [number, number, number][];
  radius: number;
  color: string;
  emissive: string;
  emissiveIntensity: number;
}

/**
 * Render a single neural-net layer as an instanced mesh of spheres.
 * drei's <Instances> keeps this cheap even for 196+ nodes.
 */
export function Layer({ positions, radius, color, emissive, emissiveIntensity }: LayerProps) {
  return (
    <Instances limit={positions.length} range={positions.length}>
      <sphereGeometry args={[radius, 12, 12]} />
      <meshStandardMaterial
        color={color}
        emissive={emissive}
        emissiveIntensity={emissiveIntensity}
        roughness={0.35}
        metalness={0.1}
      />
      {positions.map((p, i) => (
        <Instance key={i} position={p} />
      ))}
    </Instances>
  );
}

interface EdgesProps {
  from: [number, number, number][];
  to: [number, number, number][];
  count: number;
  color: string;
}

/**
 * Sample `count` edges between two layers and render them as thin cylinders.
 * We pick endpoints pseudo-randomly but deterministically so the scene looks
 * consistent across renders.
 */
export function Edges({ from, to, count, color }: EdgesProps) {
  const edges = useMemo(() => {
    // deterministic LCG so edges don't jitter between re-renders
    let seed = 1337;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    const out: { position: THREE.Vector3; quaternion: THREE.Quaternion; length: number }[] = [];
    const upAxis = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i < count; i += 1) {
      const a = from[Math.floor(rand() * from.length)];
      const b = to[Math.floor(rand() * to.length)];
      const va = new THREE.Vector3(...a);
      const vb = new THREE.Vector3(...b);
      const dir = new THREE.Vector3().subVectors(vb, va);
      const length = dir.length();
      const mid = new THREE.Vector3().addVectors(va, vb).multiplyScalar(0.5);
      const q = new THREE.Quaternion().setFromUnitVectors(upAxis, dir.clone().normalize());
      out.push({ position: mid, quaternion: q, length });
    }
    return out;
  }, [from, to, count]);

  return (
    <Instances limit={edges.length} range={edges.length}>
      <cylinderGeometry args={[0.006, 0.006, 1, 6, 1]} />
      <meshBasicMaterial color={color} transparent opacity={0.35} />
      {edges.map((e, i) => (
        <Instance
          key={i}
          position={e.position}
          quaternion={e.quaternion}
          scale={[1, e.length, 1]}
        />
      ))}
    </Instances>
  );
}
