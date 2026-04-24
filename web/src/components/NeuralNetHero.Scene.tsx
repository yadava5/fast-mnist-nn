import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

export default function NeuralNetHeroScene() {
  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [3, 1.5, 4], fov: 45 }}
      gl={{ powerPreference: 'high-performance', antialias: false }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={1.2} />
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="oklch(0.7 0.22 280)" />
      </mesh>
      <OrbitControls enablePan={false} enableZoom={false} />
    </Canvas>
  );
}
