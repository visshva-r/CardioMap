import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { VESSELS, type VesselId } from "../anatomy/registry";

function riskColor(probability: number): THREE.Color {
  const t = Math.min(1, Math.max(0, probability));
  return new THREE.Color().setHSL(0.33 * (1 - t), 0.75, 0.45);
}

type VesselTubeProps = {
  curve: THREE.CatmullRomCurve3;
  probability: number;
  selected: boolean;
  onSelect: () => void;
};

function VesselTube({ curve, probability, selected, onSelect }: VesselTubeProps) {
  const color = useMemo(() => riskColor(probability), [probability]);
  const radius = selected ? 0.11 : 0.09;
  const handleClick = (e: { stopPropagation: () => void }) => {
    e.stopPropagation();
    onSelect();
  };
  return (
    <group>
      <mesh onClick={handleClick}>
        <tubeGeometry args={[curve, 64, radius, 12, false]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={selected ? 0.35 : 0.15}
        />
      </mesh>
      <mesh visible={false} onClick={handleClick}>
        <tubeGeometry args={[curve, 48, 0.14, 8, false]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
    </group>
  );
}

type HeartSceneProps = {
  probabilities: Record<VesselId, number>;
  selected: VesselId | "Cath";
  onSelect: (id: VesselId) => void;
};

function HeartModel({
  probabilities,
  selected,
  onSelect,
}: HeartSceneProps) {
  const lad = useMemo(
    () =>
      new THREE.CatmullRomCurve3([
        new THREE.Vector3(-0.2, 0.6, 0.35),
        new THREE.Vector3(-0.05, 0.35, 0.55),
        new THREE.Vector3(0.05, -0.05, 0.62),
        new THREE.Vector3(0.08, -0.45, 0.48),
      ]),
    []
  );
  const lcx = useMemo(
    () =>
      new THREE.CatmullRomCurve3([
        new THREE.Vector3(-0.25, 0.55, 0.25),
        new THREE.Vector3(-0.55, 0.35, 0.15),
        new THREE.Vector3(-0.62, 0.05, -0.05),
        new THREE.Vector3(-0.45, -0.25, -0.15),
      ]),
    []
  );
  const rca = useMemo(
    () =>
      new THREE.CatmullRomCurve3([
        new THREE.Vector3(0.15, 0.5, 0.15),
        new THREE.Vector3(0.45, 0.25, 0.05),
        new THREE.Vector3(0.55, -0.05, -0.05),
        new THREE.Vector3(0.35, -0.35, -0.2),
      ]),
    []
  );

  const curves: Record<VesselId, THREE.CatmullRomCurve3> = { LAD: lad, LCX: lcx, RCA: rca };

  return (
    <group rotation={[0.2, -0.45, 0]}>
      <mesh>
        <sphereGeometry args={[0.75, 48, 48]} />
        <meshStandardMaterial color="#5a2030" roughness={0.85} />
      </mesh>
      <mesh scale={[1.05, 0.95, 0.9]}>
        <sphereGeometry args={[0.78, 32, 32]} />
        <meshStandardMaterial color="#2d1218" transparent opacity={0.25} />
      </mesh>
      {VESSELS.map((v) => (
        <VesselTube
          key={v.id}
          curve={curves[v.id]}
          probability={probabilities[v.id]}
          selected={selected === v.id}
          onSelect={() => onSelect(v.id)}
        />
      ))}
    </group>
  );
}

export function HeartScene({ probabilities, selected, onSelect }: HeartSceneProps) {
  return (
    <div className="heart-scene">
      <Canvas dpr={[1, 1.5]}>
        <PerspectiveCamera makeDefault position={[0, 0.2, 2.4]} fov={45} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[2, 3, 2]} intensity={1.1} />
        <HeartModel probabilities={probabilities} selected={selected} onSelect={onSelect} />
        <OrbitControls enablePan={false} minDistance={1.4} maxDistance={3.5} />
      </Canvas>
      <p className="scene-hint">Drag to rotate · Click a vessel (LAD, LCX, RCA)</p>
    </div>
  );
}
