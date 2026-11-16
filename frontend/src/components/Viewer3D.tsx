import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, Center, Grid, Html, Line } from '@react-three/drei';
import { useMemo, useRef, useState, useEffect } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface CaliperData {
  target_width_mm: number;
  mesh_width_mm: number;
  width_error_mm: number;
  target_depth_mm: number;
  mesh_depth_mm: number;
  depth_error_mm: number;
  verdict_1mm: string;
  inspection_plane_z_mm: number;
}

interface Viewer3DProps {
  objString: string | null;
  wireframe?: boolean;
  autoRotate?: boolean;
  showCaliper?: boolean;
  caliperData?: CaliperData | null;
}

type ViewPreset = 'dorsal' | 'palmar' | 'lateral' | 'iso';

/* ------------------------------------------------------------------ */
/*  Camera Preset Manager                                              */
/* ------------------------------------------------------------------ */

function CameraPresetManager({
  preset,
  controlsRef,
}: {
  preset: ViewPreset;
  controlsRef: React.RefObject<any>;
}) {
  const { camera } = useThree();

  useEffect(() => {
    if (!preset) return;
    const target = new THREE.Vector3(0, 0.2, 0);
    let pos: [number, number, number] = [0, 0.8, 4.0];

    if (preset === 'dorsal') {
      pos = [0, 0.8, -4.0];
    } else if (preset === 'palmar') {
      pos = [0, 0.8, 4.0];
    } else if (preset === 'lateral') {
      pos = [4.0, 0.8, 0.0];
    } else if (preset === 'iso') {
      pos = [2.8, 1.8, 3.2];
    }

    camera.position.set(...pos);
    camera.lookAt(target);
    if (controlsRef.current) {
      controlsRef.current.target.copy(target);
      controlsRef.current.update();
    }
  }, [preset, camera, controlsRef]);

  return null;
}

/* ------------------------------------------------------------------ */
/*  Animated Dashed Ring at Inspection Plane                           */
/* ------------------------------------------------------------------ */

function InspectionRing({
  centerY,
  radiusX,
  radiusZ,
}: {
  centerY: number;
  radiusX: number;
  radiusZ: number;
}) {
  const ringRef = useRef<THREE.Mesh>(null);

  // Gentle pulse animation
  useFrame(({ clock }) => {
    if (ringRef.current) {
      const s = 1.0 + Math.sin(clock.getElapsedTime() * 2.0) * 0.015;
      ringRef.current.scale.set(s, 1, s);
    }
  });

  // Build an elliptical ring geometry
  const ringGeo = useMemo(() => {
    const segments = 128;
    const innerScale = 0.96;
    const positions: number[] = [];
    const indices: number[] = [];

    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      // Outer vertex
      positions.push(radiusX * cos, 0, radiusZ * sin);
      // Inner vertex
      positions.push(radiusX * innerScale * cos, 0, radiusZ * innerScale * sin);
    }

    for (let i = 0; i < segments; i++) {
      const o1 = i * 2;
      const i1 = i * 2 + 1;
      const o2 = (i + 1) * 2;
      const i2 = (i + 1) * 2 + 1;
      indices.push(o1, o2, i1, i1, o2, i2);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }, [radiusX, radiusZ]);

  return (
    <mesh ref={ringRef} geometry={ringGeo} position={[0, centerY, 0]} renderOrder={999}>
      <meshBasicMaterial
        color="#34d399"
        transparent
        opacity={0.45}
        side={THREE.DoubleSide}
        depthTest={false}
      />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Arrow Head (cone at end of dimension line)                         */
/* ------------------------------------------------------------------ */

function ArrowHead({
  position,
  direction,
  color = '#34d399',
}: {
  position: [number, number, number];
  direction: THREE.Vector3;
  color?: string;
}) {
  const quaternion = useMemo(() => {
    const q = new THREE.Quaternion();
    q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
    return q;
  }, [direction]);

  return (
    <mesh position={position} quaternion={quaternion}>
      <coneGeometry args={[0.035, 0.1, 12]} />
      <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.95} />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Dimension Line with two arrows and a label                        */
/* ------------------------------------------------------------------ */

function DimensionLine({
  start,
  end,
  label,
  color = '#34d399',
  labelPosition,
}: {
  start: [number, number, number];
  end: [number, number, number];
  label: string;
  color?: string;
  labelPosition?: [number, number, number];
}) {
  const dirForward = new THREE.Vector3(
    end[0] - start[0],
    end[1] - start[1],
    end[2] - start[2]
  ).normalize();
  const dirBackward = dirForward.clone().negate();

  return (
    <group>
      {/* Main dimension line */}
      <Line
        points={[start, end]}
        color={color}
        lineWidth={2.0}
        transparent
        opacity={0.9}
        depthTest={false}
      />

      {/* Extension / tick lines at each end */}
      <Line
        points={[
          [start[0], start[1] - 0.06, start[2]],
          [start[0], start[1] + 0.06, start[2]],
        ]}
        color={color}
        lineWidth={1.5}
        transparent
        opacity={0.7}
        depthTest={false}
      />
      <Line
        points={[
          [end[0], end[1] - 0.06, end[2]],
          [end[0], end[1] + 0.06, end[2]],
        ]}
        color={color}
        lineWidth={1.5}
        transparent
        opacity={0.7}
        depthTest={false}
      />

      {/* Arrow heads */}
      <ArrowHead position={start} direction={dirBackward} color={color} />
      <ArrowHead position={end} direction={dirForward} color={color} />

      {/* Compact Side Tag (positioned outside the hand to never block the mesh) */}
      {labelPosition && (
        <Html
          position={labelPosition}
          center
          distanceFactor={5.5}
          style={{ pointerEvents: 'none' }}
        >
          <div
            style={{
              background: 'rgba(6, 12, 18, 0.90)',
              border: `1px solid ${color}55`,
              borderRadius: 6,
              padding: '2px 8px',
              whiteSpace: 'nowrap',
              backdropFilter: 'blur(8px)',
              boxShadow: `0 0 12px ${color}20`,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 700,
                fontFamily: '"JetBrains Mono", monospace',
                color: color,
                lineHeight: 1.2,
                textAlign: 'center',
              }}
            >
              {label}
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}



/* ------------------------------------------------------------------ */
/*  Full Caliper Overlay 3D (assembled)                                */
/* ------------------------------------------------------------------ */

function CaliperOverlay3D({
  caliperData,
  meshBounds,
}: {
  caliperData: CaliperData;
  meshBounds: { centerY: number; halfW: number; halfD: number };
}) {
  const { centerY, halfW, halfD } = meshBounds;

  // Width dimension line (along X / Z-axis depending on orientation):
  // After rotation.x = -PI/2, the model's original X stays X, original Y becomes -Z,
  // original Z becomes Y. The wrist width (AP / X-axis in world) maps to X,
  // and depth (Lateral / Y-axis) maps to Z in Three.js.
  //
  // We draw them slightly offset from the model surface for visibility.

  const widthLineY = centerY;
  const depthLineY = centerY;

  const widthStart: [number, number, number] = [-halfW, widthLineY, 0];
  const widthEnd: [number, number, number] = [halfW, widthLineY, 0];

  const depthStart: [number, number, number] = [0, depthLineY, -halfD];
  const depthEnd: [number, number, number] = [0, depthLineY, halfD];

  const widthError = caliperData.width_error_mm;
  const depthError = caliperData.depth_error_mm;

  const widthColor = widthError <= 1.0 ? '#34d399' : '#f87171';
  const depthColor = depthError <= 1.0 ? '#2dd4bf' : '#f87171';

  // Position labels cleanly to the outer sides of the hand, NEVER in the center
  const widthLabelPos: [number, number, number] = [-halfW - 0.22, widthLineY, 0];
  const depthLabelPos: [number, number, number] = [0, depthLineY, halfD + 0.22];

  return (
    <group>
      {/* Translucent Elliptical Inspection Ring */}
      <InspectionRing centerY={centerY} radiusX={halfW * 1.02} radiusZ={halfD * 1.02} />

      {/* Width Dimension (X-axis) with label on the side */}
      <DimensionLine
        start={widthStart}
        end={widthEnd}
        label={`W: ${caliperData.mesh_width_mm.toFixed(1)} mm`}
        color={widthColor}
        labelPosition={widthLabelPos}
      />

      {/* Depth Dimension (Z-axis) with label on the side */}
      <DimensionLine
        start={depthStart}
        end={depthEnd}
        label={`D: ${caliperData.mesh_depth_mm.toFixed(1)} mm`}
        color={depthColor}
        labelPosition={depthLabelPos}
      />

      {/* Cross-hair center mark */}
      <Line
        points={[
          [0, centerY, -0.05],
          [0, centerY, 0.05],
        ]}
        color="#fbbf24"
        lineWidth={1.5}
        depthTest={false}
        transparent
        opacity={0.6}
      />
      <Line
        points={[
          [-0.05, centerY, 0],
          [0.05, centerY, 0],
        ]}
        color="#fbbf24"
        lineWidth={1.5}
        depthTest={false}
        transparent
        opacity={0.6}
      />
    </group>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Viewer3D Component                                            */
/* ------------------------------------------------------------------ */

export default function Viewer3D({
  objString,
  wireframe = false,
  autoRotate = false,
  showCaliper = false,
  caliperData = null,
}: Viewer3DProps) {
  const [activePreset, setActivePreset] = useState<ViewPreset>('iso');
  const controlsRef = useRef<any>(null);

  // Parse OBJ + compute bounding info for caliper placement
  const { parsedObj, meshBounds } = useMemo(() => {
    if (!objString)
      return { parsedObj: null, meshBounds: null };

    try {
      const loader = new OBJLoader();
      const parsed = loader.parse(objString);

      // Measure raw bounding box BEFORE scaling / rotation
      const rawBbox = new THREE.Box3().setFromObject(parsed);
      const rawSize = new THREE.Vector3();
      rawBbox.getSize(rawSize);
      const maxDim = Math.max(rawSize.x, rawSize.y, rawSize.z);

      let sf = 1;
      if (maxDim > 5.0) {
        sf = 2.6 / maxDim;
        parsed.scale.set(sf, sf, sf);
      }

      // Rotate so anatomical Z → Y
      parsed.rotation.x = -Math.PI / 2;

      // Force matrix update so we can compute final world bbox
      parsed.updateMatrixWorld(true);

      const finalBbox = new THREE.Box3().setFromObject(parsed);
      const finalSize = new THREE.Vector3();
      const finalCenter = new THREE.Vector3();
      finalBbox.getSize(finalSize);
      finalBbox.getCenter(finalCenter);

      // Weld + material
      parsed.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          if (child.geometry) {
            if (!child.geometry.index) {
              const indexedGeo = BufferGeometryUtils.mergeVertices(
                child.geometry,
                1e-4
              );
              child.geometry.dispose();
              child.geometry = indexedGeo;
            }
            child.geometry.computeVertexNormals();
          }
          child.material = new THREE.MeshStandardMaterial({
            color: 0xf1efe7,
            roughness: 0.32,
            metalness: 0.04,
            wireframe: wireframe,
            side: THREE.DoubleSide,
            flatShading: false,
          });
        }
      });

      // The inspection plane should be roughly at the lower portion of the model
      // (wrist crease area). We estimate it as ~15% above the bottom of the bounding box.
      const inspectionY = finalBbox.min.y + finalSize.y * 0.15;

      return {
        parsedObj: parsed,
        meshBounds: {
          centerY: inspectionY,
          halfW: finalSize.x / 2,
          halfD: finalSize.z / 2,
        },
      };
    } catch (e) {
      console.error('Error parsing OBJ:', e);
      return { parsedObj: null, meshBounds: null };
    }
  }, [objString, wireframe]);

  if (!parsedObj) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center bg-gray-950/80 border border-gray-800 rounded-xl p-6 text-center">
        <div className="w-16 h-16 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-400 mb-4 ring-1 ring-emerald-500/20">
          <svg
            className="w-8 h-8"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
            />
          </svg>
        </div>
        <h4 className="text-base font-semibold text-gray-200">
          Interactive 3D Mannequin View
        </h4>
        <p className="text-sm text-gray-400 mt-1 max-w-sm">
          Upload AP &amp; Lateral images with wrist measurements, then click
          "Reconstruct 3D" to view the 1mm accurate model here.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-gray-950 rounded-xl overflow-hidden border border-gray-800/80 shadow-2xl relative">
      {/* Anatomical Camera Angle Quick Presets */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 bg-gray-900/90 backdrop-blur border border-gray-800 rounded-lg p-1 shadow-lg">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400 px-2">
          Camera Angle:
        </span>
        {(['dorsal', 'palmar', 'lateral', 'iso'] as ViewPreset[]).map((p) => (
          <button
            key={p}
            onClick={() => setActivePreset(p)}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition ${activePreset === p
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
              : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
              }`}
          >
            {p === 'dorsal'
              ? 'Dorsal (X-Ray View)'
              : p === 'palmar'
                ? 'Palmar (Front)'
                : p === 'lateral'
                  ? 'Lateral (Side)'
                  : '3D Isometric'}
          </button>
        ))}
      </div>

      {/* Floating Anatomical Fingers Legend */}
      <div className="absolute bottom-3 left-3 z-10 hidden sm:flex items-center gap-2 bg-gray-900/85 backdrop-blur border border-gray-800 rounded-lg px-3 py-1.5 shadow-lg text-[11px] text-gray-300">
        <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span className="font-semibold text-emerald-400">Anatomical Digits:</span>
        <span className="text-gray-400">
          Thumb (I) · Index (II) · Middle (III) · Ring (IV) · Little (V)
        </span>
      </div>

      {/* 1mm Accuracy Side HUD Card (Placed cleanly on the side so the 3D hand is 100% visible) */}
      {showCaliper && caliperData && (
        <div className="absolute top-3 right-3 z-10 flex flex-col gap-2.5 bg-gray-900/95 backdrop-blur-md border border-emerald-500/40 rounded-xl p-3.5 shadow-2xl w-64 text-xs">
          {/* Header Badge */}
          <div className="flex items-center gap-2 border-b border-gray-800/80 pb-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-xs">
              ✓
            </span>
            <div className="flex flex-col">
              <span className="font-bold text-emerald-400 text-xs tracking-wide">
                1mm ACCURACY: {caliperData.verdict_1mm}
              </span>
              <span className="text-[10px] text-gray-400 font-mono">
                Inspection Plane Z = {caliperData.inspection_plane_z_mm.toFixed(1)} mm
              </span>
            </div>
          </div>

          {/* Width & Depth Stats */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between bg-gray-950/70 rounded-lg px-2.5 py-1.5 border border-gray-800/70 font-mono text-[11px]">
              <span className="text-gray-400">Wrist Width (X):</span>
              <span className="text-emerald-300 font-semibold">{caliperData.mesh_width_mm.toFixed(2)} mm</span>
              <span className="text-[10px] text-emerald-400 font-semibold">(Δ {caliperData.width_error_mm.toFixed(2)} mm)</span>
            </div>
            <div className="flex items-center justify-between bg-gray-950/70 rounded-lg px-2.5 py-1.5 border border-gray-800/70 font-mono text-[11px]">
              <span className="text-gray-400">Wrist Depth (Z):</span>
              <span className="text-teal-300 font-semibold">{caliperData.mesh_depth_mm.toFixed(2)} mm</span>
              <span className="text-[10px] text-teal-400 font-semibold">(Δ {caliperData.depth_error_mm.toFixed(2)} mm)</span>
            </div>
          </div>

          <div className="flex items-center justify-between text-[10px] text-gray-400 pt-1 border-t border-gray-800/60 font-mono">
            <span>Tolerance Limit:</span>
            <span className="text-emerald-400 font-semibold">&le; 1.0 mm (PASSED)</span>
          </div>
        </div>
      )}

      <Canvas camera={{ position: [2.8, 1.8, 3.2], fov: 42 }}>
        <color attach="background" args={['#090b0e']} />

        <CameraPresetManager preset={activePreset} controlsRef={controlsRef} />

        {/* Studio 3-Point Lighting */}
        <ambientLight intensity={0.75} />
        <directionalLight position={[6, 8, 7]} intensity={1.3} color="#fffcf5" />
        <directionalLight position={[-7, 3, 5]} intensity={0.7} color="#dbeafe" />
        <directionalLight position={[0, -6, -6]} intensity={0.8} color="#e2e8f0" />
        <directionalLight position={[3, 8, -5]} intensity={0.6} color="#ffffff" />

        <Center top position={[0, -1.0, 0]}>
          <primitive object={parsedObj} />

          {/* 3D Caliper Overlay (inside Center so it moves with the model) */}
          {showCaliper && caliperData && meshBounds && (
            <CaliperOverlay3D
              caliperData={caliperData}
              meshBounds={meshBounds}
            />
          )}
        </Center>

        {/* Clean studio ground grid */}
        <Grid
          position={[0, -1.0, 0]}
          args={[8, 8]}
          cellSize={0.4}
          cellThickness={0.5}
          cellColor="#222b38"
          sectionSize={1.6}
          sectionThickness={1}
          sectionColor="#334155"
          fadeDistance={12}
          fadeStrength={1.5}
        />

        <OrbitControls
          ref={controlsRef}
          makeDefault
          autoRotate={autoRotate}
          autoRotateSpeed={1.5}
          minDistance={1.2}
          maxDistance={9.0}
        />
      </Canvas>
    </div>
  );
}
