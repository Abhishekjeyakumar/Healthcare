import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Center, Grid } from '@react-three/drei';
import { useMemo, useRef, useState, useEffect } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';

interface Viewer3DProps {
  objString: string | null;
  wireframe: boolean;
  autoRotate?: boolean;
}

type ViewPreset = 'dorsal' | 'palmar' | 'lateral' | 'iso';

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
      // Dorsal / PA X-ray angle: Thumb on left, pinky on right
      pos = [0, 0.8, -4.0];
    } else if (preset === 'palmar') {
      // Front / Palmar: Thumb on right
      pos = [0, 0.8, 4.0];
    } else if (preset === 'lateral') {
      // Lateral / side profile view
      pos = [4.0, 0.8, 0.0];
    } else if (preset === 'iso') {
      // 3/4 isometric perspective
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

export default function Viewer3D({ objString, wireframe, autoRotate = false }: Viewer3DProps) {
  const [activePreset, setActivePreset] = useState<ViewPreset>('iso');
  const controlsRef = useRef<any>(null);

  const obj = useMemo(() => {
    if (!objString) return null;
    try {
      const loader = new OBJLoader();
      const parsedObj = loader.parse(objString);

      // Auto-scale to comfortable view size (fits in ~2.6 units)
      const bbox = new THREE.Box3().setFromObject(parsedObj);
      const size = new THREE.Vector3();
      bbox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      if (maxDim > 5.0) {
        const s = 2.6 / maxDim;
        parsedObj.scale.set(s, s, s);
      }

      // Rotate so anatomical height (Z) points upright (+Y in Three.js)
      parsedObj.rotation.x = -Math.PI / 2;

      // Weld vertices and apply satin mannequin finish
      parsedObj.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          if (child.geometry) {
            if (!child.geometry.index) {
              const indexedGeo = BufferGeometryUtils.mergeVertices(child.geometry, 1e-4);
              child.geometry.dispose();
              child.geometry = indexedGeo;
            }
            child.geometry.computeVertexNormals();
          }

          child.material = new THREE.MeshStandardMaterial({
            color: 0xf1efe7, // Premium medical porcelain mannequin finish
            roughness: 0.32,  // Soft satin sheen
            metalness: 0.04,
            wireframe: wireframe,
            side: THREE.DoubleSide,
            flatShading: false,
          });
        }
      });

      return parsedObj;
    } catch (e) {
      console.error("Error parsing OBJ:", e);
      return null;
    }
  }, [objString, wireframe]);

  if (!obj) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center bg-gray-950/80 border border-gray-800 rounded-xl p-6 text-center">
        <div className="w-16 h-16 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-400 mb-4 ring-1 ring-emerald-500/20">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
        </div>
        <h4 className="text-base font-semibold text-gray-200">Interactive 3D Mannequin View</h4>
        <p className="text-sm text-gray-400 mt-1 max-w-sm">
          Upload AP & Lateral images with wrist measurements, then click "Reconstruct 3D" to view the 1mm accurate model here.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-gray-950 rounded-xl overflow-hidden border border-gray-800/80 shadow-2xl relative">
      {/* Anatomical Camera Angle Quick Presets */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 bg-gray-900/90 backdrop-blur border border-gray-800 rounded-lg p-1 shadow-lg">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400 px-2">Camera Angle:</span>
        <button
          onClick={() => setActivePreset('dorsal')}
          className={`px-2.5 py-1 text-xs font-medium rounded-md transition ${
            activePreset === 'dorsal'
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
              : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
          }`}
          title="Matches the AP/PA Radiograph view (Thumb on Left, Pinky on Right)"
        >
          Dorsal (X-Ray View)
        </button>
        <button
          onClick={() => setActivePreset('palmar')}
          className={`px-2.5 py-1 text-xs font-medium rounded-md transition ${
            activePreset === 'palmar'
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
              : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
          }`}
          title="Front Palm View (Thumb on Right)"
        >
          Palmar (Front)
        </button>
        <button
          onClick={() => setActivePreset('lateral')}
          className={`px-2.5 py-1 text-xs font-medium rounded-md transition ${
            activePreset === 'lateral'
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
              : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
          }`}
          title="Lateral profile view"
        >
          Lateral (Side)
        </button>
        <button
          onClick={() => setActivePreset('iso')}
          className={`px-2.5 py-1 text-xs font-medium rounded-md transition ${
            activePreset === 'iso'
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
              : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
          }`}
          title="3/4 Isometric 3D angle"
        >
          3D Isometric
        </button>
      </div>

      {/* Floating Anatomical Fingers Legend */}
      <div className="absolute bottom-3 left-3 z-10 hidden sm:flex items-center gap-2 bg-gray-900/85 backdrop-blur border border-gray-800 rounded-lg px-3 py-1.5 shadow-lg text-[11px] text-gray-300">
        <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span className="font-semibold text-emerald-400">Anatomical Digits:</span>
        <span className="text-gray-400">Thumb (I) · Index (II) · Middle (III) · Ring (IV) · Little (V)</span>
      </div>

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
          <primitive object={obj} />
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
