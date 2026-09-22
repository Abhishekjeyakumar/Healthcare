import { Canvas } from '@react-three/fiber';
import { OrbitControls, Center } from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import * as BufferGeometryUtils from 'three/examples/jsm/utils/BufferGeometryUtils.js';

interface Viewer3DProps {
  objString: string | null;
  wireframe: boolean;
}

export default function Viewer3D({ objString, wireframe }: Viewer3DProps) {
  const obj = useMemo(() => {
    if (!objString) return null;
    try {
      const loader = new OBJLoader();
      const parsedObj = loader.parse(objString);
      
      // Weld vertices and apply satin mannequin finish
      parsedObj.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          if (child.geometry) {
            // Merge coincident vertices so normals interpolate smoothly across triangles
            if (!child.geometry.index) {
              const indexedGeo = BufferGeometryUtils.mergeVertices(child.geometry, 1e-4);
              child.geometry.dispose();
              child.geometry = indexedGeo;
            }
            child.geometry.computeVertexNormals();
          }

          child.material = new THREE.MeshStandardMaterial({
            color: 0xedeae2, // Premium satin porcelain mannequin tone
            roughness: 0.35,  // Soft matte sheen without harsh specular streaks
            metalness: 0.05,
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
      <div className="flex h-full w-full items-center justify-center bg-gray-900 border border-gray-700 rounded-lg">
        <p className="text-gray-400">Reconstruct to view 3D model</p>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-gray-950 rounded-lg overflow-hidden border border-gray-800 shadow-2xl relative">
      <Canvas camera={{ position: [0, 0, 4.2], fov: 42 }}>
        <color attach="background" args={['#0c0e12']} />
        
        {/* Studio Mannequin 3-Point Lighting */}
        <ambientLight intensity={0.75} />
        {/* Key light: Upper front-right with warm tint */}
        <directionalLight position={[6, 8, 7]} intensity={1.3} color="#fffcf5" />
        {/* Fill light: Front-left with soft cool tint */}
        <directionalLight position={[-7, 3, 5]} intensity={0.7} color="#dbeafe" />
        {/* Rim / Backlights: Accentuates organic limb silhouettes */}
        <directionalLight position={[0, -6, -6]} intensity={0.8} color="#e2e8f0" />
        <directionalLight position={[3, 8, -5]} intensity={0.6} color="#ffffff" />
        
        <Center>
          <primitive object={obj} />
        </Center>
        
        <OrbitControls makeDefault autoRotate={false} minDistance={1} maxDistance={12} />
      </Canvas>
    </div>
  );
}
