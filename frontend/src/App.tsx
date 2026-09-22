import { useState } from 'react'
import axios from 'axios'
import Viewer3D from './components/Viewer3D'
import { UploadCloud, CheckCircle2, Bone, Download } from 'lucide-react'

function App() {
  const [apImage, setApImage] = useState<File | null>(null)
  const [latImage, setLatImage] = useState<File | null>(null)
  
  const [apPreview, setApPreview] = useState<string>('')
  const [latPreview, setLatPreview] = useState<string>('')
  
  const [loading, setLoading] = useState(false)
  const [statusText, setStatusText] = useState('')
  const [objString, setObjString] = useState<string | null>(null)
  const [wireframe, setWireframe] = useState(false)
  const [debugInfo, setDebugInfo] = useState<any>(null)

  const loadDemoData = async () => {
    try {
      const apRes = await fetch('/knee_ap_mask.png');
      const apBlob = await apRes.blob();
      const apF = new File([apBlob], 'knee_ap_mask.png', { type: 'image/png' });
      setApImage(apF);
      setApPreview('/knee_ap_mask.png');

      const latRes = await fetch('/knee_lat_mask.png');
      const latBlob = await latRes.blob();
      const latF = new File([latBlob], 'knee_lat_mask.png', { type: 'image/png' });
      setLatImage(latF);
      setLatPreview('/knee_lat_mask.png');
    } catch (err) {
      console.error('Failed to load demo images:', err);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>, type: 'ap' | 'lat') => {
    const file = e.target.files?.[0]
    if (file) {
      if (type === 'ap') {
        setApImage(file)
        setApPreview(URL.createObjectURL(file))
      } else {
        setLatImage(file)
        setLatPreview(URL.createObjectURL(file))
      }
    }
  }

  const handleReconstruct = async () => {
    if (!apImage || !latImage) return;
    
    setLoading(true)
    setStatusText('Preprocessing & Extracting features...')
    setObjString(null)
    setDebugInfo(null)
    
    try {
      const formData = new FormData()
      formData.append('ap_image', apImage)
      formData.append('lat_image', latImage)
      formData.append('threshold', '0.5')
      formData.append('sigma', '1.5')
      formData.append('size', '128')

      // Simulate a bit of process states to match requirement STEP 5
      setTimeout(() => setStatusText('Fusing AP + Lateral & Generating 3D volume...'), 400)
      setTimeout(() => setStatusText('Applying organic mannequin smoothing...'), 800)

      const response = await axios.post('http://localhost:8000/reconstruct', formData)
      
      setStatusText('Rendering mannequin in 3D canvas...')
      
      setTimeout(() => {
        setObjString(response.data.obj)
        setDebugInfo(response.data.debug)
        setLoading(false)
        setStatusText('Smooth Mannequin Reconstruction Complete')
      }, 300)
      
    } catch (error) {
      console.error(error)
      setStatusText('Error during reconstruction')
      setLoading(false)
    }
  }

  const exportObj = () => {
    if (!objString) return;
    const blob = new Blob([objString], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'reconstruction.obj';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-200 font-sans p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <header className="flex items-center space-x-3 pb-4 border-b border-gray-800">
          <Bone className="w-8 h-8 text-blue-500" />
          <div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
              Biplanar X-Ray to 3D
            </h1>
            <p className="text-xs text-gray-500 uppercase tracking-widest mt-1">Research Prototype — Visual Hull Reconstruction</p>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Panel - Controls */}
          <div className="lg:col-span-4 space-y-6">
            
            <div className="bg-gray-900/50 p-5 rounded-xl border border-gray-800 backdrop-blur-sm">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold flex items-center"><CheckCircle2 className="w-5 h-5 mr-2 text-blue-500"/> Inputs</h2>
                <button
                  type="button"
                  id="load-demo-btn"
                  onClick={loadDemoData}
                  className="text-xs px-2.5 py-1 rounded bg-blue-950/60 text-blue-400 border border-blue-800/60 hover:bg-blue-900/60 transition-colors"
                >
                  Load Sample Masks
                </button>
              </div>
              
              <div className="space-y-4">
                {/* AP Upload */}
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">AP X-ray (Front)</label>
                  <div className="relative border-2 border-dashed border-gray-700 rounded-lg p-4 text-center hover:bg-gray-800/50 transition-colors cursor-pointer group">
                    <input type="file" accept="image/*" onChange={(e) => handleImageUpload(e, 'ap')} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                    {apPreview ? (
                       <img src={apPreview} alt="AP Preview" className="h-32 mx-auto object-contain rounded" />
                    ) : (
                      <div className="flex flex-col items-center py-4 text-gray-500 group-hover:text-blue-400 transition-colors">
                        <UploadCloud className="w-8 h-8 mb-2" />
                        <span className="text-sm">Drop image or click</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Lateral Upload */}
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Lateral X-ray (Side)</label>
                  <div className="relative border-2 border-dashed border-gray-700 rounded-lg p-4 text-center hover:bg-gray-800/50 transition-colors cursor-pointer group">
                    <input type="file" accept="image/*" onChange={(e) => handleImageUpload(e, 'lat')} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                    {latPreview ? (
                       <img src={latPreview} alt="Lat Preview" className="h-32 mx-auto object-contain rounded" />
                    ) : (
                      <div className="flex flex-col items-center py-4 text-gray-500 group-hover:text-blue-400 transition-colors">
                        <UploadCloud className="w-8 h-8 mb-2" />
                        <span className="text-sm">Drop image or click</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <button 
                id="reconstruct-btn"
                onClick={handleReconstruct}
                disabled={!apImage || !latImage || loading}
                className="w-full mt-6 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-medium py-3 rounded-lg transition-all shadow-lg hover:shadow-blue-500/20"
              >
                {loading ? 'Processing...' : 'Reconstruct 3D Leg'}
              </button>
              
              {statusText && (
                <div className="mt-4 p-3 bg-gray-900 rounded-lg border border-gray-800 text-sm text-center text-blue-400 animate-pulse">
                  {statusText}
                </div>
              )}
            </div>

            {/* Debug Panel */}
            {debugInfo && (
              <div className="bg-gray-900/50 p-5 rounded-xl border border-gray-800">
                <h2 className="text-sm font-semibold mb-3 text-gray-400 uppercase tracking-wider">Debug Info</h2>
                <div className="space-y-2 text-xs font-mono text-gray-300">
                  <div className="flex justify-between border-b border-gray-800 pb-1"><span>AP Res:</span> <span>{debugInfo.ap_resolution}</span></div>
                  <div className="flex justify-between border-b border-gray-800 pb-1"><span>Lat Res:</span> <span>{debugInfo.lat_resolution}</span></div>
                  <div className="flex justify-between border-b border-gray-800 pb-1"><span>Volume:</span> <span>{debugInfo.volume_dimensions}</span></div>
                  <div className="flex justify-between border-b border-gray-800 pb-1"><span>Vertices:</span> <span className="text-emerald-400">{debugInfo.vertices_count}</span></div>
                  <div className="flex justify-between border-b border-gray-800 pb-1"><span>Faces:</span> <span className="text-emerald-400">{debugInfo.faces_count}</span></div>
                  <div className="flex justify-between border-b border-gray-800 pb-1"><span>Time:</span> <span>{debugInfo.processing_time_sec}s</span></div>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - Viewer */}
          <div className="lg:col-span-8 flex flex-col space-y-4">
            
            <div className="flex justify-end space-x-3">
              <button 
                onClick={() => setWireframe(!wireframe)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${wireframe ? 'bg-blue-600/20 text-blue-400 border-blue-500/30' : 'bg-gray-900 border-gray-700 text-gray-400 hover:text-gray-200'}`}
              >
                Wireframe
              </button>
              <button 
                onClick={exportObj}
                disabled={!objString}
                className="px-4 py-2 bg-gray-900 border border-gray-700 disabled:opacity-50 text-gray-300 rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors flex items-center"
              >
                <Download className="w-4 h-4 mr-2" /> Export OBJ
              </button>
            </div>
            
            <div className="flex-1 min-h-[600px] relative">
              <Viewer3D objString={objString} wireframe={wireframe} />
            </div>

          </div>
        </div>
      </div>
    </div>
  )
}

export default App
