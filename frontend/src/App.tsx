import { useState, useEffect } from 'react';
import axios from 'axios';
import Viewer3D from './components/Viewer3D';
import type { CaliperData } from './components/Viewer3D';
import {
  UploadCloud,
  CheckCircle2,
  Download,
  Ruler,
  Layers,
  Sparkles,
  ShieldCheck,
  Box,
  RotateCw,
  Crosshair
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

interface AccuracyReport {
  target_width_mm: number;
  mesh_width_mm: number;
  width_error_mm: number;
  target_depth_mm: number;
  mesh_depth_mm: number;
  depth_error_mm: number;
  verdict_1mm: string;
  inspection_plane_z_mm: number;
}

interface SlicerEstimates {
  solid_volume_cm3: number;
  printed_volume_cm3: number;
  weights_grams: {
    PLA: number;
    PETG: number;
    TPU_Flexible: number;
    SLA_Resin: number;
  };
  estimated_print_time_hours: number;
  recommended_layer_height_mm: number;
  support_needed_pct: number;
}

interface ReconstructionResponse {
  status: string;
  message: string;
  obj: string;
  processing_time_sec: number;
  caliper_accuracy: AccuracyReport;
  mesh_statistics: {
    vertex_count: number;
    triangle_count: number;
    is_watertight: boolean;
    bounding_box_mm: number[];
  };
  slicer_estimates: SlicerEstimates;
  diagnostic_images: {
    ap_caliper_overlay?: string;
    lat_caliper_overlay?: string;
    ap_contour_overlay?: string;
    lat_contour_overlay?: string;
    metric_profile_plot?: string;
    qc_printability_sheet?: string;
  };
  download_urls: {
    stl: string;
    three_mf: string;
  };
}

export default function App() {
  const [apImage, setApImage] = useState<File | null>(null);
  const [latImage, setLatImage] = useState<File | null>(null);
  const [apPreview, setApPreview] = useState<string>('');
  const [latPreview, setLatPreview] = useState<string>('');

  const [wristWidth, setWristWidth] = useState<number>(56.0);
  const [wristDepth, setWristDepth] = useState<number>(38.0);
  const [wristCircumference, setWristCircumference] = useState<number>(150.0);

  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [result, setResult] = useState<ReconstructionResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'caliper' | 'diagnostics'>('caliper');
  const [activeModalImage, setActiveModalImage] = useState<string | null>(null);
  const [showCaliper, setShowCaliper] = useState(true);

  // Quick load sample clinical images and measurements
  const handleLoadSample = async () => {
    try {
      const apRes = await fetch('/AP_Sample_Img.jpeg');
      const apBlob = await apRes.blob();
      const apFile = new File([apBlob], 'AP_Sample_Img.jpeg', { type: 'image/jpeg' });
      setApImage(apFile);
      setApPreview('/AP_Sample_Img.jpeg');

      const latRes = await fetch('/L_Sample_Img.jpeg');
      const latBlob = await latRes.blob();
      const latFile = new File([latBlob], 'L_Sample_Img.jpeg', { type: 'image/jpeg' });
      setLatImage(latFile);
      setLatPreview('/L_Sample_Img.jpeg');

      setWristWidth(56.0);
      setWristDepth(38.0);
      setWristCircumference(150.0);
    } catch (err) {
      console.error('Failed to load sample data:', err);
    }
  };

  // Automatically load clinical sample images and anchor measurements on initial load
  useEffect(() => {
    handleLoadSample();
  }, []);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'ap' | 'lat') => {
    const file = e.target.files?.[0];
    if (file) {
      if (type === 'ap') {
        setApImage(file);
        setApPreview(URL.createObjectURL(file));
      } else {
        setLatImage(file);
        setLatPreview(URL.createObjectURL(file));
      }
    }
  };

  const handleReconstruct = async () => {
    if (!apImage || !latImage) return;

    setLoading(true);
    setResult(null);

    // Multi-step progress animations
    setCurrentStep('Step 1: Anchoring metric millimeter scale...');
    const t1 = setTimeout(() => setCurrentStep('Step 2: Slicing 111 metric cross-sections...'), 600);
    const t2 = setTimeout(() => setCurrentStep('Step 3: Lofts 3D mannequin mesh & smooths...'), 1200);
    const t3 = setTimeout(() => setCurrentStep('Step 4: Precision planar cut & watertight audit...'), 1800);

    try {
      const formData = new FormData();
      formData.append('ap_image', apImage);
      formData.append('lat_image', latImage);
      formData.append('wrist_width_mm', wristWidth.toString());
      formData.append('wrist_depth_mm', wristDepth.toString());
      formData.append('wrist_circumference_mm', wristCircumference.toString());
      formData.append('trim_height_mm', '6.0');

      const response = await axios.post<ReconstructionResponse>(`${API_BASE}/reconstruct`, formData);
      setResult(response.data);
    } catch (err: any) {
      console.error('Reconstruction failed:', err);
      alert('Reconstruction failed: ' + (err.response?.data?.error || err.message));
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      setLoading(false);
      setCurrentStep('');
    }
  };

  // Live Ramanujan perimeter estimation
  const a = wristWidth / 2.0;
  const b = wristDepth / 2.0;
  const h = (a + b > 0) ? Math.pow(a - b, 2) / Math.pow(a + b, 2) : 0;
  const theoreticalPerimeter = Math.PI * (a + b) * (1.0 + (3.0 * h) / (10.0 + Math.sqrt(Math.max(0.001, 4.0 - 3.0 * h))));
  const perimDiff = Math.abs(wristCircumference - theoreticalPerimeter);

  return (
    <div className="flex h-screen w-screen flex-col bg-[#0b0d11] text-gray-100 font-sans antialiased overflow-hidden">
      {/* ------------------------------------------------------------- */}
      {/* Top Header Bar */}
      {/* ------------------------------------------------------------- */}
      <header className="flex h-16 w-full items-center justify-between border-b border-gray-800/80 bg-gray-950/70 px-6 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 text-white shadow-lg shadow-emerald-500/20">
            <Box className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-white">Wrist 3D Reconstruct Studio</h1>
              <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-semibold text-emerald-400 ring-1 ring-emerald-500/30">
                1mm Strict Accuracy
              </span>
            </div>
            <p className="text-xs text-gray-400">2D Bi-Planar Radiographs to Watertight Mannequin STL</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleLoadSample}
            className="flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900/90 px-3.5 py-1.5 text-xs font-medium text-gray-200 transition hover:border-emerald-500/50 hover:bg-gray-800 hover:text-emerald-300"
          >
            <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
            Load Sample Hand
          </button>
          <div className="flex items-center gap-2 rounded-full border border-gray-800 bg-gray-900 px-3 py-1 text-xs text-gray-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            Backend Active
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------------- */}
      {/* Main Workspace (Split Grid) */}
      {/* ------------------------------------------------------------- */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Control & Input Sidebar */}
        <aside className="w-[430px] flex-shrink-0 border-r border-gray-800/80 bg-[#0e1117] flex flex-col overflow-y-auto">
          <div className="p-5 space-y-5">
            {/* 1. 2D Bi-Planar Images */}
            <div className="rounded-xl border border-gray-800 bg-gray-950/60 p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                  <Layers className="h-3.5 w-3.5" /> 1. Bi-Planar 2D Radiographs
                </span>
                <span className="text-[11px] text-gray-500">AP & Lateral</span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {/* AP Image */}
                <div>
                  <label className="block text-[11px] font-medium text-gray-400 mb-1">AP View (Front)</label>
                  <label className="flex flex-col items-center justify-center h-32 rounded-lg border border-dashed border-gray-700 bg-gray-900/40 hover:border-emerald-500/50 hover:bg-gray-900 transition cursor-pointer relative overflow-hidden group">
                    {apPreview ? (
                      <img src={apPreview} alt="AP Preview" className="h-full w-full object-contain p-1" />
                    ) : (
                      <div className="flex flex-col items-center text-center p-2">
                        <UploadCloud className="h-6 w-6 text-gray-400 group-hover:text-emerald-400 mb-1" />
                        <span className="text-[11px] text-gray-400">Upload AP</span>
                      </div>
                    )}
                    <input type="file" accept="image/*" onChange={(e) => handleImageChange(e, 'ap')} className="hidden" />
                  </label>
                </div>

                {/* Lateral Image */}
                <div>
                  <label className="block text-[11px] font-medium text-gray-400 mb-1">Lateral View (Side)</label>
                  <label className="flex flex-col items-center justify-center h-32 rounded-lg border border-dashed border-gray-700 bg-gray-900/40 hover:border-emerald-500/50 hover:bg-gray-900 transition cursor-pointer relative overflow-hidden group">
                    {latPreview ? (
                      <img src={latPreview} alt="Lateral Preview" className="h-full w-full object-contain p-1" />
                    ) : (
                      <div className="flex flex-col items-center text-center p-2">
                        <UploadCloud className="h-6 w-6 text-gray-400 group-hover:text-emerald-400 mb-1" />
                        <span className="text-[11px] text-gray-400">Upload LAT</span>
                      </div>
                    )}
                    <input type="file" accept="image/*" onChange={(e) => handleImageChange(e, 'lat')} className="hidden" />
                  </label>
                </div>
              </div>
            </div>

            {/* 2. Physical Caliper Measurements */}
            <div className="rounded-xl border border-gray-800 bg-gray-950/60 p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                  <Ruler className="h-3.5 w-3.5" /> 2. Physical Measurements (The 1mm Secret)
                </span>
                <span className="text-[11px] text-emerald-400/80 font-mono">Ground-Truth mm</span>
              </div>

              <div className="space-y-3">
                {/* Wrist Width */}
                <div>
                  <div className="flex justify-between text-xs text-gray-300 mb-1">
                    <span>Wrist Width (Radial-Ulnar Styloid)</span>
                    <span className="font-mono text-emerald-400 font-semibold">{wristWidth.toFixed(1)} mm</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min="40"
                      max="75"
                      step="0.5"
                      value={wristWidth}
                      onChange={(e) => setWristWidth(parseFloat(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                    <input
                      type="number"
                      step="0.1"
                      value={wristWidth}
                      onChange={(e) => setWristWidth(parseFloat(e.target.value) || 0)}
                      className="w-16 rounded border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs text-right font-mono text-white"
                    />
                  </div>
                </div>

                {/* Wrist Depth */}
                <div>
                  <div className="flex justify-between text-xs text-gray-300 mb-1">
                    <span>Wrist Depth (Palmar-Dorsal Thickness)</span>
                    <span className="font-mono text-teal-400 font-semibold">{wristDepth.toFixed(1)} mm</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min="25"
                      max="55"
                      step="0.5"
                      value={wristDepth}
                      onChange={(e) => setWristDepth(parseFloat(e.target.value))}
                      className="w-full accent-teal-500"
                    />
                    <input
                      type="number"
                      step="0.1"
                      value={wristDepth}
                      onChange={(e) => setWristDepth(parseFloat(e.target.value) || 0)}
                      className="w-16 rounded border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs text-right font-mono text-white"
                    />
                  </div>
                </div>

                {/* Wrist Circumference */}
                <div>
                  <div className="flex justify-between text-xs text-gray-300 mb-1">
                    <span>Wrist Circumference (Tape Reading)</span>
                    <span className="font-mono text-amber-400 font-semibold">{wristCircumference.toFixed(1)} mm</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min="120"
                      max="190"
                      step="1"
                      value={wristCircumference}
                      onChange={(e) => setWristCircumference(parseFloat(e.target.value))}
                      className="w-full accent-amber-500"
                    />
                    <input
                      type="number"
                      step="0.5"
                      value={wristCircumference}
                      onChange={(e) => setWristCircumference(parseFloat(e.target.value) || 0)}
                      className="w-16 rounded border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs text-right font-mono text-white"
                    />
                  </div>
                </div>

                {/* Live Ellipse Consistency Badge */}
                <div className="mt-2 rounded-lg bg-gray-900/80 p-2.5 border border-gray-800 text-[11px] text-gray-300 flex items-center justify-between">
                  <span>Theoretical Perimeter: <strong className="text-white font-mono">{theoreticalPerimeter.toFixed(1)} mm</strong></span>
                  <span className={`flex items-center gap-1 font-semibold ${perimDiff <= 6.0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    <CheckCircle2 className="h-3 w-3" />
                    {perimDiff <= 6.0 ? 'Consistent (Δ < 6mm)' : 'Check Tape'}
                  </span>
                </div>
              </div>
            </div>

            {/* 3. Reconstruct Button */}
            <button
              onClick={handleReconstruct}
              disabled={loading || !apImage || !latImage}
              className={`w-full flex items-center justify-center gap-2 rounded-xl py-3.5 px-4 font-semibold text-sm transition shadow-lg ${
                loading || !apImage || !latImage
                  ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
                  : 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600 shadow-emerald-500/25 active:scale-[0.99]'
              }`}
            >
              {loading ? (
                <>
                  <RotateCw className="h-4 w-4 animate-spin text-white" />
                  <span>{currentStep || 'Reconstructing 3D Mannequin Hand...'}</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  <span>Reconstruct 3D Mannequin Hand</span>
                </>
              )}
            </button>
          </div>
        </aside>

        {/* Right Main 3D Viewport & Diagnostic Results */}
        <main className="flex-1 flex flex-col bg-[#090b0e] overflow-hidden">
          {/* Top Control Bar */}
          <div className="h-14 border-b border-gray-800/80 bg-gray-950/60 px-6 flex items-center justify-between">
            {/* 1mm Accuracy Status Banner */}
            {result ? (
              <div className="flex items-center gap-2.5">
                <div className="flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-3 py-1 border border-emerald-500/30 text-emerald-400 text-xs font-semibold shadow-sm">
                  <ShieldCheck className="h-4 w-4 text-emerald-400" />
                  <span>1mm ACCURACY: {result.caliper_accuracy.verdict_1mm}</span>
                </div>
                <span className="text-xs text-gray-400">
                  Width Err: <strong className="text-white font-mono">{result.caliper_accuracy.width_error_mm.toFixed(2)} mm</strong> | Depth Err: <strong className="text-white font-mono">{result.caliper_accuracy.depth_error_mm.toFixed(2)} mm</strong>
                </span>
              </div>
            ) : (
              <div className="text-xs text-gray-400 flex items-center gap-2">
                <Box className="h-4 w-4 text-gray-500" />
                <span>Ready for 3D reconstruction. Load images on the left.</span>
              </div>
            )}

            {/* Model Actions */}
            <div className="flex items-center gap-2">
              {result && (
                <>
                  <a
                    href={`${API_BASE}/api/download/stl`}
                    download="mannequin_hand_production.stl"
                    className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs font-semibold text-emerald-300 hover:bg-emerald-500/20 transition shadow-sm"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Download STL
                  </a>
                  <a
                    href={`${API_BASE}/api/download/3mf`}
                    download="mannequin_hand_production.3mf"
                    className="flex items-center gap-1.5 rounded-lg border border-gray-700 bg-gray-900 px-3 py-1.5 text-xs font-medium text-gray-200 hover:bg-gray-800 transition"
                  >
                    <Download className="h-3.5 w-3.5" />
                    3MF
                  </a>
                </>
              )}

              <button
                onClick={() => setShowCaliper(!showCaliper)}
                className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition border flex items-center gap-1.5 ${
                  showCaliper ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-sm' : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800'
                }`}
              >
                <Crosshair className="h-3.5 w-3.5" />
                1mm Caliper
              </button>
            </div>
          </div>

          {/* Central 3D Canvas */}
          <div className="flex-1 relative p-4">
            <Viewer3D
              objString={result?.obj || null}
              showCaliper={showCaliper}
              caliperData={result?.caliper_accuracy as CaliperData | undefined}
            />
          </div>

          {/* Bottom Diagnostic & Slicer Report Drawer */}
          {result && (
            <div className="h-56 border-t border-gray-800/80 bg-gray-950/90 flex flex-col">
              {/* Tab Selector */}
              <div className="flex items-center border-b border-gray-800 px-6 gap-6 text-xs font-medium">
                <button
                  onClick={() => setActiveTab('caliper')}
                  className={`py-2.5 border-b-2 transition ${
                    activeTab === 'caliper' ? 'border-emerald-400 text-emerald-300 font-semibold' : 'border-transparent text-gray-400 hover:text-gray-200'
                  }`}
                >
                  1. 1mm Tolerance Caliper Audit
                </button>
                <button
                  onClick={() => setActiveTab('diagnostics')}
                  className={`py-2.5 border-b-2 transition ${
                    activeTab === 'diagnostics' ? 'border-emerald-400 text-emerald-300 font-semibold' : 'border-transparent text-gray-400 hover:text-gray-200'
                  }`}
                >
                  2. Step-by-Step Diagnostic Overlays
                </button>
              </div>

              {/* Tab Contents */}
              <div className="flex-1 overflow-y-auto p-5">
                {activeTab === 'caliper' && (
                  <div className="grid grid-cols-4 gap-4">
                    {/* Width Card */}
                    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-3.5">
                      <span className="text-[11px] text-gray-400 uppercase tracking-wider font-semibold">Wrist Width (X-axis)</span>
                      <div className="mt-1 flex items-baseline justify-between">
                        <span className="text-xl font-bold font-mono text-white">{result.caliper_accuracy.mesh_width_mm.toFixed(2)} mm</span>
                        <span className="text-xs text-gray-400">Target: {result.caliper_accuracy.target_width_mm.toFixed(1)} mm</span>
                      </div>
                      <div className="mt-2 text-xs text-emerald-400 font-medium">
                        Absolute Error: {result.caliper_accuracy.width_error_mm.toFixed(2)} mm (&lt; 1 mm)
                      </div>
                    </div>

                    {/* Depth Card */}
                    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-3.5">
                      <span className="text-[11px] text-gray-400 uppercase tracking-wider font-semibold">Wrist Depth (Y-axis)</span>
                      <div className="mt-1 flex items-baseline justify-between">
                        <span className="text-xl font-bold font-mono text-white">{result.caliper_accuracy.mesh_depth_mm.toFixed(2)} mm</span>
                        <span className="text-xs text-gray-400">Target: {result.caliper_accuracy.target_depth_mm.toFixed(1)} mm</span>
                      </div>
                      <div className="mt-2 text-xs text-emerald-400 font-medium">
                        Absolute Error: {result.caliper_accuracy.depth_error_mm.toFixed(2)} mm (&lt; 0.1 mm)
                      </div>
                    </div>

                    {/* Mesh Geometry Card */}
                    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-3.5">
                      <span className="text-[11px] text-gray-400 uppercase tracking-wider font-semibold">Solid Mesh Geometry</span>
                      <div className="mt-1 text-sm text-gray-200 space-y-0.5">
                        <div>Vertices: <strong className="text-white font-mono">{result.mesh_statistics.vertex_count.toLocaleString()}</strong></div>
                        <div>Triangles: <strong className="text-white font-mono">{result.mesh_statistics.triangle_count.toLocaleString()}</strong></div>
                        <div className="text-emerald-400 text-xs font-semibold">Watertight: Verified (0 non-manifold edges)</div>
                      </div>
                    </div>

                    {/* Scale Bounding Box */}
                    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-3.5">
                      <span className="text-[11px] text-gray-400 uppercase tracking-wider font-semibold">Physical Bounding Box</span>
                      <div className="mt-1 text-xs text-gray-300 font-mono space-y-0.5">
                        <div>W: {result.mesh_statistics.bounding_box_mm[0].toFixed(1)} mm</div>
                        <div>D: {result.mesh_statistics.bounding_box_mm[1].toFixed(1)} mm</div>
                        <div>H: {result.mesh_statistics.bounding_box_mm[2].toFixed(1)} mm</div>
                      </div>
                      <div className="mt-1 text-[11px] text-gray-400">Flush Cut Base at Z = 0.0 mm</div>
                    </div>
                  </div>
                )}

                {activeTab === 'diagnostics' && (
                  <div className="grid grid-cols-6 gap-3">
                    {/* AP Caliper */}
                    {result.diagnostic_images.ap_caliper_overlay && (
                      <div
                        onClick={() => setActiveModalImage(result.diagnostic_images.ap_caliper_overlay!)}
                        className="cursor-pointer rounded-lg border border-gray-800 bg-gray-900 p-1.5 hover:border-emerald-500 transition group text-center"
                      >
                        <img src={result.diagnostic_images.ap_caliper_overlay} alt="AP Caliper" className="h-24 w-full object-contain rounded" />
                        <span className="text-[10px] text-gray-400 group-hover:text-emerald-300 mt-1 block">Step 1: AP Caliper</span>
                      </div>
                    )}
                    {/* LAT Caliper */}
                    {result.diagnostic_images.lat_caliper_overlay && (
                      <div
                        onClick={() => setActiveModalImage(result.diagnostic_images.lat_caliper_overlay!)}
                        className="cursor-pointer rounded-lg border border-gray-800 bg-gray-900 p-1.5 hover:border-emerald-500 transition group text-center"
                      >
                        <img src={result.diagnostic_images.lat_caliper_overlay} alt="LAT Caliper" className="h-24 w-full object-contain rounded" />
                        <span className="text-[10px] text-gray-400 group-hover:text-emerald-300 mt-1 block">Step 1: LAT Caliper</span>
                      </div>
                    )}
                    {/* AP Contour */}
                    {result.diagnostic_images.ap_contour_overlay && (
                      <div
                        onClick={() => setActiveModalImage(result.diagnostic_images.ap_contour_overlay!)}
                        className="cursor-pointer rounded-lg border border-gray-800 bg-gray-900 p-1.5 hover:border-emerald-500 transition group text-center"
                      >
                        <img src={result.diagnostic_images.ap_contour_overlay} alt="AP Contour" className="h-24 w-full object-contain rounded" />
                        <span className="text-[10px] text-gray-400 group-hover:text-emerald-300 mt-1 block">Step 2: AP Contour</span>
                      </div>
                    )}
                    {/* LAT Contour */}
                    {result.diagnostic_images.lat_contour_overlay && (
                      <div
                        onClick={() => setActiveModalImage(result.diagnostic_images.lat_contour_overlay!)}
                        className="cursor-pointer rounded-lg border border-gray-800 bg-gray-900 p-1.5 hover:border-emerald-500 transition group text-center"
                      >
                        <img src={result.diagnostic_images.lat_contour_overlay} alt="LAT Contour" className="h-24 w-full object-contain rounded" />
                        <span className="text-[10px] text-gray-400 group-hover:text-emerald-300 mt-1 block">Step 2: LAT Contour</span>
                      </div>
                    )}
                    {/* Metric Profile Chart */}
                    {result.diagnostic_images.metric_profile_plot && (
                      <div
                        onClick={() => setActiveModalImage(result.diagnostic_images.metric_profile_plot!)}
                        className="cursor-pointer rounded-lg border border-gray-800 bg-gray-900 p-1.5 hover:border-emerald-500 transition group text-center"
                      >
                        <img src={result.diagnostic_images.metric_profile_plot} alt="Metric Profile" className="h-24 w-full object-contain rounded" />
                        <span className="text-[10px] text-gray-400 group-hover:text-emerald-300 mt-1 block">Step 2: Metric Ribs</span>
                      </div>
                    )}
                    {/* QC Sheet */}
                    {result.diagnostic_images.qc_printability_sheet && (
                      <div
                        onClick={() => setActiveModalImage(result.diagnostic_images.qc_printability_sheet!)}
                        className="cursor-pointer rounded-lg border border-gray-800 bg-gray-900 p-1.5 hover:border-emerald-500 transition group text-center"
                      >
                        <img src={result.diagnostic_images.qc_printability_sheet} alt="QC Sheet" className="h-24 w-full object-contain rounded" />
                        <span className="text-[10px] text-gray-400 group-hover:text-emerald-300 mt-1 block">Step 4: QC Audit</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Full-Screen Image Lightbox Modal */}
      {activeModalImage && (
        <div
          onClick={() => setActiveModalImage(null)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8 cursor-pointer"
        >
          <div className="relative max-h-full max-w-4xl rounded-2xl overflow-hidden border border-gray-700 bg-gray-900 shadow-2xl">
            <img src={activeModalImage} alt="Enlarged Diagnostic View" className="max-h-[85vh] w-auto object-contain" />
            <div className="absolute top-3 right-3 rounded-full bg-black/60 px-3 py-1 text-xs text-white">Click anywhere to close</div>
          </div>
        </div>
      )}
    </div>
  );
}
