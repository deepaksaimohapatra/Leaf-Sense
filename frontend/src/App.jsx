import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'https://leaf-sense-kr7k.onrender.com';
// const API_BASE_URL = 'http://localhost:8000'; // Using local backend

const PLANTS = [
  { id: 'apple', name: 'Apple', icon: '🍎' },
  { id: 'tomato', name: 'Tomato', icon: '🍅' },
  { id: 'potato', name: 'Potato', icon: '🥔' },
];

const MODELS = [
  { id: 'resnet50', name: 'ResNet-50 (Default)' },
  { id: 'cnn', name: 'Custom CNN' },
  { id: 'mobilenet', name: 'MobileNet (Fast)' },
  { id: 'efficientnet', name: 'EfficientNet (Accurate)' }
];

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [step, setStep] = useState('select'); // select, upload, diagnosis
  const [loading, setLoading] = useState(false);
  const [selectedPlant, setSelectedPlant] = useState(null);
  const [selectedModel, setSelectedModel] = useState('resnet50');
  const [diagnosis, setDiagnosis] = useState(null);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const diagnoseHealth = async () => {
    if (!image || !selectedPlant) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image', image);
    formData.append('confirmed_plant', selectedPlant);
    formData.append('model', selectedModel);

    try {
      const response = await axios.post(`${API_BASE_URL}/diagnose-health`, formData, {
        headers: {
          'ngrok-skip-browser-warning': 'true'
        }
      });
      if (response.data.status === 'success') {
        setDiagnosis(response.data.data);
        setStep('diagnosis');
      } else {
        setError(response.data.message || 'Failed to diagnose health.');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Error connecting to server.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setPreview(null);
    setStep('select');
    setSelectedPlant(null);
    setSelectedModel('resnet50');
    setDiagnosis(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white font-sans selection:bg-primary-500 selection:text-white">
      {/* Header */}
      <header className="py-6 md:py-8 px-4 md:px-6 text-center border-b border-white/5 bg-slate-900/50 backdrop-blur-xl sticky top-0 z-50">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-primary-400 to-emerald-400 bg-clip-text text-transparent mb-2">
          LEAF SENSE
        </h1>
        <p className="text-slate-400 text-lg">Intelligent Leaf Diagnosis & Plant Care</p>
      </header>

      <main className="max-w-4xl mx-auto py-6 md:py-12 px-4 md:px-6">
        <div className="bg-slate-800/40 border border-white/10 rounded-2xl md:rounded-3xl p-4 md:p-8 backdrop-blur-md shadow-2xl">

          {error && (
            <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-2xl flex items-center justify-center gap-3 animate-pulse">
              <span className="text-xl font-bold">!</span>
              {error}
            </div>
          )}

          {/* Stepper (Visual Only) */}
          <div className="flex justify-between mb-12 px-4 max-w-md mx-auto">
            {['Select', 'Upload', 'Diagnose'].map((s, i) => (
              <div key={s} className="flex flex-col items-center gap-2 relative">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-500 ${(i === 0 && step === 'select') || (i === 1 && step === 'upload') || (i === 2 && step === 'diagnosis')
                  ? 'border-primary-500 bg-primary-500 shadow-[0_0_15px_rgba(34,197,94,0.4)]'
                  : i < ['select', 'upload', 'diagnosis'].indexOf(step)
                    ? 'border-primary-500/50 bg-primary-500/20'
                    : 'border-white/20 bg-white/5 opacity-50'
                  }`}>
                  <span className="text-sm font-bold">{i + 1}</span>
                </div>
                <span className={`text-xs font-medium uppercase tracking-widest ${(i === 0 && step === 'select') || (i === 1 && step === 'upload') || (i === 2 && step === 'diagnosis')
                  ? 'text-primary-400'
                  : 'text-slate-500'
                  }`}>{s}</span>
              </div>
            ))}
          </div>

          {/* Step content */}
          <div className="min-h-[400px] flex flex-col items-center justify-center transition-all duration-700 ease-out">

            {step === 'select' && (
              <div className="w-full max-w-2xl space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                <div className="text-center space-y-2">
                  <h2 className="text-3xl font-bold">Choose Your Plant</h2>
                  <p className="text-slate-400">Select the type of plant you want to analyze</p>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                  {PLANTS.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setSelectedPlant(p.id);
                        setStep('upload');
                      }}
                      className="group p-6 md:p-8 bg-white/5 border border-white/10 rounded-3xl md:rounded-4xl hover:border-primary-500/50 hover:bg-primary-500/5 transition-all text-center space-y-4 active:scale-95"
                    >
                      <div className="text-5xl group-hover:scale-110 transition-transform">{p.icon}</div>
                      <div className="text-xl font-bold">{p.name}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {step === 'upload' && (
              <div className="w-full max-w-lg space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                <div className="flex items-center justify-between">
                  <button onClick={() => setStep('select')} className="text-slate-400 hover:text-white flex items-center gap-2 text-sm font-medium">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path></svg>
                    Back to Select
                  </button>
                  <div className="px-4 py-1.5 bg-primary-500/10 border border-primary-500/20 rounded-full text-primary-400 text-xs font-bold uppercase tracking-wider">
                    {PLANTS.find(p => p.id === selectedPlant)?.name}
                  </div>
                </div>

                <div
                  className={`group relative border-2 border-dashed rounded-3xl md:rounded-4xl p-8 md:p-12 text-center transition-all cursor-pointer overflow-hidden ${preview ? 'border-primary-500/40 bg-primary-500/5' : 'border-white/10 hover:border-primary-500/30 hover:bg-white/5'
                    }`}
                  onClick={() => document.getElementById('file-upload').click()}
                >
                  {preview ? (
                    <img src={preview} alt="Preview" className="w-full h-48 md:h-64 object-cover rounded-2xl shadow-lg brightness-90 group-hover:brightness-100 transition-all duration-500" />
                  ) : (
                    <div className="space-y-4">
                      <div className="w-20 h-20 bg-primary-500/10 rounded-3xl flex items-center justify-center mx-auto text-primary-400 group-hover:scale-110 group-hover:bg-primary-500/20 transition-all duration-500">
                        <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
                      </div>
                      <div className="space-y-1">
                        <p className="text-xl font-bold text-white">Upload leaf Image</p>
                        <p className="text-slate-400">Drag and drop or click to browse</p>
                      </div>
                    </div>
                  )}
                  <input id="file-upload" type="file" className="hidden" onChange={handleImageChange} accept="image/*" />
                </div>

                <div className="w-full text-left space-y-2">
                  <label className="text-sm font-bold text-slate-400 uppercase tracking-wider ml-2">Select Classification Model</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full bg-slate-800 border border-white/10 rounded-2xl px-4 py-3 text-white focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all appearance-none cursor-pointer hover:border-white/20"
                    style={{ backgroundImage: `url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 1rem center', backgroundSize: '1.2em' }}
                  >
                    {MODELS.map(m => (
                      <option key={m.id} value={m.id}>{m.name}</option>
                    ))}
                  </select>
                </div>

                <button
                  disabled={!image || loading}
                  onClick={diagnoseHealth}
                  className="w-full py-5 bg-gradient-to-r from-primary-600 to-emerald-600 hover:from-primary-500 hover:to-emerald-500 disabled:opacity-30 disabled:cursor-not-allowed rounded-3xl font-bold text-lg shadow-xl hover:shadow-primary-500/20 transition-all active:scale-95 flex items-center justify-center gap-3"
                >
                  {loading ? (
                    <span className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></span>
                  ) : (
                    <>
                      Analyze Health
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                    </>
                  )}
                </button>
              </div>
            )}

            {step === 'diagnosis' && diagnosis && (
              <div className="w-full text-center space-y-10 animate-in slide-in-from-top-4 fade-in duration-1000">
                <div className="relative inline-block">
                  <div className={`absolute inset-0 blur-3xl rounded-full ${diagnosis.prediction === 'Healthy' ? 'bg-emerald-500/30' : 'bg-red-500/30'}`}></div>
                  <div className={`relative px-6 md:px-12 py-6 rounded-3xl md:rounded-4xl border-2 backdrop-blur-2xl ${diagnosis.prediction === 'Healthy'
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                    : 'bg-red-500/10 border-red-500/30 text-red-400'
                    }`}>
                    <div className="uppercase tracking-widest text-xs font-black mb-2">Final Diagnosis</div>
                    <h2 className="text-4xl md:text-6xl font-black">{diagnosis.prediction}</h2>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-8 text-left mb-8">
                  <div className="bg-white/5 border border-white/10 p-4 md:p-6 rounded-2xl md:rounded-3xl space-y-4">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-slate-500">Diagnosis Details</h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-end">
                        <span className="text-slate-400">Model Certainty</span>
                        <span className="text-2xl font-bold">{(diagnosis.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-white/10 h-3 rounded-full overflow-hidden">
                        <div className={`h-full transition-all duration-1000 ${diagnosis.prediction === 'Healthy' ? 'bg-emerald-500' : 'bg-red-500'}`} style={{ width: `${diagnosis.confidence * 100}%` }}></div>
                      </div>
                      <p className="text-sm text-slate-400 italic">
                        The prediction was made using the {MODELS.find(m => m.id === selectedModel)?.name} algorithm optimized for {PLANTS.find(p => p.id === selectedPlant)?.name} leaf images.
                      </p>
                    </div>
                  </div>

                  <div className="bg-white/5 border border-white/10 p-4 md:p-6 rounded-2xl md:rounded-3xl space-y-4">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-slate-500">Leaf Analysis</h3>
                    <div className="aspect-video w-full rounded-2xl overflow-hidden grayscale-[0.5] opacity-80 border border-white/10">
                      <img src={preview} alt="Analyzed Leaf" className="w-full h-full object-cover" />
                    </div>
                  </div>
                </div>

                {diagnosis.metrics_chart_base64 ? (
                  <div className="bg-white/5 border border-white/10 p-4 md:p-6 rounded-2xl md:rounded-3xl mb-8 flex justify-center items-center overflow-hidden animate-in slide-in-from-bottom-2 fade-in duration-700 delay-100">
                    <img src={`data:image/png;base64,${diagnosis.metrics_chart_base64}`} alt="Model Performance Metrics" className="w-full max-w-lg h-auto drop-shadow-2xl mix-blend-screen opacity-100" />
                  </div>
                ) : diagnosis.metrics && (
                  <div className="bg-white/5 border border-white/10 p-4 md:p-6 rounded-2xl md:rounded-3xl mb-8 space-y-6 text-left animate-in slide-in-from-bottom-2 fade-in duration-700 delay-100">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-slate-500">Model Performance Graph</h3>
                    <div className="space-y-4">
                      {Object.entries(diagnosis.metrics).map(([key, value]) => {
                        const score = value * 100;
                        const label = key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                        let colorClass = 'bg-primary-500';
                        if (score >= 95) colorClass = 'bg-emerald-500';
                        else if (score >= 90) colorClass = 'bg-emerald-400';
                        else if (score >= 85) colorClass = 'bg-primary-400';
                        else colorClass = 'bg-amber-400';

                        return (
                          <div key={key} className="space-y-1">
                            <div className="flex justify-between items-center">
                              <span className="text-xs font-bold text-slate-400 tracking-wider w-24 truncate">{label}</span>
                              <div className="flex-1 mx-4">
                                <div className="w-full bg-white/10 h-3 rounded-full overflow-hidden shadow-inner">
                                  <div className={`h-full transition-all duration-1000 ${colorClass}`} style={{ width: `${score}%` }}></div>
                                </div>
                              </div>
                              <span className="text-sm font-bold text-slate-200 text-right w-12">{score.toFixed(1)}%</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {diagnosis.recommendation && (
                  <div className="text-left bg-white/5 border border-white/10 p-4 md:p-6 rounded-2xl md:rounded-3xl mb-8 space-y-6 animate-in slide-in-from-bottom-4 fade-in duration-700">
                    <h3 className="text-lg font-bold uppercase tracking-widest pb-4 border-b border-white/10 flex items-center gap-3">
                      {diagnosis.prediction === 'Healthy' 
                        ? <><span className="text-emerald-400">Optimal Plant Care Guidelines</span></> 
                        : <><span className="text-red-400">Treatment & Recovery Plan</span></>}
                    </h3>
                    
                    {diagnosis.recommendation.status === 'Optimal' ? (
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="text-sm font-bold text-slate-400 uppercase mb-2">Maintenance</h4>
                          <ul className="list-disc pl-5 text-slate-300 space-y-1 text-sm">
                            {diagnosis.recommendation.guidelines?.maintenance_tips?.map((t, i) => <li key={i}>{t}</li>)}
                          </ul>
                        </div>
                        <div>
                          <h4 className="text-sm font-bold text-slate-400 uppercase mb-2">Fertilizer</h4>
                          <ul className="list-disc pl-5 text-slate-300 space-y-1 text-sm">
                            {diagnosis.recommendation.guidelines?.fertilizer_suggestions?.map((t, i) => <li key={i}>{t}</li>)}
                          </ul>
                        </div>
                        <div>
                          <h4 className="text-sm font-bold text-slate-400 uppercase mb-2">Watering & Sunlight</h4>
                          <ul className="list-disc pl-5 text-slate-300 space-y-1 text-sm">
                            <li>{diagnosis.recommendation.guidelines?.watering_schedule}</li>
                            <li>{diagnosis.recommendation.guidelines?.sunlight_requirements}</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="text-sm font-bold text-slate-400 uppercase mb-2">Yield Tips</h4>
                          <ul className="list-disc pl-5 text-slate-300 space-y-1 text-sm">
                            {diagnosis.recommendation.guidelines?.yield_improvement_tips?.map((t, i) => <li key={i}>{t}</li>)}
                          </ul>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-6">
                        <div className="space-y-2">
                          <h4 className="text-sm font-bold text-red-400 uppercase">Possible Causes</h4>
                          <div className="flex flex-wrap gap-2">
                            {diagnosis.recommendation.causes?.map((c, i) => (
                              <span key={i} className="px-3 py-1 bg-red-500/10 border border-red-500/20 text-red-300 text-sm rounded-full">{c}</span>
                            ))}
                          </div>
                        </div>
                        
                        <div className="grid md:grid-cols-2 gap-6">
                          <div className="space-y-2">
                            <h4 className="text-sm font-bold text-emerald-400 uppercase flex items-center gap-2">
                              <span className="text-xl">🌿</span> Organic Solutions
                            </h4>
                            <ul className="list-disc pl-5 text-slate-300 space-y-1 text-sm">
                              {diagnosis.recommendation.solutions?.organic?.map((s, i) => <li key={i}>{s}</li>)}
                            </ul>
                          </div>
                          
                          <div className="space-y-2">
                            <h4 className="text-sm font-bold text-amber-400 uppercase flex items-center gap-2">
                              <span className="text-xl">🧪</span> Chemical Treatment
                            </h4>
                            <ul className="list-disc pl-5 text-slate-300 space-y-1 text-sm">
                              {diagnosis.recommendation.solutions?.chemical?.map((s, i) => <li key={i}>{s}</li>)}
                            </ul>
                          </div>
                        </div>

                        <div className="bg-primary-500/10 border border-primary-500/20 p-4 rounded-2xl">
                          <h4 className="text-sm font-bold text-primary-400 uppercase mb-2">Future Prevention</h4>
                          <ul className="list-disc pl-5 text-primary-300 space-y-1 text-sm">
                            {diagnosis.recommendation.prevention_tips?.map((p, i) => <li key={i}>{p}</li>)}
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <button
                  onClick={reset}
                  className="py-5 px-12 bg-white text-slate-900 rounded-3xl font-black hover:bg-slate-200 transition-all uppercase tracking-tighter"
                >
                  Start New Scan
                </button>
              </div>
            )}

          </div>
        </div>
      </main>

      <footer className="max-w-4xl mx-auto py-12 px-6 text-center border-t border-white/5 space-y-4">
        <p className="text-slate-500 text-sm font-medium">© 2026 LEAF SENSE</p>
        <div className="flex justify-center gap-6 grayscale opacity-50">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-400">
            <div className="w-2 h-2 rounded-full bg-primary-500"></div> System Active
          </div>
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-400">
            <div className="w-2 h-2 rounded-full bg-primary-500"></div> Beta v1.0
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
