"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";
import Profiling from "./components/profiling";

export default function Home() {
  const [view, setView] = useState<"meeting" | "profiling">("meeting");
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [minutes, setMinutes] = useState<string | null>(null);
  const [attendeesInput, setAttendeesInput] = useState("");

  // --- NEW: State for the discovery popup ---
  const [discoveredProfiles, setDiscoveredProfiles] = useState<Record<string, number[]>>({});
  const [selectedToSave, setSelectedToSave] = useState<Set<string>>(new Set());

  const startMeeting = async () => {
    setMinutes(null);
    setTranscript([]);
    setDiscoveredProfiles({});
    
    const attendees = attendeesInput.split(",").map(n => n.trim()).filter(n => n);
    await fetch("http://127.0.0.1:5000/api/start", { 
      method: "POST", 
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attendees })
    });
    setIsRecording(true);
  };

  const stopMeeting = async () => {
    const res = await fetch("http://127.0.0.1:5000/api/stop", { method: "POST" });
    const data = await res.json();
    
    // --- NEW: Catch any new voices the AI learned during the meeting ---
    if (data.new_profiles_discovered && Object.keys(data.new_profiles_discovered).length > 0) {
      setDiscoveredProfiles(data.new_profiles_discovered);
      // Pre-select all of them by default
      setSelectedToSave(new Set(Object.keys(data.new_profiles_discovered)));
    }
    
    setIsRecording(false);
  };

  // --- NEW: Save the selected profiles back to the backend ---
  const saveSelectedProfiles = async () => {
    const profilesToSave: Record<string, number[]> = {};
    selectedToSave.forEach(name => {
      profilesToSave[name] = discoveredProfiles[name];
    });

    await fetch("http://127.0.0.1:5000/api/save_discovered_profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profiles: profilesToSave })
    });

    setDiscoveredProfiles({}); // Close the popup
  };

  const clearMeeting = async () => {
    await fetch("http://127.0.0.1:5000/api/clear", { method: "POST" });
    setTranscript([]);
    setMinutes(null);
    setDiscoveredProfiles({});
  };

  const downloadMinutes = () => {
    if (!minutes) return;
    const blob = new Blob([minutes], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "Meeting_Minutes.md";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleSelection = (name: string) => {
    const newSet = new Set(selectedToSave);
    if (newSet.has(name)) newSet.delete(name);
    else newSet.add(name);
    setSelectedToSave(newSet);
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://127.0.0.1:5000/api/transcript");
        const data = await res.json();
        setTranscript(data.transcript);
        if (!data.is_recording && transcript.length > 0) {
          const mRes = await fetch("http://127.0.0.1:5000/api/minutes");
          const mData = await mRes.json();
          if (mData.minutes) setMinutes(mData.minutes);
        }
      } catch (e) {}
    }, 2000);
    return () => clearInterval(interval);
  }, [transcript.length]);

  return (
    <div className="flex h-screen bg-[#eaf4f4] text-black overflow-hidden font-sans relative">
      
      {/* --- NEW: Discovery Popup Modal --- */}
      <AnimatePresence>
        {Object.keys(discoveredProfiles).length > 0 && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 20 }} 
              animate={{ opacity: 1, scale: 1, y: 0 }} 
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="bg-white w-[400px] p-8 rounded-[32px] shadow-2xl border border-[#a4c3b2]/30"
            >
              <div className="flex items-center gap-3 mb-2">
                <div className="w-10 h-10 bg-[#eaf4f4] rounded-xl flex items-center justify-center text-xl shadow-sm">🎙️</div>
                <h2 className="text-xl font-bold tracking-tight">New Voices Detected</h2>
              </div>
              <p className="text-sm text-black/60 font-medium mb-6 leading-relaxed">
                The AI identified these new speakers during your session. Select the ones you want to save permanently.
              </p>
              
              <div className="space-y-2 mb-8 max-h-[200px] overflow-y-auto pr-2 scrollbar-thin">
                {Object.keys(discoveredProfiles).map(name => (
                  <label key={name} className="flex items-center gap-3 p-3 bg-[#eaf4f4]/40 rounded-xl cursor-pointer hover:bg-[#eaf4f4] transition-colors border border-transparent hover:border-[#a4c3b2]/30">
                    <input 
                      type="checkbox" 
                      checked={selectedToSave.has(name)}
                      onChange={() => toggleSelection(name)}
                      className="w-5 h-5 accent-[#6b9080] rounded"
                    />
                    <span className="font-bold text-sm">{name}</span>
                  </label>
                ))}
              </div>

              <div className="flex gap-3">
                <button 
                  onClick={() => setDiscoveredProfiles({})} 
                  className="flex-1 py-3 rounded-xl text-sm font-bold text-gray-600 bg-[#eaf4f4] hover:bg-[#d6e8e8] transition-colors"
                >
                  Discard All
                </button>
                <button 
                  onClick={saveSelectedProfiles}
                  disabled={selectedToSave.size === 0}
                  className="flex-1 py-3 rounded-xl text-sm font-bold text-white bg-[#6b9080] hover:bg-[#587a6b] disabled:opacity-50 transition-colors shadow-md"
                >
                  Save Selected
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* NAVIGATION RAIL */}
      <nav className="w-64 bg-white border-r border-[#a4c3b2]/30 flex flex-col p-6 z-10 shadow-sm">
        <div className="mb-10">
          <div className="text-black font-black text-xl tracking-tight flex items-center gap-2">
            <div className="w-8 h-8 bg-[#6b9080] rounded-lg flex items-center justify-center text-white text-base shadow-sm">
              V
            </div>
            Vocalis.
          </div>
        </div>
        
        <div className="flex-1 space-y-2 relative">
          <button 
            onClick={() => setView("meeting")}
            className={`relative w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-colors text-sm font-semibold z-10 ${
              view === "meeting" ? 'text-[#6b9080]' : 'text-black/60 hover:text-black'
            }`}
          >
            {view === "meeting" && (
              <motion.div layoutId="active-tab" className="absolute inset-0 bg-[#a4c3b2]/30 rounded-xl" initial={false} transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}/>
            )}
            <span className="text-lg relative z-20">📅</span> 
            <span className="relative z-20">Meeting Hub</span>
          </button>

          <button 
            onClick={() => setView("profiling")}
            className={`relative w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-colors text-sm font-semibold z-10 ${
              view === "profiling" ? 'text-[#6b9080]' : 'text-black/60 hover:text-black'
            }`}
          >
            {view === "profiling" && (
              <motion.div layoutId="active-tab" className="absolute inset-0 bg-[#a4c3b2]/30 rounded-xl" initial={false} transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}/>
            )}
            <span className="text-lg relative z-20">🎙️</span> 
            <span className="relative z-20">Voice Profiling</span>
          </button>
        </div>

        <div className="pt-6 border-t border-[#eaf4f4]">
          <div className="p-3 bg-[#eaf4f4] rounded-xl">
            <p className="text-[9px] font-bold text-[#6b9080] uppercase tracking-widest mb-1 text-center">Engine Status</p>
            <p className="text-xs text-center text-black font-semibold flex items-center justify-center gap-1.5">
              <span className="w-1.5 h-1.5 bg-[#6b9080] rounded-full animate-pulse" /> Local AI Online
            </p>
          </div>
        </div>
      </nav>

      {/* WORKSPACE AREA */}
      <main className="flex-1 overflow-y-auto custom-scrollbar relative">
        <div className="max-w-5xl mx-auto p-8">
          <AnimatePresence mode="wait">
            
            {view === "profiling" ? (
              <motion.div key="profiling" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                <Profiling onProfileComplete={() => {}} />
              </motion.div>
            ) : (
              <motion.div key="meeting" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} className="space-y-6">
                
                {/* Header Card */}
                <div className="bg-white p-6 rounded-3xl shadow-sm border border-[#a4c3b2]/20 flex flex-col md:flex-row justify-between items-center gap-6">
                  <div className="space-y-2 w-full md:w-auto">
                    <h1 className="text-2xl font-bold tracking-tight text-black">Active Session</h1>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-[9px] font-bold text-[#6b9080] uppercase tracking-widest ml-1">Expected Attendees</label>
                      <input 
                        className="w-full md:w-80 px-4 py-2.5 text-sm bg-[#eaf4f4] border-none rounded-xl focus:ring-2 focus:ring-[#6b9080] transition-all outline-none font-medium placeholder:text-[#a4c3b2]"
                        placeholder="e.g. Nafis, Muyeed"
                        value={attendeesInput}
                        onChange={(e) => setAttendeesInput(e.target.value)}
                        disabled={isRecording}
                      />
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    {!isRecording && (transcript.length > 0 || minutes) && (
                      <motion.button 
                        whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                        onClick={clearMeeting}
                        className="px-6 py-3 rounded-full text-sm font-bold text-gray-600 bg-[#eaf4f4] hover:bg-[#d6e8e8] transition-colors shadow-sm"
                      >
                        Clear Session
                      </motion.button>
                    )}
                    
                    <button 
                      onClick={isRecording ? stopMeeting : startMeeting}
                      className={`px-6 py-3 rounded-full text-sm font-bold text-white transition-colors shadow-md ${
                        isRecording ? 'bg-black shadow-black/20 animate-pulse' : 'bg-[#6b9080] shadow-[#6b9080]/30 hover:bg-[#587a6b]'
                      }`}
                    >
                      <span className="flex items-center gap-2">
                        {isRecording ? "Finish & Summarize" : "Start Meeting"}
                        <span className="text-base">{isRecording ? "⏹" : "▶"}</span>
                      </span>
                    </button>
                  </div>
                </div>

                {/* Interaction Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 h-[500px]">
                  
                  {/* Live Transcript Panel */}
                  <div className="lg:col-span-2 bg-white p-6 rounded-3xl shadow-sm border border-[#a4c3b2]/20 flex flex-col overflow-hidden">
                    <h2 className="text-xs font-bold text-[#6b9080] uppercase tracking-widest mb-4">Live Stream</h2>
                    <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin">
                      {transcript.length === 0 && <p className="text-[#a4c3b2] text-sm italic font-medium">Listening for voice patterns...</p>}
                      <AnimatePresence>
                        {transcript.map((line, i) => (
                          <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="p-3 bg-[#eaf4f4]/50 rounded-2xl border border-[#eaf4f4]">
                            <span className="text-[10px] font-black text-[#6b9080] uppercase block mb-0.5 tracking-wider">
                              {line.split("]:")[0].replace("[", "")}
                            </span>
                            <p className="text-sm text-black font-medium leading-relaxed">{line.split("]:")[1]}</p>
                          </motion.div>
                        ))}
                      </AnimatePresence>
                    </div>
                  </div>

                  {/* AI Minutes Panel */}
                  <div className="lg:col-span-3 bg-white p-6 rounded-3xl shadow-sm border border-[#a4c3b2]/20 flex flex-col overflow-hidden relative">
                     <div className="flex justify-between items-center mb-4">
                        <h2 className="text-xs font-bold text-[#6b9080] uppercase tracking-widest">Meeting Minutes</h2>
                        
                        {isRecording && (
                          <span className="flex items-center gap-1.5 text-[10px] font-bold text-black bg-[#eaf4f4] px-2 py-0.5 rounded-full">
                            <motion.span animate={{ opacity: [1, 0, 1] }} transition={{ repeat: Infinity, duration: 1 }} className="w-1.5 h-1.5 bg-[#6b9080] rounded-full"/> 
                            LIVE
                          </span>
                        )}

                        {!isRecording && minutes && (
                          <motion.button 
                            initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
                            onClick={downloadMinutes} 
                            className="text-[10px] font-bold text-white bg-[#6b9080] hover:bg-[#587a6b] px-3 py-1.5 rounded-lg transition-colors shadow-sm flex items-center gap-1.5"
                          >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                            Download .md
                          </motion.button>
                        )}
                     </div>
                    
                    <div className="flex-1 overflow-y-auto bg-[#eaf4f4]/30 p-6 rounded-2xl border border-[#eaf4f4] prose prose-sm prose-p:text-black prose-headings:text-black max-w-none relative">
                      {minutes ? (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                          <ReactMarkdown components={{
                              h1: ({...props}) => <h1 className="text-xl font-bold mb-4 text-[#6b9080]" {...props} />,
                              h2: ({...props}) => <h2 className="text-base font-bold mt-6 mb-2 border-b border-[#a4c3b2]/30 pb-1" {...props} />,
                              li: ({...props}) => <li className="mb-1 font-medium text-sm" {...props} />,
                            }}>
                            {minutes}
                          </ReactMarkdown>
                        </motion.div>
                      ) : !isRecording && transcript.length > 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-center space-y-4">
                          <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }} className="w-8 h-8 border-4 border-[#a4c3b2]/30 border-t-[#6b9080] rounded-full"/>
                          <p className="text-[#6b9080] text-xs font-bold animate-pulse uppercase tracking-widest">AI is synthesizing minutes...</p>
                        </div>
                      ) : (
                        <div className="h-full flex flex-col items-center justify-center text-center space-y-3">
                          <div className="text-3xl opacity-50">📄</div>
                          <p className="text-[#a4c3b2] text-sm font-medium max-w-[200px]">Your professional summary will be generated here.</p>
                        </div>
                      )}
                    </div>
                  </div>

                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}