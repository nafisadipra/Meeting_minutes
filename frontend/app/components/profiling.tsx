"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface ProfilingProps {
  onProfileComplete: (name: string) => void;
}

export default function Profiling({ onProfileComplete }: ProfilingProps) {
  const [name, setName] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [profiles, setProfiles] = useState<string[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  // Fetch already saved profiles from the backend when the component loads
  useEffect(() => {
    const fetchExistingProfiles = async () => {
      try {
        const res = await fetch("http://127.0.0.1:5000/api/list_profiles");
        const data = await res.json();
        if (data.enrolled_profiles) {
          setProfiles(data.enrolled_profiles);
        }
      } catch (err) {
        console.error("Failed to fetch existing profiles", err);
      }
    };
    fetchExistingProfiles();
  }, []);

  const recordVoice = async () => {
    if (!name.trim()) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      setIsRecording(true);
      mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", blob, `${name}.webm`);
        formData.append("name", name);

        await fetch("http://127.0.0.1:5000/api/enroll_voice", { method: "POST", body: formData });
        
        setProfiles((prev) => [...prev, name]); // Safely update state
        onProfileComplete(name);
        setName("");
        setIsRecording(false);
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorder.start();
      setTimeout(() => { if (mediaRecorder.state === "recording") mediaRecorder.stop(); }, 5000);
    } catch (err) {
      console.error("Mic access denied", err);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="w-full h-full space-y-6"
    >
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 bg-white border border-[#a4c3b2]/30 rounded-xl flex items-center justify-center text-xl shadow-sm">
          🎙️
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-black">Voice Identity Registration</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Left Column: Enrollment Tool */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-[#a4c3b2]/30 h-fit">
          <h3 className="text-sm font-bold text-[#6b9080] uppercase tracking-widest mb-4 border-b border-[#eaf4f4] pb-2">Capture Voice Print</h3>
          <p className="text-xs text-black/60 font-medium mb-6">
            Enter an attendee's name and record a 5-second audio sample to train the local AI recognition engine.
          </p>
          
          <div className="space-y-4">
            <input 
              className="w-full px-4 py-3 text-sm bg-[#eaf4f4] border-none rounded-xl focus:ring-2 focus:ring-[#6b9080] transition-all outline-none text-black placeholder:text-[#a4c3b2] font-medium"
              placeholder="e.g. John Doe"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isRecording}
            />
            
            <button 
              onClick={recordVoice}
              disabled={isRecording || !name.trim()}
              className={`w-full py-3 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 ${
                isRecording 
                  ? 'bg-black text-[#eaf4f4] shadow-md' 
                  : 'bg-[#6b9080] text-white hover:bg-[#587a6b] disabled:bg-[#eaf4f4] disabled:text-[#a4c3b2] shadow-sm disabled:shadow-none'
              }`}
            >
              {isRecording ? (
                <>
                  <motion.span 
                    animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }} 
                    transition={{ repeat: Infinity, duration: 1.5 }}
                    className="w-2.5 h-2.5 bg-[#eaf4f4] rounded-full" 
                  />
                  Recording Audio...
                </>
              ) : (
                "Start 5s Recording"
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Active Roster */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-[#a4c3b2]/30">
          <h3 className="text-sm font-bold text-[#6b9080] uppercase tracking-widest mb-4 border-b border-[#eaf4f4] pb-2">Enrolled Roster</h3>
          
          <div className="flex flex-col gap-3">
            <AnimatePresence>
              {profiles.length === 0 && (
                <motion.div 
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center py-10 text-center space-y-2"
                >
                  <span className="text-3xl opacity-40">📭</span>
                  <p className="text-[#a4c3b2] italic text-xs font-medium">No voice profiles registered for this session.</p>
                </motion.div>
              )}
              {profiles.map((p, index) => (
                <motion.div 
                  key={p} 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center gap-3 p-3 bg-[#eaf4f4]/40 rounded-xl border border-[#eaf4f4]"
                >
                  <div className="w-10 h-10 bg-white text-[#6b9080] rounded-lg border border-[#a4c3b2]/20 flex items-center justify-center font-bold text-base shadow-sm">
                    {p[0].toUpperCase()}
                  </div>
                  <div className="flex-1 flex flex-col">
                    <span className="text-sm font-bold text-black">{p}</span>
                    <span className="text-[10px] font-medium text-[#6b9080]">Voice matched & locked</span>
                  </div>
                  <span className="text-[9px] font-bold text-white bg-[#a4c3b2] px-2.5 py-1 rounded-full uppercase tracking-wider shadow-sm">
                    Ready
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

      </div>
    </motion.div>
  );
}