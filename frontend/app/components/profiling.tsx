"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000";

interface ProfilingProps {
  onProfileComplete: (name: string) => void;
}

export default function Profiling({ onProfileComplete }: ProfilingProps) {
  const [name, setName] = useState("");
  const [status, setStatus] = useState<"idle" | "recording" | "uploading" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [profiles, setProfiles] = useState<string[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  useEffect(() => {
    const fetchProfiles = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/list_profiles`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.enrolled_profiles) setProfiles(data.enrolled_profiles);
      } catch {
        // Handle silently
      }
    };
    fetchProfiles();
  }, []);

  const recordVoice = async () => {
    const trimmedName = name.trim();
    if (!trimmedName) return;
    setErrorMsg("");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setErrorMsg("Microphone access denied. Check permissions.");
      return;
    }

    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    const chunks: BlobPart[] = [];

    setStatus("recording");
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setStatus("uploading");

      try {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", blob, `${trimmedName}.webm`);
        formData.append("name", trimmedName);

        const res = await fetch(`${API_BASE}/api/enroll`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error ?? `Server error ${res.status}`);
        }

        setProfiles((prev) =>
          prev.includes(trimmedName) ? prev : [...prev, trimmedName]
        );
        onProfileComplete(trimmedName);
        setName("");
        setStatus("idle");
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "Upload failed.");
        setStatus("error");
      }
    };

    mediaRecorder.start();
    setTimeout(() => {
      if (mediaRecorder.state === "recording") mediaRecorder.stop();
    }, 5000);
  };

  const deleteProfile = async (nameToDelete: string) => {
    // Optimistically update the UI immediately
    setProfiles((prev) => prev.filter((p) => p !== nameToDelete));
    
    try {
      // Send the delete request to your backend
      await fetch(`${API_BASE}/api/delete_profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: nameToDelete }),
      });
    } catch (err) {
      console.error("Failed to delete profile:", err);
    }
  };

  const isRecording = status === "recording";
  const isUploading = status === "uploading";
  const isBusy = isRecording || isUploading;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="max-w-4xl mx-auto w-full flex flex-col gap-10"
    >
      
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-black tracking-tight">Voice Identity Registry</h1>
        <p className="text-sm text-gray-700 mt-2">
          Manage the local database of recognized speaker profiles.
        </p>
      </div>

      {/* Enrollment Bar */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-bold text-black uppercase tracking-widest">
          Enroll New Subject
        </label>
        
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            className="flex-1 bg-white border border-black rounded-md px-4 py-3 outline-none text-black placeholder:text-gray-500 text-sm"
            placeholder="Enter subject name (e.g. John Doe)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={isBusy}
          />

          <button
            onClick={recordVoice}
            disabled={isBusy || !name.trim()}
            className={`sm:w-auto px-6 py-3 rounded-md text-sm font-bold border border-black transition-all shrink-0 ${
              isBusy
                ? "bg-gray-100 text-gray-500"
                : "bg-[#D9D9D9] text-black hover:bg-gray-300 disabled:opacity-30"
            }`}
          >
            {isRecording ? "Recording..." : isUploading ? "Processing..." : "Capture 5s Voice Print"}
          </button>
        </div>

        <AnimatePresence>
          {errorMsg && (
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="text-xs text-red-600 font-bold mt-1">
              {errorMsg}
            </motion.p>
          )}
        </AnimatePresence>
      </div>

      {/* Roster Grid */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-black pb-2">
          <h3 className="text-xs font-bold text-black uppercase tracking-widest">System Roster</h3>
          <span className="text-xs font-bold text-black bg-[#D9D9D9] px-2 py-1 rounded-sm border border-black">
            {profiles.length} Active Nodes
          </span>
        </div>

        {profiles.length === 0 ? (
          <div className="py-12 flex flex-col items-center justify-center text-gray-500 border border-black border-dashed bg-gray-50 rounded-md">
            <span className="text-sm font-bold">Registry is currently empty.</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            <AnimatePresence>
              {profiles.map((p, index) => (
                <motion.div
                  key={p}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-center gap-4 p-3 bg-white border border-black rounded-md group relative overflow-hidden"
                >
                  <div className="w-10 h-10 border border-black bg-[#D9D9D9] text-black flex items-center justify-center text-sm font-bold shrink-0">
                    {p[0].toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0 flex flex-col pr-10">
                    <span className="text-sm font-bold text-black truncate">{p}</span>
                    <span className="text-[10px] text-gray-600 tracking-wide uppercase">Model Trained</span>
                  </div>

                  {/* Delete Button (Appears on Hover) */}
                  <button
                    onClick={() => deleteProfile(p)}
                    className="absolute right-3 flex items-center justify-center w-8 h-8 text-[black] hover:text-red-600 transition-colors"
                    title={`Delete ${p}`}
                  >
                    <svg className="w-[18px] h-[18px]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

    </motion.div>
  );
}