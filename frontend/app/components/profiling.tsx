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
        // Non-critical — roster will be empty until next load.
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
      setErrorMsg("Microphone access denied.");
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

  const isRecording = status === "recording";
  const isUploading = status === "uploading";
  const isBusy = isRecording || isUploading;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="w-full h-full space-y-6"
    >
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 bg-white border border-gray-200 rounded-xl flex items-center justify-center text-xl shadow-sm">
          🎙️
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-black">Voice Identity Registration</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* Enrollment Tool */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-200 h-fit">
          <h3 className="text-sm font-bold text-[#facc15] uppercase tracking-widest mb-4 border-b border-gray-200 pb-2">Capture Voice Print</h3>
          <p className="text-xs text-black/60 font-medium mb-6">
            Enter an attendee&apos;s name and record a 5-second audio sample to train the local recognition engine.
          </p>

          <div className="space-y-4">
            <input
              className="w-full px-4 py-3 text-sm bg-gray-100 border-none rounded-xl focus:ring-2 focus:ring-[#facc15] transition-all outline-none text-black placeholder:text-gray-400 font-medium"
              placeholder="e.g. John Doe"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isBusy}
            />

            {errorMsg && (
              <p className="text-xs font-semibold text-red-500 px-1">{errorMsg}</p>
            )}

            <button
              onClick={recordVoice}
              disabled={isBusy || !name.trim()}
              className={`w-full py-3 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 ${
                isBusy
                  ? "bg-black text-gray-100 shadow-md"
                  : "bg-[#facc15] text-black hover:bg-[#eab308] disabled:bg-gray-200 disabled:text-gray-400 shadow-sm disabled:shadow-none"
              }`}
            >
              {isRecording ? (
                <>
                  <motion.span
                    animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                    className="w-2.5 h-2.5 bg-gray-100 rounded-full"
                  />
                  Recording...
                </>
              ) : isUploading ? (
                <>
                  <motion.span animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }} className="inline-block w-4 h-4 border-2 border-gray-300 border-t-white rounded-full" />
                  Saving Profile...
                </>
              ) : (
                "Start 5s Recording"
              )}
            </button>
          </div>
        </div>

        {/* Enrolled Roster */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-200">
          <h3 className="text-sm font-bold text-[#facc15] uppercase tracking-widest mb-4 border-b border-gray-200 pb-2">Enrolled Roster</h3>

          <div className="flex flex-col gap-3">
            <AnimatePresence>
              {profiles.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center py-10 text-center space-y-2"
                >
                  <span className="text-3xl opacity-40">📭</span>
                  <p className="text-gray-500 italic text-xs font-medium">No voice profiles registered.</p>
                </motion.div>
              ) : (
                profiles.map((p, index) => (
                  <motion.div
                    key={p}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center gap-3 p-3 bg-gray-100/50 rounded-xl border border-gray-200"
                  >
                    <div className="w-10 h-10 bg-white text-[#facc15] rounded-lg border border-gray-200 flex items-center justify-center font-bold text-base shadow-sm">
                      {p[0].toUpperCase()}
                    </div>
                    <div className="flex-1 flex flex-col">
                      <span className="text-sm font-bold text-black">{p}</span>
                      <span className="text-[10px] font-medium text-[#facc15]">Voice matched & locked</span>
                    </div>
                    <span className="text-[9px] font-bold text-white bg-gray-500 px-2.5 py-1 rounded-full uppercase tracking-wider shadow-sm">
                      Ready
                    </span>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </div>

      </div>
    </motion.div>
  );
}
