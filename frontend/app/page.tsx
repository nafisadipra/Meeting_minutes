"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";
import Profiling from "./components/profiling";
import DocumentVault, { SavedSession } from "./components/document";
import EditDocument from "./components/edit"; 

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000";

export default function Home() {
  const [view, setView] = useState<"meeting" | "profiling" | "documents" | "edit">("meeting");

  const [sessionToEdit, setSessionToEdit] = useState<SavedSession | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [minutes, setMinutes] = useState<string | null>(null);
  const [attendeesInput, setAttendeesInput] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  
  const [isHistoryOpen, setIsHistoryOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isExportOpen, setIsExportOpen] = useState(false);

  const [history, setHistory] = useState<SavedSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  const [discoveredProfiles, setDiscoveredProfiles] = useState<Record<string, number[]>>({});
  const [selectedToSave, setSelectedToSave] = useState<Set<string>>(new Set());

  const minutesRef = useRef<string | null>(null);
  minutesRef.current = minutes;

  useEffect(() => {
    const saved = localStorage.getItem("grammatica_history");
    if (saved) {
      setHistory(JSON.parse(saved));
    }
  }, []);

  const saveToHistory = (newHistory: SavedSession[]) => {
    setHistory(newHistory);
    localStorage.setItem("grammatica_history", JSON.stringify(newHistory));
  };

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/transcript`);
      if (!res.ok) return;
      const data = await res.json();
      setIsRecording(data.is_recording);
      setTranscript(data.transcript ?? []);

      if (!data.is_recording && data.transcript?.length > 0 && !minutesRef.current) {
        const mRes = await fetch(`${API_BASE}/api/minutes`);
        if (mRes.ok) {
          const mData = await mRes.json();
          if (mData.minutes) {
            setMinutes(mData.minutes);
            
            const newSession: SavedSession = {
              id: Date.now().toString(),
              name: `Session - ${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`,
              date: new Date().toLocaleDateString(),
              timestamp: Date.now(),
              transcript: data.transcript,
              minutes: mData.minutes,
              isPinned: false,
              isSaved: false
            };
            
            setCurrentSessionId(newSession.id);
            setHistory(prev => {
              const updated = [newSession, ...prev];
              localStorage.setItem("grammatica_history", JSON.stringify(updated));
              return updated;
            });
          }
        }
      }
    } catch {
      // Backend unreachable
    }
  }, []);

  useEffect(() => {
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [poll]);

  const startMeeting = async () => {
    setIsStarting(true);
    setMinutes(null);
    setTranscript([]);
    setDiscoveredProfiles({});
    setCurrentSessionId(null);
    try {
      const attendees = attendeesInput.split(",").map((n) => n.trim()).filter(Boolean);
      const res = await fetch(`${API_BASE}/api/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ attendees }),
      });
      if (res.ok) setIsRecording(true);
    } catch {
      // Handled by polling
    } finally {
      setIsStarting(false);
    }
  };

  const stopMeeting = async () => {
    setIsStopping(true);
    try {
      const res = await fetch(`${API_BASE}/api/stop`, { method: "POST" });
      if (!res.ok) return;
      const data = await res.json();

      if (data.new_profiles_discovered && Object.keys(data.new_profiles_discovered).length > 0) {
        setDiscoveredProfiles(data.new_profiles_discovered);
        setSelectedToSave(new Set(Object.keys(data.new_profiles_discovered)));
      }
      setIsRecording(false);
    } catch {
      setIsRecording(false);
    } finally {
      setIsStopping(false);
    }
  };

  const saveSelectedProfiles = async () => {
    const profilesToSave: Record<string, number[]> = {};
    selectedToSave.forEach((name) => {
      profilesToSave[name] = discoveredProfiles[name];
    });
    try {
      await fetch(`${API_BASE}/api/save_discovered_profiles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profiles: profilesToSave }),
      });
    } finally {
      setDiscoveredProfiles({});
    }
  };

  const clearMeeting = async () => {
    await fetch(`${API_BASE}/api/clear`, { method: "POST" });
    setTranscript([]);
    setMinutes(null);
    setDiscoveredProfiles({});
    setCurrentSessionId(null);
  };

  const generateDocx = (content: string, fileName: string) => {
    if (!content) return;
    const header = "<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'><head><meta charset='utf-8'><title>Meeting Minutes</title></head><body>";
    const footer = "</body></html>";
    const htmlContent = content.replace(/\n/g, '<br>').replace(/## (.*)/g, '<h2>$1</h2>').replace(/# (.*)/g, '<h1>$1</h1>');
    const sourceHTML = header + htmlContent + footer;
    
    const blob = new Blob(['\ufeff', sourceHTML], { type: 'application/msword' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName.replace(/[^a-z0-9]/gi, '_')}.doc`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadDocx = () => {
    if (minutes) generateDocx(minutes, "Meeting_Minutes");
    setIsExportOpen(false);
  };

  const downloadPdf = () => {
    window.print();
    setIsExportOpen(false);
  };

  const handleMenuAction = (option: string) => {
    setIsSettingsOpen(false);
    
    if (option === "Edit") {
      // NEW: Trigger the edit view with the current session data
      if (currentSessionId) {
        const session = history.find(s => s.id === currentSessionId);
        if (session) {
          setSessionToEdit(session);
          setView("edit");
        }
      }
    } else if (option === "Delete Session") {
      clearMeeting();
      if (currentSessionId) {
        const updatedHistory = history.filter(s => s.id !== currentSessionId);
        saveToHistory(updatedHistory);
      }
    } else if (option === "Pin") {
      if (currentSessionId) {
        const updatedHistory = history.map(s => 
          s.id === currentSessionId ? { ...s, isPinned: !s.isPinned } : s
        );
        saveToHistory(updatedHistory);
      }
    } else if (option === "Save") {
      if (currentSessionId) {
        const updatedHistory = history.map(s => 
          s.id === currentSessionId ? { ...s, isSaved: !s.isSaved } : s
        );
        saveToHistory(updatedHistory);
      }
    } else if (option === "Rename") {
      console.log("Rename clicked");
    }
  };

  const handleSaveEdit = (id: string, newMinutes: string) => {
    const updatedHistory = history.map(s => 
      s.id === id ? { ...s, minutes: newMinutes } : s
    );
    saveToHistory(updatedHistory);
    
    // Update the live viewing state so it reflects immediately
    if (currentSessionId === id) {
      setMinutes(newMinutes);
    }
    
    setSessionToEdit(null);
    setView("meeting"); // Return to the document view
  };

  const loadPastSession = (session: SavedSession) => {
    setTranscript(session.transcript);
    setMinutes(session.minutes);
    setCurrentSessionId(session.id);
  };

  const toggleSelection = (name: string) => {
    setSelectedToSave((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };

  const isBusy = isStarting || isStopping;

  const sortedHistory = [...history].sort((a, b) => {
    if (a.isPinned === b.isPinned) return b.timestamp - a.timestamp;
    return a.isPinned ? -1 : 1;
  });

  const savedDocuments = history.filter(s => s.isSaved).sort((a, b) => b.timestamp - a.timestamp);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="flex flex-col h-screen bg-[#FAFAFA] text-black font-sans p-10 pt-6 pb-6 overflow-hidden"
    >

      {/* DISCOVERY MODAL */}
      <AnimatePresence>
        {Object.keys(discoveredProfiles).length > 0 && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/10 backdrop-blur-sm">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white w-[440px] p-8 rounded-xl border border-[#C4C4C4] shadow-lg"
            >
              <h2 className="text-xl font-bold text-black mb-2">Unrecognized Voices Detected</h2>
              <p className="text-sm text-gray-700 mb-6 leading-relaxed">
                We identified new speakers during your session. Select the profiles to register to the local engine.
              </p>

              <div className="space-y-2 mb-8 max-h-[200px] overflow-y-auto pr-2">
                {Object.keys(discoveredProfiles).map((name) => (
                  <label key={name} className="flex items-center gap-4 p-3 bg-white border border-[#C4C4C4] rounded-md cursor-pointer hover:bg-gray-50 transition-colors">
                    <input
                      type="checkbox"
                      checked={selectedToSave.has(name)}
                      onChange={() => toggleSelection(name)}
                      className="w-4 h-4 accent-black rounded-sm border-black"
                    />
                    <span className="font-medium text-sm text-black">{name}</span>
                  </label>
                ))}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => setDiscoveredProfiles({})}
                  className="flex-1 py-3 rounded-md text-sm font-bold text-black bg-white border border-[#C4C4C4] hover:bg-gray-100 transition-colors"
                >
                  Discard All
                </button>
                <button
                  onClick={saveSelectedProfiles}
                  disabled={selectedToSave.size === 0}
                  className="flex-1 py-3 rounded-md text-sm font-bold text-black bg-[#D9D9D9] border border-[#C4C4C4] hover:bg-gray-300 disabled:opacity-50 transition-colors"
                >
                  Save Selected
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* TOP HEADER ROW */}
      <header className="flex justify-between items-start mb-6 shrink-0 print:hidden">
        
        <div className="flex items-center gap-4 ml-[10px]">
          <div className="w-[60px] h-[60px] flex items-center justify-center shrink-0">
            <img 
              src="/img/grammatica.svg" 
              alt="Grammatica Icon" 
              className="w-full h-full object-contain" 
            />
          </div>
          <div className="h-[27.2px] flex items-center shrink-0">
            <img 
              src="/img/grammatica-text.png" 
              alt="Grammatica Text" 
              className="h-full w-auto object-contain" 
            />
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3 bg-[#D9D9D9] rounded-md px-4 py-2 min-h-[40px]">
            {view === "meeting" ? (
              <>
                <input
                  className="w-64 px-3 py-1 text-sm bg-white focus:outline-none text-black placeholder:text-gray-500 rounded-sm"
                  placeholder="Expected Attendees (e.g. Nafis)"
                  value={attendeesInput}
                  onChange={(e) => setAttendeesInput(e.target.value)}
                  disabled={isRecording || isBusy}
                />
                
                {!isRecording && !isBusy && (transcript.length > 0 || minutes) && (
                  <button
                    onClick={clearMeeting}
                    className="px-4 py-1 text-sm font-bold text-black bg-white border border-[#C4C4C4] rounded-sm hover:bg-gray-100 transition-colors"
                  >
                    Clear
                  </button>
                )}

                <button
                  onClick={isRecording ? stopMeeting : startMeeting}
                  disabled={isBusy}
                  className={`px-6 py-1 text-sm font-bold border-2 border-[#C4C4C4] rounded-sm disabled:opacity-50 transition-colors ${
                    isRecording ? "bg-black text-white hover:bg-gray-800" : "bg-white text-black hover:bg-gray-100"
                  }`}
                >
                  {isBusy ? "Processing..." : isRecording ? "Stop & Summarize" : "Start Session"}
                </button>
              </>
            ) : view === "profiling" ? (
              <span className="px-4 py-1 text-sm font-bold text-black">Voice Profiling Engine Active</span>
            ) : (
              <span className="px-4 py-1 text-sm font-bold text-black">Document Vault</span>
            )}
          </div>

          {/* Dotted Square Button */}
          <div className="relative">
            <button
              onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              className={`w-10 h-10 bg-[#D9D9D9] border border-[#C4C4C4] rounded-md flex flex-col items-center justify-center gap-1 transition-colors ${
                isSettingsOpen ? "bg-gray-300" : "hover:bg-gray-400"
              }`}
            >
              <div className="w-1 h-1 bg-black rounded-full" />
              <div className="w-1 h-1 bg-[#5E5E5E] rounded-full" />
              <div className="w-1 h-1 bg-white rounded-full" />
            </button>

            <AnimatePresence>
              {isSettingsOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  transition={{ duration: 0.15 }}
                  className="absolute right-0 mt-2 w-48 bg-white border border-[#C4C4C4] rounded-md shadow-lg z-50 flex flex-col p-1"
                >
                  {["Edit", "Pin", "Rename", "Save", "Delete Session"].map((option) => {
                    const isCurrentlyPinned = currentSessionId && history.find(s => s.id === currentSessionId)?.isPinned;
                    const isCurrentlySaved = currentSessionId && history.find(s => s.id === currentSessionId)?.isSaved;
                    
                    const displayOption = option === "Pin" && isCurrentlyPinned ? "Unpin" : 
                                          option === "Save" && isCurrentlySaved ? "Remove from Vault" : option;
                    
                    return (
                      <button
                        key={option}
                        className={`w-full text-left px-3 py-2 text-sm font-bold rounded-sm transition-colors ${
                          option === "Delete Session" 
                            ? "text-red-600 hover:bg-red-50" 
                            : "text-black hover:bg-gray-100"
                        }`}
                        onClick={() => handleMenuAction(option)}
                        disabled={!currentSessionId && (option === "Pin" || option === "Delete Session" || option === "Rename" || option === "Save")}
                      >
                        {displayOption}
                      </button>
                    )
                  })}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </header>

      {/* BOTTOM WORKSPACE */}
      <div className="flex flex-1 gap-16 min-h-0">
        
        {/* Left Sidebar Complex */}
        <div className="flex shrink-0 print:hidden">
          
          <nav className="w-[80px] flex flex-col items-center py-6 shrink-0 gap-6 bg-white z-20 relative border border-[#C4C4C4] rounded-md shadow-sm">
            
            <button
              onClick={() => setIsHistoryOpen(!isHistoryOpen)}
              className={`w-10 h-10 flex flex-col items-center justify-center rounded-md transition-all bg-white ${
                isHistoryOpen 
                  ? "border-2 border-[#C4C4C4] shadow-[inset_2px_2px_4px_rgba(0,0,0,0.1)]" 
                  : "border border-gray-200 shadow-sm hover:shadow-md hover:border-gray-300"
              }`}
              title="Toggle History"
            >
              <svg className="w-5 h-5 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>

            {/* Meeting Hub Icon */}
            <button
              onClick={() => setView("meeting")}
              className={`relative w-10 h-10 flex flex-col items-center justify-center rounded-md transition-all bg-white ${
                view === "meeting" 
                  ? "" 
                  : "border border-gray-200 shadow-sm hover:shadow-md hover:border-gray-300"
              }`}
              title="Meeting Hub"
            >
              {view === "meeting" && (
                <motion.div
                  layoutId="nav-indicator"
                  className="absolute inset-0 border-2 border-[#C4C4C4] rounded-md shadow-[inset_2px_2px_4px_rgba(0,0,0,0.1)] pointer-events-none"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <svg className="w-5 h-5 text-black relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </button>

            {/* Document Vault Icon */}
            <button
              onClick={() => setView("documents")}
              className={`relative w-10 h-10 flex flex-col items-center justify-center rounded-md transition-all bg-white ${
                view === "documents" 
                  ? "" 
                  : "border border-gray-200 shadow-sm hover:shadow-md hover:border-gray-300"
              }`}
              title="Document Vault"
            >
              {view === "documents" && (
                <motion.div
                  layoutId="nav-indicator"
                  className="absolute inset-0 border-2 border-[#C4C4C4] rounded-md shadow-[inset_2px_2px_4px_rgba(0,0,0,0.1)] pointer-events-none"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <svg className="w-5 h-5 text-black relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </button>

            {/* Voice Profiling Icon */}
            <button
              onClick={() => setView("profiling")}
              className={`relative w-10 h-10 flex flex-col items-center justify-center rounded-md transition-all bg-white ${
                view === "profiling" 
                  ? "" 
                  : "border border-gray-200 shadow-sm hover:shadow-md hover:border-gray-300"
              }`}
              title="Voice Profiling"
            >
              {view === "profiling" && (
                <motion.div
                  layoutId="nav-indicator"
                  className="absolute inset-0 border-2 border-[#C4C4C4] rounded-md shadow-[inset_2px_2px_4px_rgba(0,0,0,0.1)] pointer-events-none"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <svg className="w-5 h-5 text-black relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </button>

          </nav>
          
          <AnimatePresence initial={false}>
            {isHistoryOpen && (
              <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 200, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="flex flex-col bg-white border-y-[0.10px] border-r border-[#C4C4C4] rounded-r-md overflow-hidden z-10 relative origin-left shadow-sm"
              >
                <div className="w-[200px] h-full flex flex-col shrink-0">
                  <div className="p-4 shrink-0">
                    <input
                      type="text"
                      placeholder="Search history..."
                      className="w-full bg-white border border-[#C4C4C4] rounded-md px-4 py-2 text-sm text-black placeholder:text-gray-500 outline-none focus:border-gray-400 transition-colors"
                    />
                  </div>
                  <div className="flex-1 overflow-y-auto px-4 pb-4">
                    <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4">Recent Sessions</h3>
                    
                    {sortedHistory.length === 0 ? (
                      <p className="text-xs text-gray-400 italic">No recent sessions.</p>
                    ) : (
                      sortedHistory.map((session) => (
                        <button 
                          key={session.id}
                          onClick={() => { loadPastSession(session); setView("meeting"); }}
                          className={`w-full text-left p-3 mb-2 rounded-md flex flex-col gap-1 transition-colors outline-none border ${
                            currentSessionId === session.id ? "bg-gray-50 border-[#C4C4C4] shadow-sm" : "bg-white border-transparent hover:border-gray-200"
                          }`}
                        >
                          <div className="flex justify-between items-center w-full">
                            <span className="text-sm font-bold text-black truncate pr-2">{session.name}</span>
                            <div className="flex gap-1 shrink-0">
                              {session.isSaved && (
                                <svg className="w-3 h-3 text-[#5E5E5E]" fill="currentColor" viewBox="0 0 20 20">
                                  <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                                  <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                                </svg>
                              )}
                              {session.isPinned && (
                                <svg className="w-3 h-3 text-black" fill="currentColor" viewBox="0 0 20 20">
                                  <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
                                </svg>
                              )}
                            </div>
                          </div>
                          <span className="text-xs text-gray-500 font-medium truncate">
                            {session.date} • {new Date(session.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                          </span>
                        </button>
                      ))
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Main Content Area */}
        <main className="flex-1 border border-[#C4C4C4] rounded-md bg-white overflow-hidden flex flex-col relative shadow-sm transition-all duration-300">
          <AnimatePresence mode="wait">

            {/* VIEW: DOCUMENTS VAULT */}
            {/* VIEW: EDIT DOCUMENT */}
            {view === "edit" && sessionToEdit ? (
              <EditDocument
                key="edit"
                session={sessionToEdit}
                onSave={handleSaveEdit}
                onCancel={() => {
                  setSessionToEdit(null);
                  setView("meeting");
                }}
              />
            ) :
            
            view === "documents" ? (
              <DocumentVault 
                key="documents" 
                savedDocuments={savedDocuments} 
                onViewSession={(session) => {
                  loadPastSession(session);
                  setView("meeting");
                }}
                onDownloadDocx={generateDocx}
              />
            ) : 

            /* VIEW: PROFILING */
            view === "profiling" ? (
              <motion.div 
                key="profiling" 
                initial={{ opacity: 0, y: 10 }} 
                animate={{ opacity: 1, y: 0 }} 
                exit={{ opacity: 0, y: -10 }} 
                transition={{ duration: 0.3 }}
                className="flex-1 p-10 overflow-y-auto print:hidden"
              >
                <Profiling onProfileComplete={() => {}} />
              </motion.div>
            ) : 
            
            /* VIEW: MEETING HUB */
            (
              <motion.div 
                key="meeting" 
                initial={{ opacity: 0, y: 10 }} 
                animate={{ opacity: 1, y: 0 }} 
                exit={{ opacity: 0, y: -10 }} 
                transition={{ duration: 0.3 }}
                className="flex-1 flex h-full print:block"
              >
                {/* LEFT PANE: Transcript */}
                <div className="w-1/3 border-r border-[#C4C4C4] flex flex-col overflow-hidden print:hidden bg-[#FAFAFA]">
                  <div className="p-6 pb-2 flex justify-between items-center z-10">
                    <h2 className="text-black font-bold uppercase text-xs tracking-widest">Live Transcript</h2>
                    {isRecording && <span className="text-xs font-bold text-black animate-pulse">● REC</span>}
                  </div>
                  
                  <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {transcript.length === 0 && (
                      <p className="text-gray-500 text-sm italic">
                        Waiting for audio stream...
                      </p>
                    )}
                    <AnimatePresence initial={false}>
                      {transcript.map((line, i) => {
                        const [speaker, ...rest] = line.replace(/^\[/, "").split("]:");
                        return (
                          <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="group">
                            <span className="text-xs font-bold text-black uppercase block mb-1">
                              {speaker.trim()}
                            </span>
                            <p className="text-black text-sm leading-relaxed">{rest.join("]:").trim()}</p>
                          </motion.div>
                        );
                      })}
                    </AnimatePresence>
                  </div>
                </div>

                {/* RIGHT PANE: Minutes */}
                <div className="flex-1 flex flex-col overflow-hidden bg-white print:overflow-visible">
                  <div className="p-6 pb-2 flex justify-between items-center z-10 print:hidden">
                    <h2 className="text-black font-bold uppercase text-xs tracking-widest">Synthesized Output</h2>
                    
                    {!isRecording && minutes && (
                      <div className="relative">
                        <button
                          onClick={() => setIsExportOpen(!isExportOpen)}
                          className="text-xs font-bold text-black bg-[#D9D9D9] border border-[#C4C4C4] hover:bg-gray-300 px-4 py-1.5 rounded-sm transition-colors flex items-center gap-2 shadow-sm"
                        >
                          Export
                          <svg className={`w-3 h-3 transition-transform ${isExportOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                          </svg>
                        </button>
                        
                        <AnimatePresence>
                          {isExportOpen && (
                            <motion.div
                              initial={{ opacity: 0, y: -5 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -5 }}
                              className="absolute right-0 mt-1 w-28 bg-white border border-[#C4C4C4] rounded-sm shadow-lg z-50 flex flex-col"
                            >
                              <button 
                                onClick={downloadDocx}
                                className="text-left px-3 py-2 text-xs font-bold text-black hover:bg-gray-100 transition-colors border-b border-gray-100"
                              >
                                .DOCX
                              </button>
                              <button 
                                onClick={downloadPdf}
                                className="text-left px-3 py-2 text-xs font-bold text-black hover:bg-gray-100 transition-colors"
                              >
                                .PDF
                              </button>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}
                  </div>

                  {/* Print Header */}
                  <div className="hidden print:block p-8 pb-4 border-b border-black mb-4">
                    <h1 className="text-2xl font-bold uppercase tracking-widest mb-2">Meeting Minutes</h1>
                    <p className="text-sm text-gray-500">Generated by Grammatica AI</p>
                  </div>

                  <div className="flex-1 overflow-y-auto p-6 prose max-w-none print:overflow-visible print:p-8">
                    {minutes ? (
                      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <ReactMarkdown>{minutes}</ReactMarkdown>
                      </motion.div>
                    ) : !isRecording && transcript.length > 0 ? (
                      <div className="h-full flex flex-col items-center justify-center opacity-50">
                         <span className="text-sm font-bold">Synthesizing...</span>
                      </div>
                    ) : (
                      <div className="h-full flex flex-col items-center justify-center opacity-20 print:hidden">
                         <span className="text-sm font-bold">No output generated yet</span>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </motion.div>
  );
}