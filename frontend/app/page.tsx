"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [minutes, setMinutes] = useState<string | null>(null);
  const [attendeesInput, setAttendeesInput] = useState("");

  const startMeeting = async () => {
    setMinutes(null);
    setTranscript([]);
    
    const attendees = attendeesInput
      .split(",")
      .map(name => name.trim())
      .filter(name => name.length > 0);

    await fetch("http://127.0.0.1:5000/api/start", { 
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attendees })
    });
    
    setIsRecording(true);
  };

  const stopMeeting = async () => {
    await fetch("http://127.0.0.1:5000/api/stop", { method: "POST" });
    setIsRecording(false);
  };

  const clearMeeting = async () => {
    await fetch("http://127.0.0.1:5000/api/clear", { method: "POST" });
    setTranscript([]);
    setMinutes(null);
  };

  // --- NEW: Download Function ---
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

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://127.0.0.1:5000/api/transcript");
        const data = await res.json();
        setTranscript(data.transcript);
        
        if (!data.is_recording && transcript.length > 0) {
          const minutesRes = await fetch("http://127.0.0.1:5000/api/minutes");
          const minutesData = await minutesRes.json();
          if (minutesData.minutes) {
            setMinutes(minutesData.minutes);
          }
        }
      } catch (error) {
        console.error("Make sure your Flask server is running!");
      }
    }, 2000); 

    return () => clearInterval(interval);
  }, [transcript.length]);

  return (
    <main className="min-h-screen p-10 font-sans text-gray-800 bg-gray-50">
      <div className="max-w-5xl mx-auto space-y-8">
        
        <header className="flex items-end justify-between pb-6 border-b border-gray-200">
          <div>
            <h1 className="mb-2 text-3xl font-bold tracking-tight text-gray-900">AI Meeting Assistant</h1>
            <div className="flex flex-col space-y-1">
              <label className="text-sm font-semibold text-gray-600">Expected Attendees (Comma-separated)</label>
              <input 
                type="text" 
                placeholder="e.g., Nafis, Muyeed" 
                value={attendeesInput}
                onChange={(e) => setAttendeesInput(e.target.value)}
                disabled={isRecording}
                className="w-80 px-3 py-2 border border-gray-300 rounded-md disabled:bg-gray-100 disabled:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          
          <div className="space-x-4">
            {!isRecording ? (
              <>
                {(transcript.length > 0 || minutes) && (
                  <button 
                    onClick={clearMeeting}
                    className="px-6 py-3 font-semibold text-gray-700 transition bg-gray-200 rounded-lg hover:bg-gray-300"
                  >
                    Clear Data
                  </button>
                )}
                <button 
                  onClick={startMeeting}
                  className="px-6 py-3 font-semibold text-white transition bg-blue-600 rounded-lg hover:bg-blue-700"
                >
                  Start Recording
                </button>
              </>
            ) : (
              <button 
                onClick={stopMeeting}
                className="px-6 py-3 font-semibold text-white transition bg-red-600 rounded-lg animate-pulse hover:bg-red-700"
              >
                Stop & Summarize
              </button>
            )}
          </div>
        </header>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          
          {/* Live Transcript Panel */}
          <div className="flex flex-col p-6 bg-white border border-gray-200 shadow-sm rounded-xl h-[600px]">
            <h2 className="mb-4 text-xl font-semibold">Live Transcript</h2>
            <div className="flex-1 p-3 space-y-3 overflow-y-auto rounded bg-gray-50">
              {transcript.length === 0 && <p className="italic text-gray-400">Waiting for speech...</p>}
              {transcript.map((line, idx) => (
                <div key={idx} className="p-3 bg-white border border-gray-100 rounded shadow-sm">
                  <span className="font-bold text-blue-600">{line.split("]: ")[0]}]</span>
                  <span className="text-gray-700">: {line.split("]: ")[1]}</span>
                </div>
              ))}
            </div>
          </div>

          {/* AI Minutes Panel */}
          <div className="flex flex-col p-6 bg-white border border-gray-200 shadow-sm rounded-xl h-[600px]">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">AI Meeting Minutes</h2>
              {/* --- NEW: Download Button (Only shows when minutes are ready) --- */}
              {minutes && (
                <button 
                  onClick={downloadMinutes}
                  className="flex items-center px-3 py-1.5 text-sm font-semibold text-green-700 bg-green-100 rounded hover:bg-green-200 transition"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                  Download .md
                </button>
              )}
            </div>
            
            <div className="flex-1 p-6 overflow-y-auto bg-gray-50 border border-gray-100 rounded-lg">
              {minutes ? (
                // --- NEW: React Markdown Styling ---
                <div className="text-gray-800">
                  <ReactMarkdown
                    components={{
                      h1: ({ node: _, ...props }) => <h1 className="mb-6 text-2xl font-bold text-gray-900 border-b pb-2" {...props} />,
                      h2: ({ node: _, ...props }) => <h2 className="mt-8 mb-4 text-xl font-bold text-gray-800" {...props} />,
                      h3: ({ node: _, ...props }) => <h3 className="mt-6 mb-2 text-lg font-semibold text-gray-800" {...props} />,
                      p: ({ node: _, ...props }) => <p className="mb-4 leading-relaxed" {...props} />,
                      ul: ({ node: _, ...props }) => <ul className="pl-6 mb-4 space-y-2 list-disc" {...props} />,
                      li: ({ node: _, ...props }) => <li className="leading-relaxed" {...props} />,
                      strong: ({ node: _, ...props }) => <strong className="font-semibold text-gray-900" {...props} />,
                    }}
                  >
                    {minutes}
                  </ReactMarkdown>
                </div>
              ) : !isRecording && transcript.length > 0 ? (
                <div className="flex items-center space-x-2 text-orange-600 animate-pulse">
                  <svg className="w-5 h-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                  <span className="font-semibold">Generating document via LM Studio...</span>
                </div>
              ) : (
                <p className="italic text-gray-400">Professional minutes will appear here after the meeting concludes.</p>
              )}
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}