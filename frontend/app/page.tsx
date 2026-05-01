"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import Profiling, { Attendee } from "./components/profiling"; // Adjust path if you put it in a components folder

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [minutes, setMinutes] = useState<string | null>(null);
  
  // State lifted up from profiling.tsx
  const [attendees, setAttendees] = useState<Attendee[]>([]);

  const startMeeting = async () => {
    setMinutes(null);
    setTranscript([]);
    
    const attendeeNames = attendees.map(a => a.name);

    await fetch("http://127.0.0.1:5000/api/start", { 
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attendees: attendeeNames })
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
        // Silent catch for polling
      }
    }, 2000); 

    return () => clearInterval(interval);
  }, [transcript.length]);

  return (
    <main className="min-h-screen p-10 font-sans text-gray-800 bg-gray-50">
      <div className="max-w-5xl mx-auto space-y-8">
        
        <header className="flex flex-col space-y-6 pb-6 border-b border-gray-200">
          <div className="flex items-end justify-between">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900">AI Meeting Assistant</h1>
            
            <div className="space-x-4">
              {!isRecording ? (
                <>
                  {(transcript.length > 0 || minutes) && (
                    <button onClick={clearMeeting} className="px-6 py-3 font-semibold text-gray-700 transition bg-gray-200 rounded-lg hover:bg-gray-300">
                      Clear Data
                    </button>
                  )}
                  <button onClick={startMeeting} className="px-6 py-3 font-semibold text-white transition bg-blue-600 rounded-lg hover:bg-blue-700">
                    Start Recording
                  </button>
                </>
              ) : (
                <button onClick={stopMeeting} className="px-6 py-3 font-semibold text-white transition bg-red-600 rounded-lg animate-pulse hover:bg-red-700">
                  Stop & Summarize
                </button>
              )}
            </div>
          </div>

          {/* Injecting the new component and passing down state */}
          <Profiling 
            attendees={attendees} 
            setAttendees={setAttendees} 
            isRecording={isRecording} 
          />
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
              {minutes && (
                <button onClick={downloadMinutes} className="flex items-center px-3 py-1.5 text-sm font-semibold text-green-700 bg-green-100 rounded hover:bg-green-200 transition">
                  Download .md
                </button>
              )}
            </div>
            
            <div className="flex-1 p-6 overflow-y-auto bg-gray-50 border border-gray-100 rounded-lg">
              {minutes ? (
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