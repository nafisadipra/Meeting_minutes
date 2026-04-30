"use client";

import { useState, useEffect } from "react";

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [minutes, setMinutes] = useState<string | null>(null);
  const [attendeesInput, setAttendeesInput] = useState("");

  const startMeeting = async () => {
    setMinutes(null);
    setTranscript([]);
    
    // Clean up the comma-separated string into an array of names
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

  // Polling mechanism to fetch live data from Flask
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://127.0.0.1:5000/api/transcript");
        const data = await res.json();
        setTranscript(data.transcript);
        
        // If recording just stopped and we have a transcript, check for minutes
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
    }, 2000); // Check every 2 seconds

    return () => clearInterval(interval);
  }, [transcript.length]);

  return (
    <main className="min-h-screen p-10 font-sans text-gray-800 bg-gray-50">
      <div className="max-w-5xl mx-auto space-y-8">
        
        <header className="flex items-end justify-between pb-6 border-b border-gray-200">
          <div>
            <h1 className="mb-2 text-3xl font-bold">Stateful Meeting Assistant</h1>
            <div className="flex flex-col space-y-1">
              <label className="text-sm font-semibold text-gray-600">Expected Attendees (Comma-separated)</label>
              <input 
                type="text" 
                placeholder="e.g., Nafis, Muyeed" 
                value={attendeesInput}
                onChange={(e) => setAttendeesInput(e.target.value)}
                disabled={isRecording}
                className="w-80 px-3 py-2 border border-gray-300 rounded-md disabled:bg-gray-100 disabled:text-gray-400"
              />
            </div>
          </div>
          
          <div className="space-x-4">
            {!isRecording ? (
              <button 
                onClick={startMeeting}
                className="px-6 py-3 font-semibold text-white transition bg-blue-600 rounded-lg hover:bg-blue-700"
              >
                Start Recording
              </button>
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
            <h2 className="mb-4 text-xl font-semibold">AI Meeting Minutes</h2>
            <div className="flex-1 p-5 overflow-y-auto rounded whitespace-pre-wrap bg-gray-50 text-gray-800 font-serif leading-relaxed border border-gray-100">
              {minutes ? (
                minutes
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