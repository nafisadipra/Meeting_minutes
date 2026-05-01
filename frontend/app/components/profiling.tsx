"use client";

import { useState, useRef } from "react";

export interface Attendee {
  name: string;
  enrolled: boolean;
}

interface ProfilingProps {
  attendees: Attendee[];
  setAttendees: React.Dispatch<React.SetStateAction<Attendee[]>>;
  isRecording: boolean;
}

export default function Profiling({ attendees, setAttendees, isRecording }: ProfilingProps) {
  const [newAttendeeName, setNewAttendeeName] = useState("");
  const [enrollingFor, setEnrollingFor] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const addAttendee = () => {
    if (!newAttendeeName.trim()) return;
    setAttendees([...attendees, { name: newAttendeeName.trim(), enrolled: false }]);
    setNewAttendeeName("");
  };

  const removeAttendee = (nameToRemove: string) => {
    setAttendees(attendees.filter(a => a.name !== nameToRemove));
  };

  const recordVoicePrint = async (name: string) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      const audioChunks: BlobPart[] = [];
      setEnrollingFor(name);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", audioBlob, `${name}_profile.webm`);
        formData.append("name", name);

        try {
          await fetch("http://127.0.0.1:5000/api/enroll_voice", {
            method: "POST",
            body: formData,
          });

          setAttendees(prev => 
            prev.map(a => a.name === name ? { ...a, enrolled: true } : a)
          );
        } catch (error) {
          console.error("Failed to upload voice profile", error);
          alert("Failed to connect to backend for voice enrollment.");
        }
        
        setEnrollingFor(null);
        stream.getTracks().forEach(track => track.stop()); 
      };

      mediaRecorder.start();
      
      setTimeout(() => {
        if (mediaRecorder.state === "recording") {
          mediaRecorder.stop();
        }
      }, 5000);

    } catch (err) {
      console.error("Microphone access denied", err);
      alert("Please allow microphone permissions to record a voice profile.");
    }
  };

  return (
    <div className="p-5 bg-white border border-gray-200 rounded-xl shadow-sm">
      <h3 className="mb-3 font-semibold text-gray-700">Meeting Attendees & Voice Profiles</h3>
      
      <div className="flex space-x-2 mb-4">
        <input 
          type="text" 
          placeholder="Enter attendee name..." 
          value={newAttendeeName}
          onChange={(e) => setNewAttendeeName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && addAttendee()}
          disabled={isRecording}
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
        />
        <button 
          onClick={addAttendee}
          disabled={isRecording || !newAttendeeName.trim()}
          className="px-4 py-2 text-sm font-semibold text-white bg-gray-800 rounded-md hover:bg-gray-900 disabled:opacity-50"
        >
          Add Person
        </button>
      </div>

      {attendees.length > 0 && (
        <ul className="space-y-2">
          {attendees.map((attendee) => (
            <li key={attendee.name} className="flex items-center justify-between p-3 bg-gray-50 border border-gray-100 rounded-lg">
              <span className="font-medium text-gray-800">{attendee.name}</span>
              <div className="flex space-x-3 items-center">
                {attendee.enrolled ? (
                  <span className="text-sm font-semibold text-green-600">✓ Voice Enrolled</span>
                ) : (
                  <button 
                    onClick={() => recordVoicePrint(attendee.name)}
                    disabled={isRecording || enrollingFor !== null}
                    className={`text-sm px-3 py-1.5 rounded-md font-medium transition ${
                      enrollingFor === attendee.name 
                        ? "bg-red-100 text-red-600 animate-pulse" 
                        : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                    }`}
                  >
                    {enrollingFor === attendee.name ? "Recording (5s)..." : "🎤 Record Voice (5s)"}
                  </button>
                )}
                <button 
                  onClick={() => removeAttendee(attendee.name)}
                  disabled={isRecording}
                  className="text-gray-400 hover:text-red-500"
                >
                  ✕
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}