"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function normalizeMarkdown(md: string): string {
  return md.replace(/([^|\n])\n(\|)/g, "$1\n\n$2");
}

export interface SavedSession {
  id: string;
  name: string;
  date: string;
  timestamp: number;
  transcript: string[];
  minutes: string;
  isPinned: boolean;
  isSaved: boolean;
}

interface EditDocumentProps {
  session: SavedSession;
  onSave: (id: string, newMinutes: string) => void;
  onCancel: () => void;
}

export default function EditDocument({ session, onSave, onCancel }: EditDocumentProps) {
  const [content, setContent] = useState(session.minutes);

  return (
    <motion.div
      key="edit"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.2 }}
      className="flex-1 flex flex-col h-full bg-white print:hidden"
    >
      {/* Edit Header */}
      <div className="flex justify-between items-center p-6 border-b border-[#C4C4C4] bg-[#FAFAFA] shrink-0">
        <div>
          <h2 className="text-black font-bold text-lg tracking-tight">Editing: {session.name}</h2>
          <p className="text-xs text-gray-500 uppercase tracking-widest mt-1">Live Markdown Editor</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="px-6 py-2 text-sm font-bold text-black bg-white border border-[#C4C4C4] rounded-sm hover:bg-gray-100 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onSave(session.id, content)}
            disabled={content === session.minutes}
            className="px-6 py-2 text-sm font-bold bg-black text-white rounded-sm hover:bg-gray-800 disabled:opacity-50 transition-colors"
          >
            Save Changes
          </button>
        </div>
      </div>

      {/* Split Pane Workspace */}
      <div className="flex flex-1 min-h-0">
        
        {/* LEFT: Markdown Input */}
        <div className="flex-1 border-r border-[#C4C4C4] bg-white flex flex-col relative group">
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="flex-1 w-full h-full p-8 resize-none outline-none font-mono text-sm text-black leading-relaxed bg-transparent"
            placeholder="Write your markdown here..."
            spellCheck="false"
          />
          <div className="absolute bottom-4 right-4 text-[10px] font-bold text-gray-400 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            Raw Markdown
          </div>
        </div>

        {/* RIGHT: Live Preview */}
        <div className="flex-1 bg-white overflow-y-auto p-8 prose max-w-none relative group">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalizeMarkdown(content)}</ReactMarkdown>
          <div className="absolute top-4 right-4 text-[10px] font-bold text-gray-400 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            Live Preview
          </div>
        </div>

      </div>
    </motion.div>
  );
}