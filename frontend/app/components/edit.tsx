"use client";

import { useState, useRef } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

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
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Custom Undo/Redo Stack
  const [history, setHistory] = useState<string[]>([session.minutes]);
  const [historyIndex, setHistoryIndex] = useState(0);

  // Update content and push to history stack
  const updateContentWithHistory = (newText: string) => {
    setContent(newText);
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newText);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "z") {
      e.preventDefault();
      if (e.shiftKey) {
        if (historyIndex < history.length - 1) {
          setHistoryIndex(historyIndex + 1);
          setContent(history[historyIndex + 1]);
        }
      } else {
        if (historyIndex > 0) {
          setHistoryIndex(historyIndex - 1);
          setContent(history[historyIndex - 1]);
        }
      }
    }
  };

  const [activeFormats, setActiveFormats] = useState<Record<string, boolean>>({});
  const [currentFontSize, setCurrentFontSize] = useState("16");

  const checkActiveFormats = () => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const before = content.substring(0, start);
    const after = content.substring(end);

    setActiveFormats({
      bold: before.endsWith("**") && after.startsWith("**"),
      italic: before.endsWith("*") && !before.endsWith("**") && after.startsWith("*") && !after.startsWith("**"),
      highlight: before.endsWith("<mark>") && after.startsWith("</mark>"),
    });
  };

  const applyFormatting = (prefix: string, suffix: string = "", formatKey: string) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = content.substring(start, end);
    const before = content.substring(0, start);
    const after = content.substring(end);
    
    const isWrapped = before.endsWith(prefix) && (suffix === "" || after.startsWith(suffix));

    let newText = "";
    let newCursorStart = start;
    let newCursorEnd = end;

    if (isWrapped) {
      newText = before.slice(0, -prefix.length) + selectedText + after.slice(suffix.length);
      newCursorStart = start - prefix.length;
      newCursorEnd = end - prefix.length;
    } else {
      newText = `${before}${prefix}${selectedText}${suffix}${after}`;
      newCursorStart = start + prefix.length;
      newCursorEnd = end + prefix.length;
    }

    updateContentWithHistory(newText);
    
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(newCursorStart, newCursorEnd);
      checkActiveFormats();
    }, 0);
  };

  // Helper for injecting specific Font Sizes via HTML
  const applyCustomFontSize = (size: string) => {
    setCurrentFontSize(size);
    applyFormatting(`<span style="font-size: ${size}px;">`, "</span>", "fontsize");
  };

  // Helper for injecting a Markdown Table
  const insertTable = () => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const before = content.substring(0, start);
    const after = content.substring(start);

    const tableTemplate = `\n\n| Column 1 | Column 2 | Column 3 |\n| :--- | :--- | :--- |\n| Data | Data | Data |\n| Data | Data | Data |\n\n`;
    
    const newText = `${before}${tableTemplate}${after}`;
    updateContentWithHistory(newText);

    setTimeout(() => {
      textarea.focus();
      // Put cursor inside the first data cell
      textarea.setSelectionRange(start + 73, start + 77); 
    }, 0);
  };

  return (
    <motion.div
      key="edit"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.2 }}
      className="flex-1 flex flex-col h-full bg-white print:hidden"
    >
      <div className="flex justify-between items-center p-6 border-b border-[#C4C4C4] bg-[#FAFAFA] shrink-0">
        <div>
          <h2 className="text-black font-bold text-lg tracking-tight">Editing: {session.name}</h2>
          <p className="text-xs text-gray-500 uppercase tracking-widest mt-1">Live Markdown Editor</p>
        </div>
        <div className="flex gap-3">
          <button onClick={onCancel} className="px-6 py-2 text-sm font-bold text-black bg-white border border-[#C4C4C4] rounded-sm hover:bg-gray-100 transition-colors">
            Cancel
          </button>
          <button onClick={() => onSave(session.id, content)} disabled={content === session.minutes} className="px-6 py-2 text-sm font-bold bg-black text-white rounded-sm hover:bg-gray-800 disabled:opacity-50 transition-colors">
            Save Changes
          </button>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* LEFT: Markdown Input */}
        <div className="flex-1 border-r border-[#C4C4C4] bg-white flex flex-col relative group">
          
          {/* Formatting Toolbar */}
          <div className="flex gap-2 p-3 border-b border-[#C4C4C4] bg-[#FAFAFA] shrink-0 items-center overflow-x-auto">
            <button onClick={() => applyFormatting("**", "**", "bold")} className={`w-8 h-8 shrink-0 flex items-center justify-center text-sm font-bold border border-[#C4C4C4] rounded-sm transition-colors ${activeFormats.bold ? "bg-[#D9D9D9] text-black" : "bg-white text-black hover:bg-gray-100"}`} title="Bold">B</button>
            <button onClick={() => applyFormatting("*", "*", "italic")} className={`w-8 h-8 shrink-0 flex items-center justify-center text-sm italic font-serif border border-[#C4C4C4] rounded-sm transition-colors ${activeFormats.italic ? "bg-[#D9D9D9] text-black" : "bg-white text-black hover:bg-gray-100"}`} title="Italic">I</button>
            
            <div className="w-[1px] h-6 bg-[#C4C4C4] mx-1 shrink-0" />
            
            {/* Custom Font Size Selector */}
            <div className="flex items-center bg-white border border-[#C4C4C4] rounded-sm overflow-hidden shrink-0">
              <span className="px-2 text-xs font-bold text-gray-500 border-r border-[#C4C4C4]">Size</span>
              <select 
                value={currentFontSize}
                onChange={(e) => applyCustomFontSize(e.target.value)}
                className="h-8 px-2 text-xs font-bold text-black bg-transparent outline-none cursor-pointer hover:bg-gray-50"
              >
                <option value="12">12px</option>
                <option value="14">14px</option>
                <option value="16">16px</option>
                <option value="18">18px</option>
                <option value="24">24px</option>
                <option value="32">32px</option>
              </select>
            </div>

            <div className="w-[1px] h-6 bg-[#C4C4C4] mx-1 shrink-0" />
            
            <button onClick={() => applyFormatting("<mark>", "</mark>", "highlight")} className={`px-3 h-8 shrink-0 flex items-center justify-center text-xs font-bold border border-[#C4C4C4] rounded-sm transition-colors ${activeFormats.highlight ? "bg-[#D9D9D9] text-black" : "bg-white text-black hover:bg-gray-100"}`} title="Highlight">
              Highlight
            </button>

            <div className="w-[1px] h-6 bg-[#C4C4C4] mx-1 shrink-0" />
            
            {/* Insert Table Button */}
            <button onClick={insertTable} className="px-3 h-8 shrink-0 flex items-center justify-center gap-1 text-xs font-bold border border-[#C4C4C4] bg-white text-black hover:bg-gray-100 rounded-sm transition-colors" title="Insert Table">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
              Table
            </button>
          </div>

          <textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => updateContentWithHistory(e.target.value)}
            onKeyDown={handleKeyDown}
            onKeyUp={checkActiveFormats}
            onMouseUp={checkActiveFormats}
            className="flex-1 w-full p-8 resize-none outline-none font-mono text-sm text-black leading-relaxed bg-transparent"
            placeholder="Write your markdown here..."
            spellCheck="false"
          />
        </div>

        {/* RIGHT: Live Preview */}
        <div className="flex-1 bg-white overflow-y-auto p-8 prose max-w-none relative group">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]} 
            rehypePlugins={[rehypeRaw]}
          >
            {normalizeMarkdown(content)}
          </ReactMarkdown>
        </div>
      </div>
    </motion.div>
  );
}