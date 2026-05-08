import { motion } from "framer-motion";

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

interface DocumentVaultProps {
  savedDocuments: SavedSession[];
  onViewSession: (session: SavedSession) => void;
  onDownloadDocx: (minutes: string, fileName: string) => void;
}

export default function DocumentVault({
  savedDocuments,
  onViewSession,
  onDownloadDocx,
}: DocumentVaultProps) {
  return (
    <motion.div
      key="documents"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3 }}
      className="flex-1 p-10 overflow-y-auto bg-white print:hidden"
    >
      <div className="max-w-6xl mx-auto w-full flex flex-col gap-8">
        <div>
          <h1 className="text-2xl font-bold text-black tracking-tight">
            Document Vault
          </h1>
          <p className="text-sm text-gray-600 mt-2">
            Secure, offline storage for your finalized meeting minutes.
          </p>
        </div>

        {savedDocuments.length === 0 ? (
          <div className="py-20 flex flex-col items-center justify-center text-gray-500 border border-[#C4C4C4] border-dashed bg-gray-50 rounded-md">
            <svg
              className="w-10 h-10 mb-4 text-[#C4C4C4]"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <span className="text-sm font-bold text-black">Vault is empty</span>
            <span className="text-xs mt-1">
              Open the 3-dot menu in an active session and click "Save" to store
              it here.
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {savedDocuments.map((session) => (
              <div
                key={session.id}
                className="p-5 bg-white border border-[#C4C4C4] rounded-md shadow-[2px_2px_8px_rgba(0,0,0,0.04)] hover:shadow-[4px_4px_12px_rgba(0,0,0,0.08)] transition-all flex flex-col"
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="w-8 h-8 bg-[#D9D9D9] rounded-sm flex items-center justify-center shrink-0 border border-[#C4C4C4]">
                    <svg
                      className="w-4 h-4 text-black"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                  </div>
                  {session.isPinned && (
                    <svg
                      className="w-4 h-4 text-black"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
                    </svg>
                  )}
                </div>

                <h3 className="text-sm font-bold text-black mb-1 truncate">
                  {session.name}
                </h3>
                <p className="text-[11px] font-bold text-gray-500 uppercase tracking-wider mb-6">
                  {session.date} •{" "}
                  {new Date(session.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>

                <div className="mt-auto flex gap-2">
                  <button
                    onClick={() => onViewSession(session)}
                    className="flex-1 py-1.5 text-xs font-bold bg-white text-black border border-black rounded-sm hover:bg-gray-50 transition-colors"
                  >
                    View
                  </button>
                  <button
                    onClick={() => onDownloadDocx(session.minutes, session.name)}
                    className="px-3 py-1.5 text-xs font-bold bg-[#D9D9D9] text-black border border-[#C4C4C4] rounded-sm hover:bg-gray-300 transition-colors"
                    title="Download .DOCX"
                  >
                    ↓
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}