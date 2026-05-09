import requests
import os


MINUTES_PROMPT = """You are a professional meeting minutes writer. Transform the raw meeting transcript below into a clean, structured, formal minutes document.

Always produce the output in this exact order:

1. **Meeting Title** - derive from context if not stated.
2. **Date, Time, and Location** - include timezone if available. Mark as "Not specified" if absent.
3. **Meeting Type** - e.g. standup, board meeting, sprint review, client call, retrospective.
4. **Attendees** - list all participants with names and roles. Separate Present from Absent (if mentioned).
5. **Facilitator / Chair** - the person who led the meeting.
6. **Minute Taker** - if mentioned; otherwise omit this field.
7. **Agenda Items** - numbered list of all agenda items discussed.
8. **Discussion Summary** - for each agenda item, write a concise, neutral, third-person summary. Capture key arguments, concerns, and outcomes. Do not editorialize.
9. **Decisions Made** - a dedicated list of all formal decisions or agreements. Each decision must be one clear sentence.
10. **Action Items** - a table with columns: Action | Owner | Deadline. Extract every task or follow-up assigned. If no deadline was stated, write "Not specified".
11. **Parking Lot / Deferred Items** - topics raised but intentionally deferred.
12. **Next Meeting** - date, time, and location if mentioned.
13. **Meeting Closure** - time the meeting ended, if mentioned.

Writing rules:
- Write in formal, third-person, past tense.
- Be factual. Do not infer intent or add information not present in the transcript.
- If a section has no relevant content, write "None noted".
- If attribution is unclear, summarize the point without a name.
- Mark any unclear segments as [unclear].
- Do not include filler content, pleasantries, or off-topic exchanges.
- Output the document only. No preamble, no closing statement.

Transcript:
{transcript}
"""


class OllamaSummarizer:
    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.max_lines = int(os.getenv("TRANSCRIPT_MAX_LINES", "500"))

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.Timeout:
            return "[Error: Ollama timed out after 180 seconds]"
        except Exception as e:
            return f"[Ollama error: {e}]"

    def _prepare_transcript(self, lines: list[str]) -> str:
        joined = "\n".join(line.strip() for line in lines if line.strip())
        if len(lines) > self.max_lines:
            print(
                f"[Warning] Transcript has {len(lines)} lines. "
                f"Truncating to first {self.max_lines} lines. "
                f"Consider chunking for long meetings."
            )
            joined = "\n".join(
                line.strip() for line in lines[:self.max_lines] if line.strip()
            )
        return joined

    def generate_minutes(self, full_transcript: list[str], output_path: str) -> None:
        if not full_transcript:
            print("No transcript to summarize.")
            return

        transcript_text = self._prepare_transcript(full_transcript)
        prompt = MINUTES_PROMPT.format(transcript=transcript_text)
        minutes = self._generate(prompt)

        with open(output_path, "w") as f:
            f.write(minutes)
        print(f"Minutes saved to {output_path}")