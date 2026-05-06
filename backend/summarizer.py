import requests
import os


class OllamaSummarizer:
    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

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

    def generate_minutes(self, full_transcript: list[str], output_path: str) -> None:
        if not full_transcript:
            print("No transcript to summarize.")
            return

        prompt = f"""Convert this meeting transcript into structured minutes.

Format:
# Minutes
## Summary
(one paragraph)
## Discussion points
- (bullet points with speaker names)
## Action items
- (list if any)

Transcript:
{chr(10).join(full_transcript[-150:])}
"""
        minutes = self._generate(prompt)
        with open(output_path, "w") as f:
            f.write(minutes)
        print(f"Minutes saved to {output_path}")
