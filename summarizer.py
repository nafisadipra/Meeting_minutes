import requests

class LocalLLMSummarizer:
    def __init__(self, url="http://localhost:1234/v1/chat/completions"):
        self.url = url

    def _ask_lm(self, prompt):
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": "You are a professional meeting assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"[Error connecting to LLM: {e}]"

    def generate_minutes(self, full_transcript):
        print("\n--- Generating Meeting Minutes ---")
        if not full_transcript:
            print("No transcription data to summarize.")
            return

        chunk_size = 50
        chunks = [full_transcript[i:i + chunk_size] for i in range(0, len(full_transcript), chunk_size)]
        
        mini_summaries = []
        
        for idx, chunk in enumerate(chunks):
            print(f"Summarizing chunk {idx + 1}/{len(chunks)}...")
            chunk_text = "\n".join(chunk)
            
            # Map Prompt: Force it to keep details
            map_prompt = f"""Read this chronological meeting transcript carefully. 
            Do NOT drop important details, arguments, or points made by the speakers. 
            Provide a comprehensive summary of the conversation flow.
            
            Transcript:
            {chunk_text}"""
            
            summary = self._ask_lm(map_prompt)
            mini_summaries.append(summary)

        print("Combining into final minutes...")
        combined_text = "\n\n".join(mini_summaries)
        
        # Reduce Prompt: Give it a strict format and forbid generic placeholders
        reduce_prompt = f"""You are a highly detailed professional secretary. 
        Read the following chronological summaries of a meeting and create a comprehensive final document.
        
        CRITICAL RULES:
        - Do NOT use generic placeholders like [Insert Location], [Date], or [Name].
        - Use the actual speaker names provided in the text.
        - Capture the actual substance of what was discussed, not just vague overviews.
        
        Use exactly this format:
        
        # Detailed Meeting Minutes
        
        ## 1. Executive Summary
        (Write a strong paragraph explaining the core purpose and outcome of the meeting based on the text.)
        
        ## 2. Comprehensive Discussion Points
        (Use bullet points to break down the specifics of what was discussed. Mention who made which points.)
        
        ## 3. Action Items & Decisions
        (List clear next steps and who is responsible for them, if mentioned.)
        
        Combined Summaries:
        {combined_text}"""
        
        final_minutes = self._ask_lm(reduce_prompt)
        
        with open("minutes.txt", "w") as f:
            f.write(final_minutes)
        print("Meeting minutes saved to 'minutes.txt'!")