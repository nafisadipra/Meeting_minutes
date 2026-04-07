import os
from dotenv import load_dotenv
from assistant import MeetingAssistant

# This loads the hidden variables from your .env file
load_dotenv() 

if __name__ == "__main__":
    # Now it grabs the token securely!
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    print("\n=== Meeting Setup ===")
    names_input = input("Enter the names of the attendees (comma-separated): ")
    # Clean up the input into a nice list (e.g., ["John", "Muyeed"])
    attendees_list = [name.strip() for name in names_input.split(",") if name.strip()]
    
    # Pass the list into the assistant
    assistant = MeetingAssistant(hf_token=HF_TOKEN, expected_attendees=attendees_list)
    
    try:
        assistant.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        assistant.stop()