import requests
import json

# ✅ Your OpenRouter API key
API_KEY = "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"
# Endpoint for LLaMA 3.2 chat
url = "https://openrouter.ai/api/v1/chat/completions"

# Headers with authorization
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

print("🤖 LLaMA 3.2 Chatbot (type 'exit' to quit)\n")

while True:
    # Take user input
    user_input = input("You: ")
    
    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot session ended.")
        break
    
    # Create payload for the API
    payload = {
        "model": "meta-llama/llama-3.2-3b-instruct:free",  # Updated to the correct free model
        "messages": [
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7  # Adjust for more creative or precise answers
    }
    
    try:
        # Send request to LLaMA API (using json parameter instead of data)
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        # Extract chatbot's reply
        data = response.json()
        answer = data['choices'][0]['message']['content']
        
        print(f"🤖 Bot: {answer}\n")
    
    except requests.exceptions.RequestException as e:
        print(f"⚠ API Request Error: {e}")
        print(f"Status Code: {response.status_code if 'response' in locals() else 'N/A'}")
        if 'response' in locals():
            print(f"Response: {response.text}")
    except KeyError as e:
        print(f"⚠ Unexpected response format - Missing key: {e}")
        if 'response' in locals():
            print("Full response:", response.text)
    except Exception as e:
        print(f"⚠ Unexpected error: {e}")