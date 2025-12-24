import requests
from config import OLLAMA_MODEL, OLLAMA_HOST

class LLMClient:
    def __init__(self):
        self.model = OLLAMA_MODEL
        self.host = OLLAMA_HOST

    def chat(self, system, user, temperature=0.5):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {"temperature": temperature}
        }
        r = requests.post(f"{self.host}/api/chat", json=payload)
        return r.json()["message"]["content"]
