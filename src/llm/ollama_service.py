import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class OllamaService:
    def __init__(self, model_name="llama3:8b"):
        """Initialize the Ollama service

        Args:
            model_name (str): Name of the model to use
        """
        self.api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/api")
        self.model_name = model_name

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2048):
        """Generate text using Ollama

        Args:
            prompt (str): The user prompt
            system_prompt (str, optional): System prompt for context
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate

        Returns:
            str: The generated response
        """
        url = f"{self.api_base}/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "No response")
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: {str(e)}"

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        """Chat with the Ollama model

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate

        Returns:
            str: The generated response
        """
        url = f"{self.api_base}/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "No response")
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: {str(e)}"

    def check_availability(self):
        """Check if Ollama server is available

        Returns:
            bool: True if server is available, False otherwise
        """
        try:
            response = requests.get(f"{self.api_base}/tags")
            return response.status_code == 200
        except Exception:
            return False 