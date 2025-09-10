# django_chatbot/bonechat/llms.py
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

_LLM = None

def _configure():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY/GEMINI_API_KEY in environment")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction="You are a helpful assistant."
    )

def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = _configure()
    return _LLM

def generate_reply(history_messages, user_prompt, max_output_tokens=512, temperature=0.6):
    llm = get_llm()
    chat = llm.start_chat(history=history_messages or [])
    resp = chat.send_message(
        user_prompt,
        generation_config=GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
    )
    return resp.text or ""
