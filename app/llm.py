"""Тонка обгортка над LLM API (Gemini через OpenAI-сумісний endpoint).

Цей файл можна і треба змінювати. Обмеження:
  * викликайте модель лише через офіційний `openai` SDK (без streaming);
  * модель беріть з LLM_MODEL — у фінальному прогоні організатори задають її самі;
  * ключ — тільки зі змінної середовища GEMINI_API_KEY, ніколи не хардкодьте.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=BASE_URL)
    return _client


def chat(messages, **kwargs) -> str:
    """Надсилає повідомлення моделі й повертає текст відповіді."""
    resp = get_client().chat.completions.create(model=MODEL, messages=messages, **kwargs)
    return resp.choices[0].message.content
