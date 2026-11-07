"""AI-тріаж звернень у підтримку інтернет-магазину.

Ця версія — «працює на демо». Ваше завдання — довести її до production.
Контракт (що має повертати triage) описано в README.md і в harness/schema.py.
"""
import json

from app.llm import chat

SYSTEM_PROMPT = """Ти асистент служби підтримки інтернет-магазину.
Визнач категорію і пріоритет звернення клієнта та коротко його опиши.
Відповідай у форматі JSON."""


def triage(message: str) -> dict:
    answer = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
    )
    return json.loads(answer)
