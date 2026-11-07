"""Перевірка, що все готово до воркшопу. Запустіть ДО початку:  python check_setup.py"""
import os
import sys

ok = True


def step(name, fn):
    global ok
    try:
        msg = fn()
        print(f"[OK]   {name}" + (f" — {msg}" if msg else ""))
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name} — {e}")


def py():
    if sys.version_info < (3, 10):
        raise RuntimeError(f"потрібен Python 3.10+, у вас {sys.version.split()[0]}")
    return sys.version.split()[0]


def deps():
    import dotenv, openai, pydantic, pytest  # noqa: F401
    return f"openai {openai.__version__}, pydantic {pydantic.__version__}"


def key():
    from dotenv import load_dotenv
    load_dotenv()
    k = os.getenv("GEMINI_API_KEY", "")
    if not k or k == "your-key-here":
        raise RuntimeError("немає ключа в .env (cp .env.example .env і вставте ключ з aistudio.google.com/apikey)")
    return f"{k[:4]}…{k[-4:]}"


def api():
    from app.llm import MODEL, chat
    text = chat([{"role": "user", "content": "Відповідай одним словом: працює?"}])
    return f"модель {MODEL} відповіла: {str(text).strip()[:40]!r}"


step("Python", py)
step("Залежності (pip install -r requirements.txt)", deps)
step("Ключ GEMINI_API_KEY", key)
if ok:
    step("Запит до моделі", api)
print("\nВсе готово 🎉" if ok else "\nЄ проблеми — виправте їх до початку воркшопу.")
sys.exit(0 if ok else 1)
