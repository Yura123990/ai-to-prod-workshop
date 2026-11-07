"""Перехоплювач HTTP-запитів OpenAI SDK.

Робить три речі:
  1. рахує виклики LLM і токени для кожного кейсу;
  2. (режим chaos) детерміновано підкидає збої: 429, 503, таймаути, «зламані» відповіді моделі;
  3. (режим mock) відповідає замість справжнього API — для перевірки пайплайну без ключа.

Збої детерміновані: для одного й того самого кейсу всі команди отримують однакові збої.
Працює лише з викликами через офіційний `openai` SDK (без streaming).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time

import openai._base_client as _bc

_http = getattr(_bc, "httpx2", None) or getattr(_bc, "httpx")

_local = threading.local()
_lock = threading.Lock()

STATE = {
    "installed": False,
    "chaos": False,
    "mock": False,
    "seed": "public",
    "allowed_model": None,
    "stats": {},  # case_id -> dict
}


def _case_stats(case_id):
    with _lock:
        return STATE["stats"].setdefault(
            case_id,
            {"http_calls": 0, "llm_calls_ok": 0, "tokens": 0, "chaos": [], "models": set()},
        )


def set_current_case(case_id):
    _local.case_id = case_id


def get_stats(case_id):
    s = _case_stats(case_id)
    return {**s, "models": sorted(s["models"]), "chaos": list(s["chaos"])}


def chaos_profile(case_id: str, seed: str) -> str:
    h = int(hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < 0.15:
        return "rate_limit_x3"  # 3 поспіль 429 — більше, ніж дефолтні 2 ретраї SDK
    if h < 0.25:
        return "garbage"  # модель відповіла текстом замість JSON
    if h < 0.32:
        return "empty"  # порожня відповідь моделі
    if h < 0.40:
        return "server_error"  # 503 один раз
    if h < 0.45:
        return "timeout"  # мережевий таймаут один раз
    return "none"


def _chaos_action(profile: str, attempt: int):
    if profile == "rate_limit_x3" and attempt <= 3:
        return "429"
    if profile == "garbage" and attempt == 1:
        return "garbage"
    if profile == "empty" and attempt == 1:
        return "empty"
    if profile == "server_error" and attempt == 1:
        return "503"
    if profile == "timeout" and attempt == 1:
        return "timeout"
    return None


GARBAGE_REPLIES = [
    "Звісно! Я проаналізував звернення: це схоже на питання доставки, пріоритет середній.",
    '{"category": "delivery", "priority": "medium", "needs_human": fal',
]


_MOCK_RULES = [
    (r"дим|іскр|загор|горить|горіл|здул|вибух|струм|smoke|fire", ("returns", "urgent", True)),
    (r"не я|без мого відома|шахрай|вкрал|fraud|unauthori", ("payment", "urgent", True)),
    (r"двічі|списали|оплат|кошт|refund|charged|гроші|платіж", ("payment", "high", False)),
    (r"поверн|обмін|return|exchange", ("returns", "medium", False)),
    (r"доставк|посилк|кур'єр|нова пошта|delivery|parcel|не прийшл", ("delivery", "medium", False)),
    (r"сайт|застосун|логін|пароль|вхід|помилк|app|login|error", ("technical", "high", False)),
    (r"чи є|характеристик|гарант|сумісн|warranty|in stock", ("product", "low", False)),
]


def _mock_answer(messages) -> str:
    text = ""
    for m in messages or []:
        if m.get("role") == "user":
            c = m.get("content")
            text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    low = text.lower()
    cat, pr, human = "other", "low", False
    for pattern, res in _MOCK_RULES:
        if re.search(pattern, low):
            cat, pr, human = res
            break
    if re.search(r"оператор|людин|менеджер|human|manager|суд|lawyer", low):
        human = True
    lang = "en" if re.search(r"[a-z]{4,}", low) and not re.search(r"[а-яіїєґ]", low) else "uk"
    return json.dumps(
        {"category": cat, "priority": pr, "needs_human": human, "language": lang,
         "summary": "Клієнт звернувся до підтримки (mock-відповідь)."},
        ensure_ascii=False,
    )


def _completion_json(model, content, prompt_tokens=0, completion_tokens=0):
    return {
        "id": "chatcmpl-workshop",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "unknown",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def _before(request):
    """Повертає (fake_response | exception | None, stats)."""
    case_id = getattr(_local, "case_id", "__no_case__")
    stats = _case_stats(case_id)
    is_chat = str(request.url.path).endswith("/chat/completions")
    body = {}
    if is_chat:
        try:
            body = json.loads(request.content or b"{}")
        except Exception:
            body = {}
    model = body.get("model")
    with _lock:
        stats["http_calls"] += 1
        attempt = stats["http_calls"]
        if model:
            stats["models"].add(model)

    if not is_chat:
        return None, stats, body

    if STATE["chaos"]:
        action = _chaos_action(chaos_profile(case_id, STATE["seed"]), attempt)
        if action:
            with _lock:
                stats["chaos"].append(action)
            if action == "429":
                return _http.Response(
                    429, headers={"retry-after-ms": "300"},
                    json={"error": {"code": 429, "message": "chaos: rate limit exceeded", "status": "RESOURCE_EXHAUSTED"}},
                    request=request), stats, body
            if action == "503":
                return _http.Response(
                    503, json={"error": {"code": 503, "message": "chaos: model overloaded", "status": "UNAVAILABLE"}},
                    request=request), stats, body
            if action == "timeout":
                return _http.ReadTimeout("chaos: read timeout", request=request), stats, body
            if action == "garbage":
                content = GARBAGE_REPLIES[len(case_id) % len(GARBAGE_REPLIES)]
                return _http.Response(200, json=_completion_json(model, content, 50, 20),
                                      request=request), stats, body
            if action == "empty":
                return _http.Response(200, json=_completion_json(model, "", 50, 0),
                                      request=request), stats, body

    if STATE["mock"]:
        time.sleep(0.05)
        content = _mock_answer(body.get("messages"))
        prompt_chars = len(json.dumps(body.get("messages", []), ensure_ascii=False))
        return _http.Response(200, json=_completion_json(model, content, prompt_chars // 4, 40),
                              request=request), stats, body
    return None, stats, body


def _after(response, stats):
    try:
        if response.status_code == 200:
            data = json.loads(response.content)
            usage = data.get("usage") or {}
            with _lock:
                stats["llm_calls_ok"] += 1
                stats["tokens"] += int(usage.get("total_tokens") or 0)
    except Exception:
        pass


def install(*, chaos=False, mock=False, seed="public", allowed_model=None):
    STATE.update(chaos=chaos, mock=mock, seed=seed, allowed_model=allowed_model)
    if STATE["installed"]:
        return
    STATE["installed"] = True

    orig_sync = _bc.SyncAPIClient._send_request
    orig_async = _bc.AsyncAPIClient._send_request

    def sync_send(self, request, *, stream, **kwargs):
        fake, stats, _ = _before(request)
        if isinstance(fake, Exception):
            raise fake
        if fake is not None:
            _after(fake, stats)
            return fake
        resp = orig_sync(self, request, stream=stream, **kwargs)
        if not stream:
            _after(resp, stats)
        return resp

    async def async_send(self, request, *, stream, **kwargs):
        fake, stats, _ = _before(request)
        if isinstance(fake, Exception):
            raise fake
        if fake is not None:
            _after(fake, stats)
            return fake
        resp = await orig_async(self, request, stream=stream, **kwargs)
        if not stream:
            try:
                await resp.aread()
            except Exception:
                pass
            _after(resp, stats)
        return resp

    _bc.SyncAPIClient._send_request = sync_send
    _bc.AsyncAPIClient._send_request = async_send
