"""Прогін кейсів через функцію triage(message) з вимірюванням часу, токенів і валідацією."""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from . import interceptor
from .schema import validate_output


def load_cases(path):
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        c = json.loads(line)
        if "input_repeat" in c:  # довгі входи зберігаємо компактно
            r = c.pop("input_repeat")
            c["input"] = r.get("prefix", "") + r["text"] * int(r["times"]) + r.get("suffix", "")
        cases.append(c)
    return cases


def _short(s, n=300):
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def run_case(triage_fn, case, *, timeout=20.0):
    box = {}

    def target():
        interceptor.set_current_case(case["id"])
        try:
            box["out"] = triage_fn(case["input"])
        except BaseException as e:  # noqa: BLE001 — нам потрібно зловити все
            box["err"] = f"{type(e).__name__}: {_short(e)}"
            box["tb"] = traceback.format_exc(limit=4)

    t0 = time.perf_counter()
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    latency = time.perf_counter() - t0

    rec = {"id": case["id"], "kind": case.get("kind", "standard"), "latency": round(latency, 3)}
    if th.is_alive():
        rec.update(ok=False, error=f"Timeout: довше за {timeout:.0f} с", output=None)
    elif "err" in box:
        rec.update(ok=False, error=box["err"], traceback=box.get("tb"), output=None)
    else:
        out = box.get("out")
        if hasattr(out, "model_dump"):
            out = out.model_dump()
        valid, verr = validate_output(out)
        try:
            json.dumps(out, ensure_ascii=False)
            serial = out
        except Exception:
            serial = _short(repr(out))
        rec.update(ok=valid, error=None if valid else f"Невалідна відповідь: {verr}", output=serial)
    rec["stats"] = interceptor.get_stats(case["id"])
    return rec


def run_cases(triage_fn, cases, *, timeout=20.0, delay=0.0, on_result=None):
    results = []
    for i, case in enumerate(cases):
        if i and delay:
            time.sleep(delay)
        rec = run_case(triage_fn, case, timeout=timeout)
        results.append(rec)
        if on_result:
            on_result(case, rec)
    return results
