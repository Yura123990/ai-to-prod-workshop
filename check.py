"""Локальна перевірка вашого рішення на публічних кейсах.

    python check.py              # звичайний прогін (справжній Gemini API)
    python check.py --chaos      # те саме + збої API, як у фінальному прогоні
    python check.py --case p11   # лише один кейс (можна кілька: --case p10 p11)
    python check.py --mock       # без API і ключа: фейкова модель, лише перевірка коду
    python check.py -v           # показати відповіді й помилки повністю

Фінальний прогін використовує ту саму логіку, але на ПРИХОВАНИХ кейсах і з chaos.
"""
import argparse
import os
import sys

from dotenv import load_dotenv

from harness import interceptor
from harness.runner import load_cases, run_cases
from harness.scoring import score_run

GREEN, RED, YELLOW, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
if os.name == "nt" or not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = BOLD = RESET = ""


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--chaos", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--cases-file", default="data/public_cases.jsonl")
    ap.add_argument("--delay", type=float, default=float(os.getenv("CHECK_DELAY", "4.5")))
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.mock:
        os.environ.setdefault("GEMINI_API_KEY", "mock-key")
        args.delay = 0
    elif not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your-key-here":
        sys.exit("Немає GEMINI_API_KEY. Скопіюйте .env.example у .env і вставте ключ (або запустіть з --mock).")

    interceptor.install(chaos=args.chaos, mock=args.mock, seed="public")
    from app.triage import triage  # імпорт після install — щоб перехоплювач уже працював

    cases = load_cases(args.cases_file)
    if args.case:
        cases = [c for c in cases if c["id"] in set(args.case)]
        if not cases:
            sys.exit("Кейси не знайдено")

    by_id = {c["id"]: c for c in cases}
    mode = " + ".join(m for m, on in (("MOCK", args.mock), ("CHAOS", args.chaos)) if on) or "звичайний"
    print(f"{BOLD}Кейсів: {len(cases)} · режим: {mode} · пауза: {args.delay}s{RESET}\n")

    def on_result(case, rec):
        tag = f"{GREEN}OK {RESET}" if rec["ok"] else f"{RED}ERR{RESET}"
        st = rec["stats"]
        chaos = f" {YELLOW}chaos={','.join(st['chaos'])}{RESET}" if st["chaos"] else ""
        line = f"{tag} {case['id']:<4} [{case.get('kind','standard'):<11}] {rec['latency']:5.2f}s {st['tokens']:>6} tok{chaos}"
        if rec["ok"]:
            out, exp = rec["output"], case["expected"]
            diffs = [f"{k}: {out.get(k)!r}≠{exp[k]!r}" for k in exp if out.get(k) != exp[k]]
            line += f"  {RED}{'; '.join(diffs)}{RESET}" if diffs else f"  {GREEN}✓ поля збігаються{RESET}"
            if args.verbose:
                line += f"\n      {DIM}{out}{RESET}"
        else:
            line += f"  {RED}{rec['error'][:160]}{RESET}"
            if args.verbose and rec.get("traceback"):
                line += f"\n{DIM}{rec['traceback']}{RESET}"
        print(line, flush=True)

    results = run_cases(triage, cases, timeout=args.timeout, delay=args.delay, on_result=on_result)
    rep = score_run(list(by_id.values()), results)
    p, m = rep["parts"], rep["metrics"]
    print(f"\n{BOLD}Орієнтовний бал на публічних кейсах: {rep['total']} / 100{RESET}")
    print(f"  Accuracy {p['accuracy']}/45 · Safety {p['safety']}/20 · "
          f"Reliability {p['reliability']}/20 · Efficiency {p['efficiency']}/15")
    print(f"  валідних {m['valid']}/{m['cases']} · токенів/кейс {m['avg_tokens']} · "
          f"медіана {m['p50_latency']}s · HTTP-викликів {m['llm_http_calls']}")
    if not args.chaos:
        print(f"{DIM}  Порада: запустіть з --chaos — у фіналі збої API будуть увімкнені.{RESET}")
    if args.mock:
        print(f"{DIM}  MOCK: відповіді фейкові, бал Accuracy нічого не означає.{RESET}")


if __name__ == "__main__":
    main()
