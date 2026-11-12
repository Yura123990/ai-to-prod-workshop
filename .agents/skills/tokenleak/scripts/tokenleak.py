#!/usr/bin/env python3
"""tokenleak: find where your coding agent leaks context tokens.

Reads local Claude Code and Codex transcripts, prices every tool result by the
"context tax" it imposed (its size x every later call that re-sent it), and
names the rule that would have avoided it. Local, read-only, stdlib only.
"""
from __future__ import annotations

import argparse
import bisect
import html
import io
import json
import os
import re
import shlex
import sys
import tempfile
import time
from datetime import datetime

__version__ = "1.9.0"
REPO = "https://github.com/Pazificateur69/tokenleak"

CHARS_PER_TOKEN = 4.0  # transcripts don't store per-result token counts; ~4 chars/token for code + English
IMAGE_TOKENS = 1500

# Relative price weights, fresh input = 1. Claude: published cache/output multipliers.
# Codex: GPT-5-family ratios (cached input 0.1x, output 8x).
WEIGHTS = {
    "claude": {"read": 0.1, "out": 5.0},
    "codex": {"read": 0.1, "out": 8.0, "enter": 1.0},
}

# kind: (flag when a result is bigger than N tokens, budget a bounded version would cost, fix)
KINDS = {
    "noisy-shell": (2000, 800, "Bound every command: quiet flags, `> log; tail -n 40 log`, grep the log"),
    "big-read": (4000, 2000, "Locate, then read: grep -n first, then read only that range"),
    "re-read": (0, 0, "Never re-read: the file is still in your context"),
    "wide-search": (2000, 600, "Narrow searches: scope path/type, cap results, -l for names only"),
    "big-fetch": (3000, 1500, "Fetch precisely: ask a question, select fields, cap pages"),
    "big-result": (5000, 2000, "Cap tool output: limits, filters, fields"),
    "rewrite": (1500, 0, "Edit, don't rewrite whole files"),
    "serial-call": (0, 0, "Batch independent reads/searches in one turn"),
    "wasted-call": (0, 0, "Don't repeat a call whose result you already have"),
    "verbose-reply": (2000, 600, "Answer lean: result first, no recap"),
}
SIZE_KIND = {"shell": "noisy-shell", "read": "big-read", "search": "wide-search",
             "fetch": "big-fetch", "subagent": "big-result", "other": "big-result"}

READ_CMDS = {"cat", "sed", "head", "tail", "nl", "less", "more", "bat", "batcat", "view"}
SEARCH_CMDS = {"rg", "grep", "egrep", "ag", "ack", "find", "fd", "ls", "tree", "locate"}
SHELL_TOOLS = {"bash", "bashoutput", "shell", "exec_command", "local_shell", "local_shell_call",
               "exec", "container.exec", "unified_exec", "run_terminal_cmd", "run_command"}
EDIT_TOOLS = {"edit", "multiedit", "write", "notebookedit", "apply_patch", "str_replace_based_edit_tool",
              "edit_file", "write_file", "create_file"}
MUTATES_ALL = re.compile(r"git\s+(checkout|switch|stash|reset|pull|merge|rebase|restore|apply)|--fix\b|"
                         r"prettier|black\b|ruff\s+format|gofmt|cargo\s+fmt|npm\s+run\s+format|codemod|sed\s+-i")


# ---------------------------------------------------------------- model

class Call:
    __slots__ = ("pos", "seg", "fresh", "cread", "c5m", "c1h", "out", "tools", "text_chars", "ts")

    def __init__(self, pos, seg, ts=None):
        self.pos, self.seg, self.ts = pos, seg, ts
        self.fresh = self.cread = self.c5m = self.c1h = self.out = 0
        self.tools, self.text_chars = [], 0

    @property
    def ctx_in(self):
        return self.fresh + self.cread + self.c5m + self.c1h


class ToolUse:
    __slots__ = ("id", "name", "input", "pos", "call", "result")

    def __init__(self, id, name, inp, pos):
        self.id, self.name, self.input, self.pos = id, name or "", inp, pos
        self.call = self.result = None


class Result:
    __slots__ = ("tool", "pos", "seg", "tokens", "text", "extra")

    def __init__(self, tool, pos, seg, tokens, text, extra=None):
        self.tool, self.pos, self.seg, self.tokens, self.text, self.extra = tool, pos, seg, tokens, text, extra


class Thread:
    """One transcript file = one context window lineage (main session or a subagent)."""

    def __init__(self, path, agent):
        self.path, self.agent = path, agent
        self.session = self.cwd = None
        self.subagent = "/subagents/" in path.replace(os.sep, "/")
        self.calls, self.uses, self.results, self.human_pos = [], [], [], []
        self.notes = []  # (pos, text) of user prompts: what the agent knew before calling tools
        self.cost_usd = None
        self.start = None  # ISO timestamp of the first event
        self.instructions = {}  # CLAUDE.md & co. as actually loaded: path -> chars (recent Claude Code logs them)
        self.attachments = False  # transcript format that records attachments, so an empty dict means "none loaded"
        self.start_ctx = None  # context size of the session's first call, kept when --since trims earlier calls
        self.mtime = os.path.getmtime(path)


def measure(content):
    """(text, est. tokens) of what a tool result put into context."""
    if content is None:
        return "", 0.0
    if isinstance(content, str):
        return content, len(content) / CHARS_PER_TOKEN
    if isinstance(content, dict):
        content = [content]
    parts, images = [], 0
    for c in content if isinstance(content, list) else []:
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, dict):
            if "image" in str(c.get("type", "")):
                images += 1
            elif isinstance(c.get("text"), str):
                parts.append(c["text"])
    text = "\n".join(parts)
    return text, len(text) / CHARS_PER_TOKEN + images * IMAGE_TOKENS


def _json_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for pos, line in enumerate(f, 1):
            try:
                yield pos, json.loads(line)
            except ValueError:
                continue


# ---------------------------------------------------------------- parsers

MAX_KEEP_CHARS = 200_000  # result text kept for dependency checks; sizes are measured on the full text


def parse_claude(path):
    """-> [main thread, sidechain thread]. Old versions logged subagent turns inline (isSidechain)."""
    main, side = Thread(path, "claude"), Thread(path, "claude")
    side.subagent = True
    state = {False: (main, {}, {}), True: (side, {}, {})}
    seg, seen = 0, set()
    for pos, d in _json_lines(path):
        if not isinstance(d, dict):
            continue
        typ = d.get("type")
        t, calls, pending = state[bool(d.get("isSidechain"))]
        t.start = t.start or d.get("timestamp")
        for th in (main, side):
            th.cwd = th.cwd or d.get("cwd")
            th.session = th.session or d.get("sessionId")
        if typ == "system" and d.get("subtype") == "compact_boundary":
            seg += 1
        elif typ == "cost-state":
            main.cost_usd = d.get("totalCostUSD", main.cost_usd)
        elif typ == "attachment":
            t.attachments = True
            a = d.get("attachment") or {}
            if isinstance(a, dict) and a.get("type") == "instructions":
                for f in a.get("files") or []:
                    if isinstance(f, dict) and f.get("path"):
                        t.instructions[str(f["path"])] = len(str(f.get("content") or ""))
        elif typ == "assistant":
            m = d.get("message") or {}
            u = m.get("usage") or {}
            mid = m.get("id") or d.get("requestId") or d.get("uuid")
            call = calls.get(mid)
            if call is None:
                if not any(u.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                              "cache_creation_input_tokens")):
                    continue  # synthetic / errored message: no API call billed
                call = calls[mid] = Call(pos, seg, epoch(d.get("timestamp")))
                t.calls.append(call)
            cc = u.get("cache_creation") or {}
            c5m, c1h = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
            if not (c5m or c1h):
                c5m = u.get("cache_creation_input_tokens") or 0
            # streamed blocks of one response repeat the usage; keep the final (max) values
            call.fresh = max(call.fresh, u.get("input_tokens") or 0)
            call.cread = max(call.cread, u.get("cache_read_input_tokens") or 0)
            call.c5m, call.c1h = max(call.c5m, c5m), max(call.c1h, c1h)
            call.out = max(call.out, u.get("output_tokens") or 0)
            for c in m.get("content") or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_use":
                    if c.get("id") and c.get("id") in seen:  # the same block logged again
                        continue
                    seen.add(c.get("id"))
                    tu = ToolUse(c.get("id"), c.get("name"), c.get("input") or {}, pos)
                    tu.call = call
                    call.tools.append(tu)
                    t.uses.append(tu)
                    pending[tu.id] = tu
                elif c.get("type") == "text":
                    call.text_chars += len(c.get("text") or "")
        elif typ == "user":
            content = (d.get("message") or {}).get("content")
            if isinstance(content, str):
                if not d.get("isMeta") and not d.get("isCompactSummary"):
                    t.human_pos.append(pos)
                    t.notes.append((pos, content[:MAX_KEEP_CHARS]))
                continue
            for c in content if isinstance(content, list) else []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_result":
                    tu = pending.pop(c.get("tool_use_id"), None)
                    if tu is not None:
                        text, toks = measure(c.get("content"))
                        extra = d.get("toolUseResult") if tu.name == "Write" else None
                        tu.result = Result(tu, pos, seg, toks, text[:MAX_KEEP_CHARS], extra)
                        t.results.append(tu.result)
                elif c.get("type") == "text" and not d.get("isMeta"):
                    t.human_pos.append(pos)
                    t.notes.append((pos, str(c.get("text") or "")[:MAX_KEEP_CHARS]))
                    break
    return [main, side]


def parse_codex(path):
    t = Thread(path, "codex")
    pending, done, seg = {}, {}, 0
    records, counts, last_total = [], [], None
    for pos, d in _json_lines(path):
        if not isinstance(d, dict):
            continue
        typ, p = d.get("type"), d.get("payload") or {}
        t.start = t.start or d.get("timestamp")
        if not isinstance(p, dict):
            continue
        if typ == "session_meta":
            t.cwd = t.cwd or p.get("cwd")
            t.session = t.session or p.get("id") or p.get("session_id")
            src = p.get("source")
            if isinstance(src, dict) and "subagent" in src:  # guardian / spawned threads belong to a parent
                t.subagent = True
                parent = re.search(r'"parent_thread_id":\s*"([^"]+)"', json.dumps(src))
                if parent:
                    t.session = parent.group(1)
        elif typ == "turn_context":
            t.cwd = t.cwd or p.get("cwd")
        elif typ == "compacted":
            seg += 1
        elif typ == "token_usage_record" and isinstance(p.get("usage"), dict):
            records.append((pos, seg, p["usage"], d.get("timestamp")))
        elif typ == "event_msg" and p.get("type") == "token_count" and isinstance(p.get("info"), dict):
            info = p["info"]
            total = json.dumps(info.get("total_token_usage"), sort_keys=True)
            if total != last_total and isinstance(info.get("last_token_usage"), dict):
                last_total = total
                counts.append((pos, seg, info["last_token_usage"], d.get("timestamp")))
        elif typ == "event_msg" and p.get("type") == "user_message":
            t.human_pos.append(pos)
            t.notes.append((pos, str(p.get("message") or "")[:MAX_KEEP_CHARS]))
        elif typ == "response_item":
            pt = p.get("type")
            if pt in ("function_call", "custom_tool_call", "local_shell_call"):
                raw = p.get("arguments") if p.get("arguments") is not None else p.get("input", p.get("action"))
                inp = raw
                if isinstance(raw, str):
                    try:
                        inp = json.loads(raw)
                    except ValueError:
                        inp = raw
                name = p.get("name") or pt
                if p.get("namespace"):  # MCP tools: keep the server name ("mcp__server:tool")
                    name = f"{p['namespace']}:{name}"
                tu = ToolUse(p.get("call_id") or p.get("id"), name, inp, pos)
                t.uses.append(tu)
                pending[tu.id] = tu
            elif pt in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
                cid = p.get("call_id")
                tu = pending.pop(cid, None) or done.get(cid)
                if tu is not None:
                    text, toks = measure(p.get("output"))
                    if tu.result is None:
                        tu.result = Result(tu, pos, seg, toks, text[:MAX_KEEP_CHARS])
                        t.results.append(tu.result)
                        if cid:
                            done[cid] = tu
                    else:  # a later chunk of the same call's output
                        tu.result.tokens += toks
                        tu.result.text = (tu.result.text + text)[:MAX_KEEP_CHARS]
            elif pt == "message" and p.get("role") == "assistant":
                t.uses.append(ToolUse(None, "__text__", sum(len(c.get("text") or "") for c in p.get("content") or []
                                                            if isinstance(c, dict)), pos))
    for pos, s, u, ts in records or counts:  # newer rollouts log both; prefer the per-response records
        call = Call(pos, s, epoch(ts))
        cached = u.get("cached_input_tokens") or 0
        call.cread = cached
        call.fresh = max(0, (u.get("input_tokens") or 0) - cached)
        call.out = u.get("output_tokens") or 0
        t.calls.append(call)
    # a response's tool calls are logged before its usage record: attach each to the next record
    starts = [c.pos for c in t.calls]
    for tu in list(t.uses):
        i = bisect.bisect_left(starts, tu.pos)
        call = t.calls[i] if i < len(t.calls) else None
        if tu.name == "__text__":
            t.uses.remove(tu)
            if call:
                call.text_chars += tu.input
            continue
        tu.call = call
        if call:
            call.tools.append(tu)
    return [t]


CODEX_TYPES = {"session_meta", "response_item", "event_msg", "turn_context", "compacted", "token_usage_record"}


def parse(path):
    """-> list of Threads in this transcript file."""
    for _, d in zip(range(5), (d for _, d in _json_lines(path))):
        if isinstance(d, dict) and "payload" in d and d.get("type") in CODEX_TYPES:
            return parse_codex(path)
    return parse_claude(path)


# ---------------------------------------------------------------- classification

def shell_command(inp):
    if isinstance(inp, str):
        return inp
    if not isinstance(inp, dict):
        return ""
    c = inp.get("command") or inp.get("cmd") or inp.get("code") or inp.get("input") or ""
    if isinstance(c, list):
        if len(c) >= 3 and str(c[0]).rsplit("/", 1)[-1] in ("bash", "sh", "zsh") and c[1] in ("-lc", "-c"):
            return str(c[2])
        return " ".join(map(str, c))
    return str(c)


def _main_segment(cmd):
    """First meaningful simple command: skips `cd x &&`, env assignments, `export`."""
    for seg in re.split(r"&&|\|\||;|\n", cmd):
        seg = seg.split("|")[0].strip()
        try:
            words = shlex.split(seg) if any(ch in seg for ch in "'\"\\") else seg.split()
        except ValueError:
            words = seg.split()
        while words and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]) or words[0] in ("sudo", "time", "env")):
            words = words[1:]
        if words and words[0] not in ("cd", "export", "source", ".", "set", "echo", "pushd"):
            return words
    return []


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def classify(tu):
    """-> (category, target, range) where category in read/search/shell/fetch/edit/subagent/other."""
    n, inp = tu.name.lower(), tu.input if isinstance(tu.input, dict) else {}
    if n in ("read", "read_file", "view_file"):
        path = inp.get("file_path") or inp.get("path") or inp.get("target_file") or ""
        return "read", str(path), (_num(inp.get("offset") or inp.get("start_line")), _num(inp.get("limit") or inp.get("end_line")))
    if n in ("grep", "glob", "ls", "search", "codebase_search", "grep_search", "file_search", "list_dir"):
        return "search", str(inp.get("pattern") or inp.get("query") or inp.get("path") or ""), None
    if n in EDIT_TOOLS:
        return "edit", str(inp.get("file_path") or inp.get("path") or shell_command(tu.input)), None
    if n in ("webfetch", "websearch", "web_search", "fetch", "web_fetch") or "mcp" in n:
        return "fetch", str(inp.get("url") or inp.get("query") or tu.name), None
    if n in ("task", "agent", "spawn_agent"):
        return "subagent", str(inp.get("description") or ""), None
    raw_js = isinstance(tu.input, str) and n == "exec" and re.search(r"\btools\.|\bawait\b|=>", tu.input)
    if raw_js or ("code" in inp and not ("cmd" in inp or "command" in inp)):  # Codex code mode: JS calling tools
        code = tu.input if raw_js else str(inp.get("code") or "")
        first = code.strip().splitlines()[0] if code.strip() else tu.name
        return ("fetch" if re.search(r"web__|\bfetch\(|browse|search_query", code) else "other"), first, None
    if n in SHELL_TOOLS or "command" in inp or "cmd" in inp:
        cmd = shell_command(tu.input)
        words = _main_segment(cmd)
        head = words[0].rsplit("/", 1)[-1] if words else ""
        if head in READ_CMDS and len(words) > 1:
            files = [w for w in words[1:] if not w.startswith("-") and not re.match(r"^[0-9,]+p?$|^'?[0-9,]+p'?$", w)]
            return "read", files[-1] if files else "", cmd.strip()
        if head in SEARCH_CMDS or words[:2] == ["git", "grep"]:
            return "search", cmd, None
        return "shell", cmd, None
    return "other", tu.name, None


def _covers(prev, new):
    """Does an earlier read range include a later one? Ranges are (offset, limit) or a shell command."""
    if isinstance(prev, str) or isinstance(new, str):
        if isinstance(prev, str) and isinstance(new, str) and prev == new:
            return True
        return isinstance(prev, str) and _main_segment(prev)[:1] == ["cat"] and "|" not in prev
    po, pl = prev
    no, nl = new
    if pl is None and (po in (None, 0, 1)):
        return True
    if pl is None or nl is None:
        return False
    po, no = po or 1, no or 1
    return po <= no and po + pl >= no + nl


# ---------------------------------------------------------------- analysis

class Finding:
    __slots__ = ("kind", "thread", "tool", "label", "tokens", "avoidable", "later", "cost", "pos")

    def __init__(self, kind, thread, tool, label, tokens, avoidable, later, cost, pos):
        self.kind, self.thread, self.tool, self.label = kind, thread, tool, label
        self.tokens, self.avoidable, self.later, self.cost, self.pos = tokens, avoidable, later, cost, pos


def weights(t):
    w = dict(WEIGHTS[t.agent])
    if t.agent == "claude":
        c5, c1 = sum(c.c5m for c in t.calls), sum(c.c1h for c in t.calls)
        w["enter"] = (1.25 * c5 + 2.0 * c1) / (c5 + c1) if c5 + c1 else 1.25
    return w


def call_cost(t, c, w):
    if t.agent == "claude":
        return c.fresh + 1.25 * c.c5m + 2.0 * c.c1h + w["read"] * c.cread + w["out"] * c.out
    return c.fresh + w["read"] * c.cread + w["out"] * c.out


def analyze(t):
    """-> (findings, total cost units) for one thread."""
    w = weights(t)
    by_seg = {}
    for c in t.calls:
        by_seg.setdefault(c.seg, []).append(c.pos)

    def later(pos, seg):
        ps = by_seg.get(seg, [])
        return len(ps) - bisect.bisect_right(ps, pos)

    findings = []

    def add(kind, tu, label, tokens, avoidable, pos, seg, out_w=0.0, cost=None):
        n = later(pos, seg)
        if cost is None:
            # generated tokens are paid once (out_w); then the next call writes them to the cache, later ones read
            cost = avoidable * (out_w + ((w["enter"] + (n - 1) * w["read"]) if n else 0))
        if cost > 0:
            findings.append(Finding(kind, t, tu.name if tu else "reply", label, tokens, avoidable, n, cost, pos))

    # events in transcript order: edits invalidate earlier reads
    reads, seen_calls = {}, {}
    for tu in sorted(t.uses, key=lambda u: u.pos):
        cat, target, rng = classify(tu)
        r = tu.result
        if cat == "edit":
            for p in re.findall(r"\*\*\* (?:Update|Add|Delete) File: (.+)", target) or [target]:
                reads.pop(_norm(p.strip(), t.cwd), None)
            if r and isinstance(r.extra, dict) and tu.name.lower() == "write" and r.extra.get("originalFile"):
                content = (tu.input or {}).get("content") or ""
                toks = len(content) / CHARS_PER_TOKEN
                changed = sum(len(line) for h in r.extra.get("structuredPatch") or [] if isinstance(h, dict)
                              for line in h.get("lines") or [] if line[:1] in "+-") / CHARS_PER_TOKEN
                if toks > KINDS["rewrite"][0] and changed < 0.5 * toks:
                    add("rewrite", tu, target, toks, toks - changed, tu.pos, r.seg, out_w=w["out"])
            continue
        if cat == "shell":
            if MUTATES_ALL.search(target):
                reads.clear()
            else:
                for p in [p for p in reads if os.path.basename(p) and os.path.basename(p) in target]:
                    reads.pop(p)
        if r is None:
            continue
        if r.text.startswith("Wasted call") or "file unchanged since your last Read" in r.text[:200]:
            single = tu.call is not None and len(tu.call.tools) == 1
            nxt = _next_call(t, tu.call)
            cost = nxt.ctx_in * w["read"] if single and nxt else 0
            add("wasted-call", tu, target, r.tokens, r.tokens, r.pos, r.seg, cost=cost or None)
            continue
        if cat == "read" and target:
            key, body = _norm(target, t.cwd), _body(r.text)
            prev = reads.get(key)
            # a leak only if we got back what was already in context; different text = the file changed
            same = bool(prev) and prev[0] == r.seg and _covers(prev[1], rng) and bool(body) and body in prev[2]
            if same and r.tokens >= 50:
                add("re-read", tu, target, r.tokens, r.tokens, r.pos, r.seg)
                continue
            if not same:
                reads[key] = (r.seg, rng, body)
        elif cat in ("search", "fetch", "subagent", "other"):
            # identical call AND identical answer: pure repeat (stateful tools legitimately differ)
            sig = (tu.name, json.dumps(tu.input, sort_keys=True, default=str))
            prev = seen_calls.get(sig)
            body = _body(r.text)
            digest = hash(body)
            if body and prev == (r.seg, digest) and r.tokens >= 50:  # image-only results can't be compared
                add("wasted-call", tu, target, r.tokens, r.tokens, r.pos, r.seg)
                continue
            seen_calls[sig] = (r.seg, digest)
        kind = SIZE_KIND.get(cat)
        if kind:
            flag, budget, _ = KINDS[kind]
            if r.tokens > flag:
                add(kind, tu, target, r.tokens, r.tokens - budget, r.pos, r.seg)

    # serial round-trips: call a then call b, one read/search each, where b's target was already
    # known before a was issued and nothing in a's result points to it -> could have been one turn
    human, known, ei = t.human_pos, set(), 0
    texts = sorted([(r.pos, r.text) for r in t.results] + t.notes, key=lambda x: x[0])
    for a, b in zip(t.calls, t.calls[1:]):
        while ei < len(texts) and texts[ei][0] < a.pos:
            known.update(WORD.findall(texts[ei][1]))
            ei += 1
        if a.seg != b.seg or len(a.tools) != 1 or len(b.tools) != 1:
            continue
        ta, tb = a.tools[0], b.tools[0]
        ca, tga, _ = classify(ta)
        cb, tgb, _ = classify(tb)
        if ca not in ("read", "search") or cb not in ("read", "search") or not ta.result:
            continue
        if bisect.bisect_left(human, a.pos) != bisect.bisect_left(human, b.pos):
            continue  # the user spoke in between
        key = _serial_key(cb, tgb)
        if not key or key not in known or (ca == cb == "read" and tga == tgb):
            continue
        if key.rsplit(".", 1)[0] in ta.result.text:
            continue  # b plausibly follows from what a returned
        if any(n in ta.result.text for n in re.findall(r"\d{2,}", json.dumps(tb.input, default=str))):
            continue  # e.g. `sed -n 153,160p` / offset taken from a's line numbers
        add("serial-call", tb, tgb, 0, 0, b.pos, b.seg, cost=b.cread * w["read"])

    flag, budget, _ = KINDS["verbose-reply"]
    for c in t.calls:
        toks = c.text_chars / CHARS_PER_TOKEN
        if toks > flag:
            add("verbose-reply", None, "", toks, toks - budget, c.pos, c.seg, out_w=w["out"])

    return findings, sum(call_cost(t, c, w) for c in t.calls)


WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _body(text):
    """Result text without harness-injected reminders, for content comparisons."""
    return REMINDER.sub("", text or "").strip()


def _serial_key(cat, target):
    if cat == "read":
        return os.path.basename(str(target).rstrip("/"))
    words = [w for w in re.findall(r"[A-Za-z0-9_]{4,}", str(target)) if w not in ("grep", "find", "head", "tail")]
    return max(words, key=len) if words else ""


def _next_call(t, call):
    if call is None:
        return None
    i = t.calls.index(call)
    return t.calls[i + 1] if i + 1 < len(t.calls) else None


def _norm(path, cwd):
    if not path:
        return ""
    path = os.path.expanduser(str(path))
    if cwd and not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.normpath(path)


# ---------------------------------------------------------------- discovery

def roots():
    claude = os.path.join(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude"), "projects")
    codex = os.path.join(os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex"), "sessions")
    return {"claude": claude, "codex": codex}


def find_files(paths, agent, since):
    out = []
    targets = paths or [r for a, r in roots().items() if agent in ("all", a)]
    for root in targets:
        if os.path.isfile(root):
            out.append(root)
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".jsonl"):
                    p = os.path.join(dirpath, f)
                    try:
                        if os.path.getmtime(p) >= since:
                            out.append(p)
                    except OSError:
                        pass
    return list(dict.fromkeys(os.path.realpath(p) for p in out))


def parse_since(s):
    if not s or s == "all":
        return 0.0
    m = re.fullmatch(r"(\d+)([hdw])", s)
    if m:
        return time.time() - int(m.group(1)) * {"h": 3600, "d": 86400, "w": 604800}[m.group(2)]
    ts = epoch(s) if re.search(r"[T ]\d|Z$|[+-]\d\d:?\d\d$", s) else None
    if ts is None:
        try:
            ts = datetime.fromisoformat(s).timestamp()  # plain dates = local midnight
        except ValueError:
            raise ValueError(f"can't read {s!r}: use 24h, 7d, 4w, 2026-10-01, 2026-10-01T09:00 or all") from None
    return ts


# ---------------------------------------------------------------- report

def fmt(n):
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def pct_str(p):
    return f"{p:.1f}%" if p >= 0.1 else f"{p:.2f}%" if p >= 0.01 else "<0.01%"


def grade(pct):
    for g, lim in (("A", 5), ("B", 10), ("C", 20), ("D", 30)):
        if pct < lim:
            return g
    return "F"


class Style:
    def __init__(self, on):
        self.on = on

    def __call__(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else str(s)


_INSTR_CACHE = {}


def always_loaded(agent, cwd):
    """Instruction files the agent loads into every call for this working dir: [(path, tokens)]."""
    key = (agent, cwd)
    if key in _INSTR_CACHE:
        return _INSTR_CACHE[key]
    home = os.path.expanduser("~")
    if agent == "claude":
        cfg = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(home, ".claude")
        cands = [os.path.join(cfg, "CLAUDE.md")]
        names = ("CLAUDE.md", "CLAUDE.local.md", os.path.join(".claude", "CLAUDE.md"))
        if cwd:
            cands.append(os.path.join(roots()["claude"], re.sub(r"[^A-Za-z0-9]", "-", cwd), "memory", "MEMORY.md"))
    else:
        cands = [os.path.join(os.environ.get("CODEX_HOME") or os.path.join(home, ".codex"), "AGENTS.md")]
        names = ("AGENTS.md", "AGENTS.override.md")
    d = cwd
    while d:
        cands += [os.path.join(d, n) for n in names]
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    out, seen = [], set()
    for f in cands:
        real = os.path.realpath(f)
        if real in seen or not os.path.isfile(f):
            continue
        seen.add(real)
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                size = len(fh.read()) / CHARS_PER_TOKEN
            if size >= 1:
                out.append((f, size))
        except OSError:
            pass
    _INSTR_CACHE[key] = out
    return out


def project_name(t):
    return os.path.basename((t.cwd or "").rstrip("/")) or "?"


def collect(args):
    """Parse and filter the transcripts the CLI flags point at -> (threads, since)."""
    if args.compare and args.since is None:  # default: a window of the same length before the split
        split = parse_since(args.compare)
        since = split - (time.time() - split)
    else:
        since = 0.0 if args.paths and args.since is None else parse_since(args.since or "7d")
    files = find_files(args.paths, args.agent, since)
    here = os.path.realpath(os.getcwd())
    threads = []
    parsed = []
    for f in files:
        try:
            parsed += parse(f)
        except (OSError, UnicodeError) as e:
            print(f"tokenleak: skipping {f}: {e}", file=sys.stderr)
    for t in parsed:
        f = t.path
        if args.agent not in ("all", t.agent):
            continue
        if t.calls:
            t.start_ctx = t.calls[0].ctx_in
        if since and t.calls:  # files are picked by mtime; only count the calls inside the window
            keep = [c for c in t.calls if c.ts is None or c.ts >= since]
            if len(keep) < len(t.calls):
                if t.cost_usd is not None:  # the recorded cost covers the whole session: keep the window's share
                    w = weights(t)
                    full = sum(call_cost(t, c, w) for c in t.calls)
                    t.cost_usd = t.cost_usd * sum(call_cost(t, c, w) for c in keep) / full if full else None
                first = keep[0].pos if keep else float("inf")
                t.calls = keep
                t.uses = [u for u in t.uses if u.pos >= first]
                t.results = [r for r in t.results if r.pos >= first]
        if not t.calls:
            continue
        if args.here or args.current:
            cwd = os.path.realpath(t.cwd) if t.cwd else ""
            if not (cwd == here or cwd.startswith(here + os.sep)):
                continue
        if args.project and args.project.lower() not in (t.cwd or "").lower():
            continue
        if args.session and args.session not in (t.session or "") and args.session not in f:
            continue
        threads.append(t)
    if args.current and threads:
        latest = max((t for t in threads if not t.subagent), key=lambda t: t.mtime, default=None)
        threads = [t for t in threads if latest and t.session == latest.session]
    return threads, since


COMPACT_SETTINGS = (200_000, 300_000, 400_000, 500_000)  # Claude Code accepts 100k-1M for autoCompactWindow
COMPACT_FIRES_BELOW = 30_000  # measured: auto-compact fires ~30k tokens under the window setting


def compact_whatif(threads, setting):
    """-> (cost units saved, compactions added) had Claude Code auto-compacted at `setting` (autoCompactWindow).

    Replays each main session's recorded context sizes. Every compaction is charged: its summary call reads the whole
    context uncached and writes 4k tokens, the session restarts at its start context plus 15k (summary and kept turns,
    newly cached) and spends 10k re-reading what it lost. Calibrated on measured runs (bench/results, 2026-10-13): it
    matched a short series (+60% predicted, +59% measured) and overstated the cost on a longer one (+46% vs +2%).
    """
    fire, saved, added = setting - COMPACT_FIRES_BELOW, 0.0, 0
    for t in threads:
        if t.agent != "claude" or t.subagent or not t.calls:
            continue
        w = weights(t)
        new = (t.start_ctx or t.calls[0].ctx_in) + 15_000
        segs = {}
        for c in t.calls:
            segs.setdefault(c.seg, []).append(c)
        for calls in segs.values():
            off = 0
            for c in calls:
                ctx = c.ctx_in - off
                if ctx > fire > new:
                    added += 1
                    saved -= ctx + w["out"] * 4_000 + w["enter"] * 15_000 + 10_000
                    off, ctx = c.ctx_in - new, new
                saved += w["read"] * min(c.cread, c.ctx_in - ctx)
    return saved, added


def compact_setting():
    """The auto-compact window this machine already sets, if any (env wins over settings, as in Claude Code)."""
    env = os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    if env:
        return env
    path = os.path.join(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude"), "settings.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("autoCompactWindow")
    except (OSError, ValueError, AttributeError):
        return None


def quick_calls(path):
    """Main-session calls of one Claude Code transcript, reading only assistant usage lines (fast enough for a hook)."""
    t, seen, seg = Thread(path, "claude"), {}, 0
    with open(path, "rb") as f:
        for pos, line in enumerate(f):
            if b'"compact_boundary"' in line:
                seg += 1
            elif b'"assistant"' in line and b'"usage"' in line and b'"isSidechain":true' not in line:
                try:
                    m = json.loads(line).get("message") or {}
                    u = m["usage"]
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
                c = seen.get(m.get("id")) or Call(pos, seg)
                seen[m.get("id")] = c
                c.fresh = max(c.fresh, u.get("input_tokens") or 0)
                c.cread = max(c.cread, u.get("cache_read_input_tokens") or 0)
                c.c5m = max(c.c5m, u.get("cache_creation_input_tokens") or 0)
                c.out = max(c.out, u.get("output_tokens") or 0)
    t.calls = sorted(seen.values(), key=lambda c: c.pos)
    t.start_ctx = t.calls[0].ctx_in if t.calls else None
    return t


def compact_tip(event, now=None):
    """SessionStart, Claude Code only: when this project's recent sessions show an earlier auto-compact would have
    paid (10% or more), say so once a week, to the user (never to the model). Checks at most once a day."""
    now = now or time.time()
    path = str((event or {}).get("transcript_path") or "")
    if os.environ.get("TOKENLEAK_TIPS") == "0" or compact_setting() or "/.claude" not in path.replace(os.sep, "/") \
            and not os.environ.get("CLAUDE_CONFIG_DIR"):
        return None
    cache = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "tokenleak")
    state_path = os.path.join(cache, "state.json")
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    if now - state.get("checked", 0) < 86400 or now - state.get("tipped", 0) < 7 * 86400:
        return None
    state["checked"] = now
    folder = os.path.dirname(path)
    files = sorted((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jsonl")),
                   key=os.path.getmtime, reverse=True) if os.path.isdir(folder) else []
    threads, budget = [], 80 << 20  # newest sessions first, up to ~80 MB of transcript
    for f in files:
        if budget <= 0 or now - os.path.getmtime(f) > 14 * 86400:
            break
        budget -= os.path.getsize(f)
        t = quick_calls(f)
        if t.calls:
            threads.append(t)
    units = sum(call_cost(t, c, weights(t)) for t in threads for c in t.calls)
    cw = context_window(threads, units) if units else None
    tip = None
    if cw and cw["best"] and cw["best"]["saved_pct"] >= 10:
        b = cw["best"]
        state["tipped"] = now
        tip = (f"tokenleak: in your recent sessions here, {cw['over_200k_pct']:.0f}% of calls re-sent over 200k tokens of "
               f"context. Replayed with an earlier auto-compact (autoCompactWindow {b['setting']}), they would have cost "
               f"about {b['saved_pct']:.0f}% less. Set it in /config, Auto-compact window; /tokenleak audits the details. "
               f"(TOKENLEAK_TIPS=0 to silence)")
    try:
        os.makedirs(cache, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError:
        pass
    return tip


def compact_report(cw, apply=False, out=None):
    """`tokenleak compact`: the replay table, and with --apply the recommended autoCompactWindow written to settings."""
    out = out or sys.stdout
    if not cw:
        out.write("tokenleak: no Claude Code sessions in this window.\n")
        return 1
    out.write(f"{cw['over_200k_pct']:.0f}% of {fmt(cw['calls'])} main-session calls carried over 200k tokens of context "
              f"(max {fmt(cw['max'])}). Replayed with an earlier auto-compact:\n\n  autoCompactWindow    cost   more compactions\n")
    best = cw["best"]
    for o in cw["options"]:
        mark = "   <- recommended" if best and o["setting"] == best["setting"] else ""
        out.write(f"  {o['setting']:>17}  {-o['saved_pct']:>+5.0f}%   {o['compactions']:>16}{mark}\n")
    out.write(f"\n  now: {cw['current'] or 'not set (Claude Code compacts near the model limit)'}\n")
    if not best:
        out.write("  Nothing to gain on these sessions: leave it as it is.\n")
        return 0
    if str(cw["current"]) == str(best["setting"]):
        out.write("  Already set.\n")
        return 0
    if not apply:
        out.write("  Apply it: tokenleak compact --apply (writes ~/.claude/settings.json, keeps a backup), or /config\n")
        return 0
    if os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW"):
        out.write("  CLAUDE_CODE_AUTO_COMPACT_WINDOW is set and wins over settings: change it there.\n")
        return 1
    path = os.path.join(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude"), "settings.json")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        data = json.loads(text or "{}")
    except FileNotFoundError:
        text, data = "", {}
    except (OSError, ValueError) as e:
        out.write(f"  Couldn't read {path} ({e}); nothing changed.\n")
        return 1
    if text:
        with open(path + ".bak-tokenleak", "w", encoding="utf-8") as f:
            f.write(text)
    data["autoCompactWindow"] = best["setting"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    out.write(f"  Set \"autoCompactWindow\": {best['setting']} in {path}" + (" (backup: .bak-tokenleak)" if text else "")
              + ". New sessions use it.\n")
    return 0


def run(args):
    threads, since = collect(args)
    return summarize(threads) + (since,)


def context_window(threads, units):
    """How big the main sessions' contexts got, and what an earlier auto-compact would have saved on them."""
    ctx = [c.ctx_in for t in threads if t.agent == "claude" and not t.subagent for c in t.calls]
    if not ctx or not units:
        return None
    options = []
    for setting in COMPACT_SETTINGS:
        saved, added = compact_whatif(threads, setting)
        options.append({"setting": setting, "saved_pct": round(100 * saved / units, 1), "compactions": added})
    top = max(o["saved_pct"] for o in options)
    best = max((o for o in options if o["saved_pct"] >= top - 2), key=lambda o: o["setting"])  # fewest compactions
    return {"calls": len(ctx), "over_200k_pct": round(100 * sum(1 for x in ctx if x > 200_000) / len(ctx), 1),
            "max": max(ctx), "options": options, "best": best if best["saved_pct"] >= 5 else None,
            "current": compact_setting()}


def summarize(threads):
    """-> (totals, by_kind, findings sorted by cost) for a set of threads."""
    findings, units, sess, loaded, firsts, base_cost = [], 0.0, {}, {}, [], 0.0
    projects, per_thread = {}, []
    split = {"cache_read": 0.0, "cache_write": 0.0, "output": 0.0}  # cost units; cache_write includes uncached input
    for t in threads:
        try:
            fs, u = analyze(t)
        except Exception as e:  # one odd transcript must not sink the whole report
            print(f"tokenleak: skipped {t.path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        key = (t.agent, t.session or t.path)
        per_thread.append((t.cwd or "?", key, u, sum(f.cost for f in fs)))
        pj = projects.setdefault(t.cwd or "?", {"units": 0.0, "leak": 0.0, "sessions": set()})
        pj["units"] += u
        pj["leak"] += sum(f.cost for f in fs)
        pj["sessions"].add(key)
        if not t.subagent:
            w = weights(t)
            n, nseg = len(t.calls), len({c.seg for c in t.calls})
            first = t.calls[0]
            start = t.start_ctx if t.start_ctx is not None else first.ctx_in
            firsts.append(start)
            if start == first.ctx_in:  # what the first call really paid for its input, then a cache read per call
                base_cost += call_cost(t, first, w) - w["out"] * first.out + start * w["read"] * (n - 1)
            else:  # session started before the window: every counted call re-read the start context
                base_cost += start * w["read"] * n
            files = ([(f, n / CHARS_PER_TOKEN, True) for f, n in t.instructions.items()]
                     if t.agent == "claude" and t.attachments
                     else [(f, toks, False) for f, toks in always_loaded(t.agent, t.cwd)])
            for f, toks, exact in files:
                e = loaded.setdefault(f, {"tokens": toks, "cost": 0.0, "sessions": 0, "exact": exact})
                e["cost"] += toks * (nseg * w["enter"] + (n - nseg) * w["read"])
                e["sessions"] += 1
        findings += fs
        units += u
        key = (t.agent, t.session or t.path)
        s = sess.setdefault(key, {"units": 0.0, "leak": 0.0, "usd": None})
        s["units"] += u
        s["leak"] += sum(f.cost for f in fs)
        if t.cost_usd is not None:
            s["usd"] = max(s["usd"] or 0, t.cost_usd)
        w = weights(t)
        for c in t.calls:
            split["cache_read"] += w["read"] * c.cread
            split["output"] += w["out"] * c.out
            split["cache_write"] += call_cost(t, c, w) - w["read"] * c.cread - w["out"] * c.out
            c.tools = []  # free memory early: transcripts can be huge
        for r in t.results:
            r.text = ""
    leak = sum(f.cost for f in findings)
    pct = 100 * leak / units if units else 0.0
    by_project = []
    usd_by_cwd = {}  # a session's $ split over its threads, so a subagent in a subfolder isn't counted twice
    for cwd, key, u, lk in per_thread:
        sk = sess.get(key)
        if sk and sk["usd"] and sk["units"]:
            usd_by_cwd[cwd] = usd_by_cwd.get(cwd, 0.0) + sk["usd"] * lk / sk["units"]
    for cwd, pj in projects.items():
        usd = usd_by_cwd.get(cwd, 0.0)
        by_project.append({"cwd": cwd, "sessions": len(pj["sessions"]), "cost_units": pj["units"],
                           "leak_pct": 100 * pj["leak"] / pj["units"] if pj["units"] else 0, "usd_leak": usd or None,
                           "leak_units": pj["leak"]})
    by_project.sort(key=lambda p: -p["leak_units"])
    usd_total = sum(s["usd"] for s in sess.values() if s["usd"])
    usd_leak = sum(s["usd"] * s["leak"] / s["units"] for s in sess.values() if s["usd"] and s["units"])
    totals = {
        "threads": len(threads),
        "sessions": len(sess),
        "claude_sessions": sum(1 for a, _ in sess if a == "claude"),
        "codex_sessions": sum(1 for a, _ in sess if a == "codex"),
        "api_calls": sum(len(t.calls) for t in threads),
        "context_tokens": sum(c.ctx_in for t in threads for c in t.calls),
        "cache_read_tokens": sum(c.cread for t in threads for c in t.calls),
        "cache_write_tokens": sum(c.c5m + c.c1h for t in threads for c in t.calls),
        "fresh_tokens": sum(c.fresh for t in threads for c in t.calls),
        "output_tokens": sum(c.out for t in threads for c in t.calls),
        "cost_units": units,
        "cost_split_pct": {k: round(100 * v / units, 1) if units else 0
                           for k, v in dict(split, context=split["cache_read"] + split["cache_write"]).items()},
        "leak_units": leak,
        "leak_pct": pct,
        "grade": grade(pct),
        "usd_total": usd_total or None,
        "usd_sessions": sum(1 for s in sess.values() if s["usd"]),
        "usd_leak": usd_leak or None,
        "avg_first_call_context": sum(firsts) / len(firsts) if firsts else 0,
        "base_context_pct": 100 * base_cost / units if units else 0,
        "always_loaded": sorted(({"path": f, **e} for f, e in loaded.items()), key=lambda e: -e["cost"]),
        "projects": by_project,
        "context_window": context_window(threads, units),
    }
    by_kind = {}
    for f in findings:
        k = by_kind.setdefault(f.kind, {"count": 0, "avoidable_tokens": 0.0, "cost": 0.0})
        k["count"] += 1
        k["avoidable_tokens"] += f.avoidable
        k["cost"] += f.cost
    findings.sort(key=lambda f: -f.cost)
    return totals, by_kind, findings


def tool_name(tool):
    """mcp__<server>__<tool> -> server:tool, dropping a server id that is only a UUID."""
    parts = str(tool).split("__")
    if len(parts) < 3 or parts[0] != "mcp":
        return str(tool)
    server = "" if re.fullmatch(r"[0-9a-f-]{16,}", parts[1]) else parts[1]
    return f"{server or 'mcp'}:{'__'.join(parts[2:])}"


def label_of(f, redact):
    if f.kind == "verbose-reply":
        return "assistant reply"
    tool = tool_name(f.tool)
    if redact:
        return tool
    lab = re.sub(r"\s+", " ", str(f.label)).strip()
    cwd = (f.thread.cwd or "").rstrip("/")
    if cwd:  # paths relative to the session's project
        lab = re.sub(r'cd "?' + re.escape(cwd) + r'"? *&& *', "", lab).replace(cwd + "/", "")
    lab = lab.replace(os.path.expanduser("~"), "~")
    return f"{tool} {lab}" if lab and lab != f.tool else tool


def fit(lab, width):
    """Shorten a label to width; a path keeps its end (the file name), anything else its start."""
    if len(lab) <= width:
        return lab
    tool, _, rest = lab.partition(" ")
    if rest and " " not in rest and ("/" in rest or "\\" in rest) and width - len(tool) - 2 > 0:
        return f"{tool} …{rest[len(rest) - (width - len(tool) - 2):]}"
    return lab[: width - 1] + "…"


def print_report(totals, by_kind, findings, since, args, out=None):
    out = out or sys.stdout
    st = Style((out.isatty() or bool(os.environ.get("FORCE_COLOR"))) and not args.no_color
               and not os.environ.get("NO_COLOR"))
    w = out.write
    if not totals["threads"]:
        w("tokenleak: no transcripts found for this filter. Try --since 30d, --agent all, or pass a path.\n")
        return
    src = ", ".join(x for x in (f"{totals['claude_sessions']} Claude Code" if totals["claude_sessions"] else "",
                                f"{totals['codex_sessions']} Codex" if totals["codex_sessions"] else "") if x)
    when = datetime.fromtimestamp(since).strftime("%Y-%m-%d") if since else "the beginning"
    w(f"\n{st('1', 'tokenleak')} {__version__}  {st('2', '·')}  {totals['sessions']} session{'s' * (totals['sessions'] != 1)} ({src})  "
      f"{st('2', '·')}  {fmt(totals['api_calls'])} API calls  {st('2', '·')}  since {when}\n\n")
    w(f"  context processed  {st('1', fmt(totals['context_tokens']))} tokens  "
      f"{st('2', '(cache reads ' + fmt(totals['cache_read_tokens']) + ' · writes ' + fmt(totals['cache_write_tokens']) + ' · fresh ' + fmt(totals['fresh_tokens']) + ')')}"
      f"   output {fmt(totals['output_tokens'])}\n")
    if totals["usd_total"]:
        n_usd = f"{totals['usd_sessions']} of {totals['sessions']} sessions record a cost"
        w(f"  spend              ≈ ${totals['usd_total']:,.2f} API-equivalent {st('2', '(Claude Code pricing; ' + n_usd + ')')}\n")
    sp = totals["cost_split_pct"]
    w(f"  cost split         context {sp['context']:.0f}% {st('2', '(cache reads ' + format(sp['cache_read'], '.0f') + '% · writes and uncached ' + format(sp['cache_write'], '.0f') + '%)')}"
      f"   output {sp['output']:.0f}%\n")
    pct, g = totals["leak_pct"], totals["grade"]
    color = {"A": "32", "B": "32", "C": "33", "D": "31", "F": "31"}[g]
    money = f"   ≈ ${totals['usd_leak']:,.2f} avoidable" if totals["usd_leak"] else ""
    w(f"\n  {st('1', 'LEAK SCORE')}  {st('1;' + color, f'{pct:.1f}%')}  grade {st('1;' + color, g)}{money}"
      f"   {st('2', '% of token cost that was avoidable · A<5 B<10 C<20 D<30')}\n\n")
    if not by_kind:
        w("  No leaks found. Your agent is lean.\n")
    else:
        w(st("2", f"  {'leak':<14}{'count':>7}{'avoidable':>12}{'of cost':>9}   fix\n"))
    for kind, k in sorted(by_kind.items(), key=lambda kv: -kv[1]["cost"]):
        share = 100 * k["cost"] / totals["cost_units"] if totals["cost_units"] else 0
        avoid = fmt(k["avoidable_tokens"]) if k["avoidable_tokens"] else "-"
        w(f"  {kind:<14}{k['count']:>7}{avoid:>12}{share:>8.1f}%   {st('2', KINDS[kind][2])}\n")
    tmp = tuple(os.path.realpath(d) + os.sep for d in {tempfile.gettempdir(), "/tmp", "/var/folders"} if os.path.isdir(d))
    repeat = [pj for pj in totals["projects"] if pj["sessions"] >= 2  # one-off and scratch folders are noise here
              and not os.path.realpath(pj["cwd"]).startswith(tmp)]
    if len(repeat) >= 2 and not args.redact:
        w(f"\n  {st('1', 'by project')}\n")
        home = os.path.expanduser("~")
        for pj in repeat[:6]:
            name = pj["cwd"].replace(home, "~")
            name = name if len(name) <= 44 else "…" + name[-43:]
            usd = f"≈ ${pj['usd_leak']:,.2f} avoidable" if pj["usd_leak"] else ""
            n = f"{pj['sessions']} sessions"
            w(f"  {pj['leak_pct']:>6.1f}%  {name:<44} {st('2', f'{n:>12}')}  {st('2', usd)}\n")
    cw = totals.get("context_window")
    if cw and cw["best"]:
        b = cw["best"]
        w(f"\n  {st('1', 'context window')}  {cw['over_200k_pct']:.0f}% of calls carried over 200k tokens of context"
          f" {st('2', '(max ' + fmt(cw['max']) + ')')}\n")
        key = '"autoCompactWindow": ' + str(b["setting"])
        gain = format(-b["saved_pct"], "+.0f") + "% cost"
        more = "(" + str(b["compactions"]) + " more compactions)"
        w(f"  replayed on these sessions: {st('1', key)} → {st('1;32', gain)} {st('2', more)}\n")
        now = f" (now: {cw['current']})" if cw["current"] else ""
        w(st("2", f"  set it with /config → Auto-compact window, or in ~/.claude/settings.json{now}\n"))
    if totals["always_loaded"] or totals["avg_first_call_context"]:
        note = (fmt(totals["avg_first_call_context"]) + " tok at session start (system prompt, tools, MCP"
                + (", files below" if totals["always_loaded"] else "") + "), re-sent every call")
        w(f"\n  {st('1', 'always loaded')}  ~{totals['base_context_pct']:.0f}% of cost  {st('2', note)}\n")
        home = os.path.expanduser("~")
        for e in totals["always_loaded"][:5]:
            share = pct_str(100 * e["cost"] / totals["cost_units"] if totals["cost_units"] else 0)
            path = e["path"] if not args.redact else os.path.basename(e["path"])
            path = path.replace(home, "~")
            path = path if len(path) <= 60 else "…" + path[-59:]
            size = fmt(e["tokens"]) + " tok (" + ("as loaded" if e.get("exact") else "current size") + ")"
            w(f"  {share:>7}  {path:<60} {st('2', size)}\n")
    if findings:
        w(f"\n  {st('1', 'top leaks')}\n")
    width = 46
    for i, f in enumerate(findings[: args.top], 1):
        share = pct_str(100 * f.cost / totals["cost_units"] if totals["cost_units"] else 0)
        lab = fit(label_of(f, args.redact), width)
        if f.kind == "serial-call":
            size = "extra round-trip"
        else:
            size = f"{fmt(f.tokens)} tok" + (f" × {f.later} later calls" if f.later else "")
        where = "" if args.redact else f"{project_name(f.thread)[:22]} · {(f.thread.session or '')[:8]}"
        w(f"  {i:>2}  {share:>6}  {st('33', f'{f.kind:<13}')} {lab:<{width}}  {st('2', size)}  {st('2', where)}\n")
    w(st("2", "\n  estimates: tokens ≈ chars/4 · avoidable = size above a sane budget, priced on every later call that"
              "\n  re-sent it (cache read 0.1x, write 1.25-2x, output 5x) · everything stays on this machine\n"))
    w(f"\n  Plug the leaks: {REPO}\n\n")


def epoch(iso):
    """ISO-8601 (with Z or nanoseconds, any Python 3.8+) or epoch seconds/ms -> epoch seconds, else None."""
    if iso is None:
        return None
    if isinstance(iso, (int, float)) and not isinstance(iso, bool):
        return iso / 1000 if iso > 1e11 else float(iso)
    text = re.sub(r"(\.\d{6})\d+", r"\1", str(iso).strip().replace("Z", "+00:00"))
    text = re.sub(r"\.(\d{1,5})(?=[+-]|$)", lambda m: "." + m.group(1).ljust(6, "0"), text)
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def compare(args):
    """Leak score before vs after a date: did installing the rules (or anything else) help?"""
    threads, since = collect(args)
    split = parse_since(args.compare)
    # a subagent belongs to its parent session's period
    starts = {}
    for t in threads:
        ts = epoch(t.start) or t.mtime
        key = (t.agent, t.session or t.path)
        starts[key] = min(starts.get(key, ts), ts)
    before = [t for t in threads if starts[(t.agent, t.session or t.path)] < split]
    after = [t for t in threads if starts[(t.agent, t.session or t.path)] >= split]
    return summarize(before), summarize(after), since, split


def top_kind(by_kind):
    return max(by_kind, key=lambda k: by_kind[k]["cost"]) if by_kind else "none"


def print_compare(a, b, since, split, args, out=None):
    out = out or sys.stdout
    st = Style((out.isatty() or bool(os.environ.get("FORCE_COLOR"))) and not args.no_color
               and not os.environ.get("NO_COLOR"))
    (ta, ka, _), (tb, kb, _) = a, b
    day = lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d")
    out.write(f"\n{st('1', 'tokenleak')} {__version__}  {st('2', '·')}  before vs after {day(split)}"
              f"  {st('2', '(sessions from ' + day(since) + ')')}\n\n")
    if not ta["threads"] or not tb["threads"]:
        side = "before" if not ta["threads"] else "after"
        out.write(f"  No sessions {side} {day(split)} in this window. Widen it with --since.\n\n")
        return
    per_call = lambda t: t["cost_units"] / t["api_calls"] if t["api_calls"] else 0
    rows = [
        ("sessions", str(ta["sessions"]), str(tb["sessions"]), ""),
        ("API calls", fmt(ta["api_calls"]), fmt(tb["api_calls"]), ""),
        ("leak score", f"{ta['leak_pct']:.1f}% {ta['grade']}", f"{tb['leak_pct']:.1f}% {tb['grade']}",
         f"{tb['leak_pct'] - ta['leak_pct']:+.1f} pts"),
        ("cost per call", fmt(per_call(ta)), fmt(per_call(tb)),
         f"{100 * (per_call(tb) - per_call(ta)) / per_call(ta):+.0f}%" if per_call(ta) else ""),
        ("top leak", top_kind(ka), top_kind(kb), ""),
    ]
    if ta["usd_leak"] or tb["usd_leak"]:
        rows.insert(3, ("avoidable spend", f"${ta['usd_leak'] or 0:,.2f}", f"${tb['usd_leak'] or 0:,.2f}", ""))
    out.write(st("2", f"  {'':<17}{'before':>16}{'after':>16}{'change':>12}\n"))
    for name, x, y, d in rows:
        color = "32" if d.startswith("-") else "31" if d.startswith("+") else "2"
        out.write(f"  {name:<17}{x:>16}{y:>16}{st(color, f'{d:>12}')}\n")
    out.write(st("2", "\n  cost per call = token cost of one API call (input-equivalent tokens: cache read 0.1x,"
                      "\n  write 1.25-2x, output 5x). Task mix differs between periods: read it as a trend.\n\n"))


HTML_CSS = """
:root{--bg:#09090b;--panel:#111114;--line:rgba(255,255,255,.09);--ink:#f4f4f5;--ink2:#c6c6cd;--muted:#8b8b95;--amber:#f5a524;--amber2:#ffc46b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
.w{max-width:1040px;margin:0 auto;padding:40px 20px 56px}code,.m{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
h1{font-size:15px;margin:0;font-weight:650}.meta{color:var(--muted);font-size:13px;margin:4px 0 0}
.hero{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:28px 0}
.card{background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.02));border:1px solid var(--line);border-radius:16px;padding:18px 20px}
.k{color:var(--muted);font-size:12.5px}.v{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:28px;letter-spacing:-.03em;margin-top:4px}
.v.big{font-size:40px;color:var(--amber2)}.g{display:inline-block;margin-left:8px;font-size:15px;padding:2px 9px;border-radius:999px;border:1px solid rgba(245,165,36,.4);color:var(--amber2);vertical-align:middle}
h2{font-size:16px;margin:30px 0 12px;font-weight:650}
.row{display:grid;grid-template-columns:130px 70px minmax(0,1fr);gap:14px;align-items:center;padding:9px 0;border-top:1px solid var(--line)}
.bar{height:8px;border-radius:4px;background:var(--amber);box-shadow:0 0 10px rgba(245,165,36,.4)}.fix{color:var(--muted);font-size:13px;margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:13px}th{color:var(--muted);font-weight:500;text-align:left;padding:7px 10px 7px 0;border-bottom:1px solid var(--line)}
td{padding:8px 10px 8px 0;border-bottom:1px solid var(--line);color:var(--ink2);vertical-align:top}td.n{font-family:ui-monospace,Menlo,monospace;white-space:nowrap}
.lab{word-break:break-all}.foot{color:var(--muted);font-size:12px;margin-top:30px}a{color:var(--amber2)}
@media(max-width:760px){.hero{grid-template-columns:repeat(2,minmax(0,1fr))}.row{grid-template-columns:100px 56px minmax(0,1fr)}}
"""


def write_html(totals, by_kind, findings, since, args, path):
    """Self-contained, shareable HTML report (no external assets)."""
    esc = html.escape
    when = datetime.fromtimestamp(since).strftime("%Y-%m-%d") if since else "the beginning"
    units = totals["cost_units"] or 1
    kinds = sorted(by_kind.items(), key=lambda kv: -kv[1]["cost"])
    top_share = max((k["cost"] for _, k in kinds), default=1) or 1
    money = f"${totals['usd_leak']:,.2f}" if totals["usd_leak"] else "n/a"
    spend = f"${totals['usd_total']:,.2f}" if totals["usd_total"] else "n/a"
    rows = "".join(
        f'<div class="row"><div><code>{esc(kind)}</code></div><div class="m">{100 * k["cost"] / units:.1f}%</div>'
        f'<div><div class="bar" style="width:{max(2, 100 * k["cost"] / top_share):.0f}%"></div>'
        f'<div class="fix">{k["count"]} found. {esc(KINDS[kind][2])}</div></div></div>' for kind, k in kinds)
    leaks = "".join(
        f'<tr><td class="n">{100 * f.cost / units:.2f}%</td><td><code>{esc(f.kind)}</code></td>'
        f'<td class="lab">{esc(label_of(f, args.redact))}</td><td class="n">{fmt(f.tokens)} tok'
        f'{" x " + str(f.later) + " calls" if f.later else ""}</td></tr>' for f in findings[: args.top or 10])
    projects = "" if args.redact or len(totals["projects"]) < 2 else (
        "<h2>By project</h2><table><tr><th>Leak score</th><th>Project</th><th>Sessions</th><th>Avoidable</th></tr>" + "".join(
            f'<tr><td class="n">{p["leak_pct"]:.1f}%</td><td class="lab">{esc(p["cwd"].replace(os.path.expanduser("~"), "~"))}</td>'
            f'<td class="n">{p["sessions"]}</td><td class="n">{"$%.2f" % p["usd_leak"] if p["usd_leak"] else ""}</td></tr>'
            for p in totals["projects"][:10]) + "</table>")
    cw = totals.get("context_window") or {}
    b = cw.get("best")
    cw_html = (f'<p class="meta">Context window: {cw["over_200k_pct"]:.0f}% of calls carried over 200k tokens. Replayed on '
               f'these sessions, <code>"autoCompactWindow": {b["setting"]}</code> would have cost {b["saved_pct"]:.0f}% less '
               f'({b["compactions"]} more compactions).</p>') if b else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tokenleak report</title><style>{HTML_CSS}</style></head><body><div class="w">
<h1>tokenleak report</h1><p class="meta">{totals['sessions']} sessions since {esc(when)}, {fmt(totals['api_calls'])} API calls, generated {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="hero">
<div class="card"><div class="k">Leak score</div><div class="v big">{totals['leak_pct']:.1f}%<span class="g">{totals['grade']}</span></div></div>
<div class="card"><div class="k">Avoidable spend</div><div class="v">{money}</div></div>
<div class="card"><div class="k">Spend (API-equivalent)</div><div class="v">{spend}</div></div>
<div class="card"><div class="k">Context processed</div><div class="v">{fmt(totals['context_tokens'])}</div></div>
</div>
<p class="meta">Cost split: context {totals['cost_split_pct']['context']:.0f}% (cache reads {totals['cost_split_pct']['cache_read']:.0f}%, cache writes and uncached input {totals['cost_split_pct']['cache_write']:.0f}%), output {totals['cost_split_pct']['output']:.0f}%. Context is what the model reads, output what it writes.</p>
{cw_html}<h2>Where it leaks</h2>{rows or '<p class="meta">No leaks found. Your agent is lean.</p>'}
<h2>Top leaks</h2><table><tr><th>Of cost</th><th>Kind</th><th>What</th><th>Size</th></tr>{leaks}</table>
{projects}
<p class="foot">Leak score: share of token cost spent on avoidable context (A under 5%, B under 10%, C under 20%, D under 30%).
Tokens are estimated at about 4 characters each; avoidable means the part above a sane budget, priced on every later call that re-sent it.
Generated locally by <a href="{REPO}">tokenleak</a> {__version__}.</p></div></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)


def print_share(totals, by_kind, out=None):
    out = out or sys.stdout
    top = [k for k, _ in sorted(by_kind.items(), key=lambda kv: -kv[1]["cost"])][:3]
    rows = [
        f"tokenleak · {totals['sessions']} sessions · {fmt(totals['context_tokens'])} context tokens",
        f"Leak score {totals['leak_pct']:.1f}% (grade {totals['grade']})"
        + (f" · ≈ ${totals['usd_leak']:,.0f} avoidable" if totals["usd_leak"] else ""),
        "Top leaks: " + (", ".join(top) if top else "none"),
        REPO.replace("https://", ""),
    ]
    width = max(len(r) for r in rows) + 2
    out.write("╭" + "─" * width + "╮\n")
    for r in rows:
        out.write("│ " + r.ljust(width - 1) + "│\n")
    out.write("╰" + "─" * width + "╯\n")


def redacted(totals, args):
    if not args.redact:
        return totals
    return dict(totals, always_loaded=[dict(e, path=os.path.basename(e["path"])) for e in totals["always_loaded"]],
                projects=[dict(p, cwd=None) for p in totals["projects"]])


def to_json(totals, by_kind, findings, args):
    totals = redacted(totals, args)
    return {
        "version": __version__,
        "totals": totals,
        "by_kind": by_kind,
        "findings": [{
            "kind": f.kind, "tool": f.tool, "label": None if args.redact else label_of(f, False),
            "tokens": round(f.tokens), "avoidable_tokens": round(f.avoidable), "later_calls": f.later,
            "cost_units": round(f.cost, 1),
            "share_pct": round(100 * f.cost / totals["cost_units"], 3) if totals["cost_units"] else 0,
            "agent": f.thread.agent, "session": None if args.redact else f.thread.session,
            "project": None if args.redact else f.thread.cwd, "line": f.pos,
        } for f in findings[: args.top if args.top else None]],
    }


def statusline(stdin, out=None):
    """Claude Code status line: live leak score of the current session (+ its subagents)."""
    out = out or sys.stdout
    try:
        info = json.loads(stdin.read() or "{}")
    except ValueError:
        info = {}
    path = info.get("transcript_path") or ""
    if not path or not os.path.isfile(path):
        out.write("tokenleak\n")
        return
    files = [path]
    sub = os.path.join(path[: -len(".jsonl")] if path.endswith(".jsonl") else path, "subagents")
    for dirpath, _, names in os.walk(sub):
        files += [os.path.join(dirpath, n) for n in names if n.endswith(".jsonl")]
    threads = [t for f in files for t in parse(f) if t.calls]
    if not threads:
        out.write("tokenleak: waiting for the first call\n")
        return
    totals, by_kind, _ = summarize(threads)
    main = next((t for t in threads if not t.subagent), threads[0])
    g = totals["grade"]
    color = {"A": "32", "B": "32", "C": "33", "D": "31", "F": "31"}[g]
    top = f" · top {top_kind(by_kind)}" if by_kind else ""
    out.write(f"\033[{color}mleak {totals['leak_pct']:.1f}% {g}\033[0m · ctx {fmt(main.calls[-1].ctx_in)}{top}\n")


# ---------------------------------------------------------------- suggest: project rules from your own leaks

TOOL_WRAPPERS = ("npx", "pnpm", "yarn", "bunx", "bun", "uv", "poetry", "pipenv", "python", "python3")
SUBCOMMAND_TOOLS = {"npm", "pnpm", "yarn", "bun", "git", "docker", "cargo", "go", "dotnet", "mvn", "gradle", "./gradlew",
                    "make", "kubectl", "terraform", "gh", "uv", "poetry", "composer", "php", "rails", "bundle", "mix",
                    "vitest", "jest", "playwright", "prisma", "next", "turbo", "nx", "cypress"}
SUBCOMMAND = re.compile(r"^[a-z][a-z0-9:_-]*$")  # `run`, `test:unit`; not paths like `src/a.test.ts`
def _logged(quiet, view='tail -n 40 "$L"', note=""):
    """Advice that bounds a command's output and still shows its exit status (a pipe to tail would hide it)."""
    return '`L="${{TMPDIR:-/tmp}}/out.log"; ' + quiet + ' > "$L" 2>&1; echo "exit=$?"; ' + view + '`' + note


ONE_TEST = " (iterate on the one failing test)"
QUIET = [  # (pattern on the command family, advice); first match wins
    (r"\b(pytest|py\.test)\b", _logged("{cmd} -q -x")),
    (r"\bvitest\b", _logged("{cmd} --reporter=dot", note=ONE_TEST)),
    (r"\bjest\b", _logged("{cmd} --silent", note=ONE_TEST)),
    (r"^(npm|pnpm|yarn|bun) (run )?test", _logged("{cmd} --silent", note=ONE_TEST)),
    (r"^(npm|pnpm|yarn|bun) (install|ci|add|i)\b", _logged("{cmd} --silent", 'tail -n 20 "$L"')),
    (r"^(pip3?|uv pip) install|^uv sync", _logged("{cmd} -q", 'tail -n 20 "$L"')),
    (r"^cargo (test|build|check|clippy)", _logged("{cmd} -q")),
    (r"^go (test|build|vet)", _logged("{cmd}", """grep -v '^ok' "$L" | tail -n 40""")),
    (r"^git status", "`git status -s`"),
    (r"^git (log|reflog)", "`git log --oneline -n 20` first, then `git show --stat <sha>`"),
    (r"^git (diff|show)", "`git diff --stat` first, then `git diff -- <path>` for the files you need"),
    (r"^(tsc|npx tsc|eslint|npx eslint|ruff|mypy|pyright|flake8)\b", _logged("{cmd}", 'head -n 30 "$L"', " (fix the first errors, re-run)")),
    (r"^(curl|wget|http)\b", "`{cmd} -s | head -c 2000` or pipe JSON through `jq` to pick fields"),
]
# commands worth a project rule: build/test/install/lint/vcs tools, not one-off shell snippets
PROJECT_TOOLS = {"npm", "pnpm", "yarn", "bun", "npx", "bunx", "make", "docker", "podman", "cargo", "go", "mvn", "gradle",
                 "./gradlew", "dotnet", "pytest", "py.test", "tox", "nox", "uv", "poetry", "pipenv", "pip", "pip3", "rails",
                 "bundle", "rake", "rspec", "php", "composer", "mix", "terraform", "kubectl", "helm", "tsc", "eslint",
                 "prettier", "ruff", "mypy", "pyright", "flake8", "jest", "vitest", "playwright", "cypress", "git", "gh",
                 "deno", "swift", "xcodebuild", "flutter", "dart", "gcc", "clang", "cmake", "ctest", "ninja", "next",
                 "vite", "webpack", "turbo", "nx", "prisma", "alembic", "dbt"}


def is_project_command(cmd, family):
    head = family.split(" ")[0] if family else ""
    if "<<" in cmd or not head:
        return False
    if head in ("python", "python3"):  # only `python -m <tool>` style invocations
        return " -m " in f" {family} "
    return head in PROJECT_TOOLS


LOG_HINT = _logged("{cmd}")


def command_family(cmd):
    """`cd x && FOO=1 npx vitest run src/a.test.ts` -> `npx vitest run`: what a project rule should name."""
    words = _main_segment(cmd)
    if not words:
        return ""
    words = [words[0].rsplit("/", 1)[-1] if words[0].startswith("/") else words[0]] + words[1:]
    out = [words[0]]
    rest = [w for w in words[1:] if not w.startswith("-") or w == "-m"]
    if words[0] in TOOL_WRAPPERS and rest:
        out.append(rest.pop(0))
        if out[-1] in ("-m", "run", "exec") and rest:
            out.append(rest.pop(0))
    if out[-1] in SUBCOMMAND_TOOLS and rest and SUBCOMMAND.match(rest[0]):
        out.append(rest.pop(0))
        if out[-1] in ("run", "compose", "exec") and rest and SUBCOMMAND.match(rest[0]):
            out.append(rest.pop(0))
    return " ".join(out)


def quiet_advice(family):
    for pat, advice in QUIET:
        if re.search(pat, family):
            return advice.format(cmd=family)
    return LOG_HINT.format(cmd=family)


def inside(path, root):
    """Is `path` under `root`? Separator- and drive-aware (Windows too)."""
    p, r = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
    try:
        return p != r and os.path.commonpath([p, r]) == r
    except ValueError:  # different drives
        return False


def suggest(threads, out=None, since=0.0):
    """Turn this project's recurring leaks into rules to paste into CLAUDE.md / AGENTS.md."""
    out = out or sys.stdout
    totals, by_kind, findings = summarize(threads)
    if not totals["threads"]:
        out.write("tokenleak: no sessions found for this project. Run it from the project folder, or widen --since.\n")
        return
    cwds = [t.cwd for t in threads if t.cwd]
    root = max(set(cwds), key=cwds.count) if cwds else ""
    rel = lambda p: os.path.relpath(p, root) if root and os.path.isabs(str(p)) and str(p).startswith(root) else str(p)

    shells, files, searches = {}, {}, {}
    for f in findings:
        if f.kind == "noisy-shell":
            fam = command_family(str(f.label))
            if is_project_command(str(f.label), fam):
                e = shells.setdefault(fam, [0, 0.0, 0.0])
                e[0] += 1; e[1] += f.tokens; e[2] += f.cost
        elif f.kind in ("big-read", "re-read"):
            path = _norm(f.label, f.thread.cwd)
            if not root or not inside(path, root) or not os.path.isfile(path):
                continue  # only files that exist in this project (relative reads after a `cd` can point elsewhere)
            key = rel(path)
            e = files.setdefault(key, [0, 0.0, 0.0])
            e[0] += 1; e[1] = max(e[1], f.tokens); e[2] += f.cost
        elif f.kind == "wide-search":
            words = _main_segment(str(f.label)) if f.tool.lower() in SHELL_TOOLS else [f.tool]
            e = searches.setdefault(words[0] if words else f.tool, [0, 0.0, 0.0])
            e[0] += 1; e[1] += f.tokens; e[2] += f.cost

    def top(d, n):
        return sorted(d.items(), key=lambda kv: -kv[1][2])[:n]

    lines = []
    for fam, (n, toks, _) in top(shells, 5):
        if n >= 2 or toks >= 6000:
            lines.append(f"- `{fam}` printed ~{fmt(toks / n)} tokens per run ({n}x). Run it as {quiet_advice(fam)}.")
    big = [(p, n, toks) for p, (n, toks, _) in top(files, 6) if n >= 2 or toks >= 6000]
    if big:
        listed = ", ".join(f"`{p}` (~{fmt(toks)} tokens, read {n}x)" for p, n, toks in big)
        lines.append(f"- Large files: {listed}. `grep -n` them, then read only the lines you need.")
    for tool, (n, toks, _) in top(searches, 1):
        if n >= 2:
            lines.append(f"- `{tool}` searches returned ~{fmt(toks / n)} tokens each ({n}x): scope them to a folder or file type, cap results, use `-l` for names only.")

    name = os.path.basename(root.rstrip("/")) if root else "this project"
    window = datetime.fromtimestamp(since).strftime("%Y-%m-%d") if since else "the beginning"
    spend = f", about ${totals['usd_leak']:,.2f} of avoidable spend" if totals["usd_leak"] else ""
    out.write(f"<!-- tokenleak suggest: {totals['sessions']} session{'s' * (totals['sessions'] != 1)} in {name} since {window}, "
              f"leak score {totals['leak_pct']:.1f}%{spend} -->\n")
    if not lines:
        out.write("No project-specific rules to add. Run `tokenleak --here` for the session-level leaks.\n")
        return
    out.write("## Token hygiene for this repo\n\n")
    out.write("Keep tool output out of the context; these are this repo's biggest repeat offenders:\n\n")
    out.write("\n".join(lines) + "\n")


GUARD_SKIP = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg", ".pdf", ".ipynb")
# Lines worth a jump in a big file: definitions in most languages (a keyword, a name, then what follows a name in
# code, not in prose like "class that ..."), module-level constants, Markdown headings.
OUTLINE_CODE = re.compile(
    r"^[ \t]*(?:(?:(?:export|default|pub(?:\([\w:]+\))?|async|public|private|protected|internal|abstract|final|sealed"
    r"|static|partial|open|data|unsafe)\s+)*"
    r"(?:(?:def|class|function\*?|func|fn|fun|struct|enum|trait|interface|module|mod|type)\s+[\w$]+(?:\.[\w$]+)*"
    r"(?:\s*(?:[(:<{=;\[]|$)|\s+(?:extends|implements|for|struct|interface|where)\b)|impl\b|func\s+\()"
    r"|(?:export\s+)?(?:const|let|var)\s+[\w$]+\s*=\s*(?:async\s+)?"
    r"(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[\w$]+\s*=>)"
    r"|(?:public|private|protected|internal|static|override)\s[^;=]*\()")
OUTLINE_CONST = re.compile(r"[A-Z_][A-Z0-9_]*\s*(?::[^=]*)?=(?!=)")  # module-level constants, after definitions
OUTLINE_MD = re.compile(r"#{1,4}\s+\S")
OUTLINE_MAX = 60  # ~10 tokens a row: still a fraction of the file it replaces
OUTLINE_SKIP = (".log", ".txt", ".out", ".json", ".jsonl", ".ndjson", ".xml", ".html", ".htm", ".css", ".lock", ".map",
                ".min.js", ".sql")


def outline(path, text):
    """-> the lines of a big file worth jumping to, as `line: text` (1-based, like Read's offset).

    Outermost first: whole nesting levels while they fit; a top level too long to fit is sampled evenly over the file.
    """
    low = path.lower()
    if low.endswith(OUTLINE_SKIP):  # logs and data: no structure worth a map, grep is the way in
        return []
    lines = text.lstrip("﻿").split("\n")
    if low.endswith((".csv", ".tsv")):
        first = lines[0].strip()
        return [f"1: {first[:160]}  (header)"] if first else []
    md = low.endswith((".md", ".markdown", ".mdx"))
    groups, fence = {}, False  # (depth, is_constant) -> [(line, text)]
    for n, line in enumerate(lines, 1):
        line = line.rstrip("\r")
        if md:
            if line.lstrip().startswith(("```", "~~~")):
                fence = not fence  # a "# comment" in a code block is not a heading
            elif not fence and OUTLINE_MD.match(line):
                groups.setdefault((len(line) - len(line.lstrip("#")), False), []).append((n, line[:70]))
        elif OUTLINE_CODE.match(line) or OUTLINE_CONST.match(line):
            key = (len(line) - len(line.lstrip()), not OUTLINE_CODE.match(line))
            groups.setdefault(key, []).append((n, line.strip()[:70]))
    keep, step = [], 1
    for key in sorted(groups):
        group = groups[key]
        if keep and len(keep) + len(group) > OUTLINE_MAX:
            break
        if len(group) > OUTLINE_MAX:
            step = -(-len(group) // OUTLINE_MAX)
            group = group[::step]
        keep += group
    rows = [f"{n}: {t}" for n, t in sorted(keep)]
    more = sum(map(len, groups.values())) - len(keep)
    if more:
        why = f"1 in {step} outer ones listed, the rest nested or constants" if step > 1 else "nested or constants"
        rows.append(f"... {more} more not shown ({why}); grep -n for them")
    return rows


def guard(stdin, out=None):
    """Claude Code PreToolUse hook on Read: stop whole-file reads of big text files.

    The model gets the file size, an outline with line numbers so it can read just the part it needs, and how to
    read it all anyway: a positive `limit` always goes through (an `offset` alone still reads up to 2,000 lines). Disable with TOKENLEAK_GUARD=0, tune with
    TOKENLEAK_GUARD_BYTES.
    """
    out = out or sys.stdout
    if os.environ.get("TOKENLEAK_GUARD", "1") == "0":
        return
    try:
        event = json.loads(stdin.read() or "{}")
    except ValueError:
        return
    if not isinstance(event, dict):
        return
    inp = event.get("tool_input") or {}
    if not isinstance(inp, dict):
        return
    path = str(inp.get("file_path") or "")
    limit = inp.get("limit")
    if event.get("tool_name") != "Read" or (isinstance(limit, int) and limit > 0) or not path \
            or path.lower().endswith(GUARD_SKIP):
        return
    try:
        size = os.path.getsize(path)
        if size <= int(os.environ.get("TOKENLEAK_GUARD_BYTES", "24000")):
            return
        with open(path, "rb") as fh:
            head = fh.read(1 << 22)  # outline the first 4 MB; count the rest
            if b"\0" in head[:4096]:  # binary: let the Read tool decide
                return
            head_lines = head.count(b"\n")
            if size > 1 << 26:  # over 64 MB: estimate from the first 4 MB rather than scan it all inside a hook
                lines = int(head_lines * size / len(head) * 1.1) + 1
            else:
                lines = head_lines + sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 16), b""))
    except (OSError, ValueError):
        return
    rows = outline(path, head.decode("utf-8", "replace"))
    partial = f", first {head_lines:,} lines only" if len(head) < size else ""
    reason = (f"tokenleak: {os.path.basename(path)} is {lines:,} lines (~{fmt(size / CHARS_PER_TOKEN)} tokens) and every "
              f"later call would re-send it. Read only the part you need with offset and limit"
              + (" (outline below)" if rows else "") + ", or grep -n first. "
              f"If you really need the whole file, call Read again with limit: {lines + 1}.")
    if rows:
        reason += f"\nOutline (line: text{partial}):\n" + "\n".join("  " + r for r in rows)
    json.dump({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                      "permissionDecisionReason": reason}}, out)


def print_rules(out=None):
    out = out or sys.stdout
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SKILL.md")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:  # installed as a bare CLI (uvx/pipx): the rules live in the skill
        out.write(f"The rules ship with the skill: {REPO}/blob/v{__version__}/skills/tokenleak/SKILL.md\n")
        return
    parts = text.split("\n---\n", 1)
    body = parts[1].lstrip() if text.startswith("---") and len(parts) == 2 else text
    out.write(body.split("\n## Audit", 1)[0].rstrip() + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="tokenleak",
        description="Find where your coding agent leaks context tokens (Claude Code + Codex transcripts). "
                    "Local and read-only.",
        epilog="`tokenleak rules` prints the rules to paste into AGENTS.md / CLAUDE.md / .cursorrules (from the skill's "
               "SKILL.md; a CLI-only install prints where to find them). "
               "`tokenleak suggest` turns this project's leaks into rules for its CLAUDE.md / AGENTS.md. "
               "`tokenleak statusline` is a Claude Code status line (reads its JSON on stdin).")
    ap.add_argument("paths", nargs="*", help="transcript files or folders (default: ~/.claude/projects, ~/.codex/sessions)")
    ap.add_argument("--since", help="window: 24h, 7d (default), 4w, 2026-10-01, or all")
    ap.add_argument("--agent", choices=("all", "claude", "codex"), default="all")
    ap.add_argument("--here", action="store_true", help="only sessions started in this directory (or below)")
    ap.add_argument("--current", action="store_true", help="only the most recent session in this directory")
    ap.add_argument("--project", help="only sessions whose working dir contains this text")
    ap.add_argument("--session", help="only this session id (prefix ok)")
    ap.add_argument("--compare", metavar="WHEN", help="before vs after a date (2026-10-01) or age (7d): e.g. when you installed the rules")
    ap.add_argument("--top", type=int, default=10, help="how many individual leaks to list (default 10)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--share", action="store_true", help="small anonymized card to share")
    ap.add_argument("--html", metavar="FILE", help="also write a self-contained HTML report to FILE")
    ap.add_argument("--redact", action="store_true", help="hide commands, paths and project names")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--version", action="version", version=f"tokenleak {__version__}")
    for stream in (sys.stdout, sys.stderr):  # Windows pipes default to cp1252, which has no "≈" or box drawing
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    argv = sys.argv[1:] if argv is None else argv
    try:
        return _main(ap, argv)
    except ValueError as e:  # bad --since / --compare values
        ap.error(str(e))


def _main(ap, argv):
    if argv[:1] == ["rules"]:
        print_rules()
        return 0
    if argv[:1] == ["suggest"]:  # project rules from this project's own leaks (default: here, last 30 days)
        args = ap.parse_args(argv[1:])
        if not (args.paths or args.project or args.session or args.current):
            args.here = True
        args.since = args.since or "30d"
        threads, since = collect(args)
        suggest(threads, since=since)
        return 0
    if argv[:1] == ["guard"]:  # Claude Code PreToolUse hook (see guard())
        guard(sys.stdin)
        return 0
    if argv[:1] == ["statusline"]:
        statusline(sys.stdin)
        return 0
    if argv[:1] == ["hook"]:  # Claude Code / Codex lifecycle hook: inject the rules as context
        buf = io.StringIO()
        print_rules(buf)
        event = argv[1] if len(argv) > 1 else "SessionStart"
        out = {"hookSpecificOutput": {"hookEventName": event, "additionalContext": buf.getvalue()}}
        if event == "SessionStart" and not sys.stdin.isatty():
            try:
                data = json.loads(sys.stdin.read() or "{}")
                tip = compact_tip(data) if isinstance(data, dict) and data.get("source") == "startup" else None
            except Exception:  # a tip must never break session start
                tip = None
            if tip:
                out["systemMessage"] = tip
        json.dump(out, sys.stdout)
        return 0
    if argv[:1] == ["compact"]:  # replay an earlier auto-compact; --apply writes the recommended setting
        args = ap.parse_args([a for a in argv[1:] if a != "--apply"])
        args.since = args.since or "30d"
        args.agent = "claude"
        totals = run(args)[0]
        return compact_report(totals.get("context_window"), apply="--apply" in argv)
    if argv[:1] == ["audit"]:
        argv = argv[1:]
    args = ap.parse_args(argv)
    if args.compare:
        if args.html or args.share:
            ap.error("--compare prints a text table or --json; --html and --share don't apply")
        a, b, since, split = compare(args)
        if args.json:
            json.dump({"version": __version__, "split": split, "before": redacted(a[0], args), "after": redacted(b[0], args),
                       "before_by_kind": a[1], "after_by_kind": b[1]}, sys.stdout, indent=2, default=str)
            sys.stdout.write("\n")
        else:
            print_compare(a, b, since, split, args)
        return 0
    totals, by_kind, findings, since = run(args)
    if args.html:
        write_html(totals, by_kind, findings, since, args, args.html)
        print(f"tokenleak: report written to {os.path.abspath(args.html)}", file=sys.stderr)
    if args.json:
        json.dump(to_json(totals, by_kind, findings, args), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    elif args.share:
        print_share(totals, by_kind)
    else:
        print_report(totals, by_kind, findings, since, args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)
