#!/usr/bin/env python3
"""Retrieval probe: three codes buried in a long prompt must come back in order.

The fork's open issues are all "runs fine, answers wrong" (issue #6: hipMemcpy2D
silent corruption on Windows dual-card; the cold-start wrong-answer one). A speed
number without a content check is worthless on this binary, so this runs before
production is switched over.

usage: probe_codes.py --port 8096 --tokens 8000 --reps 2
"""
import argparse
import json
import random
import string
import sys
import time
import urllib.request

FILLER = ("The quick brown fox jumps over the lazy dog while the engineer profiles "
          "memory bandwidth and kernel launch latency across two accelerators. ")


def make_code(rng):
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))


def build(tokens, rng):
    codes = [make_code(rng) for _ in range(3)]
    words_target = int(tokens / 1.3)
    words = FILLER.split()
    reps = max(1, words_target // len(words) + 1)
    text = (" ".join(words * reps))[: words_target * 6]
    cut = len(text) // 3
    body = (text[:cut] + f"\nCheckpoint A: {codes[0]}\n" + text[cut:2 * cut]
            + f"\nCheckpoint B: {codes[1]}\n" + text[2 * cut:]
            + f"\nCheckpoint C: {codes[2]}\n")
    prompt = ("Read the notes and find the three checkpoint codes. Reply with exactly the three codes "
              "in the order A, B, C, separated by single spaces, and nothing else.\n\n" + body)
    return prompt, codes


def post(port, payload, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    ok = 0
    for i in range(a.reps):
        rng = random.Random(1000 + i)
        prompt, codes = build(a.tokens, rng)
        body = post(a.port, {"messages": [{"role": "user", "content": prompt}],
                             "max_tokens": 96, "temperature": 0.0, "seed": 42, "stream": False,
                             "chat_template_kwargs": {"enable_thinking": False}})
        msg = body["choices"][0].get("message") or {}
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        hit = " ".join(codes) in text.upper()
        ok += hit
        t = body.get("timings", {})
        print(f"  rep{i}: expect {codes} got {text[:80]!r} -> {'PASS' if hit else 'FAIL'} "
              f"pp={round(t.get('prompt_per_second') or 0, 1)} tg={round(t.get('predicted_per_second') or 0, 1)}")
    print(f"[probe] {ok}/{a.reps} pass")
    return 0 if ok == a.reps else 1


if __name__ == "__main__":
    sys.exit(main())
