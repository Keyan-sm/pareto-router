#!/usr/bin/env python
"""Generate a routing dataset for any current model pool, via OpenRouter.

No public routing benchmark ships the May-2026 frontier (GPT-5.5, Opus/Sonnet 4.8,
Gemini 3.1 Pro). This builds one: it runs public exact-match tasks (MMLU-Pro, GPQA, ...)
through each model on OpenRouter, records the real cost and token usage OpenRouter
returns, grades deterministically (no LLM judge), and writes the per-(query, model)
quality + cost that `pareto_router.data.load_frontier` consumes.

    # one key reaches every provider; set your spend cap on openrouter.ai first
    export OPENROUTER_API_KEY=sk-or-...
    python bench_gen/generate_routing_dataset.py --estimate --n 200      # project cost first
    python bench_gen/generate_routing_dataset.py --n 200 --out frontier.jsonl
    pareto-router benchmark            # after pointing load_frontier at the file

`--dry-run` fakes the model calls so you can validate the whole pipeline offline, for free.
Confirm exact OpenRouter model slugs with `--list-models` (needs the key).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Best-guess OpenRouter slugs for the target pool. Confirm with --list-models and edit.
DEFAULT_MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.8",
    "google/gemini-3.1-pro",
    # top open-weight (confirm current slugs):
    "meta-llama/llama-4-maverick",
    "qwen/qwen3-max",
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2",
]

_LETTERS = "ABCDEFGHIJ"


def load_mmlu_pro(n: int, seed: int = 0):
    """Public, exact-match multiple-choice questions (TIGER-Lab/MMLU-Pro test split)."""
    import random

    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    items = []
    for r in rows[:n]:
        opts = list(r["options"])
        body = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(opts))
        prompt = (f"Answer the multiple choice question. End with 'The answer is (X).'\n\n"
                  f"{r['question']}\n{body}")
        gold = r["answer"] if isinstance(r["answer"], str) else _LETTERS[int(r["answer"])]
        items.append({"prompt": prompt, "dataset": f"mmlu-pro/{r.get('category', 'na')}", "gold": gold})
    return items


SOURCES = {"mmlu-pro": load_mmlu_pro}


def extract_letter(text: str):
    if not text:
        return None
    m = re.findall(r"answer is\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-J])\b", text)
    return m[-1].upper() if m else None


def call_openrouter(model: str, prompt: str, key: str, max_tokens: int = 1024):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "usage": {"include": True}}
    r = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    r.raise_for_status()
    d = r.json()
    text = d["choices"][0]["message"]["content"]
    usage = d.get("usage", {})
    return {
        "response": text,
        "num_input_tokens": int(usage.get("prompt_tokens", 0)),
        "num_output_tokens": int(usage.get("completion_tokens", 0)),
        "cost": float(usage.get("cost", 0.0)),  # OpenRouter returns real $ when usage.include=true
    }


def call_fake(model: str, prompt: str, key, max_tokens=1024):
    """Offline stand-in so --dry-run can validate the pipeline with no key or spend."""
    import random

    rng = random.Random(hash((model, prompt)) & 0xFFFFFFFF)
    letter = _LETTERS[rng.randrange(10)]
    nin, nout = len(prompt) // 4, rng.randint(20, 200)
    return {"response": f"The answer is ({letter}).", "num_input_tokens": nin,
            "num_output_tokens": nout, "cost": (nin * 2 + nout * 8) / 1e6}


def list_models(key: str):
    r = requests.get(MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    for mid in sorted(m["id"] for m in r.json()["data"]):
        print(mid)


def generate(models, items, key, dry_run=False, max_tokens=1024):
    call = call_fake if dry_run else call_openrouter
    out, total_cost = [], 0.0
    for qi, item in enumerate(items):
        cells = {}
        for model in models:
            try:
                res = call(model, item["prompt"], key, max_tokens)
            except Exception as exc:  # one model failing must not sink the row
                print(f"  ! {model} q{qi}: {repr(exc)[:80]}", file=sys.stderr)
                continue
            res["score"] = 1.0 if extract_letter(res["response"]) == item["gold"] else 0.0
            total_cost += res["cost"]
            cells[model] = res
        if len(cells) == len(models):  # keep only fully-answered rows
            out.append({"prompt": item["prompt"], "dataset": item["dataset"], "models": cells})
        if (qi + 1) % 25 == 0:
            print(f"  {qi + 1}/{len(items)} queries, spent ${total_cost:.2f}", file=sys.stderr)
    return out, total_cost


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate a routing dataset via OpenRouter.")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated OpenRouter slugs")
    p.add_argument("--source", default="mmlu-pro", choices=list(SOURCES))
    p.add_argument("--n", type=int, default=200, help="number of queries")
    p.add_argument("--out", default="frontier.jsonl")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="fake the model calls (no key, no spend)")
    p.add_argument("--estimate", action="store_true", help="project full-run cost from a 3-query sample")
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args(argv)

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if args.list_models:
        list_models(key); return 0
    if not key and not args.dry_run:
        print("Set OPENROUTER_API_KEY (or use --dry-run).", file=sys.stderr); return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.estimate:
        sample = load_mmlu_pro(3, args.seed) if args.source == "mmlu-pro" else SOURCES[args.source](3, args.seed)
        _, cost = generate(models, sample, key, dry_run=args.dry_run, max_tokens=args.max_tokens)
        per_q = cost / max(1, len(sample))
        print(f"sample: {len(sample)} queries x {len(models)} models = ${cost:.4f}  "
              f"(~${per_q:.4f}/query)\nprojected for --n {args.n}: ${per_q * args.n:.2f}")
        return 0

    items = SOURCES[args.source](args.n, args.seed)
    rows, cost = generate(models, items, key, dry_run=args.dry_run, max_tokens=args.max_tokens)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows x {len(models)} models -> {args.out}  (spent ${cost:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
