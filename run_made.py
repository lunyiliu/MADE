#!/usr/bin/env python3
"""Command-line entry point for MADE (Multilingual Agentic Diagnosing Engine).

MADE takes a diagnostic query plus a body of model-evaluation records and
produces a structured, evidence-grounded diagnostic report.

Before running, point MADE at an OpenAI-compatible LLM endpoint and a data
root of evaluation records:

    export MADE_API_KEY=...            # required
    export MADE_BASE_URL=...           # any OpenAI-compatible endpoint
    export MADE_MODEL=...              # default: gemini-3-flash

Examples:

    # A free-form query over your own evaluation data
    python run_made.py --query "Which languages is Qwen3-8B weakest on, and \\
how does it compare to Qwen3-32B?" --lang en --data-root data/demo

    # A query from the bundled 54-query diagnostic set, in a given language
    python run_made.py --qid Q12 --lang zh --data-root data/demo

    # The whole 54-query set in one language
    python run_made.py --all --lang en --data-root data/demo

    # Cap how many records are loaded (fixed-seed sample) to trade coverage
    # for speed; 'none' (default) loads everything.
    python run_made.py --record-cap 2000 --qid Q05 --lang en --data-root data/demo
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from made.data_loader import load_all
from made.llm_client import client_from_env
from made.pipeline import run_made_pipeline
from made.queries import LANGUAGES, RESPONSE_LANGUAGE, get_query, load_queries


def _record_cap(value: str):
    if value is None or str(value).lower() == "none":
        return None
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--record-cap must be a positive integer or 'none'")
    if n <= 0:
        raise argparse.ArgumentTypeError("--record-cap must be a positive integer or 'none'")
    return n


def apply_record_cap(records, cap, seed=42):
    if cap is None or len(records) <= cap:
        return records
    return random.Random(seed).sample(records, cap)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MADE to produce a diagnostic report for one or more queries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--record-cap", type=_record_cap, default=None, metavar="N|none",
        help="Maximum records to load per run. 'none' (default) loads all; an "
             "integer caps the substrate via a fixed-seed random sample, "
             "trading coverage for speed.",
    )
    p.add_argument(
        "--min-cell-n", type=int, default=20, metavar="N",
        help="Minimum samples per (model, benchmark, language) cell for it to "
             "appear in the per-language dashboard breakdown. Default 20 is a "
             "statistical-robustness floor; lower it for small datasets so "
             "per-language results are not filtered away.",
    )
    p.add_argument(
        "--data-root", default=None,
        help="Directory of *.jsonl evaluation records (overrides MADE_DATA_ROOT).",
    )
    p.add_argument("--lang", default="en", choices=LANGUAGES, help="Query / report language.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="A free-form diagnostic query.")
    group.add_argument("--qid", help="A query id from the bundled set, e.g. Q12.")
    group.add_argument("--all", action="store_true", help="Run all 54 bundled queries.")
    p.add_argument("--out", default="output", help="Output directory (default: output/).")
    p.add_argument("--no-agentic", action="store_true",
                   help="Use the non-agentic analyst path (no tool loop).")
    return p


def resolve_queries(args) -> list[dict]:
    if args.query:
        return [{
            "id": "Q_custom",
            "text": args.query,
            "query": args.query,
            "lang": args.lang,
            "response_language": RESPONSE_LANGUAGE.get(args.lang, "English"),
        }]
    if args.qid:
        return [get_query(args.qid, args.lang)]
    return load_queries(args.lang)


def main():
    args = build_parser().parse_args()

    os.environ["MADE_MIN_CELL_N"] = str(args.min_cell_n)

    records = load_all(args.data_root)
    records = apply_record_cap(records, args.record_cap)
    print(f"Loaded {len(records)} evaluation records.")

    client = client_from_env(caller="made_agent")
    queries = resolve_queries(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for q in queries:
        t0 = time.time()
        result = run_made_pipeline(client, q, records, verbose=True, agentic=not args.no_agentic)
        elapsed = time.time() - t0
        report = result.get("report", "") or ""
        stem = f"{q['id']}_{args.lang}"
        (out_dir / f"{stem}_report.md").write_text(report, encoding="utf-8")
        (out_dir / f"{stem}_intermediate.json").write_text(
            json.dumps(result.get("intermediate", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{q['id']}] report {len(report)} chars in {elapsed:.1f}s "
              f"-> {out_dir / (stem + '_report.md')}")


if __name__ == "__main__":
    main()
