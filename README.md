<div align="center">

# MADE: Multilingual Agentic Diagnosing Engine

**Beyond scoring — turning multilingual evaluation score tables into queryable, explainable, evidence-grounded diagnosis.**

</div>

---

## 📣 Introduction

Multilingual and multicultural benchmarks now span dozens of languages and model families, but the resulting score landscapes are **metric-rich and insight-poor**: leaderboards tell you *who ranks where*, not *why a model fails, on which slices, and what to do about it*. A single LLM (even with a long context) is easily swamped by the long, noisy diagnostic input, while open-ended agents drift off the evidence.

**MADE** is a lightweight, tool-centric, **multi-agent** engine that decomposes post-evaluation analysis into five role-specialised agents and produces a structured, claim-grounded diagnostic report:

![MADE overview: multilingual evaluation inputs flow through a five-agent workflow (Planner → Evidence Analyst / Case Analyst → Reporter, with the Language Reflector intervening on the cultural and grounding axes) to produce an actionable, claim-grounded diagnostic report.](assets/overview.png)

- **Planner** — reads the raw query only, decides the evidence level (Dataset / Instance / Iteration), which analysts to run, and which tool families to activate.
- **Evidence Analyst** — dataset- and group-level statistics in a bounded ReAct tool loop (rankings, gaps, failure-rate patterns).
- **Case Analyst** — instance-level retrieval of concrete transcripts (errors, disagreements, degenerations, representative cases).
- **Language Reflector** — the multilingual specialist; on two axes (cultural sensitivity + evidence grounding) it checks for English-centric inference, over-generalisation, and ungrounded cultural claims.
- **Reporter** — synthesises the final report; every quantitative claim is bound to a tool-call ledger entry.

---

## 🔰 Installation

```bash
git clone <ANONYMIZED-REPOSITORY-URL>   # anonymized for double-blind review
cd MADE
pip install -r requirements.txt
```

Point MADE at any OpenAI-compatible chat endpoint:

```bash
export MADE_API_KEY=...        # required
export MADE_BASE_URL=...        # any OpenAI-compatible endpoint that serves your backbone
export MADE_MODEL=gemini-3-flash   # default backbone for all agents
```

---

## 🚀 Quickstart

A small **synthetic** evaluation set ships in `data/demo/`. **Once you have configured an OpenAI-compatible endpoint (above)**, you can run MADE end to end on it (this is illustrative data, *not* real evaluation results):

```bash
# A free-form diagnostic query over the demo data
python run_made.py --data-root data/demo --lang en \
  --query "Which languages is Qwen3-8B weakest on in MMMLU, and how does it compare to Qwen3-32B?"

# A query from the bundled 54-query diagnostic set, in a chosen language
python run_made.py --data-root data/demo --qid Q12 --lang zh

# The whole 54-query set in one language
python run_made.py --data-root data/demo --all --lang en
```

Reports are written to `output/<id>_<lang>_report.md` with the full agent trace in `output/<id>_<lang>_intermediate.json`.

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--record-cap N\|none` | `none` | Cap how many records are loaded per run (fixed-seed sample). Trade coverage for speed on large substrates. |
| `--min-cell-n N` | `20` | Minimum samples per *(model, benchmark, language)* cell for it to appear in the per-language breakdown. **Lower this for small datasets** so per-language results are not filtered out by the statistical-robustness floor. |
| `--lang` | `en` | Query / report language (one of the 15 supported). |
| `--data-root` | `$MADE_DATA_ROOT` | Directory of `*.jsonl` evaluation records. |
| `--no-agentic` | off | Use the non-agentic analyst path (no tool loop). |

---

## 🗂️ Diagnostic Query Set

`data/queryset/` contains an **expert-authored diagnostic query set**: **54 executable queries × 15 languages** (`zh, en, ar, de, es, fr, it, ja, ko, ms, pl, pt, ru, th, tr`), one file per language (`queries_<lang>.jsonl`). Non-English queries are professional human translations (forward + back-translation, then audited).

Each query is organised along a three-dimensional taxonomy (**3 × 6 × 6**):

- **Evidence level** (3): `Dataset` · `Instance` · `Iteration`
- **Diagnostic category** (6): `Task/Lang` · `Capability` · `Compliance` · `Behavior` · `Culture` · `Improvement`
- **Query template** (6): single-model weakness · pairwise comparison · version/scale evolution · capability correlation · optimisation advice · application-oriented model selection

```json
{"id": "Q01", "language": "en", "query": "When DeepSeek-R1 handles Japanese MMMLU logical-reasoning items, does it show a specific triggering pattern for certain logical fallacies?", "level": "Instance", "category": "Behavior", "template": "single-model weakness"}
```

Load them programmatically:

```python
from made.queries import load_queries
queries = load_queries("zh")          # 54 query dicts in Chinese
```

---

## 📐 Data Interface — bring your own evaluation data

MADE diagnoses a **substrate** of per-`(model, sample)` evaluation records. Provide your own results as newline-delimited JSON (`*.jsonl`) under a data root, then set `MADE_DATA_ROOT` (or pass `--data-root`). Every `*.jsonl` file under the root is loaded; each line is one record:

```json
{
  "sample_id": "unique id of this (model, item) record",
  "model": "model name",
  "source_dataset": "benchmark name",
  "language": "language code or name",
  "country_or_culture": "country / culture label (optional)",
  "prompt": "the input shown to the model",
  "response_raw": "the model's raw output",
  "response_final": "the parsed / final answer (optional)",
  "gold": "the reference answer (optional)",
  "correct": true,
  "tag_category": ["fine-grained capability tag", "..."],
  "meta": {"eval_result": "win | tie | lose (only for pairwise-judged benchmarks)"}
}
```

MADE handles three families of correctness signal, not just binary accuracy:

- **Objective / accuracy benchmarks** (MCQ, exam, reading, math, …): set
  `correct` to `true` / `false`. It is **three-valued** — use `null` when a
  record has no binary verdict; `null` records are excluded from accuracy
  denominators, and tools report `known_n` / `unknown_n` / `coverage` so reports
  never compute fake accuracy.
- **Subjective / pairwise-judged benchmarks** (win-rate style, e.g. open-ended
  generation or multi-turn dialogue where a model's answer is judged win / tie /
  lose against a reference): set `correct` to `null` and put the verdict in
  `meta.eval_result ∈ {"win", "tie", "lose"}`. MADE then reports
  **`win_rate = (win + 0.5·tie) / known`** instead of accuracy for these slices.
- **Translation / generation-quality benchmarks** (e.g. FLORES-style): leave
  `correct` as `null` and keep any quality signal (scores, length ratio,
  language-match flags) under `meta`; tools fall back to response-rate,
  empty-rate and quality fields rather than accuracy.

Other fields: `tag_category` holds fine-grained capability tags (leave `[]` if
you have none); `meta` is otherwise a free-form object for any extra fields.

See `data/demo/eval_records.jsonl` for a complete, runnable example.

---

## 📊 Evaluation

MADE reports are scored by an LLM-as-judge along **seven dimensions** against a
deterministic ground-truth fact set:

1. `requirement_fulfillment` — does it answer the specific query?
2. `evidence_quality` — claims backed by specific numbers / cases?
3. `evidence_grounding` — numbers traceable to the data?
4. `readability_structure` — well-organized?
5. `multilingual_sensitivity` — genuine language / culture awareness?
6. `diagnostic_actionability` — concrete, evidence-tied recommendations?
7. `uncertainty_calibration` — honest about what it can and cannot conclude?

The judge is **format-agnostic**: a number is scored on whether it is correct
against the fact set, not on its citation style, so no system family is
rewarded for its formatting conventions. The full judge prompt is in
[`docs/judge_prompt.md`](docs/judge_prompt.md).

## 📁 Repository layout

```
MADE/
├── made/
│   ├── pipeline.py        # orchestrates the five agents
│   ├── agents.py          # Planner / Evidence / Case / Reflector / Reporter
│   ├── prompts.py         # agent system prompts
│   ├── tools.py           # deterministic analysis tools
│   ├── data_loader.py     # JSONL evaluation-record loader
│   ├── queries.py         # diagnostic query-set loader
│   ├── llm_client.py      # OpenAI-compatible chat client
│   └── agentic/           # bounded ReAct tool-loop + tool registry
├── data/
│   ├── queryset/          # 54 queries × 15 languages
│   └── demo/              # synthetic illustrative evaluation records
├── docs/
│   └── judge_prompt.md    # full LLM-as-judge prompt (7 dimensions)
├── run_made.py            # command-line entry point
└── requirements.txt
```

---

## 📜 Citation

This work is under review. A citation entry will be added upon publication.

```bibtex
@misc{made,
  title  = {MADE: Beyond Scoring via a Multilingual Agentic Diagnosing Engine for Fine-Grained Insights},
  note   = {Under review},
  year   = {2026}
}
```

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).
