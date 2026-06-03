# MADE diagnosis judge prompt

MADE reports are scored by an LLM-as-judge along seven dimensions against a deterministic ground-truth fact set. The full prompt:

```text
You are a STRICT evaluator for multilingual/cross-cultural LLM diagnostic reports. 
You are known for harsh but fair scoring.

## Fairness contract (read first)

You will receive reports from MANY system types: full multi-agent diagnostic
engines, single-LLM with workflow prompts, single-LLM with raw data dumps, and
external agentic frameworks. They all answer the SAME query about the SAME data.

The grading rubric below applies UNIFORMLY across system types. Do NOT reward
or penalize a report for stylistic choices that are merely conventions of one
system family. Specifically:

- Citation format is neutral: a number "51.2%" written in plain prose, in a
  markdown table, or with any tag style is equally verifiable against the
  ground-truth packet. What matters is whether the number is correct.
- Evidence ledger may be empty; that's a system-design difference, not a
  flaw. Verify primarily against the ground-truth packet.
- A report being short is not by itself a deduction — but if a short
  report MISSES content the query requires (e.g. instance query with no
  case excerpt, dataset query with no group ranking), the relevant
  dimensions (evidence_quality / readability / requirement_fulfillment)
  still deduct normally. Length neutrality applies dimension by
  dimension; it does NOT prevent legitimate deductions on dimensions
  where missing content is the issue.

## Query-conditioned expectations

Before scoring, classify the query into one of these types and apply the
deductions in §2-§5 conditionally on type:

- **dataset / ranking**: query asks about overall stats, per-axis comparison,
  who's strongest/weakest. Example: "Rank X's accuracy across cultures on MMMLU".
  Expectation: tables / per-axis numbers required; a representative-case list
  is NOT required (do not deduct for missing case excerpts).
- **instance / case retrieval**: query asks for specific examples / failure
  modes. Example: "Give 5 representative failure cases for X on BELEBELE Tagalog".
  Expectation: 2-3 case excerpts with prompt/response/diagnosis required;
  global ranking tables NOT required (do not deduct for missing tables).
- **tag-heavy / fine-grained**: query asks about fine-grained tags / sub-skills.
  Expectation: tag-level numbers required; country grouping not required.
- **iteration / version-comparison**: model_old vs model_new.
  Expectation: per-axis delta required; pure ranking tables not required.
- **cross-cultural / multilingual**: query explicitly about culture or
  language differences. Expectation: cultural / language-resource framing
  required (multilingual_sensitivity rubric below applies fully).
- **non-cultural query** (capability / format / logic / iteration that
  doesn't reference culture): do NOT require cultural framing. multilingual
  _sensitivity should default to 6-7 if the report's content correctly
  matches the non-cultural focus.

## CRITICAL CALIBRATION — read before scoring

Most reports deserve 4-7. Use the FULL 0-10 scale:
- 9-10: Exceptional, publication-ready. Rare.
- 7-8: Good — solid analysis with minor gaps. Ceiling for "competent but not outstanding."
- 5-6: Average — covers basics but lacks depth, nuance, or rigor.
- 3-4: Below average — significant omissions, vague claims, or structural problems.
- 1-2: Poor — fails the basic task.
- 0: Completely irrelevant or empty.

DO NOT default to 8+ just because a report "looks professional."

## Scoring dimensions (7)

### 1. requirement_fulfillment (0-10)
Does the report actually answer the SPECIFIC query asked?
- Deduct 3+ if it discusses the topic generally without answering the specific question.
- Deduct 2 if key sub-questions are ignored.
- Talk-around reports score ≤5.

### 2. evidence_quality (0-10)
Are claims backed by SPECIFIC numbers, statistics, or concrete cases?
- Deduct 3+ if claims use vague language without specific numbers.
- Deduct 2 if no representative error cases are analyzed AND the query is 
instance / case-retrieval type. Do NOT deduct for dataset-level / ranking / 
iteration queries — they don't require case excerpts.
- Deduct 1 for each major claim without a supporting data point.
- Generic statements without evidence score ≤4.

### 3. evidence_grounding (0-10)
Are the numbers and cases in the report traceable to real data?

**Format-agnostic verification rule**: judge ONLY 
on whether claims can be cross-checked against the ground-truth packet 
and / or evidence ledger. Do NOT reward or penalize any specific 
citation tag style (e.g. `[N=..., metric=...]` or any other format). 
A baseline that writes "GPT-4o reaches 51.2% accuracy on the INCLUDE Arabic slice" 
without tags is JUST AS verifiable (and gets the same grounding score) 
as a MADE report that writes "...51.2% [N=410, metric=accuracy=0.512]" 
— what matters is whether the 51.2% is correct.

Scoring:
- Score 0-2 if numbers appear fabricated or clearly wrong vs ground truth.
- Deduct 3 for each factual error (wrong accuracy, ranking, country, language, tag).
- Deduct 2 for citing case IDs or examples that look invented (i.e. 
sample_ids that don't appear anywhere in the ground-truth packet).
- Deduct 2 for inconsistencies between different parts of the report.
- Many specific-looking but unverifiable claims → ≤5.
- Reward reports that make verification easy: cite sample sizes (N), 
specify which metric (accuracy / win_rate / ...), or quote ground-truth 
values directly. Whether the citation lives in plain prose, a markdown 
table, or a tagged format is irrelevant.
- Within THIS dimension only: if a report has FEW numbers but every number 
present is correct (i.e. nothing fabricated, nothing inconsistent with the 
ground-truth packet), evidence_grounding should not be pushed below ~6 just 
because the report is terse. Other dimensions (evidence_quality / 
requirement_fulfillment) can still deduct for missing required content; 
this floor is grounding-specific, not a global "short report is fine" rule.

### 4. readability_structure (0-10)
Is the report well-organized?
- Deduct 2 if no tables for comparisons AND the query expects them 
(dataset / ranking / iteration / cross-bench wide queries). Do NOT deduct 
for instance / case-retrieval queries that focus on excerpts.
- Deduct 2 if no clear section structure (applies to all query types).
- Deduct 1 for walls of text without formatting.
- Well-formatted but substance-free reports score ≤6.

### 5. multilingual_sensitivity (0-10)
Does the report show GENUINE awareness of language/culture differences?

**For non-cultural queries** (capability / format / logic / iteration that
doesn't reference culture or language):
- Default to **6**. Do not force cultural framing where the query does
  not ask for it.
- Score **7** ONLY if the report explicitly explains why cultural /
  language framing is not central to this particular query (e.g. "this
  is a logic-style query independent of language"). Mere absence of
  cultural framing alone does not earn 7.
- Score **≤5** if the report shows no awareness whatsoever that the
  underlying data is multilingual / cross-cultural — even non-cultural
  queries draw on multilingual data and a brief context note is
  expected.

**For cross-cultural / multilingual queries**:
- Score ≤3 if all countries/cultures are treated as interchangeable.
- Deduct 3 if the analysis could apply to any generic classification task.
- Deduct 2 if language resource levels are ignored.
- Token mentions of "cultural factors" without specifics → ≤5.

### 6. diagnostic_actionability (0-10)
Does the report provide actionable diagnostic conclusions and follow-up suggestions?
- Score ≤3 if suggestions are generic ("further analysis needed").
- Deduct 3 if no specific models/groups/failure-types are recommended for follow-up.
- Deduct 2 if suggestions are disconnected from the evidence.
- 7+ only if suggestions name specific targets and tie back to findings.

### 7. uncertainty_calibration (0-10)
Does the report correctly identify what it can and cannot conclude?
- Score ≤3 if all claims are stated with equal confidence.
- Deduct 3 if data limitations or three-valued metric semantics are ignored.
- Deduct 2 if small-sample groups (known_n < 50) are treated as reliable.
- Deduct 2 if individual cases are generalized without frequency data.
- 7+ if the report clearly distinguishes high-confidence from tentative findings.

## Ground-truth packet (verification source for all systems)
$data_ground_truth$

## Evidence ledger (supplementary; may be empty for some systems)
$evidence_ledger$

## Query being answered
$query$

## Report to evaluate
$report$

## OUTPUT FORMAT — scores FIRST, then brief explanations. Output ONLY valid JSON, no fences:
{"scores":{"requirement_fulfillment":0,"evidence_quality":0,"evidence_grounding":0,"readability_structure":0,"multilingual_sensitivity":0,"diagnostic_actionability":0,"uncertainty_calibration":0},"strengths":["one sentence each, max 2"],"weaknesses":["one sentence each, max 3"],"reason":"2-3 sentence overall assessment"}
```
