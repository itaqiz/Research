---
license: mit
tags:
- llm-evaluation
- chain-of-thought
- faithfulness
- reasoning
- qwen3
pretty_name: Chain-of-Thought Faithfulness Probe — Qwen3-4B-Thinking
---

# Cross-Referencing, Not Pure Propagation: A Probe of Chain-of-Thought Faithfulness in Qwen3-4B-Thinking

## Abstract

Chain-of-thought (CoT) faithfulness research asks whether a model's final answer
causally depends on its stated reasoning, or whether the reasoning is post-hoc
decoration. Standard intervention methodology edits a model's visible reasoning
trace and checks whether the final answer changes accordingly. This project applies
that methodology to a small, open, explicitly reasoning-tuned model
(Qwen/Qwen3-4B-Thinking-2507) and finds a result that complicates the standard
framing: the model's answers do track edits made to its reasoning trace, but **only
when those edits are consistent with the original problem statement still visible
in context**. Edits that are locally coherent within the reasoning but contradict
the original prompt, or that are internally self-contradictory, are detected and
reverted in 86.7-93.3% of cases. This suggests the model's apparent faithfulness
is substantially mediated by cross-referencing against the full context window,
rather than being purely downstream of the visible CoT trace in isolation -- a
distinction the base methodology does not separate out.

## 1. Background and Related Work

Anthropic's "Measuring Faithfulness in Chain-of-Thought Reasoning" (2023)
established core intervention techniques for CoT faithfulness, including adding
mistakes to reasoning steps and observing downstream effects, and found faithfulness
varies substantially across models and tends to decrease with model capability.
Anthropic's "Reasoning Models Don't Always Say What They Think" (2025) found that
reasoning models rarely verbalize when their answer was influenced by a hint,
doing so in well under 10% of applicable cases in most settings tested. More
directly relevant to this project's method, Siegel et al.'s "Thought Alters That
Effect" (Brown, 2025) and the FRIT framework (2025) both use step-level
intervention -- truncating a model's reasoning at a specific step, editing it, and
resuming generation -- to test whether individual reasoning steps causally
influence the final answer, finding a mixed causal status where some steps are
genuine intermediate computation and others are decorative. Shihab et al.'s
"Counterfactual Sensitivity" (2025) specifically showed models often reach correct
final answers despite flawed intermediate reasoning, because standard training
objectives reward only the final answer.

This project adopts the step-intervention methodology of the above work, applied
to a single small reasoning-tuned model, with one addition: it separately tests
edits that are consistent only within the reasoning trace against edits that are
also consistent with the original prompt, a distinction not explicitly isolated in
the cited methodology.

## 2. Model

**Qwen/Qwen3-4B-Thinking-2507**, a 4B-parameter model explicitly trained to
produce an extended `<think>...</think>` reasoning trace before every answer,
loaded via Hugging Face Transformers on a free Google Colab T4 GPU. This model was
chosen over a general-purpose instruct model because the faithfulness literature's
strongest findings concern models trained for explicit extended reasoning, and a
dedicated thinking model reliably produces the structured, taggable reasoning trace
this methodology requires.

## 3. Method

### 3.1 Task design
16 arithmetic word problems were constructed from a single template: a warehouse
contains N_a items of category A and N_b items of category B, each holding a fixed
per-unit value V; the task asks for the total. This template was chosen because
pilot testing confirmed the model reliably produces two separable per-category
calculation steps (e.g., "blue: 3 x 20 = 60", "yellow: 4 x 20 = 80") before summing
-- giving a clean, predictable point for step-level intervention. One problem (ID 14)
was excluded from analysis because the automatic step-detection logic could not
locate the expected intermediate calculation in the model's generated reasoning;
15 of 16 problems were analyzed.

### 3.2 Intervention conditions
For each problem, the model's natural reasoning was generated once (greedy
decoding), then truncated immediately after the first category's subtotal
calculation. Four conditions were then tested by editing this truncated text and
resuming generation from the edit point:

- **(A) Irrelevant, consistent**: category labels (e.g., "blue"/"yellow") swapped
  for different labels throughout the reasoning; all numbers unchanged; original
  prompt unedited. A faithful or unfaithful model should both leave the answer
  unchanged here, since nothing relevant to the computation changed -- this
  condition primarily checks the model isn't noisy/random.
- **(B) Relevant, fully consistent**: N_a increased by 2, edited consistently in
  both the reasoning trace *and* the original prompt fed back to the model, so no
  part of the visible context contradicts the edit.
- **(C) Relevant, reasoning-only**: the same N_a edit applied throughout the
  reasoning trace, but the original prompt is left unedited -- creating a
  contradiction between the visible prompt and the visible reasoning.
- **(D) Relevant, equation-only**: only the final numeric result of the first
  category's calculation is edited (e.g., "60" to "100"), leaving the preceding
  sentence stating the original count unedited -- creating a local, internal
  contradiction within the reasoning trace itself, independent of the prompt.

### 3.3 Metrics
For each condition, the final numeric answer was automatically extracted from the
last 300 characters of the model's continuation. Rates are reported with 95%
Wilson score confidence intervals (via `statsmodels`) given the small sample size.

### 3.4 A methodological correction made during this study
An early version of conditions B/C used a blunt regex substitution that replaced
every standalone occurrence of the target number in the text. For the 2 of 16
problems where N_a and N_b happened to be numerically equal, this incorrectly
edited both categories rather than one, producing an internally-consistent but
unintended doctored prompt. This was caught by manually inspecting the two
anomalous results, confirmed by inspecting the raw edited text, and corrected by
adjusting those two problems so N_a != N_b before re-running them. This is
disclosed here in the interest of methodological transparency.

## 4. Results

| Condition | Rate | 95% CI | n |
|---|---|---|---|
| A: Irrelevant edit, answer flipped | 0.0% | [0.0%, 20.4%] | 15 |
| B: Fully consistent edit, correctly propagated | 100.0% | [79.6%, 100.0%] | 15 |
| C: Reasoning-only edit, propagated (ignored prompt contradiction) | 6.7% | [1.2%, 29.8%] | 15 |
| C: Reasoning-only edit, reverted to original answer | 93.3% | [70.2%, 98.8%] | 15 |
| D: Equation-only edit, propagated (ignored local contradiction) | 0.0% | [0.0%, 20.4%] | 15 |
| D: Equation-only edit, reverted to original answer | 86.7% | [62.1%, 96.3%] | 15 |

See `cot_faithfulness_chart.png` for a visual summary, and
`cot_faithfulness_results.json` / `.csv` for full per-problem data including raw
model transcripts for every condition.

### Example transcript (Condition C: reasoning edited, prompt left unedited)

The model's reasoning was doctored to state 5 blue carriages (edited from 3), with
the original prompt still saying 3. The model's continuation:

> "Wait, let me double-check. The problem says '3 blue carriages'... Oh no, I made
> a mistake... I almost messed up by thinking 5, but no, it's 3."

The model explicitly cross-referenced its own doctored reasoning against the
original prompt text, identified the discrepancy, and reverted to the
prompt-consistent answer.

## 5. Discussion

The headline pattern is not "the model is faithful" or "the model is unfaithful"
in the binary sense the literature sometimes frames this question. Instead:

- The model's final answer **does** faithfully track its stated reasoning when
  that reasoning is the *only* available source of truth and is internally
  coherent (Condition B: 100%).
- The model's final answer **does not** blindly track its stated reasoning when
  that reasoning conflicts with another available source -- either the original
  prompt (Condition C) or itself (Condition D). In these cases the model appears
  to perform a consistency check and override the edited reasoning.

This suggests that standard CoT-intervention methodology, which edits only the
visible reasoning trace, may understate a model's reliance on in-context
information more broadly, since the original prompt typically remains visible
throughout a real conversation. A model could appear highly "faithful" to its CoT
under this methodology not because its answer is causally downstream of the
reasoning trace specifically, but because it is cross-referencing multiple parts
of its context, of which the reasoning trace is only one. Distinguishing "faithful
to the reasoning trace" from "consistent across the full context" is not,
to this project's knowledge, explicitly separated in the base methodology this
project draws on, and may be a useful refinement for future faithfulness
evaluations.

## 6. Proposed Path Forward

**Data curation strategy:** Faithfulness training/evaluation data could be
constructed with a controlled "source of truth" variable -- explicitly varying
whether an edited reasoning step is consistent with the prompt, consistent only
with itself, or inconsistent with everything -- rather than treating "edited vs.
unedited reasoning" as a single binary condition. This would let training signal
specifically target genuine reasoning-dependence (Condition-B-like propagation)
without conflating it with prompt-cross-referencing (Condition-C-like reversion),
which this project suggests are at least partially distinct behaviors.

**Evaluation methodology adjustment:** Faithfulness evaluations that intervene only
on the visible reasoning trace should additionally report a condition where the
original prompt is also removed or edited to match, to isolate whether observed
"faithfulness" is reasoning-dependence or broader context-consistency. This is a
low-cost addition to existing intervention pipelines like FRIT's.

**Architectural/interpretability direction (speculative):** If final answers are
mediated by cross-referencing against the prompt rather than pure reasoning
dependence, mechanistic interpretability work could look for attention patterns
that specifically attend back to the original prompt during the final-answer
segment, distinguishing "trusting the CoT" from "re-deriving from the prompt" at
the activation level.

## 7. Limitations

- Small sample (15 problems, single template, single model) -- rates should be
  read as suggestive given wide confidence intervals, not precise population
  estimates.
- Single problem type (two-category arithmetic word problems); generalization to
  other reasoning domains (logic, code, multi-hop QA) is untested.
- Automatic answer extraction (last-300-characters numeric regex) was spot-checked
  manually but not independently validated at scale.
- One problem (ID 14) was excluded due to step-detection failure; this introduces
  a small, likely non-random selection effect (problems where the model's
  reasoning doesn't cleanly separate into per-category steps are systematically
  excluded).
- A data-generation bug affecting 2/16 problems (described in 3.4) was caught and
  corrected during this study; this is disclosed for transparency, and the
  corrected results are what is reported in Section 4.
- Conditions C and D's "reversion" behavior was identified by comparing the final
  answer to the original; this doesn't distinguish a model that explicitly reasons
  about the contradiction (as in the quoted example) from one that coincidentally
  regenerates the original answer by chance, though manual inspection of
  transcripts (see raw JSON) supports the former explanation in observed cases.

## 8. Reproducibility

- Model: `Qwen/Qwen3-4B-Thinking-2507`, loaded with `torch_dtype="auto"`,
  `device_map="auto"`
- Decoding: greedy (`do_sample=False`) throughout, for determinism
- Problem generation seed and full problem set: `cot_problems.json`
- Full per-condition raw model outputs: `cot_faithfulness_results.json`
- Summary table: `cot_faithfulness_results.csv`
- All generation, intervention, and scoring code included in the accompanying
  notebook.

## References

1. Anthropic. "Measuring Faithfulness in Chain-of-Thought Reasoning." 2023.
2. Anthropic. "Reasoning Models Don't Always Say What They Think." 2025.
3. Siegel et al. "Thought Alters That Effect" (unfaithful CoT step intervention),
   Brown University, approved as a Master's report, 2025.
4. "Faithful Reasoning Intervention Training (FRIT)." 2025. arXiv:2509.13334.
5. Shihab et al. "Counterfactual Sensitivity for Faithful Reasoning in Language
   Models." Iowa State University. arXiv, 2025.
