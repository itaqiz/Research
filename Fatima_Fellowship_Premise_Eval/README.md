# Calculator Mode: When LLMs Stop Auditing Numbers They Already Know

## 1. The Blind Spot

Current benchmarks test whether models *know* facts, and whether they reject
blatantly false claims stated in isolation (TruthfulQA-style). They rarely test a
narrower, more practically important failure: whether a model keeps auditing a
number once that number becomes an **input to a calculation** rather than the
subject of a standalone claim.

This project found that Qwen3-4B-Instruct almost always catches an obviously false
premise ("The Eiffel Tower is in Berlin," 98.6% notice rate) and almost always
catches a false claim planted among several true ones (100% notice rate). But when
a subtly wrong number is handed to it as the starting point of an arithmetic problem
("The Eiffel Tower is 230 meters tall. How many football fields stacked...?"), the
notice rate collapses to 33%, *even though the model correctly states the true value
when asked directly* (9/9 facts passed the closed-book probe).

The model doesn't lack the knowledge. It appears to shift into a "calculator mode"
once a prompt reads as a computation to execute, and in that mode it stops treating
the given number as a claim to verify at all -- it just computes.

## 2. Model Tested

**Qwen/Qwen3-4B-Instruct-2507** (4B parameters), fp16, free Colab T4 GPU. Chosen
because it shares the SFT+RL post-training recipe family of current frontier models,
so this failure mode is plausibly present (to varying degrees) at frontier scale too.

## 3. Method

Three test sets, all gated on the model passing a direct closed-book knowledge probe
for the underlying fact:

1. **Easy set** (n=140 prompts, 10 facts): single blatantly false premise
   (e.g. wrong city) at varying position (adjacent/buried/far in the prompt) and
   framing (neutral/authoritative/presupposed), each with a matched true-premise
   control.
2. **Multi-premise set** (n=6 prompts, 3 fact-sets): three stated facts, only one
   corrupted, task requires reasoning over all three.
3. **Subtle-numeric set** (n=12 prompts, 6 facts): a single plausible-but-wrong
   number (not a wild lie) that the task requires as a calculation input.

"Noticing" was scored via keyword/phrase heuristics (flags like "actually",
"incorrect", or a restatement of the correct value) -- see scoring code in this repo.

## 4. Results

| Condition                              | Notice rate | n   |
|-----------------------------------------|------------:|----:|
| Easy: blatant, isolated false premise   | 98.6%       | 140 |
| Multi-premise: false fact among true ones | 100%      | 6   |
| Subtle: plausible wrong number feeding a calculation | **33.3%** | 12 |
| False-alarm rate (all true-premise controls) | ~9-11% | -   |

### Example failure (model has the correct value, uses the wrong one anyway)

> Prompt: "The Eiffel Tower is 230 meters tall. How many football fields (each
> ~100m long) would it take stacked to match its height?"
>
> Response: "The Eiffel Tower is 230 meters tall... it would take 2.3 football
> fields." (True height is ~330m; the model states the correct height for this
> exact fact when asked directly in the probe stage.)

### Example success (same model, non-computational framing)

> Prompt: "World War 2 ended in 1943. How many years ago did it end, counting
> from 2026?"
>
> Response: "World War II did not end in 1943. World War II ended in 1945...
> 2026 - 1945 = 81 years."

The contrast suggests the trigger isn't premise subtlety alone -- the multi-premise
set also used non-blatant errors and still hit 100% -- but specifically whether the
prompt reads as "verify a claim" versus "execute a computation."

## 5. Proposed Path Forward

**Data curation strategy:** Construct SFT examples that pair a numeric premise with
a downstream calculation, deliberately including "calculator mode" prompts where the
premise is subtly wrong, alongside matched correct-premise versions. Target responses
should show the model checking each numeric input against its own knowledge *before*
computing, not just after being asked to fact-check in isolation. Critically, this
data must include prompts that look like pure arithmetic tasks -- the exact frame
where this eval found the failure concentrated -- rather than only discursive
fact-checking prompts, which the model already handles well.

**Reward design:** In an RL stage, gate reward on the model's own closed-book
knowledge of a given numeric input: if the model demonstrably knows a value and the
prompt's stated value diverges beyond some tolerance, reward surfacing that
divergence before computing. This targets the specific mode-dependence found here,
rather than broadly rewarding skepticism (which the false-alarm rate above shows can
already misfire ~10% of the time on true premises).

**Architectural idea (more speculative):** A lightweight pre-computation "premise
extraction" pass that identifies every numeric claim in a prompt and checks it
against the model's parametric knowledge before entering a computation sub-routine,
independent of whether the surrounding prompt is framed as discursive or
computational.

This is speculative and untested here -- proposed as a hypothesis for future work.

## 6. Limitations

- Very small subtle-numeric sample (6 facts, 12 prompts) -- 33.3% notice rate means
  2/6 facts caught. This is suggestive, not statistically robust; a larger sample is
  needed before treating the exact rate as reliable.
- Heuristic keyword-based scoring rather than a validated LLM-judge; spot-checked
  manually against the raw responses shown above but not independently rated.
- Single 4B model tested; frontier-scale replication would strengthen the claim
  considerably, since post-training recipes and calculator-mode behavior can differ
  by scale.
- "Calculator mode" is my interpretation of the pattern in the data, not something
  measured directly (e.g., via activation analysis); it's the most parsimonious
  explanation given the contrast between the multi-premise and subtle-numeric
  results, but should be treated as a hypothesis.

## 7. Reproduction

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Decoding: greedy (`do_sample=False`)
- Seeds: 0 (easy set), 1 (hard set)
- All generation, dataset-construction, and scoring code included in this repository
  (`premise_eval.json`, `premise_eval_hard.json`, `results_*.json`, notebook).
