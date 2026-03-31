# composite-data-fairness

## Overview

A growing class of alignment methods — collectively called **Seldonian algorithms** — provides high-confidence probabilistic guarantees that a deployed model satisfies fairness or safety constraints. These guarantees rest on a clean statistical structure: the dataset is split into a *candidate selection* set (used for training/optimization) and a *verification* set (used to construct a high-confidence upper bound on the constraint). As long as the candidate model is independent of the verification data and the verification scores are i.i.d., the guarantee holds with probability ≥ 1 − δ.

The paper identifies a key failure mode in **composite data regimes** — settings where the verification set is drawn from a mix of trusted ground-truth labels and model-inferred proxy scores (e.g., automated toxicity classifiers, AI reward models, or inferred demographic attributes). When proxy samples enter the verification set, they come from a *different* distribution than the ground truth, breaking the i.i.d. assumption and silently invalidating the statistical guarantee. A system can appear to satisfy a fairness constraint while actually violating it — a form of *fairness washing*.

The paper develops a unifying proof framework, proves sufficient conditions under which valid guarantees can be recovered (most importantly: restricting verification to ground-truth data only), and provides an empirical illustration of the breakdown.

---

## Experiment (`main_exp.ipynb`)

The notebook implements the empirical component in four stages:

### Stage 1 — Continuation Generation
Two base LLMs generate free-text continuations conditioned on toxic prompts:
- `meta-llama/Llama-2-7b-hf`
- `Qwen/Qwen3-8B-Base`

Prompts are drawn from the [`allenai/real-toxicity-prompts`](https://huggingface.co/datasets/allenai/real-toxicity-prompts) dataset. The top-2,000 most toxic prompts are selected from a pool of 100,000. Each model generates up to 256 new tokens per prompt (temperature = 0.7).

### Stage 2 & 3 — Proxy Scoring
Each continuation is scored by two automated proxy classifiers, representing the kind of model-inferred quantities that appear in real alignment pipelines:
- **Detoxify** (`multilingual` variant) — a multilingual toxicity classifier
- **RoBERTa** (`s-nlp/roberta_toxicity_classifier`) — a sequence classification model fine-tuned for toxicity

### Stage 4 — Ground-Truth Scoring (LLM Judge)
Each continuation is also scored by **Qwen2.5-7B-Instruct** acting as a strict toxicity judge (1–100 integer scale, normalized to [0, 1]). This serves as the *ground-truth* signal against which the proxy scores are compared.

### Simulation & Plotting
With scores from all three sources in hand, the notebook sweeps over a mixture fraction α ∈ [0, 1] (40 steps). For each α, a verification set of m = 1,000 samples is constructed where α·m scores are drawn from a proxy classifier and (1 − α)·m from the LLM judge. A high-confidence upper bound (HCUB) is computed using either **Hoeffding's inequality** or **Student's t-test** (δ = 0.05), and the mean bound is recorded across n = 50 random sub-samples.

The resulting plot shows how the HCUB drifts as more proxy data contaminates the verification set, relative to a safety threshold of τ = 0.4. This directly illustrates the theoretical failure mode: proxy contamination can push the bound across the threshold, causing the algorithm to either falsely certify an unsafe model or incorrectly abstain on a safe one.

---

## File Structure

| Path | Contents |
|------|----------|
| `qwen/contin/` | Raw continuation CSVs for the Qwen model |
| `qwen/scores/qwen_scored_as_classifier.csv` | LLM judge scores — ground truth (GT) |
| `qwen/scores/qwen_scored_results_detox.csv` | Detoxify proxy scores |
| `qwen/scores/qwen_scored_add_cl.csv` | RoBERTa proxy scores |
| `llama/contin/` | Raw continuation CSVs for the Llama model |
| `llama/scores/llama_scored_as_classifier.csv` | LLM judge scores — ground truth (GT) |
| `llama/scores/llama_scored_results.csv` | Detoxify proxy scores |
| `llama/scores/llama_scored_add_cl.csv` | RoBERTa proxy scores |
