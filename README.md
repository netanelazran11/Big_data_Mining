# Network Structure Analysis + LLM Code Generation Benchmark

This repository contains two self-contained applied projects:

1) **Network structure analysis** on multiple real-world social graphs, with classical random-graph baselines.  
2) **Code-generation evaluation pipeline** that benchmarks a code-focused Large Language Model (LLM) on a standardized dataset of programming problems.

The common theme is **empirical evaluation**: define a measurable target, build a reproducible pipeline, and compare reality vs. a baseline.

---

## Project A — Network Structure Analysis

### Goal
Understand whether simple generative models can reproduce the structural signatures observed in real networks—starting from the most fundamental observable: the **degree distribution**.

### Dataset (high-level)
We work with **eight separate social networks** (e.g., networks from different communities/domains). Each network is analyzed individually, and then we also build a unified network by merging them.

### What we compute and why it matters

#### 1) Degree distribution \( f_k \)
For each graph, we compute the relative frequency of nodes having degree \(k\).  
This is the “first diagnostic” for network structure: it tells us whether the graph looks like:
- **Erdős–Rényi-like** (Poisson-ish, narrow degree range), or
- **heavy-tailed / hub-dominated** (broad range, a few nodes with very high degree)

#### 2) Power-law fit (scale-free check)
To quantify heavy tails, we fit a power-law shape:
\[
P(k) \propto k^{\alpha}
\]
Operationally, we run a linear regression in log–log space:
- x-axis: \(\log(k)\)
- y-axis: \(\log(f_k)\)

To avoid overfitting outliers, we drop the top and bottom **5% of degrees** before fitting (so we model the bulk behavior rather than extreme tails).

#### 3) Baseline: Erdős–Rényi random graphs (ER)
For each real network, we generate an ER graph with:
- the same number of nodes \(n\)
- edge probability \(p\) chosen so that the **expected number of edges** matches the real graph

Then we compare:
- real vs. ER **degree distributions**
- real vs. ER **presence/absence of hubs**
- whether ER plausibly generates a power-law-like degree profile

In practice, ER tends to produce **narrow, Poisson-like degrees**, and fails to reproduce the broad heavy-tail patterns and hub nodes that appear in real social networks.

#### 4) Unifying networks + controlled random inter-community links
We merge the eight networks into a single graph where node IDs are relabeled to remain unique.  
Then we add **0.1m** random edges (where \(m\) is the number of edges in the unified graph) to create controlled cross-community connectivity, mimicking “random bridging” between communities.

This lets us test how robust structural properties are under noise and inter-community mixing.

---

## Project B — LLM Code Generation Benchmark

### Goal
Build a complete, reproducible evaluation pipeline that measures how well an LLM can generate **correct Python code** from natural language task descriptions—validated using **held-out executable tests**.

### Dataset: MBPP (Mostly Basic Python Problems)
We use **MBPP**, a standard benchmark dataset of Python tasks.  
Each example contains:
- a natural language **problem description**
- a list of assertion-based tests (`test_list`)
- optional setup code (`test_setup_code`)
- additional harder tests (`challenge_test_list`)
- a reference solution (used only for inspection, not for evaluation)

### Model: `Qwen2.5-Coder-1.5B-Instruct`
We use **Qwen2.5-Coder-1.5B-Instruct**, a code-specialized instruction-tuned model.

**Why this model?**
- **Coder-tuned**: trained/finetuned for program synthesis and code completion, not just general chat.
- **Small enough to run on CPU** (1.5B parameters), so the pipeline is easy to reproduce on a laptop.
- Still strong enough that **prompt design** (what information you reveal vs. hold out) materially affects pass rate.

### Core idea: provide “signature + behavior” examples, hold out evaluation
A common failure mode in code generation is producing a correct algorithm with the **wrong function name/signature**, which causes all tests to fail.

MBPP includes multiple tests per problem. We exploit that to:
- **Reveal** a small number of tests as *examples* to show the required function signature and expected behavior.
- **Hold out** at least one test to evaluate generalization.

A simple and effective split we use:
- First **2 tests** from `test_list` → included in the prompt as examples  
- The **3rd test** from `test_list` → held out for evaluation (unseen by the model)

This preserves a real evaluation signal while giving the model enough structure to match what the tests expect.

### End-to-end evaluation pipeline (what happens per task)

For each MBPP problem:

1) **Load problem**
   - `text` (natural language)
   - `test_list` (assertions)
   - optional `test_setup_code`

2) **Build the prompt**
   - Provide: problem description + example tests (signature + behavior)
   - Ask the model to output a complete Python solution

3) **Generate code**
   - Run decoding once per task (pass@1 style evaluation)

4) **Safety + execution wrapper**
   - Parse the returned code
   - Execute in a controlled environment (local Python process)
   - Define evaluation function entrypoint if needed

5) **Run held-out tests**
   - Execute the evaluation assertion(s)
   - Mark task as Pass/Fail

6) **Aggregate results**
   - Evaluate on **100 problems**
   - Report pass rate (and optionally per-category error patterns)

### What we analyze (beyond pass rate)
- **Signature errors** (wrong function name / wrong args)
- **Logic errors** (code runs but returns wrong result)
- **Runtime errors** (exceptions, missing imports, syntax errors)
- **Overfitting to examples** (passes example tests but fails held-out test)
- (Optional) difficulty gap between standard tests and `challenge_test_list`

---

## Repository Structure (suggested)
Adjust names to match your folders, but the README should map to something like:

- `networks/`  
  - scripts/notebooks to load graphs, compute degree distributions, fit power laws, generate ER baselines  
  - merge-and-perturb utilities (unification + random edges)

- `llm_code_generation/`  
  - dataset loader (MBPP via HuggingFace datasets)  
  - prompt builder (test split strategy)  
  - model runner (Qwen2.5-Coder inference)  
  - evaluator (compile + execute + assert)  
  - results + error analysis

- `reports/` or `docs/`  
  - exported HTML / figures / writeups

---

## Reproducibility

### Environment
- Python 3.10+ recommended
- Typical packages:
  - `datasets`, `transformers`, `torch`
  - `networkx`, `numpy`, `matplotlib` (for network analysis)
  - any lightweight sandboxing helpers you used for code execution

### Running (high level)
1) Network analysis:
   - load graphs → compute degree distributions → fit power law → generate ER baselines → compare plots
2) LLM benchmark:
   - load MBPP → build prompts (example/held-out split) → generate code → execute held-out tests → summarize pass rate

---

## Notes / Limitations
- LLM evaluation is **test-driven**: correctness is defined purely by passing held-out assertions.
- A model can “look correct” in natural language but still fail due to strict signature expectations—hence the explicit inclusion of example tests.
- ER is a deliberately simple baseline; its failure on heavy-tailed degrees is an expected and informative result, not a bug.

---

## Quick Summary
- **Networks**: compare real degree distributions vs. ER baselines; fit power-law exponents; study effects of merging graphs and injecting random inter-community edges.
- **LLMs**: benchmark Qwen2.5-Coder-1.5B-Instruct on MBPP using a prompt strategy that reveals signature/behavior via example tests while preserving a held-out evaluation signal.
