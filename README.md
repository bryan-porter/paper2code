# paper2code

> **arxiv URL in → citation-anchored implementation out**

```
┌─────────────────────────────┐         ┌──────────────────────────────────────┐
│                             │         │  {paper_slug}/                       │
│  /paper2code                │         │  ├── README.md                       │
│  https://arxiv.org/abs/     │  ───▶   │  ├── REPRODUCTION_NOTES.md          │
│  1706.03762                 │         │  ├── requirements.in                │
│                             │         │  ├── requirements-<target>.lock     │
│                             │         │  ├── src/                            │
│                             │         │  │   ├── model.py     # §3.2 cited  │
│                             │         │  │   ├── loss.py      # §3.4 cited  │
│                             │         │  │   ├── train.py     # §4.1 cited  │
│                             │         │  │   ├── data.py                    │
│                             │         │  │   ├── evaluate.py                │
│                             │         │  │   └── utils.py                   │
│                             │         │  ├── configs/                        │
│                             │         │  │   └── base.yaml   # all params   │
│                             │         │  └── notebooks/                      │
│                             │         │      └── walkthrough.ipynb           │
└─────────────────────────────┘         └──────────────────────────────────────┘
```

*[placeholder: animated GIF showing the full pipeline — paper fetch → parsing → ambiguity audit → code generation → walkthrough notebook]*

---

## Why this exists

**The problem:** ML papers are vague. Critical hyperparameters are buried in appendices or omitted entirely. Prose contradicts equations. "Standard settings" refers to nothing specific. When you implement a paper, you spend more time detective-working than coding.

**What LLMs get wrong:** Naive code generation fills in every gap silently and confidently. You get something that runs but doesn't match the paper. Worse, you can't tell which parts are from the paper and which were invented by the model.

**What paper2code does differently:**

1. **Citation anchoring** — every line of generated code references the exact paper section and equation it implements (`§3.2, Eq. 4`)
2. **Ambiguity auditing** — before writing a single line of code, every implementation choice is classified as `SPECIFIED`, `PARTIALLY_SPECIFIED`, or `UNSPECIFIED`
3. **Honest uncertainty** — unspecified choices are flagged with `[UNSPECIFIED]` comments at the exact line where the choice is made, with common alternatives listed
4. **Appendix mining** — appendices, footnotes, and figure captions are treated as first-class sources, not ignored

The result: code you can trust because you can verify every decision against the paper.

---

## Install

```bash
npx skills@1.5.24 add bryan-porter/paper2code/skills/paper2code
```

This pins the installer, but the GitHub source is still unpinned and follows the
repository's default branch. Review the skill and bundled scripts before
installing them, and run the installer without production credentials. For an
immutable install, quote a source ending in
`#<reviewed-full-commit-sha>` (for example,
`'bryan-porter/paper2code/skills/paper2code#<reviewed-full-commit-sha>'`).

You'll be prompted to:
1. **Select agents** — pick the coding agents you want to use this skill with (e.g., Claude Code)
2. **Choose scope** — Global (recommended) or project-level
3. **Choose method** — Symlink (recommended) or copy

Once installed, open your agent and run the skill:

```bash
claude  # or your preferred agent
```

### Supported runtime and locked installs

The maintained dependency and CI target is **64-bit Windows on CPython 3.13**, using only the canonical `https://pypi.org/simple` index. Other Python versions and operating systems may work, but they do not have a reviewed hash-locked closure in this repository and are not security-supported.

Install the acquisition helper in a new virtual environment with the complete reviewed lock:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip --isolated install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements-runtime-win-py313.lock
```

`requirements-runtime.in`, `requirements-ci.in`, and the example-specific `requirements.in` files are human-reviewed direct inputs, not install locks. The corresponding `requirements-*-win-py313.lock` files contain the complete resolved closures and artifact hashes. CI verifies their structure, installs the exact locks, runs `pip check`, tests the installed code, and audits those same lock files. Pip runs in isolated mode so user or environment configuration cannot add an unexpected package index.

---

## Usage

### Basic — generate a minimal implementation

```
/paper2code https://arxiv.org/abs/1706.03762
```

### Specify framework

```
/paper2code https://arxiv.org/abs/2006.11239 --framework jax
```

### Full mode — includes training loop and data pipeline

```
/paper2code 2106.09685 --mode full
```

### Educational mode — extra comments and pedagogical notebook

```
/paper2code https://arxiv.org/abs/2010.11929 --mode educational
```

### Using bare arxiv ID

```
/paper2code 1706.03762
```

---

## What you get

```
attention_is_all_you_need/
├── README.md                    # Paper summary, contribution statement, quick-start
├── REPRODUCTION_NOTES.md        # Ambiguity audit, unspecified choices, known deviations
├── requirements.in              # Reviewed direct dependency inputs
├── requirements-<target>.lock  # Complete hash-locked target closure
├── src/
│   ├── model.py                 # Architecture — every layer cited to paper section
│   ├── loss.py                  # Loss functions with equation references
│   ├── data.py                  # Dataset class skeleton with preprocessing TODOs
│   ├── train.py                 # Training loop (if in scope)
│   ├── evaluate.py              # Metric computation code
│   └── utils.py                 # Shared utilities (masking, positional encoding, etc.)
├── configs/
│   └── base.yaml                # All hyperparams — each one cited or flagged [UNSPECIFIED]
└── notebooks/
    └── walkthrough.ipynb        # Pedagogical notebook linking paper sections → code → sanity checks
```

### Key files explained

| File | Purpose |
|------|---------|
| `model.py` | Architecture only. Each class maps to a paper section. Variable names match paper notation. |
| `REPRODUCTION_NOTES.md` | The ambiguity audit. Lists every choice, whether the paper specified it, and what alternatives exist. |
| `base.yaml` | Single source of truth for all hyperparameters. |
| `walkthrough.ipynb` | Runnable on CPU with toy dimensions. Quotes paper passages, shows corresponding code, runs shape checks. |

---

## What this skill will NOT do

- **Won't guarantee correctness.** The implementation matches what the paper describes. If the paper is wrong, the code is wrong. If the paper is vague, the code flags it.
- **Won't invent details.** If the paper doesn't specify a hyperparameter, the code uses a common default and marks it `[UNSPECIFIED]`. It will never silently fill in gaps.
- **Won't download datasets.** The `data.py` provides a `Dataset` class skeleton with clear instructions on where to get the data and how to preprocess it.
- **Won't set up training infrastructure.** No distributed training, no experiment tracking, no checkpointing beyond what the paper's contribution requires.
- **Won't implement baselines.** Only the core contribution of the paper is implemented.
- **Won't reimplement standard components.** If the paper says "standard transformer encoder," the code imports it or notes the dependency — it doesn't reimplement attention from scratch.

---

## Security model

Treat every paper, PDF, extracted passage, supplementary file, repository link,
generated implementation, notebook, checkpoint, and package command as
**untrusted external content**. Paper2code does not automatically install
dependencies, clone linked repositories, execute generated code, or load full
Python objects from checkpoints.

The acquisition helper accepts canonical arXiv HTTPS URLs, validates every
redirect, bounds response sizes and parser state, checks expected media types,
refuses accidental output replacement, and neutralizes active HTML and Markdown
image embeds in its saved Markdown. One total monotonic deadline covers metadata,
redirects, all streamed bodies, the inert PDF download, ar5iv text acquisition,
and candidate-code lookup. A downloaded PDF is retained only as an inert artifact;
the workflow never automatically opens, parses, renders, or OCRs it. If bounded
ar5iv parsing fails, the only fallback is a bounded, regular, non-link UTF-8 text
file supplied by the user.

On POSIX, output creation uses private directories and handle-relative,
no-follow operations where the platform exposes them. Windows does not provide
equivalent Python `dir_fd`/`O_NOFOLLOW` guarantees. The helper combines reparse
point checks, regular-file checks, and pre/post-open identity checks, but a process
running as the same Windows principal could still win a path-swap race. Run it in
an ACL-isolated directory with no untrusted same-account process. These controls
do not make remote content or generated ML code trusted. Use a disposable,
project-local virtual environment without production credentials, private
datasets, or sensitive filesystem access, and inspect generated code before
running it.

See [SECURITY.md](SECURITY.md) for private vulnerability-reporting guidance.

---

## Design principles

### Citation anchoring convention

Every non-trivial code decision is anchored to the paper:

```python
# §3.2 — "We apply layer normalization before each sub-layer" (Pre-LN variant)
class TransformerBlock(nn.Module):
    def forward(self, x):
        # §3.2, Eq. 2 — attention_weights = softmax(QK^T / sqrt(d_k))
        attn_out = self.attention(self.norm1(x))  # (batch, seq_len, d_model)
        x = x + attn_out  # §3.2 — residual connection
```

### The UNSPECIFIED flag system

```python
# [UNSPECIFIED] Paper does not state epsilon for LayerNorm — using 1e-6 (common default)
# Alternatives: 1e-5 (PyTorch default), 1e-8 (some implementations)
self.norm = nn.LayerNorm(d_model, eps=1e-6)
```

```python
# [ASSUMPTION] Using pre-norm based on "we found pre-norm more stable" in §4.1
# The paper uses post-norm in Figure 1 but pre-norm in experiments — ambiguous
```

### Ambiguity classification

| Tag | Meaning |
|-----|---------|
| `§X.Y` | Directly specified in paper section X.Y |
| `§X.Y, Eq. N` | Implements equation N from section X.Y |
| `[UNSPECIFIED]` | Paper does not state this — our choice with alternatives listed |
| `[PARTIALLY_SPECIFIED]` | Paper mentions this but is ambiguous — quote included |
| `[ASSUMPTION]` | Reasonable inference from paper context — reasoning explained |
| `[FROM_OFFICIAL_CODE]` | Taken from the authors' official implementation |

---

## Contributing

### Adding worked examples

Worked examples are the most trust-building part of this project. To add one:

1. Pick a well-known paper (people should be able to verify the output)
2. Run the skill: `/paper2code https://arxiv.org/abs/XXXX.XXXXX`
3. Save the full output to `skills/paper2code/worked/{paper_slug}/`
4. Write a `review.md` that honestly evaluates:
   - What the skill got right
   - What it correctly flagged as unspecified
   - Any mistakes it made
   - Any edge cases it handled well or poorly
5. Submit a PR with all of the above

### Improving guardrails

If you find a pattern where the skill hallucinates or makes a silent assumption, add it to the appropriate file in `guardrails/`.

### Adding domain knowledge

If papers in your subfield consistently reference components that the skill doesn't know about (e.g., graph neural network primitives, RL components), add a knowledge file in `knowledge/`.

---

## Worked examples

This repo includes fully worked examples to demonstrate output quality:

| Paper | Type | Command |
|-------|------|---------|
| Attention Is All You Need (1706.03762) | Architecture | `/paper2code https://arxiv.org/abs/1706.03762` |
| DDPM (2006.11239) | Training method | `/paper2code https://arxiv.org/abs/2006.11239` |

Each includes the complete generated output plus an honest `review.md` evaluating what the skill got right and wrong.

---
