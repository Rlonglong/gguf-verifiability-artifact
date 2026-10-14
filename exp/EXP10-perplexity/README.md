# EXP10 — Quality cost of `-ffp-contract=off` (perplexity)

**Question.** Does quantizing with `-ffp-contract=off` (PR #25353) change downstream quality?

**Method (`scripts/run_ppl.sh`).** Same f16 (Qwen2.5-1.5B-Instruct, sha256 `b6eaec35…`) quantized to Q4_K_M
by the `default` and the `ffpoff` build (`b10448`, arm64 host); both scored for wikitext-2-raw perplexity with
the same `llama-perplexity` binary (ctx 512, 584 chunks).

**Result.**
| variant | Q4_K_M sha256 | PPL (± marginal SE printed by llama-perplexity) |
|---|---|---|
| default | `dd707be3…` | 10.7437 ± 0.0758 |
| ffpoff | `52e47664…` | 10.7500 ± 0.0759 |

ΔPPL = +0.0063 (+0.059%). Both models are scored on the same chunks, so the right uncertainty is the
paired one: `python scripts/paired_se.py data/ppl_default.txt data/ppl_ffpoff.txt` gives
ΔPPL = +0.0063 ± 0.0049 (paired SE, t = 1.28): negligible in magnitude and not distinguishable from zero.
Scope: one model, one type, one corpus.

Data: `data/perplexity.csv`, `data/summary.json`, `data/ppl_default.txt`, `data/ppl_ffpoff.txt` (full logs).
