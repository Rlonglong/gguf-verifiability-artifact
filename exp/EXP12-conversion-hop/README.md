# EXP12 — The conversion hop: re-deriving the published fp16 from the provider's checkpoint (paper Sec. IV)

**Question.** The publisher's fp16 GGUF (the paper's W, with the metadata M) is itself produced by a conversion
from the model provider's checkpoint. Can a third party re-derive it bit-for-bit?

**Method (`scripts/convert_repro.sh`).** Download `Qwen/Qwen2.5-1.5B-Instruct` at a pinned revision (safetensors,
JSON configs, tokenizer files, model card) into a directory named like the publisher's; convert to an f16 GGUF with
`convert_hf_to_gguf.py` at `b3772` (the release named on the publisher's model card) and at `b10448`, twice each;
record the whole-file and tensor-data sha256 (`tools/gguf_tensor_sha.py`). Needs a Python environment with torch,
transformers, safetensors, sentencepiece, and huggingface_hub, plus the two llama.cpp checkouts from EXP01's
`build_llama.sh`.

**Result (`data/convert_repro.csv`, `data/notes.md`).**
- At `b3772` the whole file equals the publisher's f16 (`bartowski/Qwen2.5-1.5B-Instruct-GGUF`, sha256
  `b6eaec3509f1d037…`), metadata and chat template included, in both runs, on an x86_64 host (the publisher used its
  own machine).
- At `b10448` the tensor data is identical (`e2333e678acf3cfc`); the newer converter adds
  `general.sampling.{temp,top_k,top_p}` to the metadata, so the whole-file hash differs.
- The descriptive metadata (`general.name`, `license`, `tags`, `languages`, `base_model.*`) comes from the model card
  and the directory name; without the model card only those keys differ.

Together with EXP09 this locates the failure: for this repository the fp16 re-derives exactly from the provider's
checkpoint, while the quantized weights do not re-derive at the disclosed release.
