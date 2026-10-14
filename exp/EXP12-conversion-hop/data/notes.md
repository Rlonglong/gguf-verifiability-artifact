# EXP12 notes (2026-09-29)
- Host: x86_64 (Ubuntu, gcc 13); the publisher's f16 was produced on the publisher's own machine.
- Tensor data: identical for both converter releases and to the published f16 (e2333e678acf3cfc).
- Whole file: identical to the published f16 at b3772 when the conversion sees the provider's model card
  (README.md) and the publisher's directory name; those feed general.name, general.basename,
  general.license(.link), general.tags, general.languages and general.base_model.* in the metadata.
  Without the model card and with another directory name, only those descriptive keys differ; the
  tokenizer arrays and tokenizer.chat_template are identical in every case.
- b10448 additionally writes general.sampling.{temp,top_k,top_p} (from generation_config.json), so its
  whole-file hash differs while the tensor data is identical.
- Conversion is deterministic: two runs per release give identical files.
