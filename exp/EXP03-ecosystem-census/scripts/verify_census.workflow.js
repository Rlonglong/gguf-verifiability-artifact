export const meta = {
  name: 'census-verify',
  description: 'Adversarially audit GGUF verifiability classifications by reading each repo model card',
  phases: [
    { title: 'Audit', detail: 'one agent per sampled repo reads its HF model card' },
    { title: 'Synthesize', detail: 'aggregate false-negatives and the robust rate' },
  ],
}

// Audit sample (inlined): 9 env-only-gap repositories + 3 with deeper gaps.
const rows = [
  { repo: 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', downloads: '9739060', base_model: 'Qwen/Qwen3-Coder-30B-A3B-Instruct', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf', downloads: '1967025', base_model: 'nvidia/nemotron-3.5-asr-streaming-0.6b', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'prism-ml/Bonsai-27B-gguf', downloads: '1638186', base_model: 'Qwen/Qwen3.6-27B', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'antirez/deepseek-v4-gguf', downloads: '1346854', base_model: 'deepseek-ai/DeepSeek-V4-Flash', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'unsloth/gemma-4-26B-A4B-it-GGUF', downloads: '1339880', base_model: 'google/gemma-4-26B-A4B-it', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF', downloads: '1269915', base_model: 'Qwen/Qwen3-30B-A3B-Thinking-2507', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'unsloth/Qwen3.6-27B-MTP-GGUF', downloads: '1224559', base_model: 'Qwen/Qwen3.6-27B', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'unsloth/Qwen3.5-9B-GGUF', downloads: '1194391', base_model: 'Qwen/Qwen3.5-9B', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'unsloth/Qwen3.5-4B-GGUF', downloads: '1188420', base_model: 'Qwen/Qwen3.5-4B', source_reconstructible: 'True', bit_exact_verifiable: 'False', primary_gap: 'env/version' },
  { repo: 'ornith-ai/Ornith-1.0-9B-GGUF', downloads: '4985686', base_model: '(none)', source_reconstructible: 'False', bit_exact_verifiable: 'False', primary_gap: 'fp16;imatrix;env/version' },
  { repo: 'mixedbread-ai/mxbai-embed-large-v1', downloads: '4162558', base_model: '(none)', source_reconstructible: 'False', bit_exact_verifiable: 'False', primary_gap: 'fp16;env/version' },
  { repo: 'ornith-ai/Ornith-1.0-35B-GGUF', downloads: '3885220', base_model: '(none)', source_reconstructible: 'False', bit_exact_verifiable: 'False', primary_gap: 'fp16;imatrix;env/version' },
]

const AUDIT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    repo: { type: 'string' },
    readme_states_llamacpp_version: { type: 'boolean', description: 'README/model card names an exact llama.cpp version or commit used to quantize' },
    readme_states_build_flags: { type: 'boolean', description: 'README states compiler/arch/-ffp-contract or other build flags' },
    imatrix_available_anywhere: { type: 'boolean', description: 'imatrix file or exact calibration corpus is retrievable (repo, README link, or named public dataset)' },
    fp16_source_available: { type: 'boolean', description: 'full-precision source weights are downloadable (public, ungated)' },
    mechanical_bitexact_holds: { type: 'boolean', description: 'true if the mechanical bit_exact_verifiable=False is confirmed (i.e., NOT actually bit-exact verifiable)' },
    false_negative: { type: 'boolean', description: 'true if the mechanical scorer understated verifiability (info exists that it missed)' },
    notes: { type: 'string', description: 'one sentence of evidence' },
  },
  required: ['repo', 'readme_states_llamacpp_version', 'imatrix_available_anywhere', 'fp16_source_available', 'mechanical_bitexact_holds', 'false_negative', 'notes'],
}

phase('Audit')
const audited = await parallel(rows.map(r => () =>
  agent(
    `You are adversarially auditing whether a HuggingFace GGUF model repo can be INDEPENDENTLY BIT-EXACT VERIFIED (i.e., someone could re-run quantization and reproduce the exact bytes).\n` +
    `Repo: ${r.repo}\n` +
    `A mechanical scorer (reading only the GGUF header + HF API) concluded: bit_exact_verifiable=${r.bit_exact_verifiable}, source_reconstructible=${r.source_reconstructible}, primary_gap=${r.primary_gap}, base_model=${r.base_model}.\n` +
    `Your job: try to REFUTE the "not bit-exact verifiable" finding by reading the human-written model card.\n` +
    `Fetch the raw model card: https://huggingface.co/${r.repo}/raw/main/README.md (if 404, try https://huggingface.co/${r.repo}).\n` +
    `Determine from the card text: (a) does it state the EXACT llama.cpp version/commit used to quantize? (b) does it state build flags (compiler/arch/-ffp-contract)? (c) is the imatrix file or exact calibration corpus retrievable? (d) are full-precision source weights downloadable?\n` +
    `A card that merely says "quantized with llama.cpp" WITHOUT a specific version does NOT count as stating the version. Bit-exact reproduction needs BOTH the exact version AND build flags AND (imatrix if used) AND fp16 source.\n` +
    `Set mechanical_bitexact_holds=true if it is indeed NOT bit-exact verifiable. Set false_negative=true only if the card supplies something the scorer missed that materially changes the verdict.`,
    { label: `audit:${r.repo.slice(0, 40)}`, phase: 'Audit', schema: AUDIT_SCHEMA }
  ).then(v => ({ ...v, downloads: r.downloads, mech_bitexact: r.bit_exact_verifiable, mech_source: r.source_reconstructible }))
     .catch(() => null)
))

const ok = audited.filter(Boolean)
phase('Synthesize')
const synth = await agent(
  `Aggregate this adversarial audit of GGUF verifiability classifications. Data (JSON):\n${JSON.stringify(ok)}\n\n` +
  `Report: (1) of ${ok.length} audited repos, how many had mechanical_bitexact_holds=true (mechanical "not verifiable" confirmed)? ` +
  `(2) how many false_negatives (scorer understated verifiability)? list them with the note. ` +
  `(3) how many model cards stated an exact llama.cpp version? how many stated build flags? ` +
  `(4) one-paragraph conclusion on how robust the "~0% bit-exact verifiable" finding is, and whether even human-readable model cards close the environment gap. Be precise and skeptical.`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return {
  audited_count: ok.length,
  bitexact_confirmed: ok.filter(a => a.mechanical_bitexact_holds).length,
  false_negatives: ok.filter(a => a.false_negative).map(a => ({ repo: a.repo, notes: a.notes })),
  cards_with_version: ok.filter(a => a.readme_states_llamacpp_version).length,
  cards_with_flags: ok.filter(a => a.readme_states_build_flags).length,
  per_repo: ok,
  synthesis: synth,
}
