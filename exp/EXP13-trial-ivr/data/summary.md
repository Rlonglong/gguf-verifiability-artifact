target                     quant   comp      FP           plain   +recipe
Qwen2.5-7B-Instruct-GGUF   Q4_K_M  armclang  default    141/339         -
Qwen2.5-7B-Instruct-GGUF   Q4_K_M  armclang  ffpoff     141/339         -
Qwen2.5-7B-Instruct-GGUF   Q4_K_M  armgcc15  default    141/339         -
Qwen2.5-7B-Instruct-GGUF   Q4_K_M  armgcc15  ffpoff     141/339         -
Qwen2.5-7B-Instruct-GGUF   Q4_K_M  x86gcc    default    170/339         -
Qwen2.5-7B-Instruct-GGUF   Q4_K_M  x86gcc    ffpoff     141/339         -
Qwen2.5-7B-Instruct-GGUF   Q8_0    armclang  default   339/339*         -
Qwen2.5-7B-Instruct-GGUF   Q8_0    armclang  ffpoff    339/339*         -
Qwen2.5-7B-Instruct-GGUF   Q8_0    armgcc15  default   339/339*         -
Qwen2.5-7B-Instruct-GGUF   Q8_0    armgcc15  ffpoff    339/339*         -
Qwen2.5-7B-Instruct-GGUF   Q8_0    x86gcc    default   339/339*         -
Qwen2.5-7B-Instruct-GGUF   Q8_0    x86gcc    ffpoff    339/339*         -
Qwen_Qwen3.5-0.8B-GGUF     Q4_K_M  armclang  default    140/335   193/335
Qwen_Qwen3.5-0.8B-GGUF     Q4_K_M  armclang  ffpoff     260/335  335/335*
Qwen_Qwen3.5-0.8B-GGUF     Q4_K_M  x86gcc    default    260/335  335/335*
Qwen_Qwen3.5-0.8B-GGUF     Q4_K_M  x86gcc    ffpoff     260/335  335/335*
Qwen_Qwen3.5-0.8B-GGUF     Q8_0    armclang  default    298/335   334/335
Qwen_Qwen3.5-0.8B-GGUF     Q8_0    armclang  ffpoff     298/335   334/335
Qwen_Qwen3.5-0.8B-GGUF     Q8_0    x86gcc    default    299/335  335/335*
Qwen_Qwen3.5-0.8B-GGUF     Q8_0    x86gcc    ffpoff     299/335  335/335*
Qwen_Qwen3.5-2B-GGUF       Q4_K_M  armclang  default    140/335   193/335
Qwen_Qwen3.5-2B-GGUF       Q4_K_M  armclang  ffpoff     260/335  335/335*
Qwen_Qwen3.5-2B-GGUF       Q4_K_M  x86gcc    default    260/335  335/335*
Qwen_Qwen3.5-2B-GGUF       Q4_K_M  x86gcc    ffpoff     260/335  335/335*
Qwen_Qwen3.5-2B-GGUF       Q8_0    armclang  default    298/335   334/335
Qwen_Qwen3.5-2B-GGUF       Q8_0    armclang  ffpoff     298/335   334/335
Qwen_Qwen3.5-2B-GGUF       Q8_0    x86gcc    default    299/335  335/335*
Qwen_Qwen3.5-2B-GGUF       Q8_0    x86gcc    ffpoff     299/335  335/335*
Qwen_Qwen3.5-4B-GGUF       Q4_K_M  armclang  default    184/441   252/441
Qwen_Qwen3.5-4B-GGUF       Q4_K_M  armclang  ffpoff     341/441  441/441*
Qwen_Qwen3.5-4B-GGUF       Q4_K_M  x86gcc    default    341/441  441/441*
Qwen_Qwen3.5-4B-GGUF       Q4_K_M  x86gcc    ffpoff     341/441  441/441*
Qwen_Qwen3.5-4B-GGUF       Q8_0    armclang  default    393/441  441/441*
Qwen_Qwen3.5-4B-GGUF       Q8_0    armclang  ffpoff     393/441  441/441*
Qwen_Qwen3.5-4B-GGUF       Q8_0    x86gcc    default    393/441  441/441*
Qwen_Qwen3.5-4B-GGUF       Q8_0    x86gcc    ffpoff     393/441  441/441*

minimal card-absent coordinates for a bit-exact re-derivation:
  Qwen2.5-7B-Instruct-GGUF   Q4_K_M  -> NOT bit-exact in any tried env (best 170/339)
  Qwen2.5-7B-Instruct-GGUF   Q8_0    -> card only (release+imatrix+preset) [arm64 armclang/default]
  Qwen_Qwen3.5-0.8B-GGUF     Q4_K_M  -> +recipe                            [x86 x86gcc/default+recipe]
  Qwen_Qwen3.5-0.8B-GGUF     Q8_0    -> +recipe                            [x86 x86gcc/default+recipe]
  Qwen_Qwen3.5-2B-GGUF       Q4_K_M  -> +recipe                            [x86 x86gcc/default+recipe]
  Qwen_Qwen3.5-2B-GGUF       Q8_0    -> +recipe                            [x86 x86gcc/default+recipe]
  Qwen_Qwen3.5-4B-GGUF       Q4_K_M  -> +recipe                            [x86 x86gcc/default+recipe]
  Qwen_Qwen3.5-4B-GGUF       Q8_0    -> +recipe                            [arm64 armclang/default+recipe]

bit-exact re-derivable with disclosed coordinates: 7/8 (target x quant)
