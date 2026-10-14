set -uo pipefail
: "${WORK:?set WORK to the directory holding the f16 GGUF, wikitext-2-raw and the two llama.cpp builds}"
cd "$WORK"; T="${TMP:-$WORK/tmp}"; mkdir -p "$T"
F=Qwen2.5-1.5B-Instruct-f16.gguf; PPL=./build-default/bin/llama-perplexity; WK=wikitext-2-raw/wiki.test.raw; OPT="-c 512 --chunks 200 -t 8"
ppl(){ nice -n 10 $PPL -m "$1" -f "$WK" $OPT 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$"; }
echo "type,ppl_default,ppl_ffpoff,delta,delta_pct" > exp21_quality.csv
for ty in Q5_K_M Q6_K IQ4_XS; do
  nice -n 10 ./build-default/bin/llama-quantize "$F" $T/q_def_$ty.gguf $ty >/dev/null 2>&1
  nice -n 10 ./build-ffpoff/bin/llama-quantize "$F" $T/q_ffp_$ty.gguf $ty >/dev/null 2>&1
  pd=$(ppl $T/q_def_$ty.gguf); pf=$(ppl $T/q_ffp_$ty.gguf)
  d=$(python3 -c "print(f'{float('$pf')-float('$pd'):.4f}')" 2>/dev/null); pct=$(python3 -c "print(f'{100*(float('$pf')-float('$pd'))/float('$pd'):.3f}')" 2>/dev/null)
  echo "$ty,$pd,$pf,$d,$pct" | tee -a exp21_quality.csv
  rm -f $T/q_def_$ty.gguf $T/q_ffp_$ty.gguf
done
echo EXP21_DONE
