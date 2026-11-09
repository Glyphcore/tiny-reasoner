#!/usr/bin/env bash
# Baseline runs for one variant (default base). Usage: bash tools/baselines.sh <variant> [--edit edits/<id>] [run names...]
# Sequential on the single GPU; each run writes runs/<id>/summary.txt. Prints one line per run.
# Budget: thinking runs cap turns at 12 and tokens at 384 (~27 s per move at 14 tok/s).
set -u
cd "$(dirname "$0")/.."
V=${1:-base}; shift || true
E=""; if [ "${1:-}" = "--edit" ]; then E="--edit $2"; shift 2; fi
ONLY="$*"
PY=.venv/bin/python
SUB="--games hanoi:2,3 puzzle8:1,2 countdown:1,2"
run() { id=$V-$1; shift; if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " ${id#$V-} "; then return; fi
        $PY tools/eval.py --variant "$V" --run "$id" $E "$@" >.q/$id.log 2>&1 && tail -1 .q/$id.log || echo "FAIL $id .q/$id.log"; }
run nothink-solo   --levels-default --seeds 0-9 --no-thinking --max-new-tokens 160 --mode solo
run think-solo     $SUB --seeds 0-4 --thinking --max-new-tokens 384 --max-turns 12 --mode solo
run nothink-chat   $SUB --seeds 0-4 --no-thinking --max-new-tokens 160 --max-turns 12 --mode chat
run think-chat     $SUB --seeds 0-2 --thinking --max-new-tokens 384 --max-turns 8 --mode chat
run text-chats     --games --thinking --max-new-tokens 384 --mode all
run nothink-cot    $SUB --seeds 0-9 --no-thinking --max-new-tokens 256 --mode solo --cot-prompt
run nothink-legal  $SUB --seeds 0-9 --no-thinking --max-new-tokens 160 --mode solo --show-legal
