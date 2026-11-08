#!/usr/bin/env bash
# Baseline runs for one variant (default base). Usage: tools/baselines.sh [variant] [--edit path]
# Sequential on the single GPU; each run writes runs/<id>/summary.txt. Prints one line per run.
set -u
cd "$(dirname "$0")/.."
V=${1:-base}; shift || true
E="$@"
PY=.venv/bin/python
run() { id=$1; shift; $PY tools/eval.py --variant "$V" --run "$id" $E "$@" >.q/$id.log 2>&1 && tail -1 .q/$id.log || echo "FAIL $id .q/$id.log"; }
run $V-nothink-solo   --levels-default --seeds 0-9 --no-thinking --max-new-tokens 160 --mode solo
run $V-think-solo     --games hanoi:2,3 puzzle8:1,2 countdown:1,2 --seeds 0-4 --thinking --max-new-tokens 512 --mode solo
run $V-nothink-chat   --games hanoi:2,3 puzzle8:1,2 countdown:1,2 --seeds 0-4 --no-thinking --max-new-tokens 160 --mode chat
run $V-text-chats     --games --thinking --max-new-tokens 512 --mode all
run $V-nothink-cot    --games hanoi:2,3 puzzle8:1,2 countdown:1,2 --seeds 0-9 --no-thinking --max-new-tokens 256 --mode solo --cot-prompt
run $V-nothink-legal  --games hanoi:2,3 puzzle8:1,2 countdown:1,2 --seeds 0-9 --no-thinking --max-new-tokens 160 --mode solo --show-legal
