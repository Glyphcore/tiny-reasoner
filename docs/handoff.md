# Handoff (overwrite, never append)

## Done
- Repo, CLAUDE.md, this handoff, `./q`.

## Goal (user)
Alter the weights, do not train. Then see the result in coherent interaction, not only toy checks.

## Next
1. Spec: `spec/interfaces.py`, `spec/diagrams.md`, `tools/spec_check.py`. Pick the model: a coherent chat/reasoning model under 1B
   (first candidate Qwen3-0.6B, thinking and non-thinking modes; fallback Qwen2.5-0.5B-Instruct). Record license and revision hash.
   Pick 3 games with exact verifiers and difficulty levels (e.g. Tower of Hanoi, sliding 8-puzzle, a grid path or small Sokoban level,
   arithmetic countdown). Write the plan into this handoff.
2. Tooling (delegate): Python env with `uv` if the image lacks Python (`make setup`: venv, torch CUDA build for the 1660 Ti, transformers,
   safetensors; no trainer libraries needed), `make check`, `tools/gpu_smoke.py` (one line: device, fp16 matmul ok, free VRAM).
   `tools/chat.py`: interactive and scripted multi-turn chat with any variant (base or an edit id), transcripts to `runs/<id>/chats/`.
3. Games, verifiers, episode runner (delegate, one subagent per game). Eval harness: game metrics, perplexity set, fixed chat set,
   games played as conversations. Baselines: base model, prompt-only.
4. Baseline numbers into the results table.
5. The lead: inspection (activations on solved vs failed steps, attention heads, logit lens), then computed edits (see CLAUDE.md), each
   applied as a `WeightEdit` with before and after hashes, evaluated on all three measures, reverted exactly.
6. The lead: compare edits, keep the best, show base vs edited chats side by side.

## Results
| edit | model | what changed | game | solve | legal | steps | ppl | chats ok |
|---|---|---|---|---|---|---|---|---|

## Plumbing
- Container `claude-tr`, tmux `tr`. GPU via `/dev/dxg` and `/usr/lib/wsl/lib` (WSL CUDA). Model cache under `.cache/` (gitignored).
