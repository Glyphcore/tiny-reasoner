# Handoff (overwrite, never append)

## Done
- Repo, CLAUDE.md, this handoff, `./q`.

## Next
1. Spec: `spec/interfaces.py`, `spec/diagrams.md`, `tools/spec_check.py`. Pick the model (candidates: Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct,
   SmolLM2-135M-Instruct; record license and revision hash) and 3 games with exact verifiers and difficulty levels (e.g. Tower of Hanoi,
   sliding 8-puzzle, a small grid path or Sokoban level, arithmetic countdown). Write the plan into this handoff.
2. Tooling (delegate): Python env (`make setup`: venv, torch CUDA build that fits the 1660 Ti, transformers, peft, safetensors), `make check`,
   GPU smoke (`tools/gpu_smoke.py` prints one line: device, fp16 matmul ok, free VRAM).
3. Games and verifiers (delegate, one subagent per game), the episode runner and the eval harness: base model and prompt-only baselines, fixed seeds.
4. Baseline numbers into the results table below.
5. The lead: LoRA design (targets, rank), data from verified traces (rejection sampling on solved episodes), train, eval, compare.
6. The lead: direct weight reads and writes (inspect, ablate, edit specific tensors), with exact revert.

## Results
| run | model | change | game | solve | legal | steps | trace |
|---|---|---|---|---|---|---|---|

## Plumbing
- Container `claude-tr`, tmux `tr`. GPU via `/dev/dxg` and `/usr/lib/wsl/lib` (WSL CUDA). Model cache under `.cache/` (gitignored).
