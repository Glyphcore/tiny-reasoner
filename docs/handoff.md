# Handoff (overwrite, never append)

## Done (branch `work/spec`, all committed, `make check` green)
- Spec (`spec/interfaces.py` 26 types, `spec/diagrams.md`, `tools/spec_check.py`), Makefile, venv, `make setup/gpu/check`.
- Games hanoi/puzzle8/countdown + verifiers + optimal solvers; `tr/episode.py` (solo and chat modes), `tr/eval.py`, `tr/run.py`, `tools/eval.py`.
- `tr/policy.py` HFPolicy (Qwen3-0.6B fp16) + `tr/fastgen.py` CUDA-graph greedy decoder (14.5 tok/s vs 3.5 eager on this WSL host; output identical to HF generate).
- Edits: `tr/edits/weightedit.py` TensorEdit (before/after copies), `tr/edits/recipes.py` RecipeEdit (taskvec/steer/orth/headscale/neuron/lora,
  streamed exact revert from the HF checkpoint), `tr/edits/lora.py`, `tr/edits/ops.py`; `tr/inspect.py`; `tools/{inspect_collect,inspect_analyse,make_edit,baselines.sh}`.
- Qwen3-0.6B-Base (rev da87bfb608c14b7cf20ba1ce41287e8de496c0cd) downloaded for task vectors.
- In flight (background, started 2026-11-08 ~21:40): `bash tools/baselines.sh base` -> `.q/baselines-base.out` (6 runs, hours);
  `tools/inspect_collect.py` -> `.q/inspect-collect.out`, `.cache/inspect/<game>.pt`.

## Goal (user)
Alter the weights, do not train. Then see the result in coherent interaction, not only toy checks.

## Plan
- Model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0. 28 layers, hidden 1024, 16 q heads / 8 kv heads,
  thinking (`<think>...</think>`) and non-thinking modes (`enable_thinking` in the chat template). fp16 ~1.2 GB VRAM.
  Fallback `Qwen/Qwen2.5-0.5B-Instruct` if Qwen3 is incoherent at fp16.
- Games (module, classes, levels):
  - `tr/games/hanoi.py` `Hanoi`/`HanoiVerifier`, name `hanoi`, levels 2..5 = disks, move text `move A C` (pegs A B C), optimal 2^n-1.
  - `tr/games/puzzle8.py` `Puzzle8`/`Puzzle8Verifier`, name `puzzle8`, levels 1..4 = optimal length buckets 2-4, 5-8, 9-14, 15-20
    (scrambled by seeded BFS distance), move text `move 5` (tile number slides into the blank), optimal by BFS.
  - `tr/games/countdown.py` `Countdown`/`CountdownVerifier`, name `countdown`, levels 1..4 = 3,4,5,6 numbers (targets grow),
    move text `3 + 4` (one binary op on two available numbers, ints only, exact division), solved when target is in the pool, optimal by search.
  - `tr/games/__init__.py` registry `GAMES: dict[name, (Game, Verifier)]`.
- Code layout: `tr/policy.py` HFPolicy, `tr/episode.py`, `tr/eval.py`, `tr/run.py`, `tr/edits/` (lead only), `tools/{chat,eval,gpu_smoke}.py`,
  `data/heldout.txt` (wikitext-2 test, 200 docs), `data/chats.json` (20 scripted multi-turn user scripts), `tests/` behaviour tests.
- Measures per variant (base, prompt-only = `show_legal`/`cot_prompt`, each edit): game solve/legal/steps/trace length over fixed seeds
  0..9 per level; ppl on heldout; chat set saved base vs edited; games in `chat` mode (model explains, verifier pushback, model corrects).

## Next
4. Baselines: when `.q/baselines-base.out` has 6 lines, copy the numbers into the Results table (per-level numbers in `runs/base-*/summary.txt`).
5. Inspection: `tools/inspect_analyse.py --label optimal` (and `legal`) -> per-layer AUC table; pick the layer with the best held AUC.
   Edits to build with `tools/make_edit.py` (each saved under `edits/<id>`, then `bash tools/baselines.sh <id> --edit edits/<id>`):
   a. taskvec lam in {0.2, 0.4, -0.2}: W_inst + lam (W_inst - W_base) (ExPO-style extrapolation of the post-training delta).
   b. steer at the best layer, scale_rel in {0.5, 1, 2} of the residual norm, baked into o_proj (and down_proj) via the mean input.
   c. orth: remove the negated optimal direction (the "wrong move" direction) from layers around the best layer.
   d. head scaling: heads whose contribution at the answer position differs most between optimal and wrong moves (tr.inspect.head_contributions).
   Each edit: all three measures, revert verified (make_edit prints revert_ok), row in Results.
6. Compare, keep the best, side-by-side chats (`tools/chat.py --script data/chats.json --run <id>` base vs `--edit`).

## Results
| edit | model | what changed | game | solve | legal | steps | ppl | chats ok |
|---|---|---|---|---|---|---|---|---|

## Pitfalls
- `/work` is an overlay where uv cannot set file perms: venv lives at `~/tr-venv`, `.venv` is a symlink (gitignored). `sed -i` warns, still works.
- `make` and `uv` are in `~/.local/bin` (PATH is set by `~/.profile`; in a bare shell `export PATH=$HOME/.local/bin:$PATH`).
- `git` needs `safe.directory` (set globally already). No pip/ensurepip in the image: use `uv pip install --python .venv/bin/python`.
- Fake policies (`tests/fake_policy.py`) find the state by matching the latest `game.render` in the messages (cache of new(level, seed<20) plus states reached); mid-game states must be reached by the policy's own moves. Pass `fallback=` when the same policy also answers non-game chats.
- Solo mode sends a fresh `[system, user]` pair per move (board re-rendered); only chat mode keeps history (capped at 24 messages, oldest user/assistant pair dropped).
- pytest cache warning on the overlay is harmless.
- `make setup`: torch must come from the cu124 index first (`TORCH_INDEX`); installing requirements first pulls the default CUDA-13 torch. `UV_HTTP_TIMEOUT=600` is set (the 30 s default timed out on nvidia wheels). Torch 2.6.0+cu124 sees the GTX 1660 Ti with `LD_LIBRARY_PATH=/usr/lib/wsl/lib` (already set in the container env).
- transformers 5.x installed; `generate` kwargs `temperature/top_p/top_k=None` are passed for greedy to avoid warnings.
- Generation is slow when other agents load the CPU: ~3.5 tok/s thinking, ~5 tok/s plain (kernel launch bound, load avg ~9 on 12 threads; GPU matmul itself is fast). Budget evals accordingly; consider a static cache or batching if needed.
- Do not `pkill -f` a pattern that appears in your own command line (kills the tool shell, exit 144); use `pkill -x` or PIDs.
- First model download is ~1.2 GB into `.cache/hf` (HF_HOME set in `tr/policy.py`); the first gpu test took 4.5 min mostly for this.

## Plumbing
- Container `claude-tr`, tmux `tr`. GPU via `/dev/dxg` and `/usr/lib/wsl/lib` (WSL CUDA, 4 libcuda files present; no nvidia-smi). Model cache under `.cache/` (gitignored).
- `make check` (spec + tests), `make gpu`, `./q <cmd>` then `tail -20 .q/last.log` on FAIL.
