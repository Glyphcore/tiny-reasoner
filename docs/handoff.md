# Handoff (overwrite, never append)

## Done (branch `work/spec`, all committed, `make check` green)
- Spec (`spec/interfaces.py` 26 types, `spec/diagrams.md`, `tools/spec_check.py`), Makefile, venv, `make setup/gpu/check`.
- Games hanoi/puzzle8/countdown + verifiers + optimal solvers; `tr/episode.py` (solo and chat modes), `tr/eval.py`, `tr/run.py`, `tools/eval.py`.
- `tr/policy.py` HFPolicy (Qwen3-0.6B fp16) + `tr/fastgen.py` CUDA-graph greedy decoder (14.5 tok/s vs 3.5 eager on this WSL host; output identical to HF generate).
- Edits: `tr/edits/weightedit.py` TensorEdit (before/after copies), `tr/edits/recipes.py` RecipeEdit (taskvec/steer/orth/headscale/neuron/lora,
  streamed exact revert from the HF checkpoint), `tr/edits/lora.py`, `tr/edits/ops.py`; `tr/inspect.py`; `tools/{inspect_collect,inspect_analyse,make_edit,baselines.sh}`.
- Qwen3-0.6B-Base (rev da87bfb608c14b7cf20ba1ce41287e8de496c0cd) downloaded for task vectors.
- Edits built and verified (apply/revert hashes ok): `edits/tv-p0.3`, `edits/tv-m0.3` (task vectors vs Qwen3-0.6B-Base, 196 tensors).
- Directions: `.cache/inspect-move/dirs-{optimal,legal,think}.pt`; heads ranking in `.q/inspect-heads.out`.
- In flight 2026-11-09 evening (detached): base think-chat, text-chats, nothink-cot, nothink-legal, then think-solo (`.q/baselines-base-*.out`);
  `tv-p0.3` then `tv-m0.3` on nothink-solo, nothink-chat, text-chats (`.q/baselines-tv-*.out`).

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
Held-out seeds 0-9; `tools/results.py` prints these tables from `runs/`. solve = solved fraction, legal = legal moves / all moves,
bad = illegal+unparseable moves per episode (episode ends after 3 in a row). Base = Qwen3-0.6B fp16, greedy.

| variant | condition | game | n | solve | legal | steps/opt | bad | reason_tok | s/ep |
|---|---|---|---|---|---|---|---|---|---|
| base | nothink-solo | countdown | 40 | 0.03 | 0.22 | 1.00 | 2.92 | 0 | 42 |
| base | nothink-solo | hanoi | 40 | 0.00 | 0.20 | - | 3.00 | 0 | 41 |
| base | nothink-solo | puzzle8 | 40 | 0.00 | 0.80 | - | 2.33 | 0 | 29 |

| base | nothink-chat | hanoi L2 | 5 | 1.00 | 1.00 | | | 0 | |
| base | nothink-chat | hanoi L3 | 5 | 0.00 | 0.42 | | | 0 | |
| base | nothink-chat | puzzle8 L1/L2 | 5 | 0.00 | 0.50/0.86 | | | 0 | |

Chat mode (verifier pushback on illegal moves) lifts hanoi L2 from 0 to 1.00 solve. Thinking mode needs the budget forcing
(policy closes </think> at max_new_tokens and answers in 48 tokens); without it every step truncated (run set aside in `.q/base-think-solo-nobudget`).

Per level (nothink-solo): hanoi legal 0.25/0.25/0.00/0.25 (L2-5); puzzle8 legal 0.90/0.62/0.77/0.78 (L1-4); countdown solve 0/0.10/0/0, legal 0.14/0.23/0.30 (L1-3).
Floor effect: non-thinking base never solves; edits are judged on legal rate and on thinking-mode solve at the easy levels.

Inspection (held-in seeds 100-139, non-thinking, residual at the LAST PROMPT token): diff-of-means directions separate
optimal vs wrong moves only weakly (held AUC <= 0.64 puzzle8, 0.69 countdown legal at L23). The move is decided one token later
(replies are "move 5"), hence the `--at move` recollection (`.cache/inspect-move/`): model rates at 96 tokens: hanoi parsed 0.92 legal 0.46 optimal 0.33 (n=39), puzzle8 legal 0.31 optimal 0.14 (n=307),
countdown legal 0.57 optimal 0.17 (n=315). Directions still weak (puzzle8 optimal AUC <= 0.65 at L9, countdown <= 0.67 at L20,
legal <= 0.65 at L18-26; hanoi has ~40 distinct states, too few). The model does not represent its own move quality; direction edits
from own-move labels are a long shot. Priority order for edits: task vectors (tv-p0.3, tv-m0.3), thinking-direction steer
(`tools/inspect_think.py` -> dirs-think.pt, baked into o_proj so non-thinking mode reasons), orth of the wrong-move direction at L18-24,
head scaling from `tools/inspect_heads.py`.

## Pitfalls
- `/work` is an overlay where uv cannot set file perms: venv lives at `~/tr-venv`, `.venv` is a symlink (gitignored). `sed -i` warns, still works.
- `make` and `uv` are in `~/.local/bin` (PATH is set by `~/.profile`; in a bare shell `export PATH=$HOME/.local/bin:$PATH`).
- `git` needs `safe.directory` (set globally already). No pip/ensurepip in the image: use `uv pip install --python .venv/bin/python`.
- Fake policies (`tests/fake_policy.py`) find the state by matching the latest `game.render` in the messages (cache of new(level, seed<20) plus states reached); mid-game states must be reached by the policy's own moves. Pass `fallback=` when the same policy also answers non-game chats.
- Solo mode sends a fresh `[system, user]` pair per move (board re-rendered); only chat mode keeps history (capped at 24 messages, oldest user/assistant pair dropped).
- pytest cache warning on the overlay is harmless.
- Background jobs started from the agent shell die when the turn ends: launch long GPU jobs with `setsid nohup ... & disown`.
- `pkill -f <pattern>` matches the agent's own shell command line; use `pkill -f "python tools/eval.py"` style patterns.
- `make setup`: torch must come from the cu124 index first (`TORCH_INDEX`); installing requirements first pulls the default CUDA-13 torch. `UV_HTTP_TIMEOUT=600` is set (the 30 s default timed out on nvidia wheels). Torch 2.6.0+cu124 sees the GTX 1660 Ti with `LD_LIBRARY_PATH=/usr/lib/wsl/lib` (already set in the container env).
- transformers 5.x installed; `generate` kwargs `temperature/top_p/top_k=None` are passed for greedy to avoid warnings.
- Generation is slow when other agents load the CPU: ~3.5 tok/s thinking, ~5 tok/s plain (kernel launch bound, load avg ~9 on 12 threads; GPU matmul itself is fast). Budget evals accordingly; consider a static cache or batching if needed.
- Do not `pkill -f` a pattern that appears in your own command line (kills the tool shell, exit 144); use `pkill -x` or PIDs.
- First model download is ~1.2 GB into `.cache/hf` (HF_HOME set in `tr/policy.py`); the first gpu test took 4.5 min mostly for this.

## Plumbing
- Container `claude-tr`, tmux `tr`. GPU via `/dev/dxg` and `/usr/lib/wsl/lib` (WSL CUDA, 4 libcuda files present; no nvidia-smi). Model cache under `.cache/` (gitignored).
- `make check` (spec + tests), `make gpu`, `./q <cmd>` then `tail -20 .q/last.log` on FAIL.
