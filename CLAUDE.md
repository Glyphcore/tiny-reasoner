# tiny-reasoner

Take a small open chat/reasoning model (< 1B params) that holds a coherent conversation, ALTER ITS WEIGHTS DIRECTLY (no training, no
gradient fine-tuning), and see the result: does step-by-step reasoning improve on challenge-based systems (games and puzzles with
verifiable moves), and does the model stay coherent in real multi-turn interaction.

Edits are computed, not trained: directions found from activations and folded into weights (steering baked into a bias or an output
projection, orthogonalising a direction out), closed-form rank-one edits, head or neuron scaling and ablation, task-vector arithmetic
between released checkpoints, and hand-built low-rank deltas in LoRA form (A and B written from analysis, not learned). Gradient training is out of scope.

## Roles
- The lead (Fable) designs, writes the spec and delegates building to subagents (Agent tool, model sonnet; haiku for lookups) with full briefs and
  disjoint files. The lead does the weight work itself: LoRA design and application, reading and writing tensors, interpreting results.

## Token rules (hard)
- Never read command output. Run through `./q` (prints `ok` or `FAIL <n> <log>`); on FAIL read `tail -20` of the log only.
- Training and eval print nothing per step; they write `runs/<id>/metrics.jsonl` and a one-line summary. Read summaries, not logs.
- No file dumps: Grep with a pattern or Read with offset/limit. Write files with Write/Edit; do not echo them back.
- Replies, commits, issues, PRs: terse, one line (`<area>: <what>`). No narration between tool calls.

## After compaction or a fresh start
- Read `docs/handoff.md` only. Do not list the tree, run git log/status, re-read specs or re-run evals to rediscover state.
- Overwrite the handoff (not append) after each step: done / next / results table / pitfalls with fixes / plumbing commands.
- Plumbing discovered twice becomes a script in `tools/` or a Make target.

## Spec first
- `spec/interfaces.py` (Protocols and dataclasses, types only) is the source of truth: Game, Episode, Move, Verifier, Policy (model wrapper),
  Trace (step-by-step reasoning record), Eval, Run, Adapter (LoRA), WeightEdit (direct tensor change with before/after hashes).
- `spec/diagrams.md`: a mermaid class box per spec type, plus the loop (generate, verify, score, train, compare). Spec, box and code move in one commit;
  `tools/spec_check.py` fails on drift.

## Science rules
- Every claim is a number from a held-out eval with a fixed seed set, against the untouched base model and against a prompt-only baseline.
- Games have exact verifiers (legal move, solved state); no model-judged scoring. Report solve rate, legal-move rate, steps to solve, and trace length.
- Coherence is measured, not assumed: perplexity on a held-out text set, a fixed set of multi-turn chats (saved transcripts, base vs edited side
  by side), and the games played as conversations (the model explains, the user pushes back, the model corrects). An edit that wins the game
  score but breaks conversation is a failure.
- Every run is reproducible from `runs/<id>/config.json` (model, revision hash, data seed, LoRA rank/targets, steps, lr). Weights edits record tensor names
  and hashes before and after; a revert must restore the base hashes exactly.
- Speed is not a goal; fit the hardware: GTX 1660 Ti 6 GB (fp16, no bf16), 12 CPU threads, about 2-3 GB RAM headroom on the host.

## Shipping
- Branch `work/<area>` from `origin/main`, one commit per green step (`make check`: spec_check, tests). No push from the container:
  say `READY: <branch>` and the orchestrator pushes. Ask the orchestrator with `NEED: <question>` only for decisions you cannot make.
- Tests are behaviour tests: a game plays to a verified end, an eval over a fixed seed set gives a stable number, a LoRA round trip and a weight
  edit plus revert restore exact hashes. No per-function unit tests.
