# Spec diagrams

One box per type in `spec/interfaces.py` (member names must match; `tools/spec_check.py`).

```mermaid
classDiagram
class Move {
  +text
}
class Game {
  <<Protocol>>
  +name
  +reset(seed, level)
  +rules()
  +hint()
  +render(state)
  +all_moves(state)
  +legal_moves(state)
  +optimal_moves(state)
  +why_illegal(state, move)
  +apply(state, move)
  +solved(state)
  +parse(text)
}
class Verifier {
  <<Protocol>>
  +legal(game, state, move)
  +replay(game, episode)
}
class Trace {
  +step
  +text
  +tokens
  +move
  +legal
  +optimal
  +after_pushback
}
class Episode {
  +game
  +seed
  +level
  +traces
  +moves
  +solved
  +verified
}
class Policy {
  <<Protocol>>
  +model_id
  +revision
  +reply(messages, max_new_tokens)
  +nll(text)
}
class Eval {
  +split
  +seeds
  +levels
  +max_new_tokens
  +chat_max_new_tokens
  +probe_states
  +solve
  +legal
  +optimal
  +recover
  +steps
  +trace_len
  +probe_legal
  +probe_opt
  +ppl
  +chats_ok
}
class WeightEdit {
  +id
  +family
  +description
  +ops
  +tensors
  +before
  +after
}
class Adapter {
  +name
  +rank
  +alpha
  +targets
}
class Run {
  +id
  +model
  +revision
  +license
  +device
  +dtype
  +threads
  +prompt
  +edit
  +eval
  +base_hash
  +reverted_hash
  +seconds
}
Game ..> Move
Verifier ..> Game
Verifier ..> Episode
Episode *-- Trace
Run *-- Eval
Run o-- WeightEdit
Adapter ..> WeightEdit : merge
```

## Loop

No gradient step anywhere: "train" in the project loop is replaced by "compute edit, apply, revert".

```mermaid
flowchart LR
  G[generate: Policy plays Game as a conversation] --> V[verify: exact Verifier, replay]
  V --> S[score: solve, legal, steps, trace len, probes, ppl, chats]
  S --> I[inspect: activations, logit lens, head attribution]
  I --> E[compute WeightEdit / Adapter from analysis]
  E --> A[apply: hashes before/after] --> G
  S --> C[compare vs base and prompt-only]
  A --> R[revert: base hashes restored exactly]
```
