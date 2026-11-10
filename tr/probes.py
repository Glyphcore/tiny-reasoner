"""Budgeted string probes: a fixed prompt, a token budget a competent answer certainly fits in, an expected pattern.

Two probe sets, both scored by the exact game verifier:
  judge: a state and a proposed move; the model must say `legal` or `illegal` (self-report vs verifier, 2x2 table).
  call:  a state; the model must emit an exact-format tool call {"tool":"move","args":{"move":"<move>"}} with a legal move.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from spec.interfaces import Probe
from tr.games import GAMES, ExactVerifier

JUDGE_BUDGET, CALL_BUDGET = 6, 24
JUDGE_TASK = ("The user shows a state and a proposed move. Decide whether the proposed move is allowed by the rules in that "
              "state. Reply with exactly one word: legal or illegal.")
CALL_TASK = ('Reply with only a JSON tool call in exactly this format and nothing else: '
             '{"tool":"move","args":{"move":"<your move>"}}')
SAID = re.compile(r"\b(illegal|legal)\b", re.I)
JUDGE_VALID = r"^\s*(illegal|legal)[.!]?\s*$"
SEED = {"dev": 5000, "heldout": 6000, "smoke": 7000, "fit": 8000}


BUCKETS = {"fit": range(0, 5), "dev": range(5, 7), "heldout": range(7, 10), "smoke": range(7, 10)}


def bucket(game, state, move=""):
    """Stable split assignment, so fit / dev / heldout never share an example even in tiny state spaces (Hanoi)."""
    return int(hashlib.sha1(f"{game.name}|{state}|{move}".encode()).hexdigest(), 16) % 10


def _states(game, split, salt, dedupe=True, tries=4000):
    rng = random.Random(f"{salt}-{game.name}-{SEED[split]}")
    seen = set()
    for _ in range(tries):
        try:
            s = game.reset(SEED[split] + rng.randrange(10**6), rng.randint(1, 5))
        except RuntimeError:
            continue
        if not dedupe or s not in seen:
            seen.add(s)
            yield s, rng


def judge_probes(split, n_per_game, prompt="plain"):
    """Balanced: n/2 legal and n/2 illegal proposals per game, deterministic per split."""
    v, out = ExactVerifier(), []
    for game in GAMES.values():
        want = {True: n_per_game // 2, False: n_per_game - n_per_game // 2}
        seen = set()
        for s, rng in _states(game, split, "judge", dedupe=False):
            if not any(want.values()):
                break
            m = rng.choice(game.all_moves(s))
            truth = v.legal(game, s, m)
            if (s, m) in seen or bucket(game, s, m.text) not in BUCKETS[split] or not want[truth]:
                continue
            seen.add((s, m))
            want[truth] -= 1
            sys = "You are checking moves in a puzzle game. " + game.rules() + "\n" + \
                  (game.hint() + "\n" if prompt == "hint" else "") + JUDGE_TASK
            msgs = [{"role": "system", "content": sys},
                    {"role": "user", "content": f"State: {game.render(s)}\nProposed move: {m.text}\nIs this move legal or illegal?"}]
            word = "legal" if truth else "illegal"
            out.append(Probe(f"judge-{game.name}-{len(out)}", msgs, rf"(?i)(?<![a-z]){word}\b", JUDGE_VALID, JUDGE_BUDGET,
                             {"game": game.name, "truth": truth, "move": m.text}))
    return out


def call_probes(split, n_per_game, prompt="plain"):
    out = []
    for game in GAMES.values():
        for s, _ in _states(game, split, "call"):
            if sum(p.meta["game"] == game.name for p in out) >= n_per_game:
                break
            if bucket(game, s) not in BUCKETS[split]:
                continue
            sys = "You are playing a puzzle game with the user. " + game.rules() + "\n" + \
                  (game.hint() + "\n" if prompt == "hint" else "") + CALL_TASK
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"State: {game.render(s)}\nYour move?"}]
            legal = "|".join(re.escape(m.text) for m in game.legal_moves(s))
            out.append(Probe(f"call-{game.name}-{len(out)}", msgs,
                             r'\{"tool":\s*"move",\s*"args":\s*\{"move":\s*"(' + legal + r')"\}\}',
                             r'^\s*\{"tool":\s*"move",\s*"args":\s*\{"move":\s*"[^"]*"\}\}\s*$', CALL_BUDGET,
                             {"game": game.name}))
    return out


def run_probe(policy, probe: Probe):
    """Generate at most `budget` tokens; hit = expected pattern appears inside the budget."""
    policy.reset_cache()
    text, n = policy.reply(probe.messages, probe.budget)
    m = re.search(probe.expect, text)
    said = SAID.search(text)
    return {"id": probe.id, "text": text, "tokens": n, "hit": bool(m), "pos": m.start() if m else None,
            "valid": bool(re.search(probe.valid, text, re.I)), "said": said.group(1).lower() if said else None, **probe.meta}


def judge_table(rows):
    """Confusion cells keyed says/is; `none` = no statement inside the budget."""
    t = {"says_legal_is_legal": 0, "says_legal_is_illegal": 0, "says_illegal_is_legal": 0, "says_illegal_is_illegal": 0,
         "none_is_legal": 0, "none_is_illegal": 0}
    for r in rows:
        t[f"{'says_' + r['said'] if r['said'] else 'none'}_is_{'legal' if r['truth'] else 'illegal'}"] += 1
    return t


def summarize(rows):
    hits = [r for r in rows if r["hit"]]
    return {"n": len(rows), "hit": len(hits) / len(rows), "valid": sum(r["valid"] for r in rows) / len(rows),
            "first_pos": sum(r["pos"] for r in hits) / len(hits) if hits else float("nan")}


def dumps(rows):
    return [json.dumps({"kind": "probe", **r}) for r in rows]
