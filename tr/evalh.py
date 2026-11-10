"""Episode runner (game as a conversation), move probes, perplexity, fixed chats, and the eval summary."""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import torch

from spec.interfaces import Episode, Eval, Move, Trace
from tr.games import GAMES, ExactVerifier

ROOT = Path(__file__).resolve().parent.parent
FORMAT = ("Each turn, reply in exactly this format and nothing else:\n"
          "THINK: <one short sentence about the state>\nMOVE: <your move>")
LEVELS = [2, 3, 4]
SPLITS = {  # fixed seed sets; dev is used for inspection and choosing edit strength, heldout for the results table
    "dev": {g: list(range(100, 106)) for g in GAMES},
    "heldout": {g: list(range(200, 206)) for g in GAMES},
    "smoke": {g: [300] for g in GAMES},
}
PROBE_SEED = {"dev": 1000, "heldout": 2000, "smoke": 3000}


def system_prompt(game, prompt):
    s = "You are playing a puzzle game with the user. " + game.rules() + "\n"
    if prompt == "hint":
        s += game.hint() + "\n"
    return s + FORMAT


def level_of(i):
    return LEVELS[i % len(LEVELS)]


def read_move(game, text):
    idx = text.rfind("MOVE:")
    if idx < 0:
        return None
    line = text[idx + 5:].strip().splitlines()
    return game.parse(line[0]) if line else None


def play(game, policy, seed, level, prompt="plain", max_new_tokens=40):
    verifier = ExactVerifier()
    state = game.reset(seed, level)
    msgs = [{"role": "system", "content": system_prompt(game, prompt)},
            {"role": "user", "content": f"State: {game.render(state)}\nYour move?"}]
    traces, moves, pushed, solved = [], [], False, False
    for step in range(2 * level + 2):
        text, n = policy.reply(msgs, max_new_tokens)
        move = read_move(game, text)
        legal = move is not None and verifier.legal(game, state, move)
        optimal = legal and move in game.optimal_moves(state)
        traces.append(Trace(step, text, n, move.text if move else None, legal, optimal, pushed))
        msgs.append({"role": "assistant", "content": text})
        if legal:
            state = game.apply(state, move)
            moves.append(move.text)
            pushed = False
            if game.solved(state):
                solved = True
                break
            msgs.append({"role": "user", "content": f"OK. State: {game.render(state)}\nYour move?"})
        else:
            pushed = True
            why = game.why_illegal(state, move) if move and move in game.all_moves(state) else \
                "I could not read a valid move in the required format"
            msgs.append({"role": "user", "content": f"That is not allowed: {why}. The state is unchanged: "
                                                    f"{game.render(state)}\nTry a different move."})
    ep = Episode(game.name, seed, level, traces, moves, solved, False)
    ep.verified = verifier.replay(game, ep)
    return ep


def probe_states(game, split, n):
    """Fixed states for the teacher-forced move probe (levels 1..5, deduplicated, disjoint seeds per split)."""
    rng = random.Random(f"probe-{game.name}-{PROBE_SEED[split]}")
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 5000:
        tries += 1
        seed, level = PROBE_SEED[split] + rng.randrange(10**6), rng.randint(1, 5)
        try:
            s = game.reset(seed, level)
        except RuntimeError:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def probe_text(game, policy, state, prompt="plain"):
    msgs = [{"role": "system", "content": system_prompt(game, prompt)},
            {"role": "user", "content": f"State: {game.render(state)}\nYour move?"}]
    return policy.render(msgs, prefix="THINK: I will pick the best legal move.\nMOVE:")


def first_token(policy, move):
    return policy.tok(" " + move.text).input_ids[0]


def probe(game, policy, states, prompt="plain"):
    """Argmax over the first tokens of all syntactic moves at the forced MOVE: position. Returns per-state rows."""
    rows = []
    for s in states:
        logits = policy.forward_text(probe_text(game, policy, s, prompt))
        toks = {m.text: first_token(policy, m) for m in game.all_moves(s)}
        legal = {toks[m.text] for m in game.legal_moves(s)}
        opt = {toks[m.text] for m in game.optimal_moves(s)}
        pick = max(set(toks.values()), key=lambda t: float(logits[t]))
        rows.append({"legal": pick in legal, "opt": pick in opt,
                     "margin": float(max(logits[t] for t in opt) -
                                     max([logits[t] for t in set(toks.values()) - opt] or [logits[pick]]))})
    return rows


def perplexity(policy, path=ROOT / "data/heldout.txt"):
    tot, n = 0.0, 0
    for para in [p.strip() for p in Path(path).read_text().split("\n\n") if p.strip()]:
        a, b = policy.nll(para)
        tot, n = tot + a, n + b
    return float(torch.exp(torch.tensor(tot / n))), n


def reply_ok(text):
    words = re.findall(r"\w+", text)
    if len(words) < 3:
        return False, "too short"
    if sum(c.isascii() for c in text) / len(text) < 0.9:
        return False, "non-ascii drift"
    tri = list(zip(words, words[1:], words[2:]))
    if len(tri) >= 8 and len(set(tri)) / len(tri) < 0.6:
        return False, "repetition"
    return True, ""


def run_chats(policy, max_new_tokens, path=ROOT / "data/chats.json"):
    """Fixed multi-turn chats. A chat is ok when every reply passes the mechanical checks and all fact checks hit."""
    results = []
    for chat in json.loads(Path(path).read_text()):
        msgs, ok, notes, turns = [], True, [], []
        for turn in chat["turns"]:
            msgs.append({"role": "user", "content": turn["user"]})
            text, _ = policy.reply(msgs, max_new_tokens)
            msgs.append({"role": "assistant", "content": text})
            good, why = reply_ok(text)
            if good and turn.get("must") and not re.search(turn["must"], text, re.I):
                good, why = False, f"missing /{turn['must']}/"
            ok = ok and good
            notes.append(why)
            turns.append({"user": turn["user"], "assistant": text, "ok": good, "why": why})
        results.append({"chat": chat["id"], "ok": ok, "turns": turns})
    return results


def write_transcripts(results, outdir, title):
    outdir.mkdir(parents=True, exist_ok=True)
    for r in results:
        lines = [f"# {r['chat']} ({title}) ok={r['ok']}", ""]
        for t in r["turns"]:
            lines += [f"**user:** {t['user']}", "", f"**assistant:** {t['assistant']}", "",
                      f"_check: {'ok' if t['ok'] else 'FAIL ' + t['why']}_", ""]
        (outdir / f"{r['chat']}.md").write_text("\n".join(lines))


def evaluate(policy, split, prompt, outdir, max_new_tokens=40, chat_max_new_tokens=48, probe_n=40, parts="gpxc"):
    """parts: g games, p probes, x perplexity, c chats. Writes metrics.jsonl rows; returns Eval."""
    ev = Eval(split, SPLITS[split], LEVELS, max_new_tokens, chat_max_new_tokens, probe_n)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, per_game = [], {}
    if "g" in parts:
        eps = []
        for name, seeds in SPLITS[split].items():
            for i, seed in enumerate(seeds):
                policy.reset_cache()
                ep = play(GAMES[name], policy, seed, level_of(i), prompt, max_new_tokens)
                eps.append(ep)
                rows.append({"kind": "episode", **asdict(ep)})
        assert all(e.verified for e in eps), "verifier disagreed with runner"

        def agg(es):
            tr = [t for e in es for t in e.traces]
            pb = [t for t in tr if t.after_pushback]
            sv = [e for e in es if e.solved]
            return {"solve": len(sv) / len(es), "legal": sum(t.legal for t in tr) / len(tr),
                    "optimal": sum(t.optimal for t in tr) / len(tr),
                    "recover": sum(t.legal for t in pb) / len(pb) if pb else float("nan"),
                    "steps": sum(len(e.moves) for e in sv) / len(sv) if sv else float("nan"),
                    "trace_len": sum(t.tokens for t in tr) / len(tr), "n_eps": len(es), "n_replies": len(tr),
                    "n_pushbacks": len(pb)}
        for name in SPLITS[split]:
            per_game[name] = agg([e for e in eps if e.game == name])
        a = agg(eps)
        ev.solve, ev.legal, ev.optimal, ev.recover, ev.steps, ev.trace_len = (
            a["solve"], a["legal"], a["optimal"], a["recover"], a["steps"], a["trace_len"])
        per_game["all"] = a
    if "p" in parts:
        pr = []
        for name in SPLITS[split]:
            r = probe(GAMES[name], policy, probe_states(GAMES[name], split, probe_n), prompt)
            per_game.setdefault(name, {}).update(probe_legal=sum(x["legal"] for x in r) / len(r),
                                                 probe_opt=sum(x["opt"] for x in r) / len(r), probe_n=len(r))
            pr += r
        ev.probe_legal, ev.probe_opt = sum(x["legal"] for x in pr) / len(pr), sum(x["opt"] for x in pr) / len(pr)
    if "x" in parts:
        ev.ppl, _ = perplexity(policy)
    if "c" in parts:
        policy.reset_cache()
        res = run_chats(policy, chat_max_new_tokens)
        write_transcripts(res, outdir / "chats", outdir.name)
        ev.chats_ok = f"{sum(r['ok'] for r in res)}/{len(res)}"
        rows += [{"kind": "chat", "chat": r["chat"], "ok": r["ok"], "why": [t["why"] for t in r["turns"]]} for r in res]
    rows.append({"kind": "per_game", **per_game})
    rows.append({"kind": "summary", **{k: v for k, v in asdict(ev).items() if k not in ("seeds", "levels")}})
    with open(outdir / "metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return ev
