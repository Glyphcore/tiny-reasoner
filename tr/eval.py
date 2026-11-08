"""Evaluator: game, text and chat suites for one policy. Silent; writes through the Run."""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Sequence

from spec.interfaces import (ChatMetrics, ChatSuite, Eval, EvalResult, GameMetrics, GameSuite, Message, Policy,
                             TextMetrics, TextSuite)
from tr.episode import GameEpisode
from tr.games import get_game

N = 20  # n-gram length for repetition


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _grams(toks, n=N):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def reply_repeats(reply: str, prev: set) -> bool:
    """Any 20-gram starting in the last 60 whitespace tokens that occurs earlier in the reply or in `prev`."""
    toks = reply.split()
    gs = _grams(toks)
    seen = set()
    lo = max(0, len(toks) - 60)
    for i, g in enumerate(gs):
        if i >= lo and (g in seen or g in prev):
            return True
        seen.add(g)
    return False


class Evaluator(Eval):
    def __init__(self, run):
        self.run = run

    def games(self, policy: Policy, suite: GameSuite) -> list[GameMetrics]:
        game, ver = get_game(suite.game)
        ep = GameEpisode(game, ver, policy, suite.episode)
        out = []
        for level in suite.levels:
            traces = []
            for seed in suite.seeds:
                t = ep.play(level, seed)
                self.run.save_trace(t)
                traces.append(t)
                self.run.log({"kind": "episode", "game": t.game, "level": level, "seed": seed, "mode": t.mode,
                              "solved": t.solved, "steps": len(t.steps), "optimal": t.optimal_steps,
                              "n_illegal": t.n_illegal, "n_unparseable": t.n_unparseable,
                              "reasoning_tokens": t.reasoning_tokens, "wall_s": t.wall_s})
            nm = sum(len(t.steps) for t in traces)
            nl = sum(1 for t in traces for s in t.steps if s.legal)
            solved = [t for t in traces if t.solved]
            m = GameMetrics(game=suite.game, level=level, n=len(traces),
                            solve_rate=len(solved) / len(traces) if traces else 0.0,
                            legal_rate=nl / nm if nm else 0.0,
                            mean_steps=_mean([len(t.steps) for t in solved]),
                            mean_optimal=_mean([t.optimal_steps for t in traces]),
                            mean_reasoning_tokens=_mean([t.reasoning_tokens for t in traces]),
                            mean_wall_s=_mean([t.wall_s for t in traces]))
            self.run.log({"kind": "level", "mode": suite.episode.mode, **dataclasses.asdict(m)})
            out.append(m)
        return out

    def text(self, policy: Policy, suite: TextSuite) -> TextMetrics:
        with open(suite.path) as f:
            docs = [l.strip() for l in f if l.strip()][:suite.max_docs]
        nll = ntok = 0
        for d in docs:
            a, b = policy.nll(d)
            nll += a
            ntok += b
        m = TextMetrics(path=suite.path, n_docs=len(docs), n_tokens=ntok,
                        ppl=math.exp(nll / ntok) if ntok else float("nan"))
        self.run.log({"kind": "text", **dataclasses.asdict(m)})
        return m

    def chats(self, policy: Policy, suite: ChatSuite) -> ChatMetrics:
        with open(suite.path) as f:
            chats = json.load(f)[:suite.max_chats]
        toks, trunc, rep, n = [], 0, 0, 0
        for c in chats:
            msgs: list[Message] = []
            prev: set = set()
            for u in c["turns"]:
                msgs.append(Message("user", u))
                g = policy.generate(msgs)
                content = f"<think>{g.reasoning}</think>\n{g.text}" if g.reasoning else g.text
                msgs.append(Message("assistant", content))
                n += 1
                toks.append(g.n_tokens)
                trunc += bool(g.truncated)
                rep += reply_repeats(g.text, prev)
                prev |= set(_grams(g.text.split()))
            self.run.save_chat(c["id"], msgs)
        m = ChatMetrics(path=suite.path, n_chats=len(chats), mean_reply_tokens=_mean(toks),
                        truncated_rate=trunc / n if n else 0.0, repeat_rate=rep / n if n else 0.0,
                        transcripts_dir=self.run.dir + "/chats")
        self.run.log({"kind": "chats", **dataclasses.asdict(m)})
        return m

    def all(self, policy: Policy, games: Sequence[GameSuite], text: TextSuite, chats: ChatSuite) -> EvalResult:
        r = EvalResult(variant=policy.config.variant)
        for s in games:
            r.games += self.games(policy, s)
        r.text = self.text(policy, text)
        r.chats = self.chats(policy, chats)
        for s in games:
            cs = dataclasses.replace(s, episode=dataclasses.replace(s.episode, mode="chat"))
            r.game_chats += self.games(policy, cs)
        self.run.summary(r)
        return r
