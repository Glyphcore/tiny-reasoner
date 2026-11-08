"""FileRun: runs/<id>/{config.json, metrics.jsonl, summary.txt, chats/, traces/}."""
from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Optional, Sequence

from spec.interfaces import (ChatSuite, EpisodeConfig, EvalResult, GameSuite, Message, PolicyConfig, Run,
                             RunConfig, TextSuite, Trace)


def _suite_from(d) -> GameSuite:
    return GameSuite(d["game"], list(d["levels"]), list(d["seeds"]), EpisodeConfig(**d["episode"]))


def config_from_dict(d) -> RunConfig:
    return RunConfig(policy=PolicyConfig(**d["policy"]), games=[_suite_from(s) for s in d["games"]],
                     text=TextSuite(**d["text"]), chats=ChatSuite(**d["chats"]), edit=d.get("edit"),
                     adapter=d.get("adapter"), note=d.get("note", ""))


def _fmt(m) -> str:
    return f"{m.game} L{m.level} solve={m.solve_rate:.2f} legal={m.legal_rate:.2f}"


class FileRun(Run):
    def __init__(self, id: str, dir: str, config: RunConfig):
        self.id, self.dir, self.config = id, dir, config

    def _p(self, *a) -> str:
        return os.path.join(self.dir, *a)

    def write_config(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with open(self._p("config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=1)

    def log(self, record) -> None:
        with open(self._p("metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self, result: EvalResult) -> str:
        parts = []
        if result.games:
            parts.append("games: " + " | ".join(_fmt(m) for m in result.games))
        if result.game_chats:
            parts.append("chat-games: " + " | ".join(_fmt(m) for m in result.game_chats))
        s = f"variant={result.variant} " + " | ".join(parts)
        if result.text:
            s += f" ppl={result.text.ppl:.1f}"
        if result.chats:
            c = result.chats
            s += f" chats={c.n_chats} trunc={c.truncated_rate:.2f} rep={c.repeat_rate:.2f}"
        s = " ".join(s.split())
        with open(self._p("summary.txt"), "w") as f:
            f.write(s + "\n")
        return s

    def save_trace(self, trace: Trace) -> str:
        os.makedirs(self._p("traces"), exist_ok=True)
        p = self._p("traces", f"{trace.game}-L{trace.level}-s{trace.seed}-{trace.mode}.json")
        with open(p, "w") as f:
            json.dump(dataclasses.asdict(trace), f, indent=1)
        return p

    def save_chat(self, chat_id: str, messages: Sequence[Message]) -> str:
        os.makedirs(self._p("chats"), exist_ok=True)
        p = self._p("chats", f"{chat_id}.json")
        with open(p, "w") as f:
            json.dump([dataclasses.asdict(m) for m in messages], f, indent=1)
        return p


def new_run(config: RunConfig, run_id: Optional[str] = None, root: str = "runs") -> FileRun:
    rid = run_id or f"{config.policy.variant}-{time.strftime('%Y%m%d-%H%M%S')}"
    r = FileRun(rid, os.path.join(root, rid), config)
    r.write_config()
    return r


def load_run(run_id: str, root: str = "runs") -> FileRun:
    d = os.path.join(root, run_id)
    with open(os.path.join(d, "config.json")) as f:
        return FileRun(run_id, d, config_from_dict(json.load(f)))
