"""GameEpisode: plays one game instance with a policy, solo or as a continuous chat."""
from __future__ import annotations

import time
from typing import Sequence

from spec.interfaces import (Episode, EpisodeConfig, Game, Message, Move, Policy, Step, Trace, Verifier)

SYSTEM_SUFFIX = "Reply with exactly one move in the format shown, after any reasoning."
CHAT_FIRST = "Let's solve this together. Give your first move and explain it briefly."
MAX_MESSAGES = 24


class GameEpisode(Episode):
    def __init__(self, game: Game, verifier: Verifier, policy: Policy, config: EpisodeConfig = EpisodeConfig()):
        self.game, self.verifier, self.policy, self.config = game, verifier, policy, config
        self._msgs: list[Message] = []

    def system(self) -> str:
        return self.game.rules() + "\n" + SYSTEM_SUFFIX

    def solo_prompt(self, state) -> str:
        g, c = self.game, self.config
        parts = [g.render(state)]
        if c.show_legal:
            parts.append("Legal moves: " + ", ".join(g.format(a) for a in g.legal_moves(state)))
        if c.cot_prompt:
            parts.append("Think step by step, then give the move.")
        return "\n".join(parts)

    def messages(self) -> Sequence[Message]:
        return list(self._msgs)

    def play(self, level: int, seed: int) -> Trace:
        g, v, c = self.game, self.verifier, self.config
        t0 = time.time()
        state = g.new(level, seed)
        tr = Trace(game=g.name, level=level, seed=seed, mode=c.mode, variant=self.policy.config.variant,
                   optimal_steps=g.optimal_steps(state))
        sysm = Message("system", self.system())
        chat = c.mode == "chat"
        hist: list[Message] = [sysm]  # chat: the live conversation
        self._msgs = [sysm]  # full transcript
        prompt = (g.render(state) + "\n" + CHAT_FIRST) if chat else self.solo_prompt(state)
        fails = 0
        solved = v.solved(state)
        turn = 0
        while not solved and turn < c.max_turns:
            um = Message("user", prompt)
            if chat:
                while len(hist) > MAX_MESSAGES:
                    del hist[1:3]
                hist.append(um)
                gen = self.policy.generate(hist)
            else:
                gen = self.policy.generate([sysm, um])
            am = Message("assistant", gen.text)
            if chat:
                hist.append(am)
            self._msgs += [um, am]
            action = g.parse(gen.text)
            error = None
            legal = False
            if action is None:
                error = "truncated" if gen.truncated else "unparseable"
                tr.n_unparseable += 1
            elif not v.legal(state, action):
                error = "illegal"
                tr.n_illegal += 1
            else:
                legal = True
            if legal:
                nxt = g.apply(state, action)
                fails = 0
            else:
                nxt = state
                fails += 1
            solved = legal and v.solved(nxt)
            if gen.reasoning:
                tr.reasoning_tokens += max(1, len(gen.reasoning) // 4)
            tr.steps.append(Step(turn=turn, prompt=prompt, reasoning=gen.reasoning, move=Move(gen.text, action),
                                 legal=legal, solved_after=solved, n_tokens=gen.n_tokens, error=error))
            turn += 1
            if solved or fails >= c.max_illegal:
                break
            if chat:
                if legal:
                    prompt = f"OK, that move is legal. Board now:\n{g.render(nxt)}\nNext move?"
                elif action is None:
                    prompt = ("I could not find a move in your reply. Give exactly one move in the format "
                              f"{g.format(g.legal_moves(state)[0])}.")
                else:
                    prompt = v.explain(state, action) + " Try again."
            else:
                prompt = self.solo_prompt(nxt)
            state = nxt
        tr.solved = solved
        tr.wall_s = time.time() - t0
        return tr
