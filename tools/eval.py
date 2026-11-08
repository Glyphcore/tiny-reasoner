"""Evaluate a policy variant. Prints only the summary line. Details in runs/<id>/."""
import argparse, dataclasses, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def seeds(s: str):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--games", nargs="*", default=["hanoi", "puzzle8", "countdown"], help="name[:l1,l2] ...")
    ap.add_argument("--levels-default", action="store_true", help="use each game's full level list")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--text", default="data/heldout.txt")
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--chats", default="data/chats.json")
    ap.add_argument("--max-chats", type=int, default=20)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--show-legal", action="store_true")
    ap.add_argument("--cot-prompt", action="store_true")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--mode", choices=["solo", "chat", "all"], default="all")
    ap.add_argument("--run")
    ap.add_argument("--edit")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    from spec.interfaces import ChatSuite, EpisodeConfig, GameSuite, RunConfig, TextSuite
    from tr.games import get_game
    from tr.run import new_run
    from tr.eval import Evaluator
    from tr.policy import default_config, load_policy

    sd = seeds(a.seeds)
    suites = []
    for spec in a.games:
        name, _, lv = spec.partition(":")
        levels = [int(x) for x in lv.split(",")] if lv and not a.levels_default else list(get_game(name)[0].levels)
        ep = EpisodeConfig(mode="chat" if a.mode == "chat" else "solo", max_turns=a.max_turns,
                           show_legal=a.show_legal, cot_prompt=a.cot_prompt)
        suites.append(GameSuite(name, levels, sd, ep))
    pc = default_config(a.variant, thinking=a.thinking, max_new_tokens=a.max_new_tokens)
    cfg = RunConfig(policy=pc, games=suites, text=TextSuite(a.text, a.max_docs), chats=ChatSuite(a.chats, a.max_chats),
                    edit=a.edit, note=a.note)
    run = new_run(cfg, a.run)
    pol = load_policy(pc)
    if a.edit:
        try:
            from tr.edits.recipes import load_any as load_edit
        except ImportError:
            sys.exit("--edit needs tr.edits.weightedit, which does not exist yet")
        load_edit(a.edit).apply(pol)
    ev = Evaluator(run)
    if a.mode == "all":
        ev.all(pol, suites, cfg.text, cfg.chats)
    else:
        from spec.interfaces import EvalResult
        r = EvalResult(variant=pc.variant)
        for s in suites:
            r.games += ev.games(pol, s)
        run.summary(r)
    print((Path(run.dir) / "summary.txt").read_text().strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
