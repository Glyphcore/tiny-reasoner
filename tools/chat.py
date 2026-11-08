"""Chat with a policy. Interactive REPL, or --script data/chats.json --ids a,b. Writes runs/<run>/chats/<chat-id>.json."""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--edit", help="path of a saved WeightEdit (tr.edits.weightedit.load_edit)")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--script")
    ap.add_argument("--ids", default="")
    ap.add_argument("--run")
    a = ap.parse_args()

    from tr.policy import default_config, load_policy
    pol = load_policy(default_config(a.variant, thinking=a.thinking, max_new_tokens=a.max_new_tokens))
    if a.edit:
        try:
            from tr.edits.recipes import load_any as load_edit
        except ImportError:
            sys.exit("--edit needs tr.edits.weightedit, which does not exist yet")
        load_edit(a.edit).apply(pol)

    from spec.interfaces import Message
    run = a.run or "chat-" + time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "runs" / run / "chats"
    out.mkdir(parents=True, exist_ok=True)

    def session(cid, user_turns):
        hist, rec = [], []
        for u in user_turns:
            hist.append(Message("user", u)); rec.append({"role": "user", "content": u, "reasoning": ""})
            g = pol.generate(hist)
            hist.append(Message("assistant", g.text)); rec.append({"role": "assistant", "content": g.text, "reasoning": g.reasoning})
        p = out / f"{cid}.json"
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return p

    if a.script:
        chats = {c["id"]: c for c in json.loads(Path(a.script).read_text())}
        ids = [i for i in a.ids.split(",") if i] or list(chats)
        paths = [session(i, chats[i]["turns"]) for i in ids]
        print(paths[-1] if len(paths) == 1 else out)
        return 0

    hist, rec = [], []
    while True:
        try:
            u = input("you> ").strip()
        except EOFError:
            break
        if not u:
            continue
        hist.append(Message("user", u)); rec.append({"role": "user", "content": u, "reasoning": ""})
        g = pol.generate(hist)
        hist.append(Message("assistant", g.text)); rec.append({"role": "assistant", "content": g.text, "reasoning": g.reasoning})
        print("bot>", g.text)
    p = out / f"{run}.json"
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
