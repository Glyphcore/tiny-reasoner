"""Print the results table (markdown rows) from runs/*/config.json + metrics.jsonl summary rows. No numbers typed by hand."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["base", "prompt-only"]


def row(rid):
    cfg = json.loads((ROOT / "runs" / rid / "config.json").read_text())
    s = [json.loads(l) for l in (ROOT / "runs" / rid / "metrics.jsonl").read_text().splitlines()
         if '"kind": "summary"' in l][0]
    c = s["confusion"]
    what = (cfg["edit"] or {}).get("description", "none") if cfg["prompt"] == "plain" else "hint added to system prompt"
    fam = (cfg["edit"] or {}).get("family", "prompt" if cfg["prompt"] != "plain" else "base")
    steps = "-" if s["steps"] != s["steps"] else f"{s['steps']:.1f}"
    return (f"| {rid} | {fam} | {what[:70]} | {s['solve']:.2f} | {s['legal']:.2f} | {steps} | "
            f"{s['judge_acc']:.2f} ({c['says_legal_is_legal']}/{c['says_legal_is_illegal']}/{c['says_illegal_is_legal']}/"
            f"{c['says_illegal_is_illegal']}) | {s['call_hit']:.2f} | {s['ppl']:.2f} | {s['chats_ok']} | "
            f"{'yes' if cfg['base_hash'] == cfg['reverted_hash'] else 'NO'} |")


ids = ORDER + sorted(p.name for p in (ROOT / "runs").iterdir() if (p / "metrics.jsonl").exists()
                     and p.name not in ORDER and (p / "config.json").exists() and p.name.startswith("e-"))
print("| run | family | what changed | solve | legal | steps | judge acc (LL/LI/IL/II) | call hit | ppl | chats | revert exact |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in ids:
    if "--ids" not in sys.argv or r in sys.argv:
        print(row(r))
