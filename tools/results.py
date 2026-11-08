"""Aggregate runs/*/metrics.jsonl into markdown tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = ["nothink-solo", "think-solo", "nothink-chat", "think-chat", "text-chats", "nothink-cot", "nothink-legal"]
HEAD = ["variant", "condition", "game"]
COLS = ["n", "solve", "legal", "steps/opt", "bad", "reason_tok", "s/ep"]
TEXT_COLS = ["variant", "ppl", "n_docs", "chats", "reply_tok", "trunc", "repeat"]


def split_id(run_id: str):
    for c in sorted(CONDITIONS, key=len, reverse=True):
        if run_id.endswith("-" + c):
            return run_id[: -len(c) - 1], c
    return run_id, ""


def load(runs: Path, variants=None):
    """-> list of (variant, condition, records)."""
    out = []
    for d in sorted(Path(runs).iterdir()) if Path(runs).is_dir() else []:
        f = d / "metrics.jsonl"
        if not f.is_file():
            continue
        v, c = split_id(d.name)
        if variants and v not in variants:
            continue
        recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        out.append((v, c, recs))
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _agg(eps):
    steps = sum(e["steps"] for e in eps)
    bad = [e["n_illegal"] + e["n_unparseable"] for e in eps]
    solved = [e for e in eps if e["solved"] and e.get("optimal")]
    legal = 1 - sum(bad) / steps if steps else 0.0
    return [str(len(eps)), f"{_mean([1.0 if e['solved'] else 0.0 for e in eps]):.2f}", f"{legal:.2f}",
            f"{_mean([e['steps'] / e['optimal'] for e in solved]):.2f}" if solved else "-",
            f"{_mean(bad):.2f}", f"{_mean([e['reasoning_tokens'] for e in eps]):.2f}",
            f"{_mean([e['wall_s'] for e in eps]):.2f}"]


def build_rows(runs, variants=None, levels=False):
    """-> (episode_rows, text_rows); rows are lists of strings."""
    groups = {}
    text = {}
    for v, c, recs in load(runs, variants):
        for r in recs:
            if r.get("kind") == "episode":
                groups.setdefault((v, c, r["game"], r["level"] if levels else None), []).append(r)
            elif r.get("kind") in ("text", "chats"):
                text.setdefault(v, {})[r["kind"]] = r
    ep_rows = [[v, c, g] + ([str(lv)] if levels else []) + _agg(eps)
               for (v, c, g, lv), eps in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0))]
    t_rows = []
    for v in sorted(text):
        t, ch = text[v].get("text"), text[v].get("chats")
        t_rows.append([v, f"{t['ppl']:.1f}" if t else "-", str(t["n_docs"]) if t else "-",
                       str(ch["n_chats"]) if ch else "-",
                       f"{ch['mean_reply_tokens']:.2f}" if ch else "-",
                       f"{ch['truncated_rate']:.2f}" if ch else "-",
                       f"{ch['repeat_rate']:.2f}" if ch else "-"])
    return ep_rows, t_rows


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render(ep_rows, t_rows, levels=False):
    head = HEAD + (["level"] if levels else []) + COLS
    parts = [table(head, ep_rows)] if ep_rows else []
    if t_rows:
        parts.append(table(TEXT_COLS, t_rows))
    return "\n\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variants", default="")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--levels", action="store_true")
    a = ap.parse_args(argv)
    vs = [x for x in a.variants.split(",") if x] or None
    e, t = build_rows(a.runs, vs, a.levels)
    print(render(e, t, a.levels))


if __name__ == "__main__":
    main()
