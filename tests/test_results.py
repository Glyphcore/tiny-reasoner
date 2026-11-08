import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import results  # noqa: E402


def ep(game, level, solved, steps, opt, ill, unp, rt, w):
    return {"kind": "episode", "game": game, "level": level, "seed": 0, "mode": "solo", "solved": solved,
            "steps": steps, "optimal": opt, "n_illegal": ill, "n_unparseable": unp, "reasoning_tokens": rt, "wall_s": w}


def mk(root, name, recs):
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_tables(tmp_path):
    mk(tmp_path, "base-nothink-solo", [ep("hanoi", 2, True, 6, 3, 0, 0, 10, 2.0), ep("hanoi", 2, False, 4, 3, 2, 1, 20, 4.0),
                                       ep("hanoi", 3, True, 7, 7, 1, 0, 0, 6.0)])
    mk(tmp_path, "tv-p0.3-think-chat", [ep("hanoi", 2, False, 5, 3, 5, 0, 0, 1.0)])
    mk(tmp_path, "base-text-chats", [
        {"kind": "text", "path": "x", "n_docs": 8, "n_tokens": 100, "ppl": 12.345},
        {"kind": "chats", "path": "y", "n_chats": 4, "mean_reply_tokens": 50.0, "truncated_rate": 0.25,
         "repeat_rate": 0.0, "transcripts_dir": "z"}])
    (tmp_path / "empty").mkdir()
    e, t = results.build_rows(tmp_path)
    assert e[0] == ["base", "nothink-solo", "hanoi", "3", "0.67", "0.76", "1.50", "1.33", "10.00", "4.00"]
    assert e[1] == ["tv-p0.3", "think-chat", "hanoi", "1", "0.00", "0.00", "-", "5.00", "0.00", "1.00"]
    assert t == [["base", "12.3", "8", "4", "50.00", "0.25", "0.00"]]
    el, _ = results.build_rows(tmp_path, ["base"], levels=True)
    assert [r[3] for r in el] == ["2", "3"] and len(el) == 2
    out = results.render(e, t)
    assert out.splitlines()[0].startswith("| variant | condition | game | n | solve")
    assert "| variant | ppl |" in out
