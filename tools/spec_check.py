"""Fail on drift between spec/interfaces.py, spec/diagrams.md and the code.

Checks: (1) interfaces.py holds types only (method bodies are `...`), (2) every spec class has a mermaid
class box with exactly the same member names, (3) every Protocol has an implementation in tr/ with all methods.
Prints one line.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMPL = {"Game": ["tr.games.Hanoi", "tr.games.Countdown", "tr.games.Grid"], "Verifier": ["tr.games.ExactVerifier"],
        "Policy": ["tr.policy.HFPolicy"]}


def spec_classes():
    tree = ast.parse((ROOT / "spec/interfaces.py").read_text())
    out, errs = {}, []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            errs.append(f"top-level function {node.name} in spec")
        if not isinstance(node, ast.ClassDef):
            continue
        members = set()
        for b in node.body:
            if isinstance(b, ast.AnnAssign):
                members.add(b.target.id)
            elif isinstance(b, ast.FunctionDef):
                members.add(b.name)
                body = [s for s in b.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                                  and isinstance(s.value.value, str))]
                if not (len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                        and body[0].value.value is Ellipsis):
                    errs.append(f"{node.name}.{b.name} has a body")
        out[node.name] = members
    return out, errs


def diagram_classes():
    text = (ROOT / "spec/diagrams.md").read_text()
    out = {}
    for m in re.finditer(r"class (\w+) \{(.*?)\}", text, re.S):
        members = set()
        for line in m.group(2).strip().splitlines():
            line = line.strip()
            if not line or line.startswith("<<"):
                continue
            members.add(re.match(r"[+\-#]?(\w+)", line).group(1))
        out[m.group(1)] = members
    return out


def main():
    spec, errs = spec_classes()
    dia = diagram_classes()
    for name in sorted(set(spec) | set(dia)):
        if name not in dia:
            errs.append(f"no diagram box for {name}")
        elif name not in spec:
            errs.append(f"diagram box {name} not in spec")
        elif spec[name] != dia[name]:
            errs.append(f"{name} members differ: {sorted(spec[name] ^ dia[name])}")
    sys.path.insert(0, str(ROOT))
    import importlib
    for proto, impls in IMPL.items():
        for path in impls:
            mod, cls = path.rsplit(".", 1)
            try:
                c = getattr(importlib.import_module(mod), cls)
            except Exception as e:  # noqa: BLE001
                errs.append(f"{path}: {type(e).__name__}")
                continue
            methods = {m for m in spec[proto] if m not in ("name", "model_id", "revision")}
            missing = [m for m in methods if not callable(getattr(c, m, None))]
            if missing:
                errs.append(f"{path} lacks {missing}")
    if errs:
        print("spec_check FAIL: " + "; ".join(errs))
        sys.exit(1)
    print(f"spec_check ok: {len(spec)} types")


if __name__ == "__main__":
    main()
