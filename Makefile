PY := .venv/bin/python
Q  := Q_QUIET=1 ./q

.PHONY: check test spec setup gpu

check: spec test

spec:
	@$(Q) $(PY) tools/spec_check.py && echo spec ok

test:
	@$(PY) -m pytest -q -x tests >.q/last.log 2>&1; c=$$?; [ $$c -eq 0 ] || [ $$c -eq 5 ] && echo tests ok || { echo "FAIL $$c .q/last.log"; exit 1; }

# setup, gpu: filled in by the tooling node (uv venv in ~/tr-venv, symlinked to .venv; torch CUDA build; gpu_smoke)
setup:
	@echo "see docs/handoff.md plumbing"

gpu:
	@$(Q) $(PY) tools/gpu_smoke.py && cat .q/last.log
