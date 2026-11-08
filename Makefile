PY := .venv/bin/python
Q  := Q_QUIET=1 ./q

.PHONY: check test spec setup gpu

check: spec test

spec:
	@$(Q) $(PY) tools/spec_check.py && echo spec ok

test:
	@$(PY) -m pytest -q -x tests >.q/last.log 2>&1; c=$$?; [ $$c -eq 0 ] || [ $$c -eq 5 ] && echo tests ok || { echo "FAIL $$c .q/last.log"; exit 1; }

# setup: idempotent. uv venv in ~/tr-venv (uv cannot write venvs under /work), .venv symlink, torch CUDA wheel, requirements.
# WSL2 GPU needs /usr/lib/wsl/lib on LD_LIBRARY_PATH (gpu_smoke.py re-execs with it if missing).
TORCH_INDEX ?= https://download.pytorch.org/whl/cu124
setup:
	@export PATH=$$HOME/.local/bin:$$PATH UV_HTTP_TIMEOUT=600; mkdir -p .q; \
	[ -x $$HOME/tr-venv/bin/python ] || uv venv $$HOME/tr-venv --python 3.12 || uv venv $$HOME/tr-venv; \
	[ -e .venv ] || ln -s $$HOME/tr-venv .venv; \
	$(PY) -c "import torch" 2>/dev/null || uv pip install --python $(PY) torch --index-url $(TORCH_INDEX) >.q/setup.log 2>&1 || { echo "FAIL torch .q/setup.log"; exit 1; }; \
	uv pip install --python $(PY) -r requirements.txt >.q/setup.log 2>&1 && echo setup ok || { echo "FAIL setup .q/setup.log"; exit 1; }

gpu:
	@$(Q) $(PY) tools/gpu_smoke.py; c=$$?; cat .q/last.log; exit $$c
