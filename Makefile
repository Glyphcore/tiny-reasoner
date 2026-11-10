PY := .venv/bin/python
export Q_QUIET := 1
.PHONY: setup check spec test
# TORCH_INDEX: CPU wheel by default; for the GTX 1660 Ti use https://download.pytorch.org/whl/cu121
setup:
	python3 -m venv .venv
	bash q .venv/bin/pip install torch --index-url $${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
	bash q .venv/bin/pip install transformers safetensors numpy pytest huggingface_hub
	bash q $(PY) tools/fetch_model.py Qwen/Qwen3-0.6B
check: spec test
spec:
	@$(PY) tools/spec_check.py
test:
	@bash q $(PY) -m pytest -q tests && echo "tests ok"
