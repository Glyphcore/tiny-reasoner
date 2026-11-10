"""Download a model snapshot into .cache/hf and print one line: repo revision path."""
import sys, os
from huggingface_hub import snapshot_download, model_info
repo = sys.argv[1]
info = model_info(repo)
p = snapshot_download(repo, revision=info.sha, cache_dir=".cache/hf",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*", "*.jinja"])
lic = (info.card_data or {}).get("license") if info.card_data else None
print(repo, info.sha, lic, p)
