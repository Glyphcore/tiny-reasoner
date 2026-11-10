"""HF model wrapper. Device-agnostic: cuda -> fp16, cpu -> fp32. Greedy decoding with KV-cache reuse across turns."""
from __future__ import annotations

import os
from pathlib import Path

import torch

MODELS = {  # name -> (repo, revision, licence)
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "apache-2.0"),
    "qwen3-0.6b-base": ("Qwen/Qwen3-0.6B-Base", "", "apache-2.0"),
}
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "hf"


def snapshot(repo, revision=""):
    d = CACHE / ("models--" + repo.replace("/", "--")) / "snapshots"
    if revision and (d / revision).exists():
        return d / revision, revision
    if d.exists() and not revision:
        rev = sorted(p.name for p in d.iterdir())[0]
        return d / rev, rev
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(repo, revision=revision or None, cache_dir=str(CACHE),
                               allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    return p, p.name


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


class HFPolicy:
    def __init__(self, name="qwen3-0.6b", model=None, tok=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo, rev, lic = MODELS[name]
        self.model_id, self.license = repo, lic
        self.device, self.dtype = pick_device()
        self.threads = int(os.environ.get("TR_THREADS", os.cpu_count() or 1))
        torch.set_num_threads(self.threads)
        if model is None:
            path, rev = snapshot(repo, rev)
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path, dtype=self.dtype).to(self.device)
        self.revision = rev
        self.model, self.tok = model.eval(), tok
        self._ids, self._cache = [], None

    def render(self, messages, prefix=""):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return text + prefix

    def reset_cache(self):
        self._ids, self._cache = [], None

    @torch.no_grad()
    def _generate(self, ids, max_new_tokens):
        n = 0
        while n < min(len(ids), len(self._ids)) and ids[n] == self._ids[n]:
            n += 1
        n = min(n, len(ids) - 1)
        if self._cache is None or n == 0:
            self._cache, n = None, 0
        elif self._cache.get_seq_length() > n:
            self._cache.crop(n - self._cache.get_seq_length())  # negative: drop that many tokens from the end
        feed = torch.tensor([ids[n:]], device=self.device)
        out, fed = [], list(ids)
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        for _ in range(max_new_tokens):
            r = self.model(input_ids=feed, past_key_values=self._cache, use_cache=True)
            self._cache = r.past_key_values
            nxt = int(r.logits[0, -1].argmax())
            if nxt in eos:
                break
            out.append(nxt)
            fed.append(nxt)
            feed = torch.tensor([[nxt]], device=self.device)
        self._ids = fed[:-1] if out and len(out) == max_new_tokens else fed
        if len(self._ids) != self._cache.get_seq_length():
            self._ids = self._ids[: self._cache.get_seq_length()]
        return out

    def reply(self, messages, max_new_tokens):
        ids = self.tok(self.render(messages)).input_ids
        out = self._generate(ids, max_new_tokens)
        return self.tok.decode(out, skip_special_tokens=True).strip(), len(out)

    @torch.no_grad()
    def nll(self, text):
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        logits = self.model(input_ids=ids).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
        return float(loss), ids.shape[1] - 1

    @torch.no_grad()
    def forward_text(self, text, hidden=False):
        """One forward over raw text; returns last-position logits (and all-layer residuals at the last position)."""
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        r = self.model(input_ids=ids, output_hidden_states=hidden)
        logits = r.logits[0, -1].float().cpu()
        if hidden:
            return logits, torch.stack([h[0, -1].float().cpu() for h in r.hidden_states])
        return logits
