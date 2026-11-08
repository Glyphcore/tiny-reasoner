"""HFPolicy: Qwen3-0.6B wrapper implementing spec.interfaces.Policy."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if os.path.isdir("/usr/lib/wsl/lib") and "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    # must be set before the process starts to matter for dlopen; harmless otherwise (torch finds libcuda via ldconfig on this host)
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from spec.interfaces import Generation, Message, PolicyConfig  # noqa: E402

transformers.logging.set_verbosity_error()

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def default_config(variant: str = "base", **kw) -> PolicyConfig:
    dev = kw.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
    return PolicyConfig(model=MODEL, revision=REVISION, variant=variant, device=dev, **kw)


class HFPolicy:
    def __init__(self, config: PolicyConfig):
        self.config = config
        dtype = getattr(torch, config.dtype)
        if config.device == "cpu" and dtype == torch.float16:
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        self.model = AutoModelForCausalLM.from_pretrained(config.model, revision=config.revision, torch_dtype=dtype)
        self.model.to(config.device).eval()
        self._gd = None          # CUDA-graph greedy decoder (tr.fastgen), built on first use
        self.no_graph = False    # set True to force HF generate

    @torch.no_grad()
    def generate(self, messages: Sequence[Message]) -> Generation:
        c = self.config
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=c.thinking)
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=c.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if c.temperature == 0:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            torch.manual_seed(c.seed)
            torch.cuda.manual_seed_all(c.seed)
            kw.update(do_sample=True, temperature=c.temperature)
        eos = self.tokenizer.eos_token_id
        stop = {eos, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        if c.temperature == 0 and self.model.device.type == "cuda" and not self.no_graph:
            if self._gd is None:
                from tr.fastgen import GraphDecoder
                self._gd = GraphDecoder(self.model)
            out = torch.tensor(self._gd.generate(ids["input_ids"], c.max_new_tokens, stop), device=self.model.device)
        else:
            out = self.model.generate(**ids, **kw)[0, ids["input_ids"].shape[1]:]
        n = int(out.shape[0])
        stopped = n > 0 and int(out[-1]) in stop
        raw = self.tokenizer.decode(out, skip_special_tokens=True)
        reasoning, text = "", raw
        m = _THINK.search(raw)
        if m:
            reasoning, text = m.group(1).strip(), (raw[:m.start()] + raw[m.end():])
        elif "</think>" in raw:
            reasoning, text = raw.split("</think>", 1)[0].replace("<think>", "").strip(), raw.split("</think>", 1)[1]
        elif c.thinking and "<think>" in raw:  # cut off inside the thinking block
            reasoning, text = raw.replace("<think>", "", 1).strip(), ""
        return Generation(text=text.strip(), reasoning=reasoning, n_tokens=n, truncated=(n >= c.max_new_tokens and not stopped))

    @torch.no_grad()
    def nll(self, text: str) -> tuple[float, int]:
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(self.model.device)
        if ids.shape[1] < 2:
            return 0.0, 0
        logits = self.model(ids).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0]
        return float(-lp.sum()), int(lp.shape[0])

    def tensor(self, name: str) -> torch.Tensor:
        return dict(self.model.named_parameters())[name]

    def tensor_names(self) -> Sequence[str]:
        return [n for n, _ in self.model.named_parameters()]


def load_policy(config: PolicyConfig) -> HFPolicy:
    return HFPolicy(config)
