"""Greedy decoding with a CUDA graph per token (no compiler needed).

Prefill runs eagerly into a StaticCache; the single-token decode step (forward, argmax, feed back, position += 1, append to an
output buffer) is captured once as a CUDA graph and replayed, which removes the ~2000 kernel launches per token that dominate
on WSL. Stop tokens are checked every `chunk` replays, so a reply may run a few tokens past the stop token; they are trimmed.
Weight edits mutate parameters in place (copy_), so the captured graph sees them. Call `invalidate()` if a parameter tensor is
ever replaced rather than copied into.
"""
from __future__ import annotations

import torch


class GraphDecoder:
    def __init__(self, model, max_len: int = 3072, chunk: int = 8):
        from transformers import StaticCache
        self.model = model
        self.device = model.device
        self.max_len = max_len
        self.chunk = chunk
        cfg = model.config
        try:
            self.cache = StaticCache(config=cfg, max_cache_len=max_len)
        except TypeError:  # older signature
            self.cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=max_len, device=self.device, dtype=model.dtype)
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.idx = torch.zeros(1, dtype=torch.long, device=self.device)
        self.buf = torch.zeros(max_len, dtype=torch.long, device=self.device)
        self.graph = None

    def invalidate(self):
        self.graph = None

    def _step(self):
        out = self.model(input_ids=self.tok, past_key_values=self.cache, cache_position=self.pos, use_cache=True)
        nxt = out.logits[0, -1].argmax().view(1)
        self.buf.index_copy_(0, self.idx, nxt)
        self.tok.copy_(nxt.view(1, 1))
        self.pos += 1
        self.idx += 1

    def _ensure_graph(self):
        if self.graph is not None:
            return
        # dummy prefill so the cache has a valid state, then warm up and capture on a side stream
        dummy = torch.full((1, 8), 1, dtype=torch.long, device=self.device)
        self.cache.reset()
        self.model(input_ids=dummy, past_key_values=self.cache, cache_position=torch.arange(8, device=self.device), use_cache=True)
        self.tok.fill_(1); self.pos.fill_(8); self.idx.fill_(0)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, stop_ids: set[int]) -> list[int]:
        """input_ids (1, L) on device. Returns generated token ids (stop token included when hit)."""
        L = input_ids.shape[1]
        if L + max_new_tokens + self.chunk + 1 > self.max_len:
            raise ValueError(f"prompt {L} + {max_new_tokens} exceeds static cache {self.max_len}")
        self._ensure_graph()
        self.cache.reset()
        out = self.model(input_ids=input_ids, past_key_values=self.cache,
                         cache_position=torch.arange(L, device=self.device), use_cache=True)
        nxt = out.logits[0, -1].argmax().view(1)
        self.buf[0] = nxt[0]
        self.tok.copy_(nxt.view(1, 1))
        self.pos.fill_(L)
        self.idx.fill_(1)
        done = 1
        if int(nxt) in stop_ids:
            return [int(nxt)]
        while done < max_new_tokens:
            n = min(self.chunk, max_new_tokens - done)
            for _ in range(n):
                self.graph.replay()
            new = self.buf[done:done + n].tolist()
            done += n
            for i, t in enumerate(new):
                if t in stop_ids:
                    return self.buf[:done - n + i + 1].tolist()
        return self.buf[:done].tolist()
