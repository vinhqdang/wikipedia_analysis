"""Binoculars zero-shot detector (Hans et al., 2024) with a small observer/performer pair.

score = log-perplexity under the performer / cross-perplexity between observer and performer.
Lower scores mean "more machine-like".
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class Binoculars:
    def __init__(self, observer="Qwen/Qwen2.5-3B", performer="Qwen/Qwen2.5-3B-Instruct", device="cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(observer)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.observer = AutoModelForCausalLM.from_pretrained(observer, **kw).to(device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(performer, **kw).to(device).eval()

    @torch.inference_mode()
    def score(self, texts, max_len=512, batch_size=4):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(texts), batch_size):
            idx = order[s : s + batch_size]
            enc = self.tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                           return_tensors="pt", padding_side="right").to(self.device)
            obs = self.observer(**enc).logits[:, :-1]
            per = self.performer(**enc).logits[:, :-1]
            target = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            for b, i in enumerate(idx):
                m = mask[b]
                o, p, t = obs[b][m].float(), per[b][m].float(), target[b][m]
                logp = F.log_softmax(p, dim=-1)
                ppl = -logp.gather(-1, t[:, None]).mean()
                xppl = -(F.softmax(o, dim=-1) * logp).sum(-1).mean()
                out[i] = (ppl / xppl).item()
        return out
