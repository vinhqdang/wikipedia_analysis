"""Generate machine-written Wikipedia-style leads for known titles (detector validation set)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = ("Write the opening section of an English Wikipedia article titled \"{title}\". "
          "Use an encyclopedic tone, plain prose, about 250 words, no headings and no bullet points.")


@torch.inference_mode()
def generate(model_name, titles, batch_size=16, max_new_tokens=380, device="cuda", seed=0):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
    outs = []
    for s in range(0, len(titles), batch_size):
        chunk = titles[s : s + batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(title=t)}], tokenize=False,
                                           add_generation_prompt=True) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return outs
