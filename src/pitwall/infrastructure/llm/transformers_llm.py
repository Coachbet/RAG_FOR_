"""LLM adapter: a small open model run with Hugging Face transformers, exactly like the Session 5.2 lab
(Qwen/Qwen2.5-1.5B-Instruct on GPU, 0.5B on CPU). Greedy decoding for reproducibility.
"""
from __future__ import annotations


class TransformersLLM:
    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.name = f"hf/{model}"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(self.device)

    def generate(self, system: str, user: str, max_tokens: int = 500) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
