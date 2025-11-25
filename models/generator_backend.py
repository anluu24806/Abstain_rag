from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFGeneratorBackend:
    """
    Wrapper HF LLM:
      - compute_logprob(prompt, answer)
      - generate(prompt)
    """
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 512,
        load_in_4bit: bool = False,
        **gen_kwargs,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.gen_kwargs = gen_kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        self.model.eval()

    @torch.no_grad()
    def compute_logprob(self, prompt: str, answer: str) -> float:
        full = prompt + " " + answer
        inputs = self.tokenizer(full, return_tensors="pt",
                                truncation=True, max_length=self.max_length)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt",
                                    truncation=True, max_length=self.max_length)["input_ids"]

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_len = prompt_ids.shape[1]

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        n_answer_tokens = (labels != -100).sum()
        if n_answer_tokens.item() == 0:
            return float("nan")
        logp = -loss.item() * n_answer_tokens.item()
        return float(logp)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.gen_kwargs,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):]
