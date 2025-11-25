import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFGeneratorBackend:
    """
    Wrapper đơn giản quanh HF causal LM để:
      - load model lớn (Mistral-7B, LLaMA, ...)
      - cung cấp hàm generate(prompt, ...) trả về text.

    Dùng được cho cả tiny-gpt2 lẫn mistralai/Mistral-7B-Instruct-v0.2.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 512,
        torch_dtype=torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model (kiểu "an toàn VRAM" giống script teacher)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",        # để HF tự map sang GPU
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # gen_kwargs mặc định (sẽ được update bởi arg truyền vào generate)
        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> str:
        """
        Sinh 1 câu trả lời text từ prompt.
        Các tham số temperature/top_p/do_sample nếu truyền vào sẽ override self.gen_kwargs.
        """
        # Cập nhật gen_kwargs local
        gen_kwargs = dict(self.gen_kwargs)
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
            # nếu temperature > 0 thì mặc định bật sampling
            if temperature > 0:
                gen_kwargs["do_sample"] = True
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if do_sample is not None:
            gen_kwargs["do_sample"] = do_sample

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

        # Bỏ phần input, chỉ lấy phần mới sinh
        gen_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()
