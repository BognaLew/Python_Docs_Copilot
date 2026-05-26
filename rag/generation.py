import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from langchain_huggingface import HuggingFacePipeline

MAX_NEW_TOKENS = os.getenv('MAX_NEW_TOKENS')
TEMPERATURE = os.getenv('TEMPERATURE')

class Generator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.llm = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
                return_full_text=False,
                repetition_penalty=1.1,
            ),
        )

        return
    