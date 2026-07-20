import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from peft import PeftModel

model_id = "Qwen/Qwen2-0.5B-Instruct"
os.environ["HF_TOKEN"] = "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"

# 2. Load Tokenizer & Model with 4-bit Quantization
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cpu")
merged_model = PeftModel.from_pretrained(base_model, "./final_adapter")
merged_model = merged_model.merge_and_unload()

# Save the unified standalone model
merged_model.save_pretrained("./qwen2-0.5b-distilled-final")
tokenizer.save_pretrained("./qwen2-0.5b-distilled-final")