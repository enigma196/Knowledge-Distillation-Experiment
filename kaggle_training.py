import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = "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"
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import TrainerCallback
from datetime import datetime

# 1. Configuration & Model Selection
model_id = "Qwen/Qwen2-0.5B-Instruct"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class LossLoggingCallback(TrainerCallback):
    def __init__(self, file_path="./training_loss.txt"):
        self.file_path = file_path
        
        # Initialize the text file with headers if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write("Timestamp | Epoch | Step | Train Loss | Eval Loss\n")
                f.write("-" * 60 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # This method triggers every time logging_steps is reached
        if logs is not None:
            # We only write if a loss value actually exists in this step
            train_loss = logs.get("loss", "N/A")
            eval_loss = logs.get("eval_loss", "N/A")
            epoch = round(state.epoch, 2) if state.epoch is not None else "N/A"
            step = state.global_step
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Format values cleanly (e.g., limit decimals if float)
            if isinstance(train_loss, float):
                train_loss = f"{train_loss:.4f}"
            if isinstance(eval_loss, float):
                eval_loss = f"{eval_loss:.4f}"

            # Only write if we actually have some training progress logged
            if train_loss != "N/A" or eval_loss != "N/A":
                log_line = f"{timestamp} | Epoch: {epoch} | Step: {step} | Train Loss: {train_loss} | Eval Loss: {eval_loss}\n"
                
                # Append to the file
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
                    f.flush()  # Force write immediately to disk
loss_logger = LossLoggingCallback(file_path="./training_loss.txt")

# 2. Load Tokenizer & Model with 4-bit Quantization
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",             # Automatically handles splitting across T4 x2
    trust_remote_code=True
)

# 3. PEFT/LoRA Setup
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Prepare Dataset
dataset = load_dataset("json", data_files="/kaggle/input/datasets/sawadogogsamuel/teacher2/generation_results_mistral01_reduce.jsonl", split="train")

# Map your "input" and "output" keys to the chat template Qwen expects
def format_to_chat(example):
    prompt = f"""Génère { example['n_questions'] } questions à partir du texte suivant.
    Texte :
    {example['input']}
    Catégories :
    {example['type']}"""
    
    return {
        "messages": [
            {"role": "system", "content": "Tu es un assistant serviable, issu d'un grand modèle enseignant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["output"]}
        ]
    }

# Apply mapping and remove old columns to keep memory clean
dataset = dataset.map(format_to_chat, remove_columns=["input", "output", "type", "n_questions"])

# 5. Training Configurations
training_args = SFTConfig(
    output_dir="./qwen2-0.5b-distilled", # Enregistrement dans le répertoire de Kaggle
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2, 
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,                
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_strategy="epoch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    # max_length=2048,                 # Utiliser la longueur maximum que ce modèle propose
    packing=True,                         
    dataset_text_field="messages",     
)

# 6. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[loss_logger]
    # dataset_kwargs={"skip_prepare_dataset": True} # FIX 2: Prevents cross-device multi-GPU crashes
)

# 7. Start the Distillation Experiment
trainer.train()

# 8. Save the trained adapter weights
trainer.model.save_pretrained("./final_adapter")

# Clear VRAM to prepare for merging cleanly
del model
del trainer
torch.cuda.empty_cache()

# 9. Load base model and merge on CPU (to avoid VRAM memory splits)
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
merged_model = PeftModel.from_pretrained(base_model, "./final_adapter")
merged_model = merged_model.merge_and_unload()

# Save the unified standalone model to Kaggle Output
merged_model.save_pretrained("./qwen2-0.5b-distilled-final")
tokenizer.save_pretrained("./qwen2-0.5b-distilled-final")

print("🎉 Finished! Your weights are safely compiled in /kaggle/working/qwen2-0.5b-distilled-final")