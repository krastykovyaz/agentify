# pip install unsloth transformers datasets trl accelerate peft pandas torch

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import pandas as pd
import torch

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen3.5-4B"
CSV_PATH = "datasets/data_multilingual_augmented.csv"
OUTPUT_DIR = "models/qwen35_4b_finetuned_no_reasoning"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# =========================
# LOAD MODEL
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# =========================
# LORA CONFIG
# =========================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

# Проверка колонок
assert "raw_text" in df.columns
assert "ready_text" in df.columns

# Опционально: если есть колонка system
has_system = "system" in df.columns

# =========================
# FORMAT DATASET (КЛЮЧЕВОЕ ИЗМЕНЕНИЕ!)
# =========================
EOS_TOKEN = tokenizer.eos_token

# Qwen использует chat template формат
def format_example(row):
    # Формат Qwen chat template
    if has_system and pd.notna(row.get("system")):
        system_msg = row["system"]
    else:
        # Дефолтный system prompt для подавления reasoning
        system_msg = "Отвечай только финальным ответом, без рассуждений, пояснений и thinking process."
    
    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{row["raw_text"]}<|im_end|>
<|im_start|>assistant
{row["ready_text"]}<|im_end|>{EOS_TOKEN}"""
    
    return {"text": prompt}

formatted_data = [format_example(row) for _, row in df.iterrows()]
dataset = Dataset.from_list(formatted_data)

# Покажем пример для проверки
print("\n=== Пример форматированных данных ===")
print(dataset[0]["text"])
print("=" * 50)

# =========================
# TRAINER
# =========================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,  # Увеличил до 3 эпох
        learning_rate=2e-4,  # Немного увеличил LR
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

# =========================
# TRAIN
# =========================
print("\n=== Начинаем обучение ===\n")
trainer.train()

# =========================
# SAVE LORA
# =========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Model saved to: {OUTPUT_DIR}")
print("\nТеперь квантизируйте модель:")
print(f"llama-quantize {OUTPUT_DIR}/model.gguf {OUTPUT_DIR}/model-q4_k_m.gguf q4_k_m")
