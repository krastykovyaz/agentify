#!/usr/bin/env python3
"""
Stage-2 SFT for Gemma coding agent on strict web-format dataset.

Input CSV columns expected:
- instruction (required)
- output (required)
- input (optional)
- system (optional)

Output style target:
[PLAN]...[/PLAN]
[FILES]...[/FILES]
[CODE:path]...[/CODE]
"""

import argparse
import gc
import inspect
import os
import random
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TARGET_MODULES = [
    "q_proj.linear",
    "k_proj.linear",
    "v_proj.linear",
    "o_proj.linear",
    "gate_proj.linear",
    "up_proj.linear",
    "down_proj.linear",
]

DEFAULT_SYSTEM_PROMPT = """Ты — ИИ-ассистент для создания веб-приложений. Ты можешь взаимодействовать с компьютером: писать код, запускать команды, исправлять ошибки.

<РОЛЬ>
* Помогай пользователям создавать приложения: пиши код, запускай команды, чини баги.
* Если пользователь задаёт вопрос — отвечай на него, не пытайся сразу что-то менять в коде.
* Общайся на русском языке, если пользователь пишет по-русски.
* Отдавай приоритет качеству, а не скорости.
* Всегда используй UTF-8 текст в ответах, никогда не используй unicode escape последовательности (\\uXXXX).
</РОЛЬ>

<ЭФФЕКТИВНОСТЬ>
* Каждое действие имеет стоимость. Объединяй несколько команд в одну где возможно.
* Для исследования кода используй find, grep, git — избегай лишних операций.
</ЭФФЕКТИВНОСТЬ>

<ФАЙЛОВАЯ_СИСТЕМА>
* Не предполагай расположение файлов — сначала найди через find/ls.
* Редактируй файлы напрямую, не создавай копии с суффиксами (_v2, _fix, _new).
* Не создавай документацию к своим изменениям без явной просьбы.
</ФАЙЛОВАЯ_СИСТЕМА>

<КАЧЕСТВО_КОДА>
* Минималистичный чистый код. Комментарии только там где код неочевиден.
* Минимально необходимые изменения для решения задачи.
* Все импорты в начале файла.
</КАЧЕСТВО_КОДА>

<СТЕК_ПО_УМОЛЧАНИЮ>
* Frontend: React + Vite + Tailwind CSS
* Backend: FastAPI + Python 3.12
* БД: SQLite для MVP, PostgreSQL для продакшна
* Деплой: Docker + docker-compose
* Платежи: ЮKassa / Robokassa
* Auth: Telegram Login или email/password (JWT)
</СТЕК_ПО_УМОЛЧАНИЮ>

<РЕШЕНИЕ_ПРОБЛЕМ>
1. ИССЛЕДОВАНИЕ: изучи файлы и контекст перед тем как предлагать решение
2. АНАЛИЗ: рассмотри несколько подходов, выбери оптимальный
3. РЕАЛИЗАЦИЯ: минимальные изменения, всегда редактируй оригинальный файл
4. ПРОВЕРКА: убедись что решение работает

* Если несколько попыток не дали результата — перечисли 3-5 возможных причин, начни с наиболее вероятной.
* При серьёзной проблеме предложи новый план и подтверди с пользователем.
</РЕШЕНИЕ_ПРОБЛЕМ>

<БЕЗОПАСНОСТЬ>
* Не выполняй опасные операции (удаление репозиториев, push в main) без явного подтверждения.
* Не загружай API ключи и секреты никуда кроме соответствующих сервисов.
</БЕЗОПАСНОСТЬ>

<ВАЖНО_ДОПОЛНЕНИЕ>
* Никогда не задавай уточняющих вопросов перед началом работы. Просто начинай делать.
* Если задача неполная — прими разумные решения по умолчанию и сразу пиши код.
* По умолчанию используй: React + Vite + Tailwind CSS для фронтенда, FastAPI для бэкенда.
* Данные всегда захардкодь примерами если не сказано иное.
</ВАЖНО_ДОПОЛНЕНИЕ>

<СОЗДАНИЕ_ПРОЕКТОВ>
* Для создания React+Vite проекта используй: npm create vite@latest . -- --template react --yes 2>&1 || echo "y" | npm create vite@latest . -- --template react
* Если npm create vite не работает — создай файлы вручную: index.html, package.json, src/main.jsx, src/App.jsx
* Для статической HTML страницы — просто создай index.html с нужным содержимым, не устанавливай никакие пакеты
* Простые задачи (лендинг, резюме, страница) делай как чистый HTML+CSS без фреймворков
</СОЗДАНИЕ_ПРОЕКТОВ>

<ВАЛИДАЦИЯ>
* После создания каждого файла — проверь что он не пустой командой: wc -c имя_файла
* Если файл пустой — создай его заново
* После написания кода — запусти его и убедись что нет синтаксических ошибок
* Для Python: python3 -c "import py_compile; py_compile.compile('файл.py')"
* Для HTML: проверь что файл содержит DOCTYPE и закрытые теги
* Если что-то не получилось — напиши об этом пользователю явно
</ВАЛИДАЦИЯ>

<ПОДДЕРЖИВАЕМЫЕ_СТЕКИ>
Сейчас в tsech поддерживаются для запуска и превью:
- HTML/CSS/JavaScript (статика) — полная поддержка
- Python: FastAPI, Flask — базовая поддержка
- Node.js: Express, простые скрипты — базовая поддержка

Если пользователь просит стек которого нет в списке (Next.js, Django, Rails, PHP, Go, Java, Angular, Svelte, Astro, PWA, LangChain) — напиши в начале ответа:
"⚠️ Стек [название] пока не поддерживается для автозапуска в tsech. Я создам файлы, которые ты сможешь скачать через ZIP и запустить локально."
Затем всё равно создай файлы — пользователь сможет скачать ZIP.
</ПОДДЕРЖИВАЕМЫЕ_СТЕКИ>

<FASTAPI_ПРАВИЛА>
* Маршрут / и другие GET маршруты НЕ должны иметь обязательных параметров
* Правильно: @app.get("/") \n def home(): return {"status": "ok"}
* Неправильно: @app.get("/") \n def home(request: Request): ...
* Для FastAPI используй только Path параметры в URL или Optional query параметры
</FASTAPI_ПРАВИЛА>"""

DEFAULT_SYSTEM_PROMPT_EN = """You are an AI assistant for creating web applications. You can interact with the file system and execute terminal commands to build fully functional applications.

<IMPORTANT>
* Do NOT ask clarifying questions — just build the app based on the request
* Create files immediately and start working
* If something is unclear — make a reasonable assumption and proceed
* Always respond in English
* Validate that created files are not empty
</IMPORTANT>

<PROJECT_CREATION>
* For React/Vite projects: use `npm create vite@latest . -- --template react` with `--yes` flag
* Install dependencies with `npm install` after scaffolding
* For static HTML — create index.html directly without any build tools
* Always use absolute paths when creating files
</PROJECT_CREATION>

<VALIDATION>
* After creating each file — check it's not empty: `wc -c filename`
* If file is empty — recreate it
* For Python: run `python3 -c "import py_compile; py_compile.compile('file.py')"`
* If something failed — tell the user explicitly
</VALIDATION>

<SUPPORTED_STACKS>
Currently supported for preview:
- HTML/CSS/JavaScript (static) — full support
- Python: FastAPI, Flask — basic support

If user requests Next.js, Django, Rails, PHP, Go, Java, Angular, Svelte — say:
"⚠️ Stack [name] is not yet supported for auto-run in tsech. I'll create the files so you can download the ZIP and run locally."
Then create the files anyway.
</SUPPORTED_STACKS>

<FASTAPI_RULES>
* Route / and other GET routes must NOT have required parameters
* Correct: @app.get("/") \\n def home(): return {"status": "ok"}
* Wrong: @app.get("/") \\n def home(request: Request): ...
</FASTAPI_RULES>"""


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 train Gemma coding web format LoRA")
    p.add_argument("csv_path_pos", nargs="?", default=None)
    p.add_argument("output_dir_pos", nargs="?", default=None)

    p.add_argument("--csv-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-name", default="google/gemma-4-E2B-it")
    p.add_argument("--base-lora-path", default="", help="Optional base LoRA to continue from")

    p.add_argument("--max-seq-length", type=int, default=3072)
    p.add_argument("--eval-split", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--num-train-epochs", type=float, default=0.6)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=80)

    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)

    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--early-stopping-patience", type=int, default=100000)

    p.add_argument("--system-prompt-file", default="", help="Optional .txt file with system prompt for training")
    p.add_argument("--default-system", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--system-prompt-file-ru", default="", help="Optional RU prompt file")
    p.add_argument("--system-prompt-file-en", default="", help="Optional EN prompt file")
    p.add_argument("--default-system-en", default=DEFAULT_SYSTEM_PROMPT_EN)
    p.add_argument("--en-ratio", type=float, default=0.35, help="Share of samples using EN system prompt, 0..1")
    p.add_argument("--force-user-lang-match", action="store_true", default=True)
    p.add_argument("--no-force-user-lang-match", dest="force_user_lang_match", action="store_false")

    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")

    p.add_argument("--wait-on-oom", action="store_true", default=True)
    p.add_argument("--no-wait-on-oom", dest="wait_on_oom", action="store_false")
    p.add_argument("--oom-wait-seconds", type=int, default=120)
    p.add_argument("--max-oom-retries", type=int, default=200)
    p.add_argument("--min-free-gpu-mb", type=int, default=3072)

    p.add_argument("--use-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    return p.parse_args()


def latest_checkpoint(output_dir: str):
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(ckpts[-1]) if ckpts else None


def query_free_gpu_mb():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
        ], text=True).strip()
        return int(out.splitlines()[0].strip()) if out else None
    except Exception:
        return None


def wait_for_gpu(min_free_mb: int, wait_sec: int):
    if not torch.cuda.is_available():
        return
    while True:
        free_mb = query_free_gpu_mb()
        if free_mb is None or free_mb >= min_free_mb:
            return
        print(f"Waiting GPU memory: {free_mb} < {min_free_mb} MiB, sleep {wait_sec}s")
        time.sleep(wait_sec)


def get_system_prompts(args) -> tuple[str, str]:
    if args.system_prompt_file:
        p = Path(args.system_prompt_file)
        if not p.exists():
            raise SystemExit(f"system prompt file not found: {p}")
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            raise SystemExit("system prompt file is empty")
        # legacy single prompt path: use same prompt for RU and EN
        return txt, txt

    ru = args.default_system.strip()
    en = args.default_system_en.strip()

    if args.system_prompt_file_ru:
        pr = Path(args.system_prompt_file_ru)
        if not pr.exists():
            raise SystemExit(f"RU prompt file not found: {pr}")
        ru = pr.read_text(encoding="utf-8").strip()
        if not ru:
            raise SystemExit("RU prompt file is empty")

    if args.system_prompt_file_en:
        pe = Path(args.system_prompt_file_en)
        if not pe.exists():
            raise SystemExit(f"EN prompt file not found: {pe}")
        en = pe.read_text(encoding="utf-8").strip()
        if not en:
            raise SystemExit("EN prompt file is empty")

    return ru, en


def looks_english(text: str) -> bool:
    if not text:
        return False
    en = len(re.findall(r"[A-Za-z]", text))
    ru = len(re.findall(r"[А-Яа-яЁё]", text))
    return en > ru and en >= 8


def pick_system_prompt(row, ru_prompt: str, en_prompt: str, rng: random.Random, en_ratio: float, force_lang_match: bool) -> str:
    explicit = str(row.get("system") or "").strip()
    if explicit:
        return explicit
    if force_lang_match and looks_english(str(row.get("instruction") or "")):
        return en_prompt
    return en_prompt if rng.random() < en_ratio else ru_prompt


def build_messages(row, ru_prompt, en_prompt, rng, en_ratio, force_lang_match):
    sys_msg = pick_system_prompt(row, ru_prompt, en_prompt, rng, en_ratio, force_lang_match)
    instr = str(row["instruction"]).strip()
    inp = str(row.get("input") or "").strip()
    out = str(row["output"]).strip()

    user_text = instr if not inp else f"{instr}\n\nInput:\n{inp}"
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": out},
    ]


def main():
    load_dotenv()
    args = parse_args()

    args.csv_path = args.csv_path or args.csv_path_pos
    args.output_dir = args.output_dir or args.output_dir_pos
    if not args.csv_path or not args.output_dir:
        raise SystemExit("Pass csv and output dir")

    if not (0.0 <= args.en_ratio <= 1.0):
        raise SystemExit("--en-ratio must be in range [0,1]")

    system_prompt_ru, system_prompt_en = get_system_prompts(args)
    rng = random.Random(args.seed)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.makedirs(args.output_dir, exist_ok=True)

    bnb = None
    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=hf_token,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    if args.base_lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.base_lora_path)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=TARGET_MODULES,
            ),
        )

    print("=== Loading/cleaning dataset ===")
    df = pd.read_csv(args.csv_path)
    if "instruction" not in df.columns or "output" not in df.columns:
        raise ValueError("CSV must contain instruction and output")
    if "input" not in df.columns:
        df["input"] = ""

    df = df.dropna(subset=["instruction", "output"]).copy()
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["input"] = df["input"].fillna("").astype(str).str.strip()
    df["output"] = df["output"].astype(str).str.strip()
    df = df[(df["instruction"] != "") & (df["output"] != "")]

    # Keep only rows that likely match target strict format in assistant output.
    df = df[df["output"].str.contains(r"\[PLAN\].*\[/PLAN\]", regex=True, na=False)]
    df = df[df["output"].str.contains(r"\[FILES\].*\[/FILES\]", regex=True, na=False)]
    df = df[df["output"].str.contains(r"\[CODE:[^\]]+\]", regex=True, na=False)]
    df = df.drop_duplicates(subset=["instruction", "input", "output"])

    # small optional balancing: if EN prompts are used, ensure some EN-like user instructions exist
    if args.force_user_lang_match:
        en_mask = df["instruction"].astype(str).apply(looks_english)
        en_count = int(en_mask.sum())
        if en_count == 0 and args.en_ratio > 0:
            print("Warning: no EN user instructions found; EN prompt will still be sampled by en-ratio.")

    df["text"] = df.apply(
        lambda r: tokenizer.apply_chat_template(
            build_messages(r, system_prompt_ru, system_prompt_en, rng, args.en_ratio, args.force_user_lang_match),
            tokenize=False,
            add_generation_prompt=False,
        ),
        axis=1,
    )

    ds = Dataset.from_pandas(df[["text"]], preserve_index=False)
    if args.eval_split > 0:
        split = ds.train_test_split(test_size=args.eval_split, seed=args.seed, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    if eval_ds is not None and args.save_steps % args.eval_steps != 0:
        args.save_steps = args.eval_steps

    ta_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": eval_ds is not None,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "eval_steps": args.eval_steps if eval_ds is not None else None,
    }

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "steps" if eval_ds is not None else "no"
    else:
        ta_kwargs["evaluation_strategy"] = "steps" if eval_ds is not None else "no"
    if "group_by_length" in sig:
        ta_kwargs["group_by_length"] = True
    if "dataloader_num_workers" in sig:
        ta_kwargs["dataloader_num_workers"] = 2

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "packing": False,
        "args": TrainingArguments(**ta_kwargs),
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if eval_ds is not None else [],
    }

    sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    filtered = {k: v for k, v in trainer_kwargs.items() if k in sft_sig}
    if "tokenizer" in sft_sig:
        filtered["tokenizer"] = tokenizer
    elif "processing_class" in sft_sig:
        filtered["processing_class"] = tokenizer

    try:
        trainer = SFTTrainer(**filtered)
    except Exception as e:
        print(f"SFTTrainer compatibility issue: {e}. Falling back to transformers.Trainer.")

        def tok_map(batch):
            return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length)

        train_tok = train_ds.map(tok_map, batched=True, remove_columns=["text"])
        eval_tok = eval_ds.map(tok_map, batched=True, remove_columns=["text"]) if eval_ds is not None else None

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**ta_kwargs),
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if eval_tok is not None else [],
        )

    print("=== Training ===")
    start_ckpt = latest_checkpoint(args.output_dir) if args.resume else None
    if start_ckpt:
        print(f"Resuming from: {start_ckpt}")

    retries = 0
    while True:
        if args.wait_on_oom:
            wait_for_gpu(args.min_free_gpu_mb, args.oom_wait_seconds)
        try:
            trainer.train(resume_from_checkpoint=start_ckpt)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if ("outofmemory" in msg or "cuda out of memory" in msg) and args.wait_on_oom:
                retries += 1
                if retries > args.max_oom_retries:
                    raise
                print(f"OOM retry {retries}/{args.max_oom_retries}. Sleep {args.oom_wait_seconds}s")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(args.oom_wait_seconds)
                start_ckpt = latest_checkpoint(args.output_dir) or start_ckpt
                continue
            raise

    print("=== Save adapter ===")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    sp_ru = Path(args.output_dir) / "SYSTEM_PROMPT_RU_USED.txt"
    sp_en = Path(args.output_dir) / "SYSTEM_PROMPT_EN_USED.txt"
    sp_ru.write_text(system_prompt_ru, encoding="utf-8")
    sp_en.write_text(system_prompt_en, encoding="utf-8")

    print(f"Saved to: {args.output_dir}")
    print(f"System prompt snapshot RU: {sp_ru}")
    print(f"System prompt snapshot EN: {sp_en}")


if __name__ == "__main__":
    main()
