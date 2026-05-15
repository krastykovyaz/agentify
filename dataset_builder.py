import os
import json
import requests
from tqdm import tqdm

# ----------------------------
# CONFIG (from env)
# ----------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://gpu1.sedan.pro:11434/api/generate")
MODEL = os.getenv("MODEL", "qwen3:30b")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
OVERLAP = int(os.getenv("OVERLAP", "200"))

EXPAND_THRESHOLD = int(os.getenv("EXPAND_THRESHOLD", "1000"))


# ----------------------------
# LOAD TEXT
# ----------------------------
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------
# CHUNKING
# ----------------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):

    chunks = []
    i = 0

    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap

    return chunks


# ----------------------------
# CALL OLLAMA
# ----------------------------
def call_llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=500)
    except:
        r = requests.post(OLLAMA_URL, json=payload, timeout=500)
    r.raise_for_status()
    return r.json()["response"]


# ----------------------------
# MODE DETECTION
# ----------------------------
def is_small(text: str) -> bool:
    return len(text) < EXPAND_THRESHOLD


# ----------------------------
# EXPAND SMALL TEXT
# ----------------------------
def expand_text(text: str) -> str:
    prompt = f"""
Ты data augmentation engine.

У тебя мало текста. Расширь его в реалистичный новостной/диалоговый контекст.

ВАЖНО:
- не выдумывай новые факты
- только правдоподобное расширение

Сделай:
1. расширенную версию текста
2. 3 перефразирования
3. краткий контекст ситуации

Текст:
{text}
"""
    return call_llm(prompt)


# ----------------------------
# CLASSIFY
# ----------------------------
def classify(text: str) -> str:
    prompt = f"""
Определи тип текста:

- dialogue
- article
- report
- mixed

Верни строго JSON:
{{"type":"..."}}

Текст:
{text}
"""
    try:
        res = call_llm(prompt)
        return "mixed"
    except:
        return "mixed"


# ----------------------------
# GENERATE DATASET
# ----------------------------
def generate_tasks(text: str, doc_type: str) -> str:
    prompt = f"""
Ты LLM dataset engine.

Тип текста: {doc_type}

Сделай 4 задачи:

1. summarization
2. telegram_post
3. qa
4. extraction

Формат: JSONL (каждая строка JSON)

Пример:
{{"task":"summary","input":"...","output":"..."}}

Текст:
{text}
"""
    return call_llm(prompt)


# ----------------------------
# VALIDATE OUTPUT
# ----------------------------
def validate(output: str) -> str:
    prompt = f"""
Ты валидатор JSONL dataset.

Исправь ошибки:
- битый JSON
- пустые ответы
- мусор
- дубли

Верни только JSONL.

Текст:
{output}
"""
    try:
        return call_llm(prompt)
    except:
        return output


# ----------------------------
# PARSE JSONL
# ----------------------------
def parse_jsonl(text: str):
    items = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        try:
            items.append(json.loads(line))
        except:
            continue

    return items


# ----------------------------
# SAVE JSONL
# ----------------------------
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ----------------------------
# PIPELINE
# ----------------------------
def run(input_file, output_file):
    text = load_txt(input_file)


    

    if is_small(text):
        print("⚡ Small input detected → expanding...")
        text = expand_text(text)
    else:
        text = text[:2000]


    print('STEP 2: chunk')
    chunks = chunk_text(text)

    dataset = []

    for chunk in tqdm(chunks):
        print('STEP 3: chunk')
        # classify
        doc_type = classify(chunk)
        print('STEP 4: chunk')
        # generate dataset
        raw = generate_tasks(chunk, doc_type)
        print('STEP 5: chunk')
        # validate
        cleaned = validate(raw)
        print('STEP 6: chunk')
        # parse
        items = parse_jsonl(cleaned)
        
        dataset.extend(items)
        print('STEP 7: chunk')

    save_jsonl(dataset, output_file)
    print(f"Saved dataset → {output_file}")


# ----------------------------
# ENTRYPOINT
# ----------------------------
if __name__ == "__main__":
    run("data/input.txt", "data/dataset.jsonl")