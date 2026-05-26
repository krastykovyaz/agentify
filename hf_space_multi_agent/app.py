import os
import requests
import gradio as gr

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip()

AGENTS = {
    "summary": {
        "model": os.getenv("OLLAMA_MODEL_SUMMARY", "agentify:summary_q3_k"),
        "system": "Краткое резюме ситуации. Сразу короткий текст по исходному тексту. Без вводных слов.",
    },
    "qa": {
        "model": os.getenv("OLLAMA_MODEL_QA", "agentify:qa_q3_k"),
        "system": "Отвечай по фактам кратко и точно.",
    },
    "extraction": {
        "model": os.getenv("OLLAMA_MODEL_EXTRACTION", "agentify:extraction_q3_k"),
        "system": "Только валидный JSON-объект строкой, без markdown и пояснений.",
    },
    "validator": {
        "model": os.getenv("OLLAMA_MODEL_VALIDATOR", "agentify:validator_q3_k"),
        "system": "Проверяй качество и риски, давай краткий список замечаний.",
    },
    "dialogue": {
        "model": os.getenv("OLLAMA_MODEL_DIALOGUE", "agentify:dialogue_q4_k_m"),
        "system": "Ты добрый и харизматичный собеседник.",
    },
    "telegram": {
        "model": os.getenv("OLLAMA_MODEL_TELEGRAM", "agentify:telegram_q4_k_m"),
        "system": "Один готовый пост для Telegram без вариантов, рекомендаций и саммари.",
    },
    "universal": {
        "model": os.getenv("OLLAMA_MODEL_UNIVERSAL", "agentify:universal_q4_k_m"),
        "system": "Универсальный ассистент: кратко, структурно, по задаче.",
    },
    "coding_web": {
        "model": os.getenv("OLLAMA_MODEL_CODING_WEB", "agentify:coding_web_q4_k_m"),
        "system": "Верни чистый код/скрипт без markdown и пояснений.",
    },
}


def ask(agent_key: str, text: str) -> str:
    if not OLLAMA_BASE_URL:
        return "Set Space secret OLLAMA_BASE_URL first."
    if not text.strip():
        return "Введите текст запроса."

    cfg = AGENTS[agent_key]
    payload = {
        "model": cfg["model"],
        "stream": False,
        "messages": [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": text},
        ],
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192},
    }

    url = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
    r = requests.post(url, json=payload, timeout=300)
    if r.status_code >= 400:
        return f"HTTP {r.status_code}: {r.text[:1000]}"
    data = r.json()
    return (data.get("message") or {}).get("content", "").strip()


with gr.Blocks() as demo:
    gr.Markdown("# Agentify Multi-Agent Chat")
    gr.Markdown("Select agent and send text.")
    agent = gr.Dropdown(choices=list(AGENTS.keys()), value="universal", label="Agent")
    inp = gr.Textbox(lines=10, label="Input")
    out = gr.Textbox(lines=16, label="Output")
    btn = gr.Button("Run")
    btn.click(fn=ask, inputs=[agent, inp], outputs=out)


demo.launch()
