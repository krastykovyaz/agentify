# augment_dataset_local.py
# pip install pandas requests tqdm

import pandas as pd
import requests
import time
import random
from tqdm import tqdm
from typing import Dict, List
import os

# ============================================
# CONFIG
# ============================================

INPUT_CSV = "data2.csv"                        # Входной файл
OUTPUT_CSV = "data_augmented2.csv"             # Выходной файл
AUGMENTATION_FACTOR = 3                       # 1, 2, 3, 4, 5 (во сколько раз увеличить)

OLLAMA_URL = "http://gpu3.sedan.pro:11434/api/generate"
MODEL = "qwen3:30b"
TEMPERATURE = 0.7
MAX_RETRIES = 3

# Стратегии аугментации
AUGMENTATION_STRATEGIES = [
    "paraphrase",      # Перефразировать ответ
    "shorter",         # Сделать короче
    "longer",          # Сделать подробнее
    "different_style", # Другой стиль
    "question_only",   # Только вопрос (для QA)
    "reverse_directive", # Новая формулировка вопроса
]

# ============================================
# PROMPTS FOR AUGMENTATION
# ============================================

PROMPTS = {
    "paraphrase": """
Перефразируй следующий ответ ассистента. Сохрани смысл, но используй другие слова и другую структуру.

Оригинальный ответ:
{text}

Перефразированный ответ (только ответ, без объяснений):
""",
    
    "shorter": """
Сделай следующий ответ ассистента КОРОЧЕ. Сохрани главную мысль, убери лишние слова.

Оригинальный ответ:
{text}

Короткая версия (только ответ, без объяснений):
""",
    
    "longer": """
Расширь следующий ответ ассистента, сделай его ПОДРОБНЕЕ. Добавь полезные объяснения.

Оригинальный ответ:
{text}

Расширенная версия (только ответ, без объяснений):
""",
    
    "different_style": """
Перепиши следующий ответ ассистента в ДРУГОМ СТИЛЕ.
Если был формальный - сделай неформальным (разговорным).
Если был кратким - сделай более развернутым.

Оригинальный ответ:
{text}

Переписанный ответ (только ответ, без объяснений):
""",
    
    "question_only": """
На основе следующего вопроса пользователя создай ТОЛЬКО ВОПРОС (без ответа).
Вопрос должен быть переформулирован иначе, но означать то же самое.

Оригинальный вопрос:
{raw_text}

Переформулированный вопрос (только вопрос, без ответа):
""",
    
    "reverse_directive": """
Придумай ДРУГУЮ формулировку вопроса пользователя, которая ведет к тому же ответу.

Текущий вопрос:
{raw_text}

Текущий ответ:
{ready_text}

Новая формулировка вопроса (только вопрос, без ответа):
""",
}

# ============================================
# FUNCTIONS
# ============================================

def call_ollama(prompt: str, temperature: float = 0.7) -> str:
    """Вызов Ollama API"""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=180
            )
            response.raise_for_status()
            data = response.json()
            answer = data["response"].strip()
            if answer and len(answer) >= 3:
                return answer
        except Exception as e:
            print(f"\n⚠️  Ошибка (попытка {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(2)
    return ""


def augment_row(raw_text: str, ready_text: str, strategy: str) -> Dict:
    """Аугментация одной строки"""
    
    if strategy == "question_only":
        prompt = PROMPTS["question_only"].format(raw_text=raw_text)
        new_raw = call_ollama(prompt, temperature=0.8)
        return {"raw_text": new_raw, "ready_text": ready_text} if new_raw else None
    
    elif strategy == "reverse_directive":
        prompt = PROMPTS["reverse_directive"].format(raw_text=raw_text, ready_text=ready_text)
        new_raw = call_ollama(prompt, temperature=0.8)
        return {"raw_text": new_raw, "ready_text": ready_text} if new_raw else None
    
    elif strategy in ["paraphrase", "shorter", "longer", "different_style"]:
        prompt = PROMPTS[strategy].format(text=ready_text)
        new_ready = call_ollama(prompt, temperature=0.7)
        return {"raw_text": raw_text, "ready_text": new_ready} if new_ready else None
    
    return None


def print_statistics(df: pd.DataFrame, original_count: int):
    """Вывод статистики"""
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)
    print(f"📁 Исходный файл: {INPUT_CSV}")
    print(f"📁 Выходной файл: {OUTPUT_CSV}")
    print(f"📊 Исходное количество строк: {original_count}")
    print(f"📈 Финальное количество строк: {len(df)}")
    print(f"🔁 Коэффициент увеличения: {len(df) / original_count:.1f}x")
    
    # Статистика по длине
    df['raw_len'] = df['raw_text'].astype(str).str.len()
    df['ready_len'] = df['ready_text'].astype(str).str.len()
    
    print(f"\n📝 Статистика по длине:")
    print(f"   raw_text (вопрос):   средняя {df['raw_len'].mean():.0f} символов")
    print(f"   ready_text (ответ):  средняя {df['ready_len'].mean():.0f} символов")
    
    # Примеры
    print("\n📌 ПРИМЕРЫ АУГМЕНТАЦИИ:")
    print("-" * 50)
    
    # Показываем несколько примеров
    orig = pd.read_csv(INPUT_CSV)
    for i in range(min(2, len(orig))):
        print(f"\nОригинал {i+1}:")
        print(f"  Вопрос: {orig.iloc[i]['raw_text'][:80]}...")
        print(f"  Ответ:  {orig.iloc[i]['ready_text'][:80]}...")
    
    # Ищем сгенерированные варианты
    new_items = df[~df['ready_text'].isin(orig['ready_text'])]
    if len(new_items) > 0:
        print(f"\nПример аугментации (перефразирование):")
        sample = new_items.iloc[0]
        print(f"  Вопрос: {sample['raw_text'][:80]}...")
        print(f"  Ответ:  {sample['ready_text'][:80]}...")


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("=" * 60)
    print("🔄 АУГМЕНТАЦИЯ DATASET (без скачивания)")
    print("=" * 60)
    
    # 1. Проверка входного файла
    if not os.path.exists(INPUT_CSV):
        print(f"\n❌ Ошибка: файл {INPUT_CSV} не найден!")
        print(f"Создайте файл с колонками 'raw_text' и 'ready_text'")
        return
    
    # 2. Загрузка данных
    print(f"\n📂 Загрузка данных из {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Проверка колонок
    if 'raw_text' not in df.columns or 'ready_text' not in df.columns:
        print(f"\n❌ Ошибка: в файле должны быть колонки 'raw_text' и 'ready_text'")
        print(f"Найдены колонки: {list(df.columns)}")
        return
    
    original_count = len(df)
    print(f"✅ Загружено {original_count} записей")
    
    # 3. Аугментация данных
    print(f"\n🔄 Аугментация данных (увеличение в {AUGMENTATION_FACTOR}x)...")
    print(f"Стратегии: {', '.join(AUGMENTATION_STRATEGIES)}")
    
    augmented_rows = []
    augmented_count = 0
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка"):
        raw_text = row['raw_text']
        ready_text = row['ready_text']
        
        # Сохраняем оригинал
        augmented_rows.append({
            'raw_text': raw_text,
            'ready_text': ready_text,
            'augmentation_type': 'original'
        })
        
        # Генерируем аугментации
        for i in range(AUGMENTATION_FACTOR - 1):
            strategy = random.choice(AUGMENTATION_STRATEGIES)
            augmented = augment_row(raw_text, ready_text, strategy)
            
            if augmented:
                augmented['augmentation_type'] = strategy
                augmented_rows.append(augmented)
                augmented_count += 1
            else:
                failed_count += 1
            
            # Небольшая пауза между запросами
            time.sleep(0.3)
    
    # 4. Создание финального DataFrame
    result_df = pd.DataFrame(augmented_rows)
    
    # Удаляем дубликаты
    before_dedup = len(result_df)
    result_df = result_df.drop_duplicates(subset=['raw_text', 'ready_text'])
    after_dedup = len(result_df)
    
    # 5. Сохранение результата
    print(f"\n💾 Сохранение в {OUTPUT_CSV}...")
    result_df[['raw_text', 'ready_text']].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    # 6. Статистика
    print("\n" + "=" * 60)
    print("✅ АУГМЕНТАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 60)
    print(f"📊 Исходных записей: {original_count}")
    print(f"🔄 Сгенерировано вариантов: {augmented_count}")
    print(f"❌ Не удалось сгенерировать: {failed_count}")
    print(f"📈 Финальных записей: {len(result_df)}")
    print(f"🔁 Реальный коэффициент: {len(result_df) / original_count:.1f}x")
    
    if before_dedup > after_dedup:
        print(f"🗑️ Удалено дубликатов: {before_dedup - after_dedup}")
    
    # Детальная статистика по типам аугментации
    print(f"\n📊 Статистика по типам аугментации:")
    type_counts = result_df[result_df['augmentation_type'] != 'original']['augmentation_type'].value_counts()
    for aug_type, count in type_counts.items():
        print(f"   {aug_type}: {count} записей")
    
    # Проверка качества
    print_statistics(result_df, original_count)
    
    print(f"\n✨ Готово! Файл сохранен: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()