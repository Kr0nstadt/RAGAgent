# AGENTS.md — инструкции для RAGAgent

## Стек
- Ядро: `graph_rag_1c_erp.py` — 4-слойный Graph RAG
- **Поиск по графу: `query_graph.py`** — быстрый поиск без LLM (JSON)
- **Интерфейс: opencode** — я задаю вам вопросы и генерирую инструкцию
- Облачная LLM: **Wormsoft** (`https://ai.wormsoft.ru/api/gpt`, OpenAI-совместимый), модель `wormsoft/agent/high`, ключ `WORMSOFT_API_KEY` в `.env`
- Запасной облачный: DeepSeek (`DEEPSEEK_API_KEY`), локальный: Ollama (`qwen2.5:7b`)
- Загружено: 48948 чанков, 48788 узлов графа (4 слоя)
- Граф строится из:
  - Markdown-документации ITS (`001--1С-ERP Управление предприятием 2, редакция 2.5/`)
  - XML-выгрузки конфигурации 1С ERP (`ERPcode/`)

## Архитектура графа — 4 слоя

| Слой | Название | Узлы | Связи |
|------|----------|------|-------|
| **L1** | Business Scenarios | «Закупка сырья», «Выпуск продукции», «Расчеты с контрагентами» (297 сценариев) | → entry_doc (L3), → clarification (L2) |
| **L2** | Clarification Nodes | «Уточнить вид номенклатуры», «Проверить настройки склада», «Уточнить соглашение» (12 типовых) | → clarifies_field (L3) |
| **L3** | UI & Metadata | Документы, Справочники, Перечисления, Реквизиты форм (из ERPcode XML) | → references (L3↔L3), → parent_child |
| **L4** | Knowledge | Фрагменты ИТС, описания логики проведения, регистры (из md-файлов) | → parent_child, → shared_terms, → references |

## Рабочий процесс (основной)

Вместо медленной Ollama:
1. **Вы** описываете задачу прямо в чате opencode
2. **Я** (opencode) запускаю `python query_graph.py "ваш запрос"` → получаю контекст из всех 4 слоёв
3. **Я** смотрю слой **L1 (сценарии)** и **L2 (уточнения)** — задаю вам **моделирующие вопросы**: какие бизнес-процессы охватить, какие документы должны создаваться
4. **Вы** отвечаете
5. **Я** иду по графу в **L3 (метаданные)** и **L4 (знания)** — нахожу точные реквизиты документов, меню, последовательности
6. **Я** генерирую итоговую инструкцию — сразу, без ожидания LLM

## Команды
```powershell
# Сборка 4-слойного графа (документация + код + сценарии + уточнения)
python graph_rag_1c_erp.py build

# Быстрый поиск по графу (без LLM, JSON на выходе, с группировкой по слоям)
python query_graph.py "настроить закупки"

# Интерактивный режим (с уточняющими вопросами через Wormsoft)
python graph_rag_1c_erp.py interactive

# Пакетный (через Wormsoft)
python graph_rag_1c_erp.py instruct "запрос"

# Пакетный с другим провайдером/моделью
python graph_rag_1c_erp.py instruct "запрос" --provider wormsoft --model wormsoft/agent/high

# Запустить FastAPI-сервер
python graph_rag_1c_erp.py serve --port 8000
```

## Режимы работы
| Режим | Команда | Что делает |
|-------|---------|------------|
| **НОВЫЙ: opencode** | (чат) | Я ищу по 4-слойному графу, задаю вопросы по моделированию, генерирую инструкцию |
| **Сборка** | `build` | Парсит .md (L4) + XML (L3) → строит сценарии (L1) + уточнения (L2) → граф + эмбеддинги |
| **Интерактивный** | `interactive` | Вводишь запрос → уточняющие вопросы от Wormsoft → инструкция |
| **Интерактивный (без уточнений)** | `interactive --mode direct` | Вводишь запрос → видишь что нашёл граф → решаешь генерировать |
| **Пакетный** | `instruct "запрос"` | Сразу генерирует инструкцию через Wormsoft |
| **Поисковый** | `query "запрос"` | Только поиск по графу, без LLM |
| **API сервер** | `serve` | FastAPI сервер для интеграций |
| **Поиск JSON** | `query_graph.py "..."` | Быстрый поиск, JSON по слоям (для opencode) |

## Важные особенности
1. **4 слоя**: L1 (сценарии) → L2 (уточнения) → L3 (метаданные) → L4 (знания). Связаны жёсткими ребрами.
2. **Основной workflow**: вы говорите мне задачу → я ищу по графу → нахожу сценарий и уточнения → задаю modeling-вопросы → нахожу документы и реквизиты → генерирую ответ.
3. **query_graph.py** использует `lightweight` режим (только метаданные + векторы) — загрузка ~7-9с
4. **LLM (Wormsoft)** — получает полный контекст всех 4 слоёв и генерирует пошаговую инструкцию (куда нажимать, какие реквизиты, как проверить)
5. **ERPcode**: XML-выгрузка конфигурации 1С ERP парсится в Layer 3 — из каждого XML извлекаются Name, Synonym, реквизиты с типами
6. **Ollama / DeepSeek** — запасные варианты, если Wormsoft недоступен

## Модели Wormsoft (ai.wormsoft.ru/api/gpt)
- `wormsoft/agent/high`, `wormsoft/agent/medium`, `wormsoft/agent/low` — агентские (рекомендуется high)
- `wormsoft/code/*`, `openai/gpt-oss:120b`, `deepseek-ai/deepseek-v4-pro`, `qwen/qwen3.6:27b`, `kimi/kimi-k2.7-code` и др.
- Список доступных: `GET https://ai.wormsoft.ru/api/gpt/models` (Authorization: Bearer WORMSOFT_API_KEY)
- Эмбеддинги: `qwen/qwen3-embedding:8b` на `/api/gpt/embedding`

## Если Ollama не отвечает
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" ps
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" stop qwen2.5:7b
```

## Проблемы с Wormsoft
```powershell
# Проверка доступных моделей (валидный ли ключ)
curl.exe -s -H "Authorization: Bearer $env:WORMSOFT_API_KEY" https://ai.wormsoft.ru/api/gpt/models

# Документация API: https://ai.wormsoft.ru/docs
# База: https://ai.wormsoft.ru/api/gpt (chat/completions, responses, embeddings, models)
# Ошибки: 400 (unsupported model), 401/403 (ключ), 429 (лимит), 500 (fallback провайдеров не сработал)
```
