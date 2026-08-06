# RAGAgent — Graph RAG по 1С:ERP 2.5

Проект содержит готовый четырёхслойный граф знаний по 1С:ERP 2.5 и инструменты для поиска, разбора файлов ТЗ и подготовки пошаговых инструкций. Большая XML-выгрузка `ERPcode` в репозиторий не входит: для работы используется уже построенный индекс.

Обычный поиск, проверка графа, создание Task Graph и построение планов работают локально, без LLM и API-ключа. Wormsoft требуется только для генерации текста через Pi/Herdr или явные LLM-команды.

## Что находится в репозитории

| Путь | Назначение |
|---|---|
| `graph_rag_data/` | Готовый граф, поисковый индекс и векторы. Позволяет работать сразу после клонирования. |
| `001--1С-ERP Управление предприятием 2, редакция 2.5/` | Markdown-документация ИТС и источник слоя L4. |
| `graph_rag_1c_erp.py` | Сборка, поиск, API-сервер и команды управления графом. |
| `query_graph.py` | Быстрый поиск по готовому графу без LLM, JSON на выходе. |
| `workspace_task.py` | Сохраняемый workflow: файл ТЗ → уточнения → план → инструкция. |
| `.pi/` | Координатор и проектные субагенты для Pi/Herdr. |

`task_data/` намеренно не публикуется: там находятся пользовательские ТЗ, ответы и результаты конкретных задач.

## Требования

- Windows 10/11 и PowerShell;
- Git;
- [Git LFS](https://git-lfs.com/);
- Python 3.11–3.14;
- [uv](https://docs.astral.sh/uv/) — рекомендуемый установщик Python-окружения;
- для агентного режима: Herdr, Node.js и Pi `@earendil-works/pi-coding-agent`.

## 1. Клонирование готового проекта

Готовый индекс хранится через Git LFS. Без `git lfs pull` в каталоге окажутся маленькие текстовые указатели вместо настоящих файлов.

```powershell
git lfs install
git clone https://github.com/Kr0nstadt/RAGAgent.git
Set-Location RAGAgent
git lfs pull
```

Готовый индекс занимает примерно 1 ГБ. Нужен дополнительный запас места для `.git`, Python-окружения и пользовательских Task Graph.

## 2. Установка Python-зависимостей

Рекомендуемый вариант с изолированным окружением:

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Далее во всех примерах используется:

```powershell
$python = ".\.venv\Scripts\python.exe"
```

## 3. Проверка готового графа

```powershell
& $python query_graph.py "настроить закупку сырья"
& $python graph_rag_1c_erp.py validate
```

`query_graph.py` должен вернуть JSON с результатами, сгруппированными по слоям. `validate` дополнительно печатает отчёт целостности. В опубликованном индексе он может показать дубли ID в сыром `chunks.json`; поиск при загрузке дедуплицирует их и отдельно выводит `unique_chunks`. Для строгого отчёта `ok: true` пересоберите индекс командой `build`. API-ключ для этих проверок не нужен.

Если `validate` сообщает об отсутствующих файлах, сначала проверьте LFS:

```powershell
git lfs pull
git lfs ls-files
```

## 4. Основные способы запуска

### Быстрый поиск без LLM

```powershell
& $python query_graph.py "создание номенклатуры и единиц измерения"
```

### Поиск через основной CLI

```powershell
& $python graph_rag_1c_erp.py query "цепочка заказа клиента до реализации"
```

### Локальный API-сервер

```powershell
& $python graph_rag_1c_erp.py serve --port 8000
```

После запуска API доступен по адресу `http://127.0.0.1:8000`.

### Разбор файла ТЗ без LLM

```powershell
& $python workspace_task.py start "C:\path\requirements.docx"
& $python workspace_task.py prepare TASK_ID --limit 10
& $python workspace_task.py status TASK_ID
```

Поддерживаются `.docx`, `.xlsx`, `.pdf`, `.md` и `.txt`. Task Graph и ответы сохраняются в `task_data/<task-id>/`.

## 5. Запуск Herdr/Pi с субагентами

Создайте локальный `.env` из примера и добавьте собственный ключ:

```powershell
Copy-Item .env.example .env
notepad .env
```

Минимальное содержимое:

```dotenv
WORMSOFT_API_KEY=ваш_ключ
```

Если Pi не установлен:

```powershell
npm install --global @earendil-works/pi-coding-agent
```

Запуск координатора:

```powershell
.\start_erp_pi.ps1
```

В Pi можно написать:

```text
/erp-task C:\path\requirements.docx
```

Координатор создаёт Task Graph, задаёт вопросы раундами и вызывает проектных субагентов последовательно. Итог сохраняется в `task_data/<task-id>/final_instruction.md`.

API-ключ не хранится в Git. Не добавляйте `.env` в коммиты.

## 6. Пересборка графа из исходных данных

Пересборка не нужна для обычной работы: `graph_rag_data/` уже содержит готовый индекс. XML-выгрузка `ERPcode` из-за размера не публикуется. Для пересборки её нужно отдельно положить в корень проекта под именем `ERPcode/`.

```powershell
& $python graph_rag_1c_erp.py build
& $python graph_rag_1c_erp.py validate
```

Сборщик читает:

- `001--1С-ERP Управление предприятием 2, редакция 2.5/` — знания ИТС;
- `ERPcode/` — метаданные, формы, реквизиты, документы и регистры ERP;
- встроенные сценарии L1 и уточнения L2.

Результат атомарно заменяется в `graph_rag_data/`. Пересборка ресурсоёмкая: предусмотрите свободное место и не прерывайте процесс во время финальной замены индекса.

Дополнительные команды обогащения:

```powershell
& $python graph_rag_1c_erp.py enhance
& $python graph_rag_1c_erp.py refresh-clarifications
& $python graph_rag_1c_erp.py compact-index
```

## 7. Тесты

```powershell
& $python -m unittest discover -s tests -v
```

Тесты графа и workflow не выполняют запросы к Wormsoft без явного включения LLM.

## Публикация изменений графа

Перед коммитом убедитесь, что Git LFS установлен:

```powershell
git lfs install
git lfs status
git add .gitattributes .gitignore README.md .env.example
git add graph_rag_data
git status
```

Готовый индекс занимает около 1 ГБ. Проверьте доступный объём Git LFS перед публикацией. `ERPcode/`, `manifest.json` и `manifest.ndjson` остаются локальными и игнорируются Git.
