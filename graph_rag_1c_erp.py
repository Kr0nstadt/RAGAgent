#!/usr/bin/env python3
"""
Graph RAG for 1C ERP ITS Documentation
========================================
Строит граф знаний из документации ITS 1C ERP и позволяет 
делать RAG-запросы с учетом связей между сущностями.

Использование:
  1. Построение графа: python graph_rag_1c_erp.py build
  2. Запрос:          python graph_rag_1c_erp.py query "как создать заказ поставщику"
  3. API сервер:      python graph_rag_1c_erp.py serve
"""

import os
import re
import json
import hashlib
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
# Поддержка UTF-8 для Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Загрузка .env (ключи API)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import pickle

import networkx as nx
import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
ITS_ROOT = Path(__file__).parent / "001--1С-ERP Управление предприятием 2, редакция 2.5"
DATA_DIR = Path(__file__).parent / "graph_rag_data"
GRAPH_FILE = DATA_DIR / "knowledge_graph.graphml"
CHUNKS_FILE = DATA_DIR / "chunks.json"
CHUNKS_JSONL_FILE = DATA_DIR / "chunks.jsonl"
VECTORS_FILE = DATA_DIR / "vectors.npy"
VECTORS_SPARSE_FILE = DATA_DIR / "vectors_tfidf.npz"
NODES_FILE = DATA_DIR / "nodes.json"
TFIDF_FILE = DATA_DIR / "tfidf_vectorizer.pkl"
GRAPH_PICKLE = DATA_DIR / "knowledge_graph.pkl"
CHUNKS_META_FILE = DATA_DIR / "chunks_meta.json"
INDEX_MANIFEST_FILE = DATA_DIR / "index_manifest.json"
INSTRUCTIONS_DIR = Path(__file__).parent / "instructions"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)


def llm_calls_enabled() -> bool:
    """Внешние/локальные LLM вызываются только после явного разрешения."""
    return os.environ.get("RAG_ENABLE_LLM", "").strip().lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Модели данных
REGEX_1C_TERMS = re.compile(
    r'(Организаци[яюи]|Номенклатур[ауы]|Контрагент[а-яё]*|'
    r'Партнер[а-яё]*|Соглашени[яе]|Договор[а-яё]*|Склад[а-яё]*|'
    r'Заказ[а-яё]*|Поступлени[яе]|Реализаци[яюи]|Счет[а-яё]*|'
    r'Документ[а-яё]*|Справочник[а-яё]*|Регистр[а-яё]*|'
    r'Отчет[а-яё]*|Обработк[ау]|Перемещени[яе]|Списани[яе]|'
    r'Оприходовани[яе]|Инвентаризаци[яюи]|Цен[а-яё]*|'
    r'Валюта[а-яё]*|Подразделени[яе]|Должност[а-яё]*|'
    r'Расход[а-яё]*|Доход[а-яё]*|Прибыль[а-яё]*|Убытк[а-яё]*)'
)

# ---------------------------------------------------------------------------
@dataclass
class DocChunk:
    """Фрагмент документации с метаданными"""
    id: str
    title: str
    content: str
    path: str               # иерархический путь
    layer: int = 4          # 1=Business Scenario, 2=Clarification, 3=UI&Metadata, 4=Knowledge
    node_type: str = 'knowledge'  # 'scenario', 'clarification', 'metadata', 'knowledge'
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)  # ссылки на другие документы
    terms: List[str] = field(default_factory=list)
    level: int = 0
    # L1→L3: какие документы создаются в этом сценарии
    entry_docs: List[str] = field(default_factory=list)
    # L1→L2: какие вопросы нужно задать для этого сценария
    clarifications: List[str] = field(default_factory=list)
    # Типизированные отношения L3/L4. Каждый элемент: target, type,
    # reverse_type, weight, evidence, properties.
    relations: List[Dict[str, Any]] = field(default_factory=list)
    # Расширенные свойства узла (обязательность поля, UI path, источник XML и т.д.)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# 1. Парсер markdown-файлов ITS
# ---------------------------------------------------------------------------
def load_ref_mapping() -> Dict[str, str]:
    """Загружает маппинг из manifest.json: URL bookmark -> relativePath"""
    mapping = {}
    manifest_path = ITS_ROOT / "manifest.json"
    if not manifest_path.exists():
        return mapping

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            url = entry.get("url", "")
            rel_path = entry.get("relativePath", "")
            # Извлекаем часть после /bookmark/
            m = re.search(r'/bookmark/(.+)$', url)
            if m:
                bookmark = m.group(1)  # например "Introduction/About"
                mapping[bookmark.lower()] = str(rel_path).replace("\\", "/").replace(".md", "")
        print(f"  Загружено {len(mapping)} маппингов из manifest.json")
    except Exception as e:
        print(f"  Ошибка загрузки manifest.json: {e}")
    
    return mapping


def parse_its_markdown() -> List[DocChunk]:
    """Парсит все .md файлы в ITS_ROOT и возвращает список DocChunk.
    
    Умный чанкинг: каждый документ разбивается на секции по заголовкам H2/##.
    Это позволяет находить именно тот раздел, который нужен, а не весь документ.
    """
    ref_mapping = load_ref_mapping()
    
    chunks = []
    md_files = sorted(ITS_ROOT.rglob("*.md"))
    
    print(f"[1/5] Найдено .md файлов: {len(md_files)}")
    
    for fpath in md_files:
        rel = fpath.relative_to(ITS_ROOT)
        path_str = str(rel).replace("\\", "/").replace(".md", "")

        try:
            text = fpath.read_text(encoding="utf-8")
        except Exception:
            continue
        
        if not text.strip():
            continue
        
        # Удаляем строку _Source
        clean_text = re.sub(r'^_Source:.*$', '', text, flags=re.MULTILINE)
        clean_text = clean_text.strip()
        
        # Извлекаем H1 (заголовок документа)
        title_match = re.search(r'^#\s+(.+?)$', clean_text, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else fpath.stem
        
        # Извлекаем ссылки и термины из всего документа (наследуются всеми секциями)
        raw_refs = list(set(re.findall(r'/db/erp25doc/bookmark/([^\s)\]]+)', clean_text)))
        all_refs = []
        for ref in raw_refs:
            ref_lower = ref.lower().strip('/')
            if ref_lower in ref_mapping:
                resolved = ref_mapping[ref_lower]
                if resolved != path_str:
                    all_refs.append(resolved)
        
        doc_terms = list(set(REGEX_1C_TERMS.findall(clean_text)))
        
        level = len([p for p in path_str.split("/") if p]) - 1
        parts = path_str.split("/")
        parent_id = "/".join(parts[:-1]) if len(parts) > 1 else None
        
        # Разбиваем документ на секции по ## (H2)
        section_pattern = re.split(r'^##\s+', clean_text, flags=re.MULTILINE)
        
        # Создаем узел-документ (id = path_str) — родитель для всех секций
        header_end = clean_text.find('\n## ')
        doc_intro = clean_text[:header_end] if header_end > 0 else clean_text
        doc_node = DocChunk(
            id=path_str,
            title=doc_title,
            content=doc_intro,
            path=path_str,
            parent_id=parent_id,
            refs=all_refs,
            terms=doc_terms,
            level=level
        )
        chunks.append(doc_node)
        
        if len(section_pattern) > 1 and len(clean_text) > 200:
            # Есть H2-секции: секции с заголовками
            
            for i, sec in enumerate(section_pattern[1:], 1):
                sec = sec.strip()
                if not sec or len(sec) < 30:
                    continue
                lines = sec.split('\n')
                sec_title = lines[0].strip()
                sec_content = '\n'.join(lines[1:]).strip()
                
                terms = list(set(REGEX_1C_TERMS.findall(sec_content)))
                
                section_id = path_str + "/" + re.sub(r'[^\wа-яёА-ЯЁ]+', '_', sec_title.lower())[:60]
                
                # Извлекаем рефы только из этой секции
                sec_refs = list(set(re.findall(r'/db/erp25doc/bookmark/([^\s)\]]+)', sec_content)))
                resolved_sec_refs = []
                for ref in sec_refs:
                    ref_lower = ref.lower().strip('/')
                    if ref_lower in ref_mapping:
                        resolved = ref_mapping[ref_lower]
                        if resolved != path_str:
                            resolved_sec_refs.append(resolved)
                
                chunks.append(DocChunk(
                    id=section_id,
                    title=doc_title + " - " + sec_title,
                    content=sec_title + "\n" + sec_content,
                    path=path_str,
                    parent_id=path_str,
                    refs=list(set(all_refs + resolved_sec_refs)),
                    terms=terms if terms else doc_terms,
                    level=level + 1
                ))
    
    # Заполняем children_ids
    chunk_map = {c.id: c for c in chunks}
    for c in chunks:
        if c.parent_id and c.parent_id in chunk_map:
            chunk_map[c.parent_id].children_ids.append(c.id)
    
    print(f"  Создано чанков: {len(chunks)}")
    return chunks

# ---------------------------------------------------------------------------
# 1b. Парсер XML-выгрузки конфигурации 1С ERP (ERPcode/)
# ---------------------------------------------------------------------------
ERPCODE_DIR = Path(__file__).parent / "ERPcode"
# Маппинг: имя папки -> человекочитаемый тип метаданных
METADATA_TYPE_NAMES = {
    "Catalogs": "Справочник",
    "Documents": "Документ",
    "InformationRegisters": "Регистр сведений",
    "AccumulationRegisters": "Регистр накопления",
    "AccountingRegisters": "Регистр бухгалтерии",
    "CalculationRegisters": "Регистр расчета",
    "Enums": "Перечисление",
    "ChartsOfAccounts": "План счетов",
    "ChartsOfCharacteristicTypes": "План видов характеристик",
    "ChartsOfCalculationTypes": "План видов расчета",
    "BusinessProcesses": "Бизнес-процесс",
    "Tasks": "Задача",
    "Reports": "Отчет",
    "DataProcessors": "Обработка",
    "Constants": "Константа",
    "Subsystems": "Подсистема",
    "CommonModules": "Общий модуль",
    "CommonForms": "Общая форма",
    "CommonCommands": "Общая команда",
    "CommonAttributes": "Общий реквизит",
    "ExchangePlans": "План обмена",
    "FunctionalOptions": "Функциональная опция",
    "Roles": "Роль",
    "DocumentJournals": "Журнал документов",
    "Filters": "Критерий отбора",
    "HTTPServices": "HTTP-сервис",
    "WebServices": "Web-сервис",
    "ScheduledJobs": "Регламентное задание",
    "SessionParameters": "Параметр сеанса",
    "SettingsStorages": "Хранилище настроек",
    "Languages": "Язык",
    "Definitions": "Определяемый тип",
}

def parse_erp_code() -> List[DocChunk]:
    """Парсит XML-выгрузку конфигурации 1С ERP (ERPcode/) в DocChunk.
    
    Из каждого XML-файла извлекает:
    - название объекта метаданных (Name)
    - синоним (Synonym)
    - реквизиты (Attributes) с их типами
    - табличные части (TabularSections)
    - ссылки на другие объекты метаданных
    """
    import xml.etree.ElementTree as ET
    
    chunks = []
    erp_dir = ERPCODE_DIR
    
    if not erp_dir.exists():
        print(f"  Папка ERPcode не найдена: {erp_dir}")
        return chunks
    
    ns = {
        'v8': 'http://v8.1c.ru/8.1/data/core',
        'cfg': 'http://v8.1c.ru/8.1/data/enterprise/current-config',
        'xr': 'http://v8.1c.ru/8.3/xcf/readable',
    }
    
    # Парсим только ключевые типы метаданных (без общих модулей, форм, стилей и т.д.)
    key_dirs = {'Catalogs', 'Documents', 'Enums', 'InformationRegisters',
                'AccumulationRegisters', 'AccountingRegisters', 'CalculationRegisters',
                'ChartsOfAccounts', 'ChartsOfCharacteristicTypes', 'ChartsOfCalculationTypes',
                'BusinessProcesses', 'Tasks', 'Reports', 'DataProcessors',
                'Constants', 'Subsystems', 'ExchangePlans', 'FunctionalOptions',
                'Roles', 'Filters', 'HTTPServices', 'WebServices', 'ScheduledJobs',
                'SessionParameters', 'SettingsStorages'}
    
    xml_files = []
    for d in key_dirs:
        dir_path = erp_dir / d
        if dir_path.exists():
            # Только .xml файлы верхнего уровня (не подпапки с формами)
            for f in dir_path.glob("*.xml"):
                # Пропускаем "ПрисоединенныеФайлы" — служебные
                if 'ПрисоединенныеФайлы' in f.stem:
                    continue
                xml_files.append(f)
    
    print(f"  Найдено XML-файлов конфигурации: {len(xml_files)}")
    
    # Регекс для быстрого извлечения информации без полного DOM-парсинга
    # Ищем: имя, синоним, реквизиты
    name_re = re.compile(r'<Name>([^<]+)</Name>')
    synonym_re = re.compile(r'<v8:content>([^<]+)</v8:content>')
    attr_re = re.compile(r'<Attribute[^>]*>.*?<Name>([^<]+)</Name>.*?<v8:content>([^<]+)</v8:content>.*?</Attribute>', re.DOTALL)
    attr_type_re = re.compile(r'cfg:(CatalogRef|DocumentRef|EnumRef|InformationRegisterRef|AccumulationRegisterRef|ChartOfAccountsRef|BusinessProcessRef|TaskRef|DataProcessorRef|ReportRef|ConstantRef|ExchangePlanRef|FilterRef|HTTPServiceRef|WebServiceRef|ScheduledJobRef|SessionParameterRef|SettingsStorageRef|ChartOfCharacteristicTypesRef|ChartOfCalculationTypesRef|AccountingRegisterRef|CalculationRegisterRef)\.([A-Za-zА-Яа-яЁё]+)')
    ts_re = re.compile(r'<TabularSection[^>]*>.*?<Name>([^<]+)</Name>.*?<v8:content>([^<]+)</v8:content>.*?</TabularSection>', re.DOTALL)
    meta_type_re = re.compile(r'<(Catalog|Document|InformationRegister|AccumulationRegister|AccountingRegister|CalculationRegister|Enum|ChartOfAccounts|ChartOfCharacteristicTypes|ChartOfCalculationTypes|BusinessProcess|Task|Report|DataProcessor|Constant|Subsystem|ExchangePlan|FunctionalOption|Role|Filter|HTTPService|WebService|ScheduledJob|SessionParameter|SettingsStorage)\b')
    comment_re = re.compile(r'<Comment([^>]*)/>|<Comment>([^<]*)</Comment>')
    
    objects_info = []
    for fpath in xml_files:
        meta_type_dir = fpath.parent.name
        meta_type_name = METADATA_TYPE_NAMES.get(meta_type_dir, meta_type_dir)
        
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        
        # Быстрая проверка, что это нужный тип метаданных
        if not meta_type_re.search(text):
            continue
        
        # Имя объекта
        name_match = name_re.search(text)
        if not name_match:
            continue
        obj_name = name_match.group(1)
        
        # Иерархический путь
        meta_path = f"ERPcode/{meta_type_dir}/{obj_name}"
        
        # Синоним (первый русский)
        synonyms = synonym_re.findall(text)
        synonym = synonyms[0] if synonyms else obj_name
        
        # Комментарий
        comment_match = comment_re.search(text)
        comment = comment_match.group(2) if comment_match and comment_match.group(2) else ''
        if not comment and comment_match:
            comment = comment_match.group(0) if comment_match.group(1) == '' else ''
            # <Comment/> or <Comment></Comment> means empty comment
            comment = ''
        
        # Реквизиты
        attributes = []
        refs = []
        for attr_match in attr_re.finditer(text):
            attr_name = attr_match.group(1)
            attr_synonym = attr_match.group(2)
            
            # Ищем тип этого реквизита: 200 символов после <Name>
            attr_pos = text.find(f'<Name>{attr_name}</Name>', attr_match.start())
            type_search = text[attr_pos:attr_pos+500]
            
            type_refs = attr_type_re.findall(type_search)
            type_str = '; '.join(f"{ref[0]}.{ref[1]}" for ref in type_refs) if type_refs else ''
            
            for ref_type, ref_obj in type_refs:
                # Преобразуем название типа папки
                folder_name = ref_type.replace('Ref', '') + 's'
                ref_target = f"ERPcode/{folder_name}/{ref_obj}"
                if ref_target not in refs:
                    refs.append(ref_target)
            
            attr_desc = f"{attr_synonym} ({attr_name})"
            if type_str:
                simple = type_str.replace('CatalogRef.', 'Спр.').replace('DocumentRef.', 'Док.').replace('EnumRef.', 'Переч.').replace('InformationRegisterRef.', 'РС.').replace('AccumulationRegisterRef.', 'РН.')
                attr_desc += f" — {simple}"
            attributes.append(attr_desc)
        
        # Табличные части
        tabular_sections = []
        for ts_match in ts_re.finditer(text):
            tabular_sections.append(ts_match.group(2))
        
        # Формируем содержимое
        content_parts = [f"Объект метаданных: {meta_type_name}"]
        content_parts.append(f"Имя: {obj_name}")
        content_parts.append(f"Синоним: {synonym}")
        if comment:
            content_parts.append(f"Описание: {comment}")
        if attributes:
            content_parts.append(f"\nРеквизиты ({len(attributes)}):")
            for a in attributes[:50]:  # максимум 50 реквизитов
                content_parts.append(f"  - {a}")
        if tabular_sections:
            content_parts.append(f"\nТабличные части ({len(tabular_sections)}):")
            for ts in tabular_sections[:10]:
                content_parts.append(f"  - {ts}")
        
        content = "\n".join(content_parts)
        
        objects_info.append({
            'id': meta_path,
            'title': f"{meta_type_name} {synonym}",
            'content': content,
            'path': meta_path,
            'refs': refs,
            'level': 2,
        })
    
    for obj in objects_info:
        chunks.append(DocChunk(
            id=obj['id'],
            title=obj['title'],
            content=obj['content'],
            path=obj['path'],
            layer=3,
            node_type='metadata',
            refs=obj['refs'],
            terms=[],
            level=obj['level'],
        ))
    
    print(f"  Создано чанков из кода: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# 1.5. Бизнес-сценарии (Layer 1)
# ---------------------------------------------------------------------------
def parse_business_scenarios() -> List[DocChunk]:
    """Извлекает бизнес-сценарии из структуры ИТС."""
    scenarios = []
    scenario_dirs = sorted(ITS_ROOT.glob("???--*.md")) if ITS_ROOT.exists() else []
    
    for md_file in scenario_dirs:
        scenario_id = f"scenario_{md_file.stem}"
        title = re.sub(r'^\d+--\d+\.\s*', '', md_file.stem)
        title = title.replace('--', ': ')
        
        # Собираем подразделы (под-сценарии)
        dir_name = md_file.with_suffix('').name
        scenario_dir = ITS_ROOT / dir_name
        sub_scenarios = []
        if scenario_dir.exists():
            for sub in sorted(scenario_dir.glob("*.md")):
                sub_title = re.sub(r'^\d+--', '', sub.stem)
                # Имя подраздела повторяется в разных главах ИТС, поэтому ID
                # обязан включать родительский сценарий.
                sub_id = f"{scenario_id}/{sub.stem}"
                sub_scenarios.append(DocChunk(
                    id=sub_id,
                    title=sub_title,
                    content=f"Под-сценарий: {sub_title}. Входит в: {title}",
                    path=f"scenarios/{dir_name}/{sub.name}",
                    layer=1,
                    node_type='scenario',
                    parent_id=scenario_id,
                    level=1,
                ))
        
        scenario_chunk = DocChunk(
            id=scenario_id,
            title=title,
            content=f"Бизнес-сценарий: {title}. Включает под-сценарии: {len(sub_scenarios)}",
            path=f"scenarios/{md_file.name}",
            layer=1,
            node_type='scenario',
            children_ids=[s.id for s in sub_scenarios],
            level=0,
        )
        scenarios.append(scenario_chunk)
        scenarios.extend(sub_scenarios)
    
    print(f"  Создано сценариев: {len(scenarios)} (Layer 1)")
    return scenarios


def _find_entry_docs(scenario: DocChunk, metadata_chunks: List[DocChunk]) -> List[str]:
    """Находит метаданные Layer 3, релевантные сценарию."""
    keywords = re.findall(r'[А-ЯЁ][а-яё]+', scenario.title)
    matches = []
    for mc in metadata_chunks:
        if any(kw.lower() in mc.title.lower() for kw in keywords):
            if mc.node_type == 'metadata' and ('Document' in mc.path or 'Document' in str(type(mc))):
                matches.append(mc.id)
    return matches[:5]


# ---------------------------------------------------------------------------
# 1.6. Уточняющие узлы (Layer 2)
# ---------------------------------------------------------------------------
CLARIFY_TEMPLATES = [
    ("Организация", "Какие организации и контуры учёта входят в процесс?", "ERPcode/Catalogs/Организации"),
    ("Склад", "Какие склады, помещения и кладовые участвуют?", "ERPcode/Catalogs/Склады"),
    ("Подразделение", "Какие подразделения исполняют и контролируют этапы?", "ERPcode/Catalogs/СтруктураПредприятия"),
    ("Номенклатура", "Какие категории номенклатуры участвуют: сырьё, полуфабрикаты, продукция, тара?", "ERPcode/Catalogs/Номенклатура"),
    ("ВидНоменклатуры", "Какие виды номенклатуры и правила учёта характеристик/серий нужны?", "ERPcode/Catalogs/ВидыНоменклатуры"),
    ("Единицы", "В каких единицах ведутся хранение, производство, закупка и продажа?", "ERPcode/Catalogs/УпаковкиЕдиницыИзмерения"),
    ("Контрагент", "Какие партнёры и контрагенты участвуют и как различаются их роли?", "ERPcode/Catalogs/Контрагенты"),
    ("Соглашение", "Какие соглашения, договоры, условия оплаты и отгрузки используются?", "ERPcode/Catalogs/СоглашенияСКлиентами"),
    ("Вид цены", "Какие виды цен, скидки и правила расчёта применяются?", "ERPcode/Catalogs/ВидыЦен"),
    ("Валюта", "В каких валютах ведутся цены, расчёты и учёт?", "ERPcode/Catalogs/Валюты"),
    ("Период", "За какой период (месяц, квартал, год)?", None),
    ("СтатусЗаказа", "Какие состояния и правила подтверждения заказа нужны?", "ERPcode/Enums/СтатусыЗаказовНаПроизводство"),
    ("ВидОбеспечения", "Как выбирается способ обеспечения: производство, закупка, перемещение, остаток?", "ERPcode/Enums/ВариантыОбеспечения"),
    ("Партия", "Нужно ли обособленное обеспечение (учёт по назначениям)?", None),
    ("КаналПродаж", "Чем различаются процессы по каналам продаж и типам клиентов?", None),
    ("ПроизводствоПодЗаказ", "Производство работает под заказ, на склад или по смешанной схеме?", None),
    ("Полуфабрикаты", "Учитываются ли полуфабрикаты отдельным выпуском и передачей между этапами?", "ERPcode/Catalogs/Номенклатура"),
    ("РесурсныеСпецификации", "Какие рецептуры и ресурсные спецификации используются?", "ERPcode/Catalogs/РесурсныеСпецификации"),
    ("Планирование", "Какой горизонт и уровень производственного планирования требуется?", None),
    ("ПополнениеЗапасов", "Как рассчитываются минимальные запасы и количество закупки?", None),
    ("ВыборПоставщика", "По каким правилам выбирается поставщик и график поставок?", "ERPcode/Catalogs/Партнеры"),
    ("ОрдернаяСхема", "Нужна ли ордерная схема при приёмке, перемещении и отгрузке?", None),
    ("FIFO", "Как должен обеспечиваться FIFO: по партиям, сериям или организационным правилом?", None),
    ("СерииСроки", "Нужно ли вести серии, даты производства и сроки годности?", "ERPcode/Catalogs/СерииНоменклатуры"),
    ("КонтрольКачества", "Где фиксируются входной, производственный и выходной контроль качества?", None),
    ("Брак", "Как оформляются брак, изоляция, возврат и утилизация?", None),
    ("Тара", "Как учитывается упаковка и многооборотная тара?", "ERPcode/Catalogs/Номенклатура"),
    ("Доставка", "Доставка и маршруты ведутся в ERP или во внешней системе?", None),
    ("Возвраты", "Какие причины, финансовые последствия и дальнейшие действия предусмотрены для возврата?", None),
    ("КонтурУчета", "Нужен оперативный, управленческий, регламентированный учёт или несколько контуров?", None),
    ("Себестоимость", "Как должна рассчитываться себестоимость и какие затраты включаются?", None),
    ("Интеграции", "Какие данные поступают автоматически и из каких систем?", None),
    ("Роли", "Кто создаёт, согласует, проводит и контролирует документы?", None),
    ("Исключения", "Какие исключительные ветки разрешены и кто их согласует?", None),
    ("КритерииПриемки", "Какие отчёты и показатели подтверждают успешность процесса?", None),
]

CLARIFICATION_KEYWORDS = {
    "Склад": ("склад", "запас", "хранен", "логист"),
    "Номенклатура": ("номенклат", "товар", "продук", "сырь", "материал"),
    "ВидНоменклатуры": ("номенклат", "товар", "продук", "сырь"),
    "Единицы": ("номенклат", "товар", "продук", "сырь", "упаков"),
    "Контрагент": ("продаж", "закуп", "клиент", "постав", "расчет"),
    "Соглашение": ("продаж", "закуп", "клиент", "постав", "расчет"),
    "Вид цены": ("продаж", "цен", "скид"),
    "Валюта": ("валют", "расчет", "казнач"),
    "СтатусЗаказа": ("заказ", "производ"),
    "ВидОбеспечения": ("обеспеч", "закуп", "производ", "склад"),
    "Партия": ("обеспеч", "парт", "сер", "склад", "производ"),
    "КаналПродаж": ("продаж", "клиент", "рознич"),
    "ПроизводствоПодЗаказ": ("производ",),
    "Полуфабрикаты": ("производ", "полуфаб"),
    "РесурсныеСпецификации": ("производ", "спецификац", "рецепт"),
    "Планирование": ("план", "производ"),
    "ПополнениеЗапасов": ("закуп", "запас", "постав"),
    "ВыборПоставщика": ("закуп", "постав"),
    "ОрдернаяСхема": ("склад", "прием", "отгруз"),
    "FIFO": ("склад", "запас", "хранен"),
    "СерииСроки": ("склад", "производ", "товар", "номенклат"),
    "КонтрольКачества": ("качеств", "производ", "закуп", "склад"),
    "Брак": ("качеств", "производ", "возврат"),
    "Тара": ("склад", "упаков", "тара", "продаж"),
    "Доставка": ("логист", "достав", "продаж"),
    "Возвраты": ("возврат", "продаж"),
    "Себестоимость": ("себесто", "затрат", "производ"),
    "Интеграции": ("интеграц", "обмен"),
}

def generate_clarification_nodes() -> List[DocChunk]:
    """Создаёт узлы-уточнения (Layer 2) на основе типовых вопросов."""
    nodes = []
    for cls_id, question, ref_metadata in CLARIFY_TEMPLATES:
        node = DocChunk(
            id=f"clarify_{cls_id}",
            title=question,
            content=f"Уточнение: {question}",
            path=f"clarifications/{cls_id}",
            layer=2,
            node_type='clarification',
        )
        if ref_metadata:
            node.entry_docs.append(ref_metadata)
        nodes.append(node)
    print(f"  Создано уточняющих узлов: {len(nodes)} (Layer 2)")
    return nodes


def link_scenario_to_clarifications(scenarios: List[DocChunk], clarifications: List[DocChunk],
                                    metadata_chunks: List[DocChunk]) -> None:
    """Связывает сценарии (L1) с уточнениями (L2) и метаданными (L3)."""
    for sc in scenarios:
        scenario_text = f"{sc.title} {sc.content}".lower().replace("ё", "е")
        selected = []
        core = {"Организация", "Подразделение", "КонтурУчета", "Роли", "КритерииПриемки"}
        for clarification in clarifications:
            clarification_id = clarification.id.removeprefix("clarify_")
            keywords = CLARIFICATION_KEYWORDS.get(clarification_id, ())
            if clarification_id in core or any(keyword in scenario_text for keyword in keywords):
                selected.append(clarification.id)
        sc.clarifications = selected
        
        # Ищем релевантные документы Layer 3 по ключевым словам из названия сценария
        keywords = re.findall(r'[А-ЯЁ][а-яё]+', sc.title)
        doc_ids = []
        for mc in metadata_chunks:
            if mc.node_type != 'metadata':
                continue
            # Проверяем по заголовку и содержанию
            title_lower = mc.title.lower()
            path_lower = mc.path.lower()
            for kw in keywords:
                if kw.lower() in title_lower or kw.lower() in path_lower:
                    if mc.id not in doc_ids:
                        doc_ids.append(mc.id)
                    break
        sc.entry_docs = doc_ids[:10]
def build_knowledge_graph(chunks: List[DocChunk]) -> nx.MultiDiGraph:
    """Строит 4-слойный граф знаний."""
    # MultiDiGraph сохраняет несколько независимых смыслов между одной парой
    # узлов (например references + requires + may_write_register).
    G = nx.MultiDiGraph()
    chunk_map = {c.id: c for c in chunks}

    def add_edge(source: str, target: str, rel: str, weight: float,
                 evidence: str = "", properties: Optional[Dict[str, Any]] = None) -> bool:
        if source not in chunk_map or target not in chunk_map or source == target:
            return False
        key_base = rel
        key = key_base
        counter = 2
        # Одинаковое отношение с теми же свойствами повторно не добавляем.
        props_json = json.dumps(properties or {}, ensure_ascii=False, sort_keys=True, default=str)
        while G.has_edge(source, target, key=key):
            existing = G.get_edge_data(source, target, key)
            if existing and existing.get("relation") == rel and existing.get("properties", "{}") == props_json:
                return False
            key = f"{key_base}:{counter}"
            counter += 1
        G.add_edge(
            source, target, key=key, relation=rel, weight=float(weight),
            evidence=str(evidence or ""), properties=props_json,
        )
        return True
    
    print("  Добавление узлов с атрибутами слоёв...")
    for c in chunks:
        G.add_node(c.id,
                   title=c.title,
                   layer=c.layer,
                   node_type=c.node_type,
                   level=c.level,
                   path=c.path,
                   terms=",".join(c.terms[:20]),
                   content_preview=c.content[:500],
                   metadata=json.dumps(c.metadata or {}, ensure_ascii=False, default=str))
    
    print("  Ребра parent-child...")
    for c in chunks:
        if c.parent_id and c.parent_id in chunk_map:
            add_edge(c.parent_id, c.id, "parent_child", 1.0)
            add_edge(c.id, c.parent_id, "child_parent", 0.8)
    
    print("  Ребра между слоями (inter-layer)...")
    layer_edges = 0
    for c in chunks:
        # L1 → L3: entry_docs (сценарий → документы)
        if c.layer == 1 and c.entry_docs:
            for target_id in c.entry_docs:
                if add_edge(c.id, target_id, "entry_doc", 0.9):
                    layer_edges += 1
        # L1 → L2: clarifications (сценарий → уточняющие вопросы)
        if c.layer == 1 and c.clarifications:
            for target_id in c.clarifications:
                if add_edge(c.id, target_id, "clarification", 0.8):
                    layer_edges += 1
        # L2 → L3: entry_docs на clarification-узлах (уточнение → реквизит)
        if c.layer == 2 and c.entry_docs:
            for target_id in c.entry_docs:
                if add_edge(c.id, target_id, "clarifies_field", 0.7):
                    layer_edges += 1
        # Точные типизированные отношения из XML/UI.
        for item in c.relations:
            target_id = item.get("target")
            relation_type = item.get("type")
            if not target_id or not relation_type:
                continue
            if add_edge(
                c.id, target_id, relation_type,
                float(item.get("weight", 0.7)),
                evidence=item.get("evidence", ""),
                properties=item.get("properties", {}),
            ):
                layer_edges += 1
            reverse_type = item.get("reverse_type")
            if reverse_type and add_edge(
                target_id, c.id, reverse_type,
                float(item.get("weight", 0.7)),
                evidence=item.get("evidence", ""),
                properties=item.get("properties", {}),
            ):
                layer_edges += 1
    
    print(f"  Добавлено {layer_edges} межслойных ребер")
    
    print("  Ребра cross-references...")
    ref_count = 0
    for c in chunks:
        for target_id in c.refs:
            if add_edge(c.id, target_id, "references", 0.7):
                ref_count += 1
    print(f"  Добавлено {ref_count} ребер cross-references")
    
    print("  Ребра shared-terms (через инвертированный индекс)...")
    term_index = {}
    for c in chunks:
        seen_terms = set()
        for t in c.terms:
            t_lower = t.lower().strip()
            if len(t_lower) > 2 and t_lower not in seen_terms:
                seen_terms.add(t_lower)
                if t_lower not in term_index:
                    term_index[t_lower] = []
                term_index[t_lower].append(c.id)
    
    shared_edges = 0
    chunk_term_count = {c.id: len(c.terms) for c in chunks}
    
    max_shared_edges = int(os.environ.get("RAG_MAX_SHARED_TERM_EDGES", "20000"))
    for term, cids in sorted(term_index.items()):
        if len(cids) < 2:
            continue
        cids = sorted(set(cids))
        for i in range(len(cids)):
            for j in range(i+1, len(cids)):
                c1, c2 = cids[i], cids[j]
                score = 1.0 / max(chunk_term_count.get(c1, 1), chunk_term_count.get(c2, 1))
                if score > 0.05:
                    added = add_edge(c1, c2, "shared_terms", round(score, 3))
                    add_edge(c2, c1, "shared_terms", round(score, 3))
                    if added:
                        shared_edges += 1
                if shared_edges >= max_shared_edges:
                    break
            if shared_edges >= max_shared_edges:
                break
        if shared_edges >= max_shared_edges:
            break
    print(f"  Добавлено {shared_edges} ребер shared-terms")
    
    print(f"\n[2/5] Граф построен: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер, "
          f"{len([n for n in G.nodes if G.nodes[n].get('layer')==1])} сценариев, "
          f"{len([n for n in G.nodes if G.nodes[n].get('layer')==2])} уточнений, "
          f"{len([n for n in G.nodes if G.nodes[n].get('layer')==3])} метаданных, "
          f"{len([n for n in G.nodes if G.nodes[n].get('layer')==4])} чанков знаний")
    return G

# ---------------------------------------------------------------------------
# 3. Векторизация (embedding) с использованием sentence-transformers
# ---------------------------------------------------------------------------
class Embedder:
    """Обертка для создания эмбеддингов текста.
    
    По умолчанию использует TF-IDF (работает сразу).
    Для лучшего качества установите sentence-transformers:
        pip install sentence-transformers
    """
    
    def __init__(self, model_name: str = None, vectorizer_path: Optional[Path] = None):
        self.model_name = model_name
        self._model = None
        self._vectorizer = None
        self.vectorizer_path = vectorizer_path
        # Пробуем загрузить существующий vectorizer
        if vectorizer_path and vectorizer_path.exists():
            try:
                import pickle
                with open(vectorizer_path, "rb") as f:
                    self._vectorizer = pickle.load(f)
                print(f"  Загружен TF-IDF vectorizer из {vectorizer_path}")
            except Exception:
                pass
    
    def _try_load_sentence(self):
        if self.model_name is None:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Загрузка sentence-transformers: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print("  Модель загружена")
            return True
        except Exception as e:
            print(f"  Не удалось загрузить sentence-transformers: {e}")
            return False
    
    def encode(self, texts: List[str], fit: bool = False):
        if self._model is None:
            if not self._try_load_sentence():
                return self._encode_tfidf(texts, fit=fit)
            else:
                return self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
        else:
            return self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
    
    def _encode_tfidf(self, texts: List[str], fit: bool = False) -> np.ndarray:
        if fit or self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                analyzer='word',
                token_pattern=r'(?u)\b[a-zA-Zа-яА-ЯёЁ0-9]+\b',
                ngram_range=(1, 2),
                stop_words=None,
                sublinear_tf=True
            )
            vectors = self._vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        else:
            vectors = self._vectorizer.transform(texts).tocsr().astype(np.float32)
        return normalize(vectors, norm="l2", axis=1, copy=False)
    
    def save_vectorizer(self, target_path: Optional[Path] = None):
        path = target_path or self.vectorizer_path
        if self._vectorizer and path:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self._vectorizer, f)


def create_embeddings(chunks: List[DocChunk], embedder: Embedder) -> Tuple[Any, List[str]]:
    """Создает эмбеддинги для всех чанков"""
    print(f"\n[3/5] Создание эмбеддингов для {len(chunks)} чанков...")
    
    texts = []
    node_ids = []
    for c in chunks:
        # Берем заголовок + первые 1500 символов (для больших документов)
        text = f"{c.title}\n{c.content[:1500]}"
        texts.append(text)
        node_ids.append(c.id)
    
    vectors = embedder.encode(texts, fit=True)
    print(f"  Размерность эмбеддингов: {vectors.shape}")
    return vectors, node_ids

# ---------------------------------------------------------------------------
# 4. Сохранение/загрузка данных
# ---------------------------------------------------------------------------
def save_data(chunks: List[DocChunk], graph: nx.Graph, vectors: Any,
              node_ids: List[str], embedder: Optional[Embedder] = None):
    """Сохраняет все данные на диск"""
    print(f"\n[4/5] Сохранение данных...")
    import uuid
    stage = DATA_DIR / f".build-staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        stage_chunks = stage / CHUNKS_JSONL_FILE.name
        with open(stage_chunks, "w", encoding="utf-8", newline="\n") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False, default=str) + "\n")

        chunks_meta = {
            c.id: {
                "title": c.title, "path": c.path, "layer": c.layer,
                "node_type": c.node_type, "level": c.level,
                "content_preview": c.content[:500], "metadata": c.metadata,
            }
            for c in chunks
        }
        stage_meta = stage / CHUNKS_META_FILE.name
        stage_meta.write_text(json.dumps(chunks_meta, ensure_ascii=False), encoding="utf-8")
        stage_nodes = stage / NODES_FILE.name
        stage_nodes.write_text(json.dumps(node_ids), encoding="utf-8")

        if sp.issparse(vectors):
            vector_target = VECTORS_SPARSE_FILE
            stage_vectors = stage / vector_target.name
            sp.save_npz(stage_vectors, vectors.astype(np.float32), compressed=True)
            vector_format = "scipy-csr-npz"
        else:
            vector_target = VECTORS_FILE
            stage_vectors = stage / vector_target.name
            np.save(stage_vectors, np.asarray(vectors, dtype=np.float32))
            vector_format = "numpy-float32"

        stage_graph = stage / GRAPH_PICKLE.name
        with open(stage_graph, "wb") as f:
            pickle.dump(graph, f)
        staged_files = [
            (stage_chunks, CHUNKS_JSONL_FILE), (stage_meta, CHUNKS_META_FILE),
            (stage_nodes, NODES_FILE), (stage_vectors, vector_target),
            (stage_graph, GRAPH_PICKLE),
        ]
        if embedder is not None:
            stage_vectorizer = stage / TFIDF_FILE.name
            embedder.save_vectorizer(stage_vectorizer)
            staged_files.append((stage_vectorizer, TFIDF_FILE))
        if os.environ.get("RAG_WRITE_GRAPHML", "").strip().lower() in {"1", "true", "yes"}:
            stage_graphml = stage / GRAPH_FILE.name
            nx.write_graphml(graph, stage_graphml)
            staged_files.append((stage_graphml, GRAPH_FILE))

        stage_manifest = stage / INDEX_MANIFEST_FILE.name
        stage_manifest.write_text(json.dumps({
            "schema_version": 2,
            "chunks": len(chunks), "unique_nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(), "vector_rows": int(vectors.shape[0]),
            "vector_columns": int(vectors.shape[1]), "vector_format": vector_format,
            "vector_file": vector_target.name,
            "graph_type": type(graph).__name__,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        staged_files.append((stage_manifest, INDEX_MANIFEST_FILE))

        # os.replace атомарен для файлов на одном томе. Старый рабочий индекс
        # остаётся целым, пока весь новый набор не подготовлен в staging.
        for source, target in staged_files:
            os.replace(source, target)
    finally:
        resolved_stage = stage.resolve()
        if resolved_stage.parent == DATA_DIR.resolve() and resolved_stage.name.startswith(".build-staging-"):
            shutil.rmtree(resolved_stage, ignore_errors=True)

    print(f"  Чанки: {CHUNKS_JSONL_FILE}")
    print(f"  Векторы: {vector_target}")
    print(f"  Граф: {GRAPH_PICKLE}")


def compact_search_index(vectors: Any, node_ids: List[str],
                         valid_ids: set[str], batch_size: int = 512):
    """Убирает дубли node_ids и конвертирует старый dense TF-IDF в CSR по частям."""
    last_index: Dict[str, int] = {}
    for index, node_id in enumerate(node_ids):
        if node_id in valid_ids:
            last_index[node_id] = index
    ordered = sorted(last_index.items(), key=lambda item: item[1])
    unique_ids = [node_id for node_id, _ in ordered]
    indices = [index for _, index in ordered]
    if sp.issparse(vectors):
        return vectors[indices].tocsr().astype(np.float32), unique_ids
    parts = []
    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset:offset + batch_size]
        dense = np.asarray(vectors[batch_indices], dtype=np.float32)
        part = sp.csr_matrix(dense)
        part.eliminate_zeros()
        parts.append(part)
    matrix = sp.vstack(parts, format="csr", dtype=np.float32) if parts else sp.csr_matrix((0, vectors.shape[1]), dtype=np.float32)
    return matrix, unique_ids


def save_enhanced_index(chunks: List[DocChunk], graph: nx.Graph,
                        vectors: Any, node_ids: List[str],
                        remove_dense_after_success: bool = False):
    """Атомарно сохраняет расширенную топологию, не перезаписывая полный корпус ИТС."""
    import uuid
    stage = DATA_DIR / f".enhance-staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        stage_meta = stage / CHUNKS_META_FILE.name
        meta_payload = {
            c.id: {
                "title": c.title, "path": c.path, "layer": c.layer,
                "node_type": c.node_type, "level": c.level,
                "content_preview": c.content[:500], "metadata": c.metadata,
            }
            for c in chunks
        }
        stage_meta.write_text(json.dumps(meta_payload, ensure_ascii=False), encoding="utf-8")
        stage_nodes = stage / NODES_FILE.name
        stage_nodes.write_text(json.dumps(node_ids), encoding="utf-8")
        stage_vectors = stage / VECTORS_SPARSE_FILE.name
        sp.save_npz(stage_vectors, vectors.tocsr().astype(np.float32), compressed=True)
        stage_graph = stage / GRAPH_PICKLE.name
        with stage_graph.open("wb") as handle:
            pickle.dump(graph, handle)
        stage_manifest = stage / INDEX_MANIFEST_FILE.name
        stage_manifest.write_text(json.dumps({
            "schema_version": 2, "build_mode": "incremental-enhance",
            "chunks": len(chunks), "unique_nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(), "vector_rows": int(vectors.shape[0]),
            "vector_columns": int(vectors.shape[1]), "vector_format": "scipy-csr-npz",
            "vector_file": VECTORS_SPARSE_FILE.name,
            "graph_type": type(graph).__name__,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        staged = [
            (stage_meta, CHUNKS_META_FILE), (stage_nodes, NODES_FILE),
            (stage_vectors, VECTORS_SPARSE_FILE), (stage_graph, GRAPH_PICKLE),
            (stage_manifest, INDEX_MANIFEST_FILE),
        ]
        if os.environ.get("RAG_WRITE_GRAPHML", "").strip().lower() in {"1", "true", "yes"}:
            stage_graphml = stage / GRAPH_FILE.name
            nx.write_graphml(graph, stage_graphml)
            staged.append((stage_graphml, GRAPH_FILE))
        for source, target in staged:
            os.replace(source, target)
    finally:
        resolved = stage.resolve()
        if resolved.parent == DATA_DIR.resolve() and resolved.name.startswith(".enhance-staging-"):
            shutil.rmtree(resolved, ignore_errors=True)
    if remove_dense_after_success and VECTORS_FILE.exists():
        # Это производный индекс, восстановимый из исходников; удаляем только
        # после успешной атомарной установки CSR и manifest.
        VECTORS_FILE.unlink()


def save_compact_vectors_only(vectors: Any, node_ids: List[str], graph: nx.Graph,
                              remove_dense_after_success: bool = True):
    """Атомарно устанавливает CSR-индекс, освобождая место для enhance."""
    import uuid
    stage = DATA_DIR / f".compact-staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        stage_vectors = stage / VECTORS_SPARSE_FILE.name
        sp.save_npz(stage_vectors, vectors.tocsr().astype(np.float32), compressed=True)
        stage_nodes = stage / NODES_FILE.name
        stage_nodes.write_text(json.dumps(node_ids), encoding="utf-8")
        stage_manifest = stage / INDEX_MANIFEST_FILE.name
        stage_manifest.write_text(json.dumps({
            "schema_version": 2, "build_mode": "compact-base",
            "chunks": graph.number_of_nodes(), "unique_nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(), "vector_rows": int(vectors.shape[0]),
            "vector_columns": int(vectors.shape[1]), "vector_format": "scipy-csr-npz",
            "vector_file": VECTORS_SPARSE_FILE.name,
            "graph_type": type(graph).__name__,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        for source, target in (
            (stage_vectors, VECTORS_SPARSE_FILE),
            (stage_nodes, NODES_FILE),
            (stage_manifest, INDEX_MANIFEST_FILE),
        ):
            os.replace(source, target)
    finally:
        resolved = stage.resolve()
        if resolved.parent == DATA_DIR.resolve() and resolved.name.startswith(".compact-staging-"):
            shutil.rmtree(resolved, ignore_errors=True)
    if remove_dense_after_success and VECTORS_FILE.exists():
        VECTORS_FILE.unlink()
    if remove_dense_after_success and GRAPH_FILE.exists() and not os.environ.get("RAG_WRITE_GRAPHML"):
        GRAPH_FILE.unlink()


def _load_full_chunks() -> List[DocChunk]:
    if CHUNKS_JSONL_FILE.exists():
        chunks = []
        with open(CHUNKS_JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(DocChunk(**json.loads(line)))
        return chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return [DocChunk(**d) for d in json.load(f)]


def _load_vectors():
    if INDEX_MANIFEST_FILE.exists():
        try:
            manifest = json.loads(INDEX_MANIFEST_FILE.read_text(encoding="utf-8"))
            vector_file = manifest.get("vector_file")
            if vector_file == VECTORS_SPARSE_FILE.name and VECTORS_SPARSE_FILE.exists():
                return sp.load_npz(VECTORS_SPARSE_FILE).tocsr()
            if vector_file == VECTORS_FILE.name and VECTORS_FILE.exists():
                return np.load(VECTORS_FILE, mmap_mode="r")
        except (OSError, json.JSONDecodeError):
            pass
    if VECTORS_SPARSE_FILE.exists():
        return sp.load_npz(VECTORS_SPARSE_FILE).tocsr()
    if VECTORS_FILE.exists():
        return np.load(VECTORS_FILE, mmap_mode="r")
    return None


def load_data(lightweight=False) -> Tuple[List[DocChunk], Optional[nx.Graph], Any, Optional[List[str]]]:
    """Загружает данные с диска.
    
    lightweight=True: не загружает полный текст чанков и граф (быстрый поиск).
    """
    if not CHUNKS_JSONL_FILE.exists() and not CHUNKS_FILE.exists():
        return [], None, None, None
    
    if lightweight:
        # Быстрая загрузка: только метаданные + векторы
        chunks_meta = {}
        if CHUNKS_META_FILE.exists():
            with open(CHUNKS_META_FILE, "r", encoding="utf-8") as f:
                chunks_meta = json.load(f)
        else:
            # Fallback: загружаем полные чанки, но берём только метаданные
            chunks_data = [asdict(chunk) for chunk in _load_full_chunks()]
            chunks_meta = {d["id"]: {"title": d["title"], "path": d["path"]} for d in chunks_data}
        
        # Топология графа нужна и быстрому поиску; pickle ~50 МБ и не требует
        # загрузки полного текста 700+ МБ.
        graph = None
        if GRAPH_PICKLE.exists():
            with open(GRAPH_PICKLE, "rb") as f:
                graph = pickle.load(f)
        elif GRAPH_FILE.exists():
            graph = nx.read_graphml(GRAPH_FILE)

        chunks = []
        for key, value in chunks_meta.items():
            graph_data = graph.nodes[key] if graph is not None and key in graph else {}
            metadata = value.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            chunks.append(DocChunk(
                id=key, title=value["title"], path=value["path"],
                content=value.get("content_preview", graph_data.get("content_preview", "")),
                layer=int(value.get("layer", graph_data.get("layer", 4)) or 4),
                node_type=value.get("node_type", graph_data.get("node_type", "knowledge")),
                level=int(value.get("level", graph_data.get("level", 0)) or 0),
                metadata=metadata,
            ))
        
        vectors = None
        node_ids = None
        if (VECTORS_SPARSE_FILE.exists() or VECTORS_FILE.exists()) and NODES_FILE.exists():
            vectors = _load_vectors()
            with open(NODES_FILE, "r") as f:
                node_ids = json.load(f)

        print(f"Загружено (быстро): {len(chunks)} чанков (метаданные), векторы {vectors.shape if vectors is not None else '—'}")
        return chunks, graph, vectors, node_ids

    chunks = _load_full_chunks()
    # Инкрементальный enhance хранит новые поля/UI-узлы в compact meta, а
    # полный текст старого корпуса остаётся в chunks.json. Добавляем previews
    # расширений, не теряя полный текст L4.
    if CHUNKS_META_FILE.exists():
        compact_meta = json.loads(CHUNKS_META_FILE.read_text(encoding="utf-8"))
        existing_ids = {chunk.id for chunk in chunks}
        for key, value in compact_meta.items():
            if key in existing_ids:
                continue
            metadata = value.get("metadata", {})
            chunks.append(DocChunk(
                id=key, title=value.get("title", key),
                content=value.get("content_preview", ""), path=value.get("path", key),
                layer=int(value.get("layer", 4) or 4),
                node_type=value.get("node_type", "knowledge"),
                level=int(value.get("level", 0) or 0), metadata=metadata if isinstance(metadata, dict) else {},
            ))
    
    graph = None
    if GRAPH_PICKLE.exists():
        with open(GRAPH_PICKLE, "rb") as f:
            graph = pickle.load(f)
    elif GRAPH_FILE.exists():
        graph = nx.read_graphml(GRAPH_FILE)
    
    vectors = None
    node_ids = None
    if (VECTORS_SPARSE_FILE.exists() or VECTORS_FILE.exists()) and NODES_FILE.exists():
        vectors = _load_vectors()
        with open(NODES_FILE, "r") as f:
            node_ids = json.load(f)
    
    print(f"Загружено: {len(chunks)} чанков, {graph.number_of_nodes() if graph else 0} узлов графа")
    return chunks, graph, vectors, node_ids

# ---------------------------------------------------------------------------
# 5. Graph RAG запрос
# ---------------------------------------------------------------------------
class GraphRAG:
    """Graph RAG движок: объединяет поиск по графу и семантический поиск"""
    
    def __init__(self, chunks: List[DocChunk], graph: nx.Graph,
                 vectors: Any, node_ids: List[str],
                 embedder: Embedder):
        self.chunks = chunks
        self.chunk_map = {c.id: c for c in chunks}
        self.graph = graph
        self.vectors = vectors
        self.node_ids = node_ids
        self.node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        self.embedder = embedder
    
    def _similarities(self, query: str) -> np.ndarray:
        query_matrix = self.embedder.encode([query])
        if sp.issparse(self.vectors):
            if not sp.issparse(query_matrix):
                query_matrix = sp.csr_matrix(np.asarray(query_matrix, dtype=np.float32))
            result = self.vectors @ query_matrix.T
            return result.toarray().ravel().astype(np.float32, copy=False)
        query_vector = (
            query_matrix.toarray().ravel() if sp.issparse(query_matrix)
            else np.asarray(query_matrix).reshape(-1)
        )
        dtype = getattr(self.vectors, "dtype", np.float32)
        query_vector = query_vector.astype(dtype, copy=False)
        # Векторы нормализованы при сборке, поэтому dot == cosine similarity и
        # не создаёт копию float64 размером ~1.8 ГБ.
        return np.asarray(self.vectors @ query_vector, dtype=np.float32).ravel()

    def _relation_edges(self, node_id: str, direction: str = "out"):
        if self.graph is None or node_id not in self.graph:
            return []
        result = []
        if self.graph.is_multigraph():
            iterator = (
                self.graph.out_edges(node_id, keys=True, data=True)
                if direction == "out" else self.graph.in_edges(node_id, keys=True, data=True)
            )
            for source, target, _key, data in iterator:
                result.append((source, target, data))
        else:
            iterator = (
                self.graph.out_edges(node_id, data=True)
                if direction == "out" else self.graph.in_edges(node_id, data=True)
            )
            result.extend(iterator)
        return result

    def search(self, query: str, top_k: int = None, graph_expand: int = 30,
               graph_depth: int = 2) -> Dict:
        """
        4-слойный Graph RAG поиск:
        1. Векторный поиск по query → находит сценарии (L1), метаданные (L3), знания (L4)
        2. Расширение по графу через межслойные ребра (entry_doc, clarification, parent_child, references)
        3. Возвращает результаты, сгруппированные по слоям
        """
        if top_k is None:
            word_count = len(query.split())
            top_k = max(5, min(20, word_count * 3))
        
        if self.vectors is None or not self.node_ids:
            raise RuntimeError("Поисковый индекс не загружен. Выполните build.")
        sims = self._similarities(query)
        
        top_indices = np.argsort(sims)[::-1]
        top_indices = [i for i in top_indices if sims[i] > 0.03][:top_k]
        
        # Собираем результаты векторного поиска
        vector_results = []
        seen_vector_nodes = set()
        for idx in top_indices:
            node_id = self.node_ids[idx]
            if node_id in seen_vector_nodes:
                continue
            chunk = self.chunk_map.get(node_id)
            if chunk:
                seen_vector_nodes.add(node_id)
                vector_results.append({
                    "node_id": node_id,
                    "title": chunk.title,
                    "path": chunk.path,
                    "score": float(sims[idx]),
                    "content": chunk.content,
                    "source": "vector",
                    "layer": chunk.layer,
                    "node_type": chunk.node_type,
                })
        
        # Расширение по графу: межслойные и внутрислойные связи
        relation_priority = {
            "entry_doc": 0.98, "clarification": 0.95, "clarifies_field": 0.92,
            "requires": 1.0, "required_reference": 1.0, "can_create_inline": 0.92,
            "creates_on_basis": 0.96, "can_be_created_on_basis_of": 0.9,
            "may_write_register": 0.92, "has_registrator": 0.92,
            "opened_by_command": 0.95, "opens_object": 0.95,
            "in_subsystem": 0.88, "contains_object": 0.86,
            "has_field": 0.82, "field_type": 0.86,
            "parent_child": 0.76, "child_parent": 0.7,
            "references": 0.58, "shared_terms": 0.36,
        }
        graph_expanded_nodes: Dict[str, Dict[str, Any]] = {}
        vector_node_ids = {item["node_id"] for item in vector_results}
        for seed in vector_results:
            seed_id = seed["node_id"]
            queue = [(seed_id, max(float(seed["score"]), 0.35), 0, [seed_id], [])]
            best_depth = {seed_id: 0}
            while queue:
                current, path_score, depth, path, relations = queue.pop(0)
                if depth >= graph_depth:
                    continue
                candidates = []
                for direction in ("out", "in"):
                    for source, target, edge_data in self._relation_edges(current, direction):
                        neighbor = target if direction == "out" else source
                        rel = edge_data.get("relation", "")
                        edge_weight = float(edge_data.get("weight", relation_priority.get(rel, 0.5)))
                        semantic_weight = relation_priority.get(rel, edge_weight)
                        if direction == "in":
                            semantic_weight *= 0.88
                        candidates.append((semantic_weight, neighbor, rel, direction, edge_data))
                candidates.sort(key=lambda item: (-item[0], str(item[1]), item[2]))
                for semantic_weight, neighbor, rel, direction, edge_data in candidates[:60]:
                    if neighbor in path or neighbor not in self.chunk_map:
                        continue
                    new_depth = depth + 1
                    new_score = path_score * semantic_weight * (0.88 ** (new_depth - 1))
                    existing = graph_expanded_nodes.get(neighbor)
                    if existing is None or new_score > existing["path_score"]:
                        graph_expanded_nodes[neighbor] = {
                            "path_score": new_score, "relation": rel,
                            "direction": direction, "path": path + [neighbor],
                            "relations": relations + [rel],
                            "evidence": edge_data.get("evidence", ""),
                        }
                    if new_depth < best_depth.get(neighbor, graph_depth + 1):
                        best_depth[neighbor] = new_depth
                        queue.append((
                            neighbor, new_score, new_depth,
                            path + [neighbor], relations + [rel],
                        ))
        
        graph_results = []
        for nid, expansion in graph_expanded_nodes.items():
            if nid in vector_node_ids:
                continue
            chunk = self.chunk_map.get(nid)
            if chunk:
                idx = self.node_to_idx.get(nid)
                vector_score = float(sims[idx]) if idx is not None and idx < len(sims) else 0.0
                score = max(float(expansion["path_score"]), vector_score * 0.45 + float(expansion["path_score"]) * 0.55)
                graph_results.append({
                    "node_id": nid,
                    "title": chunk.title,
                    "path": chunk.path,
                    "score": score,
                    "content": chunk.content,
                    "source": "graph",
                    "layer": chunk.layer,
                    "node_type": chunk.node_type,
                    "relation": expansion["relation"],
                    "graph_path": expansion["path"],
                    "graph_relations": expansion["relations"],
                    "evidence": expansion["evidence"],
                })
        
        graph_results.sort(key=lambda x: x["score"], reverse=True)
        graph_results = graph_results[:graph_expand]
        
        all_results = vector_results + graph_results
        
        # Строим контекст
        paths = set()
        for r in all_results:
            parts = r["path"].split("/")
            for i in range(1, len(parts)+1):
                p = "/".join(parts[:i])
                if p in self.chunk_map:
                    paths.add(self.chunk_map[p].title)
        
        context_parts = []
        context_parts.append("=== ИЕРАРХИЯ РАЗДЕЛОВ ===")
        context_parts.append(" > ".join(sorted(paths)))
        context_parts.append("")
        context_parts.append("=== РЕЛЕВАНТНЫЕ ДОКУМЕНТЫ (ПОЛНЫЙ ТЕКСТ) ===")
        for i, r in enumerate(all_results):
            context_parts.append(f"\n--- [{i+1}] {r['title']} (score={r['score']:.3f}, source={r['source']}) ---")
            context_parts.append(f"Путь: {r['path']}")
            # Показываем до 3000 символов контента
            content = r['content'][:3000].rstrip()
            if len(content) == 3000:
                content += "..."
            context_parts.append(content)
        
        context = "\n".join(context_parts)
        
        workflow = self._extract_workflow(all_results, query)
        
        # Группируем результаты по слоям
        by_layer = {1: [], 2: [], 3: [], 4: []}
        for r in all_results:
            layer = r.get("layer", 4)
            if layer in by_layer:
                by_layer[layer].append(r)
        
        return {
            "query": query,
            "vector_results": vector_results,
            "graph_expanded": graph_results,
            "context": context,
            "workflow": workflow,
            "all_nodes": list(set(r["node_id"] for r in all_results)),
            "by_layer": {
                "scenarios": by_layer[1][:5],
                "clarifications": by_layer[2][:10],
                "metadata": by_layer[3][:10],
                "knowledge": by_layer[4][:10],
            }
        }
    
    def _extract_workflow(self, results: List[Dict], query: str) -> List[str]:
        """Извлекает последовательность действий из результатов.
        
        Ищет:
        - Нумерованные списки (1. 2. 3. или 1) 2) 3))
        - Маркеры последовательности
        - Реквизиты и поля для заполнения
        """
        sorted_results = sorted(results, key=lambda r: len(r["path"].split("/")))
        
        # Собираем все нумерованные шаги со всего контента
        all_steps = []
        seen_steps = set()
        
        for r in sorted_results:
            chunk = self.chunk_map.get(r["node_id"])
            if not chunk:
                continue
            text = chunk.content
            
            # 1. Нумерованные списки (1. text, 2. text, или 1) text)
            numbered = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s*([^\n]+)', text)
            for s in numbered:
                s_clean = s.strip()
                if s_clean and len(s_clean) > 10 and s_clean not in seen_steps:
                    all_steps.append(f"  {len(all_steps)+1}. {s_clean}")
                    seen_steps.add(s_clean)
            
            # 2. Маркированные списки (- text или * text) - важные пункты
            bullets = re.findall(r'(?:^|\n)\s*[-‣•]\s*([^\n]+)', text)
            for s in bullets:
                s_clean = s.strip()
                if s_clean and len(s_clean) > 20 and s_clean not in seen_steps:
                    all_steps.append(f"  - {s_clean}")
                    seen_steps.add(s_clean)
            
            # 3. Маркеры "Последовательность действий", "Порядок действий"
            markers = re.findall(
                r'(?:Последовательность|Порядок|Этап[а-я]*|'
                r'Необходимо|Важно|Внимание|Примечание|'
                r'Реквизит[а-я]*|Обязательно|'
                r'Настройк[аи]|Параметр[а-я]*):?\s*([^\n]+)',
                text, re.IGNORECASE
            )
            for s in markers:
                s_clean = s.strip()
                if s_clean and len(s_clean) > 15 and s_clean not in seen_steps:
                    all_steps.append(f"  * {s_clean}")
                    seen_steps.add(s_clean)
        
        return all_steps[:30]
    
    def generate_instruction(self, query: str, provider: str = "wormsoft",
                              model: str = "wormsoft/agent/high") -> Dict:
        """Полный пайплайн: поиск по 4 слоям → промпт → генерация через LLM"""
        result = self.search(query)
        
        if provider == "ollama":
            prompt = self._build_compact_prompt(query, result)
        else:
            context = self._format_layer_context(result)
            prompt = f"""Ты — эксперт 1С:ERP. Пользователь: {query}

Контекст из графа знаний (4 слоя):
{context[:6000]}

Составь подробную пошаговую инструкцию по настройке сквозного бизнес-процесса:
1. Какие объекты/документы 1С создать и в какой последовательности
2. Где находится каждый пункт в меню (раздел, подраздел, пункт)
3. Какие реквизиты заполнить и какими значениями
4. Какие настройки включить и где
5. Как проверить, что всё работает"""
        
        system = "Ты — эксперт-консультант по 1С:ERP Управление предприятием 2.5. Давай точные пошаговые инструкции на основе документации: куда нажимать, что заполнять, как настроить."
        
        llm = LlmClient(provider=provider, model=model)
        try:
            instruction = llm.prompt(prompt, system=system)
        except Exception as e:
            instruction = f"Ошибка генерации инструкции: {e}"
        finally:
            llm.close()
        
        return {
            "query": query,
            "instruction": instruction,
            "context": result["context"],
            "workflow": result["workflow"],
            "vector_results": result["vector_results"][:5],
            "graph_expanded": result["graph_expanded"]
        }
    
    def _build_compact_prompt(self, query: str, result: Dict) -> str:
        """Компактный промпт для маленьких локальных моделей (Ollama)"""
        sections = []
        for r in result["vector_results"][:3]:
            sections.append(r["title"])
        sections_text = "; ".join(sections)
        
        return f"Ты эксперт 1С ERP. На основе документации ({sections_text}) ответь на вопрос: {query}\n\nДай пошаговую инструкцию по настройке: что создать, где найти в меню, какие реквизиты заполнить."

    def _format_layer_context(self, result: Dict) -> str:
        """Форматирует контекст из всех 4 слоёв для DeepSeek."""
        parts = []
        by_layer = result.get("by_layer", {})
        
        # L1: Сценарии
        scens = by_layer.get("scenarios", [])
        if scens:
            parts.append("=== [L1] БИЗНЕС-СЦЕНАРИИ ===")
            for s in scens:
                parts.append(f"- {s['title']}")
        
        # L2: Уточнения
        cls = by_layer.get("clarifications", [])
        if cls:
            parts.append("\n=== [L2] УТОЧНЯЮЩИЕ ВОПРОСЫ ===")
            for c in cls:
                parts.append(f"- {c['title']}")
        
        # L3: Метаданные
        meta = by_layer.get("metadata", [])
        if meta:
            parts.append("\n=== [L3] ОБЪЕКТЫ МЕТАДАННЫХ ===")
            for m in meta[:8]:
                content = m.get("content", "")[:600]
                parts.append(f"- {m['title']}: {content}")
        
        # L4: Знания
        kn = by_layer.get("knowledge", [])
        if kn:
            parts.append("\n=== [L4] ФРАГМЕНТЫ ДОКУМЕНТАЦИИ ===")
            for k in kn[:5]:
                content = k.get("content", "")[:300].strip()
                parts.append(f"- {k['title']}: {content}")
        
        return "\n".join(parts)

    def generate_clarifying_questions(self, query: str, result: Dict,
                                       llm: 'LlmClient') -> str:
        """Генерирует уточняющие вопросы по моделированию (не по реализации)."""
        context = self._format_layer_context(result)
        system = "Ты — методолог по 1С:ERP. Твоя задача — понять бизнес-процесс пользователя, а не давать инструкции."
        prompt = f"""Пользователь хочет настроить в 1С:ERP: {query}

Вот что нашлось в графе знаний (все 4 слоя):
{context[:4000]}

Сформулируй 3-5 уточняющих вопросов, чтобы понять:
- КАКОЙ именно бизнес-процесс нужно автоматизировать (не какой документ, а какой процесс)
- Какие документы/объекты должны быть задействованы
- Какие есть особые требования (организация, подразделение, номенклатура)

Вопросы должны быть про бизнес-моделирование, не про реализацию.
Пример: «Какие документы закупки должны создаваться: Заказ поставщику или Поступление товаров?» — НЕ верно.
«Какой бизнес-процесс закупок: закупка сырья для производства или закупка товаров для перепродажи?» — верно.

Напиши ТОЛЬКО вопросы, по одному на строку."""
        return llm.prompt(prompt, system=system)

    def generate_instruction_with_context(self, query: str, answers: str,
                                           result: Dict, llm: 'LlmClient') -> str:
        """Генерирует итоговую инструкцию с учётом уточнений (полный контекст 4 слоёв)."""
        context = self._format_layer_context(result)
        system = "Ты — эксперт-консультант по 1С:ERP Управление предприятием 2.5. Составляй подробные пошаговые инструкции по настройке сквозных бизнес-процессов: куда нажимать, что заполнять, как проверить."
        
        prompt = f"""Пользователь хочет: {query}

Дополнительный контекст от пользователя (ответы на уточняющие вопросы):
{answers}

Контекст из графа знаний 1С ERP (все 4 слоя):
{context[:6000]}

Составь подробную пошаговую инструкцию по настройке сквозного бизнес-процесса:

1. Какой бизнес-процесс настраивается и какие документы 1С в нём участвуют (в порядке создания)
2. Для каждого шага: ГДЕ в меню 1С:ERP найти нужный пункт (Раздел → подраздел → команда)
3. Для каждого документа: КАКИЕ реквизиты заполнить и какими значениями (виды номенклатуры, склады, соглашения, цены)
4. Какие настройки/справочники предварительно заполнить (ставки НДС, виды цен, склады и т.п.)
5. Как проверить результат (какие отчёты/регистры проверить)

Пиши как методист для начинающего пользователя: точно, без абстракций, только то, что есть в контексте документации. Если данных не хватает — укажи, что именно нужно уточнить."""
        return llm.prompt(prompt, system=system)
    
    def build_prompt(self, query: str, result: Optional[Dict] = None) -> str:
        """Строит промпт для LLM на основе контекста из 4 слоёв."""
        if result is None:
            result = self.search(query)
        
        context = self._format_layer_context(result)
        workflow_text = ""
        if result.get("workflow"):
            workflow_text = "\n=== ПОСЛЕДОВАТЕЛЬНОСТЬ ДЕЙСТВИЙ ===\n" + "\n".join(result["workflow"])
        
        return f"""Ты — эксперт-консультант по 1С:ERP Управление предприятием 2.5.

Вопрос: {query}

{context}
{workflow_text}

На основе этих данных составь подробную пошаговую инструкцию:
1. Какие объекты/документы 1С создать и где в меню
2. Какие реквизиты заполнить
3. Последовательность шагов
4. Какие настройки включить"""


# ---------------------------------------------------------------------------
# 6. LLM клиент для генерации инструкций (ai.wormsoft.ru / DeepSeek / Ollama)
# ---------------------------------------------------------------------------
class LlmClient:
    """Клиент для вызова LLM: Wormsoft (https://ai.wormsoft.ru), DeepSeek API или Ollama.
    
    Wormsoft — основной режим (OpenAI-совместимый, ключ WORMSOFT_API_KEY в .env).
    DeepSeek — запасной облачный (DEEPSEEK_API_KEY).
    Ollama — локальный для тестов.
    """
    
    DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
    WORMSOFT_URL = "https://ai.wormsoft.ru/api/gpt/chat/completions"
    
    def __init__(self, provider: str = "wormsoft", model: str = "wormsoft/agent/high",
                 timeout: int = 300):
        self.provider = provider
        self.model = model
        self.timeout = timeout
        if provider == "wormsoft":
            self.api_key = os.environ.get("WORMSOFT_API_KEY", "")
        elif provider == "deepseek":
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        else:
            self.api_key = ""
    
    def prompt(self, message: str, system: str = "") -> str:
        if not llm_calls_enabled():
            return (
                "[Ошибка] LLM-вызовы отключены. Для намеренного вызова запустите "
                "команду с --allow-llm или установите RAG_ENABLE_LLM=1."
            )
        if self.provider == "wormsoft" and self.api_key:
            return self._prompt_openai(self.WORMSOFT_URL, "Wormsoft", message, system)
        elif self.provider == "deepseek" and self.api_key:
            return self._prompt_openai(self.DEEPSEEK_URL, "DeepSeek", message, system)
        elif self.provider == "ollama":
            return self._prompt_ollama(message)
        else:
            return (f"[Ошибка] Нет API-ключа Wormsoft (WORMSOFT_API_KEY) "
                    f"или DeepSeek (DEEPSEEK_API_KEY), Ollama не выбран")
    
    def _prompt_openai(self, url: str, name: str, message: str, system: str = "") -> str:
        """Универсальный вызов OpenAI-совместимого chat/completions (Wormsoft / DeepSeek)."""
        try:
            import urllib.request
            import urllib.error
            
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})
            
            body = json.dumps({
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 16384,
                "stream": False,
            }).encode("utf-8")
            
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8") if e.fp else str(e)
            return f"[Ошибка {name} API {e.code}] {detail[:200]}"
        except Exception as e:
            return f"[Ошибка {name}] {e}"
    
    def _prompt_ollama(self, message: str) -> str:
        """Отправляет промпт в локальную модель Ollama (запасной вариант)."""
        try:
            import urllib.request
            import urllib.error
            
            data = json.dumps({
                "model": self.model,
                "prompt": message,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 4096
                }
            }).encode("utf-8")
            
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            return result.get("response", "").strip()
        except Exception as e:
            return f"[Ошибка Ollama] {e}"
    
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# 7. CLI команды
# ---------------------------------------------------------------------------
def cmd_build(include_fields: bool = True, include_ui: bool = True,
              include_forms: bool = False):
    """Сборка 4-слойного графа знаний"""
    print("=" * 60)
    print("  1С ERP Graph RAG - Построение 4-слойного графа знаний")
    print("=" * 60)
    
    # Layer 4: Парсим документацию ИТС
    print("\n[Layer 4] Парсинг документации...")
    chunks = parse_its_markdown()
    
    # Layer 3: Парсим код конфигурации 1С ERP
    print("\n[Layer 3] Парсинг метаданных...")
    code_chunks = parse_erp_code()
    chunks.extend(code_chunks)

    # Точный структурный проход по XML: обязательные поля, зависимости,
    # BasedOn, движения по регистрам, подсистемы, команды и формы.
    print("\n[Layer 3+] Типизированные связи метаданных и интерфейса...")
    try:
        from erp_graph_enhancements import enrich_erp_metadata
        typed_chunks, typed_stats = enrich_erp_metadata(
            ERPCODE_DIR, code_chunks, DocChunk,
            include_fields=include_fields,
            include_ui=include_ui,
            include_forms=include_forms,
        )
        chunks.extend(typed_chunks)
        print(f"  Добавлено типизированных L3-узлов: {len(typed_chunks)}")
        print(f"  Статистика: {json.dumps(typed_stats, ensure_ascii=False)}")
    except Exception as exc:
        # Базовый граф можно собрать, но ошибка явно попадает в отчёт/консоль.
        print(f"  Ошибка типизированного прохода L3: {exc}")
        raise
    
    # Layer 1: Парсим бизнес-сценарии
    print("\n[Layer 1] Извлечение бизнес-сценариев...")
    metadata_chunks = [c for c in chunks if c.node_type == 'metadata']
    scenarios = parse_business_scenarios()
    chunks.extend(scenarios)
    
    # Layer 2: Генерация уточняющих узлов
    print("\n[Layer 2] Генерация уточняющих вопросов...")
    clarifications = generate_clarification_nodes()
    chunks.extend(clarifications)
    
    # Связываем слои
    print("\n[Inter-layer] Связывание сценариев с уточнениями и документами...")
    link_scenario_to_clarifications(scenarios, clarifications, metadata_chunks)

    from erp_graph_enhancements import ensure_unique_chunks, validate_graph
    chunks, dedup_report = ensure_unique_chunks(chunks)
    print(f"\n[Integrity] Уникализация ID: {json.dumps(dedup_report, ensure_ascii=False)}")

    graph = build_knowledge_graph(chunks)
    
    embedder = Embedder(vectorizer_path=TFIDF_FILE)
    vectors, node_ids = create_embeddings(chunks, embedder)

    validation = validate_graph(chunks, graph, vectors, node_ids)
    validation["deduplication"] = dedup_report
    if not validation.get("ok"):
        raise RuntimeError("Граф собран с ошибками целостности: " + "; ".join(validation.get("errors", [])))

    save_data(chunks, graph, vectors, node_ids, embedder=embedder)

    validation_path = DATA_DIR / "validation_report.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Отчёт целостности: {validation_path}")
    
    # Статистика
    print(f"\n[5/5] Итоговая статистика:")
    print(f"  Всего чанков: {len(chunks)}")
    print(f"  Узлов графа: {graph.number_of_nodes()}")
    print(f"  Ребер графа: {graph.number_of_edges()}")
    
    by_layer = {}
    for c in chunks:
        by_layer.setdefault(c.layer, 0)
        by_layer[c.layer] += 1
    for layer in sorted(by_layer):
        print(f"  Слой {layer}: {by_layer[layer]} узлов")
    
    density = nx.density(graph)
    print(f"  Плотность графа: {density:.4f}")
    
    inter_layer = sum(1 for _, _, d in graph.edges(data=True) 
                      if d.get("relation") in ("entry_doc", "clarification", "clarifies_field"))
    shared = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") == "shared_terms")
    parent_child = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") in ("parent_child", "child_parent"))
    refs = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") == "references")
    print(f"  Ребра parent-child: {parent_child}")
    print(f"  Ребра cross-references: {refs}")
    print(f"  Ребра shared-terms: {shared}")
    print(f"  Ребра межслойные: {inter_layer}")
    
    print("\n4-слойный граф знаний построен!")


def cmd_enhance(include_ui: bool = True, include_forms: bool = False,
                keep_dense: bool = False):
    """Инкрементально обогащает существующий индекс без повторного парсинга ИТС."""
    import gc
    from erp_graph_enhancements import (
        enrich_erp_metadata, ensure_unique_chunks, validate_graph,
    )
    print("=" * 60)
    print("  Инкрементальное обогащение Graph RAG (без LLM)")
    print("=" * 60)
    chunks, base_graph, vectors, node_ids = load_data(lightweight=True)
    if not chunks or base_graph is None or vectors is None or not node_ids:
        raise RuntimeError("Базовый индекс не найден. Сначала выполните build.")
    base_ids = {chunk.id for chunk in chunks}
    metadata_chunks = [
        chunk for chunk in chunks
        if chunk.layer == 3 and chunk.node_type == "metadata"
    ]
    print(f"  Базовые узлы: {len(chunks)}, L3-объекты: {len(metadata_chunks)}")
    additions, stats = enrich_erp_metadata(
        ERPCODE_DIR, metadata_chunks, DocChunk,
        include_fields=True, include_ui=include_ui,
        include_forms=include_forms, relevant_fields_only=True,
    )
    merged_chunks, dedup = ensure_unique_chunks(chunks + additions)
    print(f"  Новые L3/UI-узлы: {len(additions)}")
    print(f"  Статистика L3+: {json.dumps(stats, ensure_ascii=False)}")
    print(f"  Уникализация: {json.dumps(dedup, ensure_ascii=False)}")

    extension_chunks = [chunk for chunk in merged_chunks if chunk.layer == 3]
    extension_graph = build_knowledge_graph(extension_chunks)
    enhanced_graph = nx.compose(nx.MultiDiGraph(base_graph), extension_graph)
    del extension_graph
    gc.collect()

    print("  Конвертация плотного TF-IDF в разреженный CSR и удаление дублей ID...")
    compact_vectors, compact_ids = compact_search_index(vectors, node_ids, base_ids)
    validation = validate_graph(merged_chunks, enhanced_graph, compact_vectors, compact_ids)
    validation["enhancement_stats"] = stats
    validation["deduplication"] = dedup
    if not validation.get("ok"):
        raise RuntimeError("Ошибка целостности enhance: " + "; ".join(validation.get("errors", [])))

    save_enhanced_index(
        merged_chunks, enhanced_graph, compact_vectors, compact_ids,
        remove_dense_after_success=not keep_dense,
    )
    if not keep_dense and GRAPH_FILE.exists() and not os.environ.get("RAG_WRITE_GRAPHML"):
        GRAPH_FILE.unlink()
    validation_path = DATA_DIR / "validation_report.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(json.dumps({
        "ok": True, "chunks": len(merged_chunks),
        "graph_nodes": enhanced_graph.number_of_nodes(),
        "graph_edges": enhanced_graph.number_of_edges(),
        "vector_rows": compact_vectors.shape[0],
        "vector_nonzero": int(compact_vectors.nnz),
        "validation_report": str(validation_path),
    }, ensure_ascii=False, indent=2))


def cmd_refresh_clarifications():
    """Обновляет рабочий L2 и его связи без пересборки L3/L4 и векторов."""
    from erp_graph_enhancements import validate_graph

    chunks, graph, vectors, node_ids = load_data(lightweight=True)
    if not chunks or graph is None or vectors is None or not node_ids:
        raise RuntimeError("Базовый индекс не найден. Сначала выполните build.")

    old_l2 = [chunk for chunk in chunks if chunk.layer == 2]
    scenarios = [chunk for chunk in chunks if chunk.layer == 1]
    metadata_chunks = [
        chunk for chunk in chunks
        if chunk.layer == 3 and chunk.node_type == "metadata"
    ]
    clarifications = generate_clarification_nodes()
    link_scenario_to_clarifications(scenarios, clarifications, metadata_chunks)

    refreshed = nx.MultiDiGraph(graph)
    old_l2_ids = {
        node for node, data in refreshed.nodes(data=True)
        if int(data.get("layer", 4) or 4) == 2
    }
    refreshed.remove_nodes_from(old_l2_ids)

    def add_relation(source: str, target: str, relation: str, weight: float) -> None:
        if source not in refreshed or target not in refreshed or source == target:
            return
        for _source, _target, data in refreshed.out_edges(source, data=True):
            if _target == target and data.get("relation") == relation:
                return
        key = relation
        suffix = 2
        while refreshed.has_edge(source, target, key=key):
            key = f"{relation}:{suffix}"
            suffix += 1
        refreshed.add_edge(
            source, target, key=key, relation=relation, weight=float(weight),
            evidence="static clarification catalog", properties="{}",
        )

    for clarification in clarifications:
        refreshed.add_node(
            clarification.id, title=clarification.title, layer=2,
            node_type="clarification", level=clarification.level,
            path=clarification.path, terms="",
            content_preview=clarification.content[:500], metadata="{}",
        )
    for scenario in scenarios:
        for target in scenario.clarifications:
            add_relation(scenario.id, target, "clarification", 0.8)
        for target in scenario.entry_docs:
            add_relation(scenario.id, target, "entry_doc", 0.9)
    for clarification in clarifications:
        for target in clarification.entry_docs:
            add_relation(clarification.id, target, "clarifies_field", 0.7)

    merged_chunks = [chunk for chunk in chunks if chunk.layer != 2] + clarifications
    compact_vectors, compact_ids = compact_search_index(
        vectors, node_ids, {chunk.id for chunk in merged_chunks},
    )
    validation = validate_graph(merged_chunks, refreshed, compact_vectors, compact_ids)
    if not validation.get("ok"):
        raise RuntimeError(
            "Ошибка целостности L2: " + "; ".join(validation.get("errors", []))
        )
    save_enhanced_index(merged_chunks, refreshed, compact_vectors, compact_ids)
    validation["clarification_refresh"] = {
        "old_nodes": len(old_l2), "new_nodes": len(clarifications),
        "scenario_links": sum(len(item.clarifications) for item in scenarios),
        "field_links": sum(len(item.entry_docs) for item in clarifications),
    }
    validation_path = DATA_DIR / "validation_report.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({
        "ok": True, **validation["clarification_refresh"],
        "graph_nodes": refreshed.number_of_nodes(),
        "graph_edges": refreshed.number_of_edges(),
        "validation_report": str(validation_path),
    }, ensure_ascii=False, indent=2))


def cmd_compact_index(keep_dense: bool = False):
    """Конвертирует старый dense TF-IDF в CSR без изменения графа."""
    chunks, graph, vectors, node_ids = load_data(lightweight=True)
    if not chunks or graph is None or vectors is None or not node_ids:
        raise RuntimeError("Базовый индекс не найден")
    compact_vectors, compact_ids = compact_search_index(
        vectors, node_ids, {chunk.id for chunk in chunks},
    )
    if compact_vectors.shape[0] != len(compact_ids) or len(compact_ids) != len(set(compact_ids)):
        raise RuntimeError("Компактный индекс не прошёл проверку ID")
    save_compact_vectors_only(
        compact_vectors, compact_ids, graph,
        remove_dense_after_success=not keep_dense,
    )
    print(json.dumps({
        "ok": True, "rows": compact_vectors.shape[0],
        "columns": compact_vectors.shape[1], "nonzero": int(compact_vectors.nnz),
        "duplicates_removed": len(node_ids) - len(compact_ids),
        "saved_to": str(VECTORS_SPARSE_FILE),
        "dense_removed": not keep_dense,
    }, ensure_ascii=False, indent=2))


def cmd_query(query: str, top_k: int = 10):
    """Выполняет Graph RAG запрос"""
    chunks, graph, vectors, node_ids = load_data()
    if not chunks:
        print("Данные не найдены. Сначала выполните: python graph_rag_1c_erp.py build")
        return
    
    embedder = Embedder(vectorizer_path=TFIDF_FILE)
    rag = GraphRAG(chunks, graph, vectors, node_ids, embedder)
    
    result = rag.search(query, top_k=top_k)
    
    print(f"\n{'='*60}")
    print(f"  Запрос: {query}")
    print(f"{'='*60}\n")
    
    print("Релевантные документы (векторный поиск):")
    for i, r in enumerate(result["vector_results"][:5]):
        print(f"  [{i+1}] {r['title']} (score={r['score']:.3f})")
        print(f"       Путь: {r['path']}")
    
    if result["graph_expanded"]:
        print(f"\nСвязанные разделы (расширение графа):")
        for i, r in enumerate(result["graph_expanded"][:5]):
            print(f"  [{i+1}] {r['title']}")
            print(f"       Путь: {r['path']}")
    
    if result["workflow"]:
        print(f"\nВозможная последовательность действий:")
        for s in result["workflow"][:10]:
            print(f"  {s}")
    
    print(f"\nПолный контекст ({len(result['context'])} символов, показано первые 3000):")
    print(result["context"][:3000])
    print("... (контекст обрезан для вывода в терминал)")


def cmd_serve(host: str = "127.0.0.1", port: int = 8321):
    """Запускает API сервер для Graph RAG"""
    from fastapi import FastAPI, HTTPException, Request
    import uvicorn
    
    chunks, graph, vectors, node_ids = load_data(lightweight=True)
    if not chunks:
        print("Данные не найдены. Сначала выполните: python graph_rag_1c_erp.py build")
        return
    
    embedder = Embedder(vectorizer_path=TFIDF_FILE)
    rag = GraphRAG(chunks, graph, vectors, node_ids, embedder)
    
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="1C ERP Graph RAG API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    
    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(chunks), "graph_nodes": graph.number_of_nodes()}
    
    @app.get("/query")
    @app.post("/query")
    def query_endpoint(q: str = "", top_k: int = 10):
        if not q:
            return {"error": "Parameter 'q' is required (например: /query?q=как создать заказ поставщику)"}
        result = rag.search(q, top_k=top_k)
        prompt = rag.build_prompt(q, result=result)
        return {
            "query": q,
            "results": result["vector_results"][:5],
            "graph_expanded": result["graph_expanded"],
            "workflow": result["workflow"][:15],
            "prompt": prompt
        }
    
    @app.get("/instruct")
    @app.post("/instruct")
    def instruct_endpoint(q: str = "", provider: str = "wormsoft", model: str = "wormsoft/agent/high"):
        if not q:
            return {"error": "Parameter 'q' is required (например: /instruct?q=как создать заказ поставщику)"}
        if not llm_calls_enabled():
            raise HTTPException(
                status_code=403,
                detail="LLM отключена. Запустите сервер с RAG_ENABLE_LLM=1 только при намеренном разрешении.",
            )
        result = rag.generate_instruction(q, provider=provider, model=model)
        return result

    @app.post("/tasks/upload")
    async def upload_task(request: Request, filename: str, max_file_mb: int = 250):
        """Принимает raw binary body; не требует python-multipart и не держит файл в памяти."""
        import uuid
        from tz_pipeline import SUPPORTED_SUFFIXES, TASK_DATA_DIR, ingest_requirement_file
        safe_name = Path(filename).name
        if not safe_name or Path(safe_name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail="Недопустимое имя или формат файла")
        max_bytes = max_file_mb * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Лимит файла: {max_file_mb} МБ")
        upload_dir = TASK_DATA_DIR / "_uploads" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / safe_name
        total = 0
        try:
            with upload_path.open("wb") as handle:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(status_code=413, detail=f"Лимит файла: {max_file_mb} МБ")
                    handle.write(chunk)
            task, saved = ingest_requirement_file(upload_path, max_file_mb=max_file_mb)
            return {**task.summary(), "saved_to": str(saved), "upload_bytes": total}
        except HTTPException:
            upload_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/tasks/{task_id}/model")
    def task_model(task_id: str):
        from tz_pipeline import TASK_DATA_DIR, TaskGraphDocument
        task_path = TASK_DATA_DIR / Path(task_id).name / "task_graph.json"
        if not task_path.is_file():
            raise HTTPException(status_code=404, detail="Task Graph не найден")
        task = TaskGraphDocument.load(task_path)
        return {
            **task.summary(),
            "nodes": [{"id": node, **data} for node, data in task.graph.nodes(data=True)],
            "edges": [
                {"source": source, "target": target, "key": key, **data}
                for source, target, key, data in task.graph.edges(keys=True, data=True)
            ],
        }

    @app.get("/tasks/{task_id}/questions")
    def task_questions(task_id: str, limit: int = 10):
        from tz_pipeline import (
            TASK_DATA_DIR, GraphOnlyTzNormalizer, QuestionPlanner, TaskGraphDocument,
            apply_mapping_answers,
        )
        task_path = TASK_DATA_DIR / Path(task_id).name / "task_graph.json"
        if not task_path.is_file():
            raise HTTPException(status_code=404, detail="Task Graph не найден")
        task = TaskGraphDocument.load(task_path)
        mappings = GraphOnlyTzNormalizer().normalize(task, graph)
        answers = _load_task_answers(task_path.parent)
        mappings = apply_mapping_answers(mappings, answers, graph)
        planner = QuestionPlanner()
        answered_ids = _answer_ids(answers)
        questions = planner.plan(
            task, mappings=mappings, answered_ids=answered_ids,
            limit=max(1, min(limit, 100)),
        )
        return {
            "task_id": task_id,
            "candidate_count": len(planner.plan(
                task, mappings=mappings, answered_ids=answered_ids,
            )),
            "questions": [asdict(question) for question in questions],
        }

    @app.post("/tasks/{task_id}/answers")
    def task_answers(task_id: str, answers: Dict[str, Any]):
        from tz_pipeline import TASK_DATA_DIR
        task_dir = TASK_DATA_DIR / Path(task_id).name
        if not (task_dir / "task_graph.json").is_file():
            raise HTTPException(status_code=404, detail="Task Graph не найден")
        target = task_dir / "answers.json"
        existing = _load_task_answers(task_dir)
        existing_map = existing.get("answers", existing) if isinstance(existing, dict) else {}
        new_map = answers.get("answers", answers) if isinstance(answers, dict) else {}
        merged = {**existing_map, **new_map}
        target.write_text(
            json.dumps({"answers": merged}, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return {"task_id": task_id, "saved_to": str(target), "answers": len(merged)}

    @app.get("/tasks/{task_id}/plan")
    def task_plan(task_id: str, question_limit: int = 10):
        from erp_graph_enhancements import EndToEndProcessPlanner
        from tz_pipeline import (
            TASK_DATA_DIR, GraphOnlyTzNormalizer, QuestionPlanner, TaskGraphDocument,
            apply_mapping_answers,
        )
        task_path = TASK_DATA_DIR / Path(task_id).name / "task_graph.json"
        if not task_path.is_file():
            raise HTTPException(status_code=404, detail="Task Graph не найден")
        task = TaskGraphDocument.load(task_path)
        mappings = GraphOnlyTzNormalizer().normalize(task, graph)
        answers = _load_task_answers(task_path.parent)
        mappings = apply_mapping_answers(mappings, answers, graph)
        questions = QuestionPlanner().plan(
            task, mappings=mappings, answered_ids=_answer_ids(answers),
            limit=max(1, min(question_limit, 100)),
        )
        process = EndToEndProcessPlanner(graph).plan(task.graph, mappings)
        return {
            "task": task.summary(), "normalization": mappings,
            "process": process,
            "questions": [asdict(question) for question in questions],
            "ready_for_instruction": process.get("ready", False) and not any(question.blocking for question in questions),
        }
    
    @app.get("/graph/stats")
    def graph_stats():
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "graph_type": type(graph).__name__,
        }
    
    print(f"API сервер запущен на http://{host}:{port}")
    print(f"  GET/POST /query?q=<вопрос>  - Graph RAG запрос")
    print(f"  GET  /graph/stats           - статистика графа")
    print(f"  GET  /health                - проверка")
    print(f"  POST /tasks/upload?filename=<имя> - потоковая загрузка ТЗ (raw body)")
    print(f"  GET  /tasks/<id>/questions  - моделирующие вопросы")
    print()
    print(f"Пример в браузере: http://{host}:{port}/query?q=как создать заказ поставщику")
    uvicorn.run(app, host=host, port=port)


def save_instruction(query: str, instruction: str, sources: List[str] = None,
                     answers: str = "") -> Path:
    """Сохраняет сгенерированную инструкцию в файл instructions/<дата>_<слаг>.md."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    slug = re.sub(r'[^\w\-]+', '_', query.strip())[:60].strip('_')
    if not slug:
        slug = "instruction"
    fpath = INSTRUCTIONS_DIR / f"{ts}_{slug}.md"
    
    lines = []
    lines.append(f"# Инструкция: {query}")
    lines.append(f"\nДата: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Источник: graph_rag_1c_erp.py (Graph RAG, 4 слоя)")
    if answers:
        lines.append(f"\n## Контекст пользователя\n{answers}")
    lines.append("\n---\n")
    lines.append(instruction)
    if sources:
        lines.append("\n## Источники")
        for s in sources:
            lines.append(f"- {s}")
    lines.append("")
    
    fpath.write_text("\n".join(lines), encoding="utf-8")
    return fpath


def _build_template_instruction(query: str, answers: str, result: Dict, rag: GraphRAG) -> str:
    """Генерирует инструкцию на основе графа без LLM (шаблон)."""
    by_layer = result.get("by_layer", {})
    meta = by_layer.get("metadata", [])
    kn = by_layer.get("knowledge", [])
    scens = by_layer.get("scenarios", [])
    workflow = result.get("workflow", [])
    
    parts = []
    parts.append(f"Инструкция: {query}")
    parts.append("=" * 60)
    
    if scens:
        parts.append(f"\n[1] Бизнес-сценарий: {scens[0]['title']}")
    
    if meta:
        parts.append(f"\n[2] Какие объекты 1С задействованы:")
        for m in meta:
            parts.append(f"  • {m['title']}")
            # Достаём реквизиты из контента
            content = m.get("content", "")
            reqs = re.findall(r'- ([^(]+)\(([^)]+)\)', content)
            for req_name, req_type in reqs[:5]:
                parts.append(f"    - {req_name.strip()} ({req_type.strip()})")
    
    if workflow:
        parts.append(f"\n[3] Последовательность действий (из документации):")
        for s in workflow[:10]:
            parts.append(f"  {s}")
    
    if kn:
        parts.append(f"\n[4] Описание из документации:")
        for k in kn[:3]:
            content = k.get("content", "")[:500].strip()
            if content:
                parts.append(f"  {content}")
    
    parts.append(f"\n[5] Контекст пользователя:")
    parts.append(f"  {answers}")
    
    parts.append("\n" + "=" * 60)
    parts.append("Для более точной инструкции настройте API-ключ Wormsoft в .env (WORMSOFT_API_KEY)")
    
    return "\n".join(parts)


def cmd_clarify(query: str = ""):
    """Режим с уточняющими вопросами: пользователь -> вопросы -> ответы -> инструкция"""
    chunks, graph, vectors, node_ids = load_data()
    if not chunks:
        print("Данные не найдены. Сначала выполните: python graph_rag_1c_erp.py build")
        return
    
    embedder = Embedder(vectorizer_path=TFIDF_FILE)
    rag = GraphRAG(chunks, graph, vectors, node_ids, embedder)
    
    print("=" * 60)
    print("  1С ERP — Уточняющий режим")
    print("=" * 60)
    print("  Опишите что хотите сделать — я задам уточняющие")
    print("  вопросы и сформирую точную инструкцию.")
    print("  'exit' для выхода.")
    print()
    
    first = True
    while True:
        try:
            if first and query:
                q = query.strip()
            else:
                q = input("\nВаша задача: ").strip()
            first = False
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            
            # 1. Поиск по графу
            print("\n--- АНАЛИЗ ЗАПРОСА ---")
            result = rag.search(q)
            
            print(f"\nНайдено разделов: {len(result['vector_results'])}")
            for i, r in enumerate(result["vector_results"][:3]):
                print(f"  [{i+1}] {r['title']}")
            
            # 2. Уточняющие вопросы: пробуем LLM, если нет — берём из Layer 2 графа
            by_layer = result.get("by_layer", {})
            l2_questions = [r["title"] for r in by_layer.get("clarifications", [])]
            
            questions = None
            api_key = None
            if llm_calls_enabled():
                api_key = os.environ.get("WORMSOFT_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                provider, model = "wormsoft", "wormsoft/agent/high"
                print(f"\n--- УТОЧНЯЮЩИЕ ВОПРОСЫ (Wormsoft) ---")
                print("  (генерация...)")
                with LlmClient(provider=provider, model=model) as llm:
                    questions_text = rag.generate_clarifying_questions(q, result, llm)
                    questions = [s.strip() for s in questions_text.split('\n') 
                                 if s.strip() and not s.startswith('---') and 'Ошибка' not in s]
                    if questions and len(questions) < 2:
                        questions = None
            
            if not questions:
                # Fallback: вопросы из Layer 2 графа
                if l2_questions:
                    questions = l2_questions[:5]
                    print(f"\n--- УТОЧНЯЮЩИЕ ВОПРОСЫ (из графа, Layer 2) ---")
                else:
                    questions = [
                        "Для какой организации / предприятия настраиваете?",
                        "Какой период (год, месяц)?",
                        "Есть ли особые требования или специфика?"
                    ]
                    print(f"\n--- УТОЧНЯЮЩИЕ ВОПРОСЫ (типовые) ---")
            
            print(f"({len(questions)} вопросов)")
            print()
            
            answers = {}
            for i, question in enumerate(questions):
                ans = input(f"  Вопрос {i+1}: {question}\n  Ответ: ").strip()
                answers[f"q{i+1}"] = {"question": question, "answer": ans or "(не указано)"}
            
            # 3. Финальная инструкция
            print(f"\n--- ФОРМИРУЮ ИНСТРУКЦИЮ ---")
            answers_text = "\n".join(f"Вопрос: {v['question']}\nОтвет: {v['answer']}" for v in answers.values())
            
            instruction = None
            if api_key:
                print("  (Wormsoft, ожидайте...)")
                with LlmClient(provider="wormsoft", model="wormsoft/agent/high") as llm:
                    instruction = rag.generate_instruction_with_context(q, answers_text, result, llm)
            
            if not instruction or instruction.startswith("[Ошибка"):
                print("  (шаблон из графа)")
                instruction = _build_template_instruction(q, answers_text, result, rag)
            
            print("\n" + "=" * 60)
            print("  ИНСТРУКЦИЯ")
            print("=" * 60)
            print()
            print(instruction)
            print()
            print("=" * 60)
            print(f"  Длина: {len(instruction)} символов")
            print()
            
            sources = [f"{r['title']} ({r['path']})" for r in result["vector_results"]]
            fpath = save_instruction(q, instruction, sources, answers=answers_text)
            print(f"  Сохранено: {fpath}")
            print()
            
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Ошибка: {e}")


def cmd_interactive(query: str = ""):
    """Интерактивный режим: поиск + генерация инструкции (старый, без уточнений)"""
    chunks, graph, vectors, node_ids = load_data()
    if not chunks:
        print("Данные не найдены. Сначала выполните: python graph_rag_1c_erp.py build")
        return
    
    embedder = Embedder(vectorizer_path=TFIDF_FILE)
    rag = GraphRAG(chunks, graph, vectors, node_ids, embedder)
    
    print("=" * 60)
    print("  1С ERP Graph RAG - Интерактивный режим")
    print("=" * 60)
    print("  Введите вопрос или 'exit' для выхода")
    print()
    
    first = True
    while True:
        try:
            if first and query:
                q = query.strip()
            else:
                q = input("> ").strip()
            first = False
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            
            # 1. Поиск по графу
            print("\n--- ПОИСК ПО ГРАФУ ---")
            result = rag.search(q)
            
            print(f"\nРелевантные документы:")
            for i, r in enumerate(result["vector_results"][:3]):
                print(f"  [{i+1}] {r['title']} (score={r['score']:.3f})")
            
            if result["graph_expanded"]:
                print(f"\nСвязанные разделы:")
                for i, r in enumerate(result["graph_expanded"][:3]):
                    print(f"  [{i+1}] {r['title']}")
            
            if result["workflow"]:
                print(f"\nПоследовательность действий:")
                for s in result["workflow"][:8]:
                    print(f"  {s}")
            
            print(f"\nКонтекст для LLM ({len(result['context'])} символов):")
            print(result["context"][:2000])
            if len(result["context"]) > 2000:
                print(f"  ... (ещё {len(result['context']) - 2000} символов)")
            
            # 2. Предпросмотр промпта
            prompt = rag._build_compact_prompt(q, result)
            print(f"\n--- ПРОМПТ ДЛЯ LLM ({len(prompt)} символов) ---")
            print(prompt)
            
            # 3. Запрос на генерацию
            ans = input("\nСгенерировать инструкцию? [Д/Н]: ").strip().lower()
            if ans not in ("д", "y", "yes", "да", ""):
                print()
                continue
            
            provider = "ollama"
            model = "qwen2.5:7b"
            
            print(f"\n--- ГЕНЕРАЦИЯ ИНСТРУКЦИИ ({provider}/{model}) ---")
            print("  (ожидайте 30-60 секунд)")
            
            llm = LlmClient(provider=provider, model=model)
            try:
                instruction = llm.prompt(prompt)
            except Exception as e:
                instruction = f"[Ошибка] {e}"
            finally:
                llm.close()
            
            print("\n" + "=" * 60)
            print("  ИНСТРУКЦИЯ")
            print("=" * 60)
            print()
            print(instruction)
            print()
            print("=" * 60)
            print(f"  Длина: {len(instruction)} символов")
            print()
            
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Ошибка: {e}")


def cmd_instruct(query: str, provider: str = "wormsoft", model: str = "wormsoft/agent/high"):
    """Генерирует пошаговую инструкцию через LLM на основе Graph RAG"""
    chunks, graph, vectors, node_ids = load_data()
    if not chunks:
        print("Данные не найдены. Сначала выполните: python graph_rag_1c_erp.py build")
        return
    
    embedder = Embedder(vectorizer_path=TFIDF_FILE)
    rag = GraphRAG(chunks, graph, vectors, node_ids, embedder)
    
    print(f"Генерация инструкции для запроса: {query}")
    print(f"Провайдер: {provider}, Модель: {model}")
    print()
    
    result = rag.generate_instruction(query, provider=provider, model=model)
    
    print("=" * 60)
    print(f"  ИНСТРУКЦИЯ ПО НАСТРОЙКЕ")
    print("=" * 60)
    print()
    print(result["instruction"])
    print()
    print("=" * 60)
    print(f"\nИсточники:")
    for r in result["vector_results"]:
        print(f"  - {r['title']} ({r['path']})")
    print(f"\nДлина инструкции: {len(result['instruction'])} символов")
    
    sources = [f"{r['title']} ({r['path']})" for r in result["vector_results"]]
    fpath = save_instruction(query, result["instruction"], sources)
    print(f"\n💾 Сохранено: {fpath}")


def cmd_ingest_tz(file_path: str, task_id: str = "", max_file_mb: int = 250):
    """Локально разбирает ТЗ и сохраняет персональный Task Graph."""
    from tz_pipeline import ingest_requirement_file
    task, saved = ingest_requirement_file(
        Path(file_path), task_id=task_id or None, max_file_mb=max_file_mb,
    )
    print(json.dumps({**task.summary(), "saved_to": str(saved)}, ensure_ascii=False, indent=2))


def _resolve_task_graph_path(task: str) -> Path:
    from tz_pipeline import TASK_DATA_DIR
    candidate = Path(task)
    if candidate.is_file():
        return candidate
    return TASK_DATA_DIR / task / "task_graph.json"


def _load_task_answers(task_dir: Path) -> Dict[str, Any]:
    path = task_dir / "answers.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _answer_ids(answers: Dict[str, Any]) -> set[str]:
    payload = answers.get("answers", answers) if isinstance(answers, dict) else {}
    return set(payload) if isinstance(payload, dict) else set()


def cmd_normalize_tz(task: str, output: str = ""):
    """Переводит Task Graph на язык ERP детерминированно, без LLM."""
    from tz_pipeline import GraphOnlyTzNormalizer, TaskGraphDocument, apply_mapping_answers
    task_path = _resolve_task_graph_path(task)
    task_graph = TaskGraphDocument.load(task_path)
    _, graph, _, _ = load_data(lightweight=True)
    if graph is None:
        raise RuntimeError("Статический граф не найден. Выполните build.")
    payload = GraphOnlyTzNormalizer().normalize(task_graph, graph)
    payload = apply_mapping_answers(payload, _load_task_answers(task_path.parent), graph)
    output_path = Path(output) if output else task_path.parent / "erp_mapping.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "task_id": task_graph.task_id,
        "mappings": len(payload.get("mappings", [])),
        "mapping_gaps": len(payload.get("mapping_gaps", [])),
        "saved_to": str(output_path),
    }, ensure_ascii=False, indent=2))


def cmd_task_questions(task: str, limit: int = 10, all_questions: bool = False,
                       materialize: bool = False):
    from tz_pipeline import (
        GraphOnlyTzNormalizer, QuestionPlanner, TaskGraphDocument, apply_mapping_answers,
    )
    task_path = _resolve_task_graph_path(task)
    task_graph = TaskGraphDocument.load(task_path)
    _, graph, _, _ = load_data(lightweight=True)
    if graph is None:
        raise RuntimeError("Статический граф не найден. Выполните build.")
    mappings = GraphOnlyTzNormalizer().normalize(task_graph, graph)
    answers = _load_task_answers(task_path.parent)
    mappings = apply_mapping_answers(mappings, answers, graph)
    planner = QuestionPlanner()
    questions = planner.plan(
        task_graph, mappings=mappings, answered_ids=_answer_ids(answers),
        limit=None if all_questions else limit,
    )
    if materialize:
        planner.materialize(task_graph, questions)
        task_graph.save(task_path.parent.parent)
    print(json.dumps({
        "task_id": task_graph.task_id,
        "candidate_count": len(planner.plan(
            task_graph, mappings=mappings, answered_ids=_answer_ids(answers),
        )),
        "returned": len(questions),
        "questions": [asdict(question) for question in questions],
    }, ensure_ascii=False, indent=2))


def cmd_answer_task(task: str, question_id: str, answer: str):
    """Сохраняет ответ и делает следующий раунд вопросов адаптивным."""
    task_path = _resolve_task_graph_path(task)
    if not task_path.is_file():
        raise FileNotFoundError(f"Task Graph не найден: {task_path}")
    current = _load_task_answers(task_path.parent)
    answer_map = current.get("answers", current) if isinstance(current, dict) else {}
    if not isinstance(answer_map, dict):
        answer_map = {}
    answer_map[str(question_id)] = answer
    target = task_path.parent / "answers.json"
    target.write_text(
        json.dumps({"answers": answer_map}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "task_id": task, "question_id": question_id,
        "answers_saved": len(answer_map), "saved_to": str(target),
    }, ensure_ascii=False, indent=2))


def cmd_plan_task(task: str, output: str = "", dependency_depth: int = 3):
    """Строит сквозной офлайн-план: ERP-объекты, документы, регистры, НСИ и вопросы."""
    from erp_graph_enhancements import EndToEndProcessPlanner, render_offline_instruction
    from tz_pipeline import (
        GraphOnlyTzNormalizer, QuestionPlanner, TaskGraphDocument, apply_mapping_answers,
    )
    task_path = _resolve_task_graph_path(task)
    task_graph = TaskGraphDocument.load(task_path)
    _, graph, _, _ = load_data(lightweight=True)
    if graph is None:
        raise RuntimeError("Статический граф не найден. Выполните build.")
    normalization = GraphOnlyTzNormalizer().normalize(task_graph, graph)
    answers = _load_task_answers(task_path.parent)
    normalization = apply_mapping_answers(normalization, answers, graph)
    questions = QuestionPlanner().plan(
        task_graph, mappings=normalization, answered_ids=_answer_ids(answers),
    )
    process = EndToEndProcessPlanner(graph).plan(
        task_graph.graph, normalization, dependency_depth=dependency_depth,
    )
    payload = {
        "task": task_graph.summary(), "normalization": normalization,
        "process": process, "questions": [asdict(question) for question in questions],
        "ready_for_instruction": process.get("ready", False) and not any(q.blocking for q in questions),
    }
    output_path = Path(output) if output else task_path.parent / "process_plan.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    instruction_path = output_path.parent / "instruction_draft.md"
    instruction_path.write_text(
        render_offline_instruction(task_graph.summary(), process, questions), encoding="utf-8",
    )
    print(json.dumps({
        "task_id": task_graph.task_id,
        "erp_objects": len(process.get("erp_objects", [])),
        "documents": len(process.get("documents_in_requirement_order", [])),
        "chain_gaps": len(process.get("chain_gaps", [])),
        "questions": len(questions), "saved_to": str(output_path),
        "instruction_draft": str(instruction_path),
    }, ensure_ascii=False, indent=2))


def cmd_dependencies(target: str, max_depth: int = 5, include_optional: bool = False):
    from erp_graph_enhancements import DependencyPlanner
    _, graph, _, _ = load_data(lightweight=True)
    if graph is None:
        raise RuntimeError("Граф не найден. Выполните build.")
    result = DependencyPlanner(graph).plan(
        target, max_depth=max_depth, include_optional=include_optional,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def cmd_document_chain(start: str, end: str = "", max_depth: int = 10):
    from erp_graph_enhancements import DocumentChainPlanner
    _, graph, _, _ = load_data(lightweight=True)
    if graph is None:
        raise RuntimeError("Граф не найден. Выполните build.")
    result = DocumentChainPlanner(graph).plan(start, end or None, max_depth=max_depth)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def cmd_validate():
    from erp_graph_enhancements import validate_graph
    chunks, graph, vectors, node_ids = load_data()
    if not chunks or graph is None:
        raise RuntimeError("Граф не найден. Выполните build.")
    report = validate_graph(chunks, graph, vectors, node_ids)
    target = DATA_DIR / "validation_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "saved_to": str(target)}, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1C ERP ITS Graph RAG")
    sub = parser.add_subparsers(dest="command", help="Команда")
    
    p_build = sub.add_parser("build", help="Построить граф знаний")
    p_build.add_argument("--no-fields", action="store_true", help="Не создавать отдельные узлы реквизитов L3")
    p_build.add_argument("--no-ui", action="store_true", help="Не разбирать подсистемы и CommandInterface")
    p_build.add_argument("--with-forms", action="store_true",
                         help="Дополнительно разобрать поля форм (существенно увеличивает граф)")

    p_enhance = sub.add_parser("enhance", help="Обогатить существующий индекс типизированными L3/UI-связями")
    p_enhance.add_argument("--no-ui", action="store_true", help="Не разбирать подсистемы и CommandInterface")
    p_enhance.add_argument("--with-forms", action="store_true", help="Разобрать поля форм (ресурсоёмко)")
    p_enhance.add_argument("--keep-dense", action="store_true", help="Не удалять старый vectors.npy после проверки CSR")

    p_compact = sub.add_parser("compact-index", help="Преобразовать старый dense TF-IDF в разреженный CSR")
    p_compact.add_argument("--keep-dense", action="store_true")

    sub.add_parser(
        "refresh-clarifications",
        help="Обновить расширенный каталог L2-вопросов без пересборки корпуса",
    )
    
    p_query = sub.add_parser("query", help="Выполнить запрос")
    p_query.add_argument("query", type=str, help="Текст запроса")
    p_query.add_argument("--top-k", type=int, default=10, help="Количество результатов")
    
    p_serve = sub.add_parser("serve", help="Запустить API сервер")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8321)
    
    p_interactive = sub.add_parser("interactive", help="Интерактивный режим (поиск + генерация)")
    p_interactive.add_argument("--mode", type=str, default="clarify",
                               choices=["clarify", "direct"],
                               help="clarify — с уточняющими вопросами (по умолчанию), direct — прямой ответ")
    p_interactive.add_argument("--file", type=str, default="",
                               help="Путь к файлу с задачей (ТЗ). Содержимое файла станет первым запросом")
    p_interactive.add_argument("--allow-llm", action="store_true",
                               help="Явно разрешить LLM-вызовы в интерактивном режиме")
    p_interactive.add_argument("query", type=str, nargs="*",
                               help="Задача (можно без кавычек — все слова объединяются). Пусто — ввод с клавиатуры")
    
    p_instruct = sub.add_parser("instruct", help="Сгенерировать инструкцию через LLM")
    p_instruct.add_argument("query", type=str, help="Текст запроса")
    p_instruct.add_argument("--provider", type=str, default="wormsoft", help="Провайдер LLM (wormsoft/deepseek/ollama)")
    p_instruct.add_argument("--model", type=str, default="wormsoft/agent/high", help="Модель LLM")
    p_instruct.add_argument("--allow-llm", action="store_true",
                            help="Явно разрешить обращение к выбранной LLM")

    p_ingest = sub.add_parser("ingest-tz", help="Локально разобрать файл ТЗ в Task Graph")
    p_ingest.add_argument("file", type=str, help="DOCX/TXT/MD/XLSX/PDF с ТЗ")
    p_ingest.add_argument("--task-id", type=str, default="")
    p_ingest.add_argument("--max-file-mb", type=int, default=250)

    p_normalize = sub.add_parser("normalize-tz", help="Офлайн-перевод Task Graph на язык ERP")
    p_normalize.add_argument("task", type=str, help="Task ID или путь к task_graph.json")
    p_normalize.add_argument("--output", type=str, default="")

    p_questions = sub.add_parser("task-questions", help="Адаптивные моделирующие вопросы по ТЗ")
    p_questions.add_argument("task", type=str, help="Task ID или путь к task_graph.json")
    p_questions.add_argument("--limit", type=int, default=10, help="Размер одного раунда")
    p_questions.add_argument("--all", action="store_true", help="Вернуть все релевантные вопросы")
    p_questions.add_argument("--materialize", action="store_true", help="Сохранить вопросы узлами Task Graph")

    p_answer = sub.add_parser("answer-task", help="Сохранить ответ на моделирующий вопрос")
    p_answer.add_argument("task", type=str, help="Task ID или путь к task_graph.json")
    p_answer.add_argument("question_id", type=str)
    p_answer.add_argument("answer", type=str, help="Текст ответа или точный ERP node ID")

    p_plan_task = sub.add_parser("plan-task", help="Сквозной офлайн-план процесса по Task Graph")
    p_plan_task.add_argument("task", type=str, help="Task ID или путь к task_graph.json")
    p_plan_task.add_argument("--output", type=str, default="")
    p_plan_task.add_argument("--dependency-depth", type=int, default=3)

    p_dependencies = sub.add_parser("dependencies", help="Обратная цепочка зависимостей объекта ERP")
    p_dependencies.add_argument("target", type=str, help="ID или название объекта")
    p_dependencies.add_argument("--max-depth", type=int, default=5)
    p_dependencies.add_argument("--include-optional", action="store_true")

    p_chain = sub.add_parser("document-chain", help="Цепочка документов и регистров")
    p_chain.add_argument("start", type=str, help="Начальный документ")
    p_chain.add_argument("end", type=str, nargs="?", default="", help="Конечный документ")
    p_chain.add_argument("--max-depth", type=int, default=10)

    sub.add_parser("validate", help="Проверить целостность графа и поискового индекса")
    
    args = parser.parse_args()
    
    if args.command == "build":
        cmd_build(
            include_fields=not args.no_fields,
            include_ui=not args.no_ui,
            include_forms=args.with_forms,
        )
    elif args.command == "enhance":
        cmd_enhance(
            include_ui=not args.no_ui,
            include_forms=args.with_forms,
            keep_dense=args.keep_dense,
        )
    elif args.command == "compact-index":
        cmd_compact_index(keep_dense=args.keep_dense)
    elif args.command == "refresh-clarifications":
        cmd_refresh_clarifications()
    elif args.command == "query":
        cmd_query(args.query, args.top_k)
    elif args.command == "serve":
        cmd_serve(args.host, args.port)
    elif args.command == "interactive":
        query = " ".join(args.query) if args.query else ""
        if args.file:
            try:
                from tz_pipeline import RequirementFileLoader
                query = RequirementFileLoader().load(args.file).text.strip()
            except Exception as e:
                print(f"Ошибка чтения файла {args.file}: {e}")
                sys.exit(2)
        if args.allow_llm:
            os.environ["RAG_ENABLE_LLM"] = "1"
        if args.mode == "clarify":
            cmd_clarify(query)
        else:
            cmd_interactive(query)
    elif args.command == "instruct":
        if args.allow_llm:
            os.environ["RAG_ENABLE_LLM"] = "1"
        cmd_instruct(args.query, args.provider, args.model)
    elif args.command == "ingest-tz":
        cmd_ingest_tz(args.file, args.task_id, args.max_file_mb)
    elif args.command == "normalize-tz":
        cmd_normalize_tz(args.task, args.output)
    elif args.command == "task-questions":
        cmd_task_questions(args.task, args.limit, args.all, args.materialize)
    elif args.command == "answer-task":
        cmd_answer_task(args.task, args.question_id, args.answer)
    elif args.command == "plan-task":
        cmd_plan_task(args.task, args.output, args.dependency_depth)
    elif args.command == "dependencies":
        cmd_dependencies(args.target, args.max_depth, args.include_optional)
    elif args.command == "document-chain":
        cmd_document_chain(args.start, args.end, args.max_depth)
    elif args.command == "validate":
        cmd_validate()
    else:
        parser.print_help()
