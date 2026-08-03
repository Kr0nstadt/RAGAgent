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
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import pickle

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
ITS_ROOT = Path(__file__).parent / "001--1С-ERP Управление предприятием 2, редакция 2.5"
DATA_DIR = Path(__file__).parent / "graph_rag_data"
GRAPH_FILE = DATA_DIR / "knowledge_graph.graphml"
CHUNKS_FILE = DATA_DIR / "chunks.json"
VECTORS_FILE = DATA_DIR / "vectors.npy"
NODES_FILE = DATA_DIR / "nodes.json"
TFIDF_FILE = DATA_DIR / "tfidf_vectorizer.pkl"
GRAPH_PICKLE = DATA_DIR / "knowledge_graph.pkl"
CHUNKS_META_FILE = DATA_DIR / "chunks_meta.json"
INSTRUCTIONS_DIR = Path(__file__).parent / "instructions"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)

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
                sub_id = f"scenario_{sub.stem}"
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
    ("Организация", "Для какой организации настраиваете процесс?", "Спр.Организации"),
    ("Склад", "Какой склад используется? Производственный, оптовый, розничный?", "Спр.Склады"),
    ("Подразделение", "Какое подразделение-исполнитель?", "Спр.СтруктураПредприятия"),
    ("Номенклатура", "Какой вид номенклатуры: сырьё, материал, продукция?", "Спр.Номенклатура"),
    ("Контрагент", "Какой контрагент / партнёр участвует?", "Спр.Контрагенты"),
    ("Соглашение", "По какому соглашению / договору работаем?", "Спр.СоглашенияСКлиентами"),
    ("Вид цены", "Какой вид цены применяется?", "Спр.ВидыЦен"),
    ("Валюта", "В какой валюте ведётся учёт?", "Спр.Валюты"),
    ("Период", "За какой период (месяц, квартал, год)?", None),
    ("СтатусЗаказа", "Какой статус заказа: Формируется, К производству, Закрыт?", "Переч.СтатусыЗаказовНаПроизводство"),
    ("ВидОбеспечения", "Какой способ обеспечения: производство, закупка, перемещение?", "Переч.ВариантыОбеспечения"),
    ("Партия", "Нужно ли обособленное обеспечение (учёт по назначениям)?", None),
]

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
        if sc.level > 0:
            continue
        # Все типовые уточнения применимы к любому сценарию
        sc.clarifications = [c.id for c in clarifications[:8]]
        
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
def build_knowledge_graph(chunks: List[DocChunk]) -> nx.DiGraph:
    """Строит 4-слойный граф знаний."""
    G = nx.DiGraph()
    chunk_map = {c.id: c for c in chunks}
    
    print("  Добавление узлов с атрибутами слоёв...")
    for c in chunks:
        G.add_node(c.id,
                   title=c.title,
                   layer=c.layer,
                   node_type=c.node_type,
                   level=c.level,
                   path=c.path,
                   terms=",".join(c.terms[:20]),
                   content_preview=c.content[:200])
    
    print("  Ребра parent-child...")
    for c in chunks:
        if c.parent_id and c.parent_id in chunk_map:
            G.add_edge(c.parent_id, c.id, relation="parent_child", weight=1.0)
            G.add_edge(c.id, c.parent_id, relation="child_parent", weight=0.8)
    
    print("  Ребра между слоями (inter-layer)...")
    layer_edges = 0
    for c in chunks:
        # L1 → L3: entry_docs (сценарий → документы)
        if c.layer == 1 and c.entry_docs:
            for target_id in c.entry_docs:
                if target_id in chunk_map and not G.has_edge(c.id, target_id):
                    G.add_edge(c.id, target_id, relation="entry_doc", weight=0.9)
                    layer_edges += 1
        # L1 → L2: clarifications (сценарий → уточняющие вопросы)
        if c.layer == 1 and c.clarifications:
            for target_id in c.clarifications:
                if target_id in chunk_map and not G.has_edge(c.id, target_id):
                    G.add_edge(c.id, target_id, relation="clarification", weight=0.8)
                    layer_edges += 1
        # L2 → L3: entry_docs на clarification-узлах (уточнение → реквизит)
        if c.layer == 2 and c.entry_docs:
            for target_id in c.entry_docs:
                if target_id in chunk_map and not G.has_edge(c.id, target_id):
                    G.add_edge(c.id, target_id, relation="clarifies_field", weight=0.7)
                    layer_edges += 1
        # L3 → L4: metadata → knowledge (через shared-terms / refs)
    
    print(f"  Добавлено {layer_edges} межслойных ребер")
    
    print("  Ребра cross-references...")
    ref_count = 0
    for c in chunks:
        for target_id in c.refs:
            if target_id in chunk_map and target_id != c.id and not G.has_edge(c.id, target_id):
                G.add_edge(c.id, target_id, relation="references", weight=0.7)
                ref_count += 1
                if ref_count > 5000:
                    break
        if ref_count > 5000:
            break
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
    
    for term, cids in term_index.items():
        if len(cids) < 2:
            continue
        for i in range(len(cids)):
            for j in range(i+1, len(cids)):
                c1, c2 = cids[i], cids[j]
                if not G.has_edge(c1, c2):
                    score = 1.0 / max(chunk_term_count.get(c1, 1), chunk_term_count.get(c2, 1))
                    if score > 0.05:
                        G.add_edge(c1, c2, relation="shared_terms", weight=round(score, 3))
                        G.add_edge(c2, c1, relation="shared_terms", weight=round(score, 3))
                        shared_edges += 1
                        if shared_edges > 10000:
                            break
            if shared_edges > 10000:
                break
        if shared_edges > 10000:
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
    
    def encode(self, texts: List[str], fit: bool = False) -> np.ndarray:
        if self._model is None:
            if not self._try_load_sentence():
                return self._encode_tfidf(texts, fit=fit)
            else:
                return self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        else:
            return self._model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
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
            vectors = self._vectorizer.fit_transform(texts).toarray()
            # Сохраняем vectorizer
            if self.vectorizer_path:
                import pickle
                with open(self.vectorizer_path, "wb") as f:
                    pickle.dump(self._vectorizer, f)
                print(f"  TF-IDF vectorizer сохранен в {self.vectorizer_path}")
        else:
            vectors = self._vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
    
    def save_vectorizer(self):
        if self._vectorizer and self.vectorizer_path:
            import pickle
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self._vectorizer, f)


def create_embeddings(chunks: List[DocChunk], embedder: Embedder) -> Tuple[np.ndarray, List[str]]:
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
def save_data(chunks: List[DocChunk], graph: nx.DiGraph, vectors: np.ndarray, node_ids: List[str]):
    """Сохраняет все данные на диск"""
    print(f"\n[4/5] Сохранение данных...")
    
    # Чанки (полные)
    chunks_data = []
    for c in chunks:
        d = asdict(c)
        chunks_data.append(d)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=1)
    
    # Чанки (только метаданные для быстрой загрузки)
    chunks_meta = {c.id: {"title": c.title, "path": c.path} for c in chunks}
    with open(CHUNKS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False)
    
    # Node IDs
    with open(NODES_FILE, "w", encoding="utf-8") as f:
        json.dump(node_ids, f)
    
    # Векторы (float32 для быстрой загрузки)
    np.save(VECTORS_FILE, vectors.astype(np.float32))
    
    # Граф (pickle — быстрее graphml)
    with open(GRAPH_PICKLE, "wb") as f:
        pickle.dump(graph, f)
    nx.write_graphml(graph, GRAPH_FILE)
    
    print(f"  Чанки: {CHUNKS_FILE}")
    print(f"  Векторы: {VECTORS_FILE}")
    print(f"  Граф: {GRAPH_PICKLE}")


def load_data(lightweight=False) -> Tuple[List[DocChunk], Optional[nx.DiGraph], Optional[np.ndarray], Optional[List[str]]]:
    """Загружает данные с диска.
    
    lightweight=True: не загружает полный текст чанков и граф (быстрый поиск).
    """
    if not CHUNKS_FILE.exists():
        return [], None, None, None
    
    if lightweight:
        # Быстрая загрузка: только метаданные + векторы
        chunks_meta = {}
        if CHUNKS_META_FILE.exists():
            with open(CHUNKS_META_FILE, "r", encoding="utf-8") as f:
                chunks_meta = json.load(f)
        else:
            # Fallback: загружаем полные чанки, но берём только метаданные
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            chunks_meta = {d["id"]: {"title": d["title"], "path": d["path"]} for d in chunks_data}
        
        chunks = [DocChunk(id=k, title=v["title"], path=v["path"], content="")
                  for k, v in chunks_meta.items()]
        
        vectors = None
        node_ids = None
        if VECTORS_FILE.exists() and NODES_FILE.exists():
            vectors = np.load(VECTORS_FILE)
            with open(NODES_FILE, "r") as f:
                node_ids = json.load(f)
        
        print(f"Загружено (быстро): {len(chunks)} чанков (метаданные), векторы {vectors.shape if vectors is not None else '—'}")
        return chunks, None, vectors, node_ids
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    chunks = [DocChunk(**d) for d in chunks_data]
    
    graph = None
    if GRAPH_PICKLE.exists():
        with open(GRAPH_PICKLE, "rb") as f:
            graph = pickle.load(f)
    elif GRAPH_FILE.exists():
        graph = nx.read_graphml(GRAPH_FILE)
    
    vectors = None
    node_ids = None
    if VECTORS_FILE.exists() and NODES_FILE.exists():
        vectors = np.load(VECTORS_FILE)
        with open(NODES_FILE, "r") as f:
            node_ids = json.load(f)
    
    print(f"Загружено: {len(chunks)} чанков, {graph.number_of_nodes() if graph else 0} узлов графа")
    return chunks, graph, vectors, node_ids

# ---------------------------------------------------------------------------
# 5. Graph RAG запрос
# ---------------------------------------------------------------------------
class GraphRAG:
    """Graph RAG движок: объединяет поиск по графу и семантический поиск"""
    
    def __init__(self, chunks: List[DocChunk], graph: nx.DiGraph,
                 vectors: np.ndarray, node_ids: List[str],
                 embedder: Embedder):
        self.chunks = chunks
        self.chunk_map = {c.id: c for c in chunks}
        self.graph = graph
        self.vectors = vectors
        self.node_ids = node_ids
        self.node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        self.embedder = embedder
    
    def search(self, query: str, top_k: int = None, graph_expand: int = 10) -> Dict:
        """
        4-слойный Graph RAG поиск:
        1. Векторный поиск по query → находит сценарии (L1), метаданные (L3), знания (L4)
        2. Расширение по графу через межслойные ребра (entry_doc, clarification, parent_child, references)
        3. Возвращает результаты, сгруппированные по слоям
        """
        if top_k is None:
            word_count = len(query.split())
            top_k = max(5, min(20, word_count * 3))
        
        query_vec = self.embedder.encode([query])[0]
        sims = cosine_similarity([query_vec], self.vectors)[0]
        
        top_indices = np.argsort(sims)[::-1]
        top_indices = [i for i in top_indices if sims[i] > 0.03][:top_k]
        
        # Собираем результаты векторного поиска
        vector_results = []
        for idx in top_indices:
            node_id = self.node_ids[idx]
            chunk = self.chunk_map.get(node_id)
            if chunk:
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
        graph_expanded_nodes = {}
        for vr in vector_results:
            nid = vr["node_id"]
            if self.graph is not None and nid in self.graph:
                for pred in self.graph.predecessors(nid):
                    if pred not in graph_expanded_nodes:
                        graph_expanded_nodes[pred] = 0.6
                for succ in self.graph.successors(nid):
                    edge_data = self.graph.get_edge_data(nid, succ)
                    if edge_data:
                        rel = edge_data.get("relation", "")
                        # Приоритет: межслойные связи > parent_child > references > shared_terms
                        if rel == "entry_doc":
                            graph_expanded_nodes[succ] = 0.95
                        elif rel == "clarification":
                            graph_expanded_nodes[succ] = 0.9
                        elif rel == "clarifies_field":
                            graph_expanded_nodes[succ] = 0.85
                        elif rel == "parent_child":
                            graph_expanded_nodes[succ] = 0.7
                        elif rel == "references":
                            graph_expanded_nodes[succ] = 0.5
                        elif rel == "shared_terms":
                            graph_expanded_nodes[succ] = 0.4
        
        graph_results = []
        for nid, boost in graph_expanded_nodes.items():
            if nid not in self.node_to_idx:
                continue
            if any(vr["node_id"] == nid for vr in vector_results):
                continue
            chunk = self.chunk_map.get(nid)
            if chunk:
                idx = self.node_to_idx[nid]
                score = float(sims[idx]) if idx < len(sims) else boost
                graph_results.append({
                    "node_id": nid,
                    "title": chunk.title,
                    "path": chunk.path,
                    "score": score * boost,
                    "content": chunk.content,
                    "source": "graph",
                    "layer": chunk.layer,
                    "node_type": chunk.node_type,
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
def cmd_build():
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
    
    graph = build_knowledge_graph(chunks)
    
    embedder = Embedder(vectorizer_path=TFIDF_FILE)
    vectors, node_ids = create_embeddings(chunks, embedder)
    embedder.save_vectorizer()
    
    save_data(chunks, graph, vectors, node_ids)
    
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
    from fastapi import FastAPI
    import uvicorn
    
    chunks, graph, vectors, node_ids = load_data()
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
        result = rag.generate_instruction(q, provider=provider, model=model)
        return result
    
    @app.get("/graph/stats")
    def graph_stats():
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph)
        }
    
    print(f"API сервер запущен на http://{host}:{port}")
    print(f"  GET/POST /query?q=<вопрос>  - Graph RAG запрос")
    print(f"  GET  /graph/stats           - статистика графа")
    print(f"  GET  /health                - проверка")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1C ERP ITS Graph RAG")
    sub = parser.add_subparsers(dest="command", help="Команда")
    
    p_build = sub.add_parser("build", help="Построить граф знаний")
    
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
    p_interactive.add_argument("query", type=str, nargs="*",
                               help="Задача (можно без кавычек — все слова объединяются). Пусто — ввод с клавиатуры")
    
    p_instruct = sub.add_parser("instruct", help="Сгенерировать инструкцию через LLM")
    p_instruct.add_argument("query", type=str, help="Текст запроса")
    p_instruct.add_argument("--provider", type=str, default="wormsoft", help="Провайдер LLM (wormsoft/deepseek/ollama)")
    p_instruct.add_argument("--model", type=str, default="wormsoft/agent/high", help="Модель LLM")
    
    args = parser.parse_args()
    
    if args.command == "build":
        cmd_build()
    elif args.command == "query":
        cmd_query(args.query, args.top_k)
    elif args.command == "serve":
        cmd_serve(args.host, args.port)
    elif args.command == "interactive":
        query = " ".join(args.query) if args.query else ""
        if args.mode == "clarify":
            cmd_clarify(query)
        else:
            cmd_interactive(query)
    elif args.command == "instruct":
        cmd_instruct(args.query, args.provider, args.model)
    else:
        parser.print_help()
