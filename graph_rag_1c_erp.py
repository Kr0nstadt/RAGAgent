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
import sys
# Поддержка UTF-8 для Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
ITS_ROOT = Path(r"C:\Users\Y.Karpova\Desktop\RAW")
DATA_DIR = Path(__file__).parent / "graph_rag_data"
GRAPH_FILE = DATA_DIR / "knowledge_graph.graphml"
CHUNKS_FILE = DATA_DIR / "chunks.json"
VECTORS_FILE = DATA_DIR / "vectors.npy"
NODES_FILE = DATA_DIR / "nodes.json"
TFIDF_FILE = DATA_DIR / "tfidf_vectorizer.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(DATA_DIR, exist_ok=True)

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
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)  # ссылки на другие документы
    terms: List[str] = field(default_factory=list)
    level: int = 0

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
# 2. Построение графа знаний
# ---------------------------------------------------------------------------
def build_knowledge_graph(chunks: List[DocChunk]) -> nx.DiGraph:
    """Строит направленный граф знаний из чанков документации"""
    G = nx.DiGraph()
    chunk_map = {c.id: c for c in chunks}
    
    print("  Добавление узлов...")
    for c in chunks:
        G.add_node(c.id,
                   title=c.title,
                   level=c.level,
                   path=c.path,
                   terms=",".join(c.terms[:20]),
                   content_preview=c.content[:200])
    
    print("  Ребра parent-child...")
    for c in chunks:
        if c.parent_id and c.parent_id in chunk_map:
            G.add_edge(c.parent_id, c.id, relation="parent_child", weight=1.0)
            G.add_edge(c.id, c.parent_id, relation="child_parent", weight=0.8)
    
    print("  Ребра cross-references...")
    # Теперь refs уже разрешены через manifest.json -> chunk_id
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
    # Инвертированный индекс: термин -> список chunk_id
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
        # Для каждого термина соединяем все чанки, где он встречается
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
    
    print(f"\n[2/5] Граф построен: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер")
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
    
    # Чанки
    chunks_data = []
    for c in chunks:
        d = asdict(c)
        chunks_data.append(d)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=1)
    
    # Node IDs
    with open(NODES_FILE, "w", encoding="utf-8") as f:
        json.dump(node_ids, f)
    
    # Векторы
    np.save(VECTORS_FILE, vectors)
    
    # Граф
    nx.write_graphml(graph, GRAPH_FILE)
    
    print(f"  Чанки: {CHUNKS_FILE}")
    print(f"  Векторы: {VECTORS_FILE}")
    print(f"  Граф: {GRAPH_FILE}")


def load_data() -> Tuple[List[DocChunk], Optional[nx.DiGraph], Optional[np.ndarray], Optional[List[str]]]:
    """Загружает данные с диска"""
    if not CHUNKS_FILE.exists():
        return [], None, None, None
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    chunks = [DocChunk(**d) for d in chunks_data]
    
    graph = None
    if GRAPH_FILE.exists():
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
        Graph RAG поиск:
        1. Векторный поиск по query → релевантные узлы
        2. Расширение по графу: соседи релевантных узлов
        3. Сбор контекста с полным текстом
        """
        # Авто-подбор top_k в зависимости от вопроса
        if top_k is None:
            word_count = len(query.split())
            top_k = max(5, min(20, word_count * 3))
        
        query_vec = self.embedder.encode([query])[0]
        
        sims = cosine_similarity([query_vec], self.vectors)[0]
        
        top_indices = np.argsort(sims)[::-1]
        # Отсекаем по порогу и берем top_k
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
                    "source": "vector"
                })
        
        # Расширение по графу: предки, потомки, shared-terms
        graph_expanded_nodes = {}
        for vr in vector_results:
            nid = vr["node_id"]
            if nid in self.graph:
                for pred in self.graph.predecessors(nid):
                    if pred not in graph_expanded_nodes:
                        graph_expanded_nodes[pred] = 0.6
                for succ in self.graph.successors(nid):
                    edge_data = self.graph.get_edge_data(nid, succ)
                    if edge_data:
                        if edge_data.get("relation") == "parent_child":
                            graph_expanded_nodes[succ] = 0.7
                        elif edge_data.get("relation") == "references":
                            graph_expanded_nodes[succ] = 0.5
                        elif edge_data.get("relation") == "shared_terms":
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
                    "source": "graph"
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
        
        return {
            "query": query,
            "vector_results": vector_results,
            "graph_expanded": graph_results,
            "context": context,
            "workflow": workflow,
            "all_nodes": list(set(r["node_id"] for r in all_results))
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
    
    def build_prompt(self, query: str) -> str:
        """Строит промпт для LLM на основе Graph RAG контекста"""
        result = self.search(query)
        
        workflow_text = ""
        if result["workflow"]:
            workflow_text = "\n=== ИЗВЛЕЧЕННАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ДЕЙСТВИЙ ===\n" + "\n".join(result["workflow"])
        
        prompt = f"""Ты — эксперт-консультант по 1С:ERP Управление предприятием 2.5.
Твоя задача — дать пользователю полную пошаговую инструкцию на основе документации ITS 1C ERP.

Вопрос пользователя: {query}

НИЖЕ ПРИВЕДЕНА ИНФОРМАЦИЯ ИЗ ДОКУМЕНТАЦИИ ITS 1C ERP.
Используй ТОЛЬКО эту информацию для ответа. Если в документации нет каких-то деталей — не выдумывай, а честно скажи, что в документации это не описано.

{result['context']}
{workflow_text}

На основе документации составь доскональную пошаговую инструкцию. Для каждого шага укажи:

1. Какой объект/документ создать и в каком разделе меню он находится (конкретный путь в интерфейсе 1С)
2. Какие реквизиты (поля) нужно заполнить и какие значения в них указать
3. Какие настройки предварительно должны быть включены (функциональные опции)
4. Какие справочники должны быть预先 созданы (организации, партнеры, номенклатура и т.д.)
5. В какой последовательности выполнять шаги — что нужно создать СНАЧАЛА, а что ПОТОМ

Инструкция должна быть максимально подробной и практической, чтобы пользователь мог сразу выполнить все действия в 1С.

ВАЖНО: Перечисли все реквизиты с пояснениями. Например: "В поле 'Организация' выберите созданную ранее организацию", "В поле 'Вид цены' укажите 'Закупочная'".
"""
        return prompt


# ---------------------------------------------------------------------------
# 6. CLI команды
# ---------------------------------------------------------------------------
def cmd_build():
    """Сборка графа знаний и эмбеддингов"""
    print("=" * 60)
    print("  1С ERP Graph RAG - Построение графа знаний")
    print("=" * 60)
    
    chunks = parse_its_markdown()
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
    
    # Плотность графа
    density = nx.density(graph)
    print(f"  Плотность графа: {density:.4f}")
    
    # Количество связей
    shared = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") == "shared_terms")
    parent_child = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") in ("parent_child", "child_parent"))
    refs = sum(1 for _, _, d in graph.edges(data=True) if d.get("relation") == "references")
    print(f"  Ребра parent-child: {parent_child}")
    print(f"  Ребра cross-references: {refs}")
    print(f"  Ребра shared-terms: {shared}")
    
    print("\nГраф знаний построен!")


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
        prompt = rag.build_prompt(q)
        return {
            "query": q,
            "results": result["vector_results"][:5],
            "graph_expanded": result["graph_expanded"],
            "workflow": result["workflow"][:15],
            "prompt": prompt
        }
    
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


def cmd_interactive():
    """Интерактивный режим"""
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
    
    while True:
        try:
            q = input("> ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            
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
            
            print(f"\nКонтекст для LLM ({len(result['context'])} символов, показано первые 2000):")
            print(result["context"][:2000])
            print()
            
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Ошибка: {e}")


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
    
    p_interactive = sub.add_parser("interactive", help="Интерактивный режим")
    
    args = parser.parse_args()
    
    if args.command == "build":
        cmd_build()
    elif args.command == "query":
        cmd_query(args.query, args.top_k)
    elif args.command == "serve":
        cmd_serve(args.host, args.port)
    elif args.command == "interactive":
        cmd_interactive()
    else:
        parser.print_help()
