#!/usr/bin/env python3
"""Локальная загрузка ТЗ и построение task graph без LLM и сетевых вызовов."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import uuid
import zipfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import networkx as nx


TASK_DATA_DIR = Path(__file__).parent / "task_data"
DEFAULT_MAX_FILE_MB = 250
DEFAULT_MAX_EXTRACTED_CHARS = 8_000_000
SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".xlsx", ".pdf"}

SECTION_ALIASES = {
    "входные данные": "input",
    "входы": "input",
    "действия": "action",
    "этапы": "action",
    "результат": "output",
    "результаты": "output",
    "выходные данные": "output",
    "контрольные точки": "control",
    "контроль": "control",
    "особенности": "constraint",
    "ограничения": "constraint",
    "исключения": "exception",
}


def _slug(value: str, limit: int = 80) -> str:
    value = re.sub(r"[^\w\-]+", "_", value.strip(), flags=re.UNICODE).strip("_")
    return value[:limit] or "task"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(v) for v in value]
    return value


@dataclass
class SourceParagraph:
    text: str
    index: int
    style: str = ""
    source_part: str = "body"


@dataclass
class ParsedRequirementDocument:
    source_path: str
    sha256: str
    size_bytes: int
    mime_type: str
    paragraphs: List[SourceParagraph]
    warnings: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)


@dataclass
class PlannedQuestion:
    id: str
    category: str
    text: str
    priority: float
    blocking: bool = False
    source_node: Optional[str] = None
    relates_to: List[str] = field(default_factory=list)
    answer_type: str = "text"
    options: List[str] = field(default_factory=list)
    reason: str = ""


class RequirementFileLoader:
    """Безопасно читает ТЗ и возвращает нормализованные абзацы.

    DOCX/XLSX разбираются стандартной библиотекой. PDF поддерживается через
    необязательный pypdf; отсутствие пакета даёт понятную локальную ошибку.
    """

    def __init__(self, max_file_mb: int = DEFAULT_MAX_FILE_MB,
                 max_extracted_chars: int = DEFAULT_MAX_EXTRACTED_CHARS):
        self.max_file_bytes = max_file_mb * 1024 * 1024
        self.max_extracted_chars = max_extracted_chars

    def load(self, path: Path | str) -> ParsedRequirementDocument:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Файл ТЗ не найден: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Неподдерживаемый формат {path.suffix}. "
                f"Поддерживаются: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise ValueError(
                f"Файл {size / 1024 / 1024:.1f} МБ превышает лимит "
                f"{self.max_file_bytes / 1024 / 1024:.0f} МБ"
            )
        digest = self._sha256(path)
        suffix = path.suffix.lower()
        warnings: List[str] = []
        if suffix in {".txt", ".md"}:
            paragraphs = self._load_text(path)
        elif suffix == ".docx":
            paragraphs = self._load_docx(path)
        elif suffix == ".xlsx":
            paragraphs = self._load_xlsx(path)
        else:
            paragraphs, warnings = self._load_pdf(path)
        total_chars = sum(len(p.text) for p in paragraphs)
        if total_chars > self.max_extracted_chars:
            raise ValueError(
                f"Из файла извлечено {total_chars:,} символов, лимит "
                f"{self.max_extracted_chars:,}. Разделите ТЗ на части."
            )
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return ParsedRequirementDocument(
            source_path=str(path), sha256=digest, size_bytes=size,
            mime_type=mime, paragraphs=paragraphs, warnings=warnings,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for part in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(part)
        return digest.hexdigest()

    @staticmethod
    def _decode_text(raw: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def _load_text(self, path: Path) -> List[SourceParagraph]:
        text = self._decode_text(path.read_bytes())
        return [
            SourceParagraph(text=line.strip(), index=i)
            for i, line in enumerate(text.splitlines()) if line.strip()
        ]

    def _checked_zip(self, path: Path) -> zipfile.ZipFile:
        archive = zipfile.ZipFile(path)
        total = sum(info.file_size for info in archive.infolist())
        # Защита от zip bomb: распакованный офисный файл не должен быть
        # несоразмерно больше пользовательского лимита.
        if total > max(self.max_file_bytes * 20, 512 * 1024 * 1024):
            archive.close()
            raise ValueError("Архив офисного файла имеет подозрительно большой распакованный размер")
        return archive

    def _load_docx(self, path: Path) -> List[SourceParagraph]:
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        result: List[SourceParagraph] = []
        with self._checked_zip(path) as archive:
            try:
                root = ET.fromstring(archive.read("word/document.xml"))
            except KeyError as exc:
                raise ValueError("DOCX не содержит word/document.xml") from exc
        index = 0
        body = root.find(".//w:body", ns)
        if body is None:
            return result
        for element in list(body):
            kind = _local_name(element.tag)
            if kind == "p":
                text = "".join(t.text or "" for t in element.findall(".//w:t", ns)).strip()
                if not text:
                    continue
                style_el = element.find("./w:pPr/w:pStyle", ns)
                style = ""
                if style_el is not None:
                    style = style_el.get(f"{{{ns['w']}}}val", "")
                result.append(SourceParagraph(text=text, index=index, style=style))
                index += 1
            elif kind == "tbl":
                for row_no, row in enumerate(element.findall("./w:tr", ns), 1):
                    cells = []
                    for cell in row.findall("./w:tc", ns):
                        value = " ".join(
                            "".join(t.text or "" for t in para.findall(".//w:t", ns)).strip()
                            for para in cell.findall(".//w:p", ns)
                        ).strip()
                        cells.append(value)
                    text = " | ".join(cells).strip(" |")
                    if text:
                        result.append(SourceParagraph(
                            text=text, index=index, style="table-row",
                            source_part=f"table-row-{row_no}",
                        ))
                        index += 1
        return result

    def _load_xlsx(self, path: Path) -> List[SourceParagraph]:
        spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
        result: List[SourceParagraph] = []
        with self._checked_zip(path) as archive:
            shared: List[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in root.iter(f"{{{spreadsheet_ns}}}si"):
                    shared.append("".join(t.text or "" for t in item.iter(f"{{{spreadsheet_ns}}}t")))
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            targets = {
                rel.get("Id"): rel.get("Target", "")
                for rel in rels.iter(f"{{{pkg_rel_ns}}}Relationship")
            }
            index = 0
            for sheet in workbook.iter(f"{{{spreadsheet_ns}}}sheet"):
                title = sheet.get("name", "Лист")
                rel_id = sheet.get(f"{{{rel_ns}}}id")
                target = targets.get(rel_id, "")
                if not target:
                    continue
                sheet_path = target.lstrip("/")
                if not sheet_path.startswith("xl/"):
                    sheet_path = f"xl/{sheet_path}"
                sheet_root = ET.fromstring(archive.read(sheet_path))
                result.append(SourceParagraph(
                    text=f"ЛИСТ: {title}", index=index, style="heading-sheet",
                    source_part=title,
                ))
                index += 1
                for row in sheet_root.iter(f"{{{spreadsheet_ns}}}row"):
                    values: List[str] = []
                    for cell in row.iter(f"{{{spreadsheet_ns}}}c"):
                        cell_type = cell.get("t", "")
                        value_el = cell.find(f"{{{spreadsheet_ns}}}v")
                        inline = cell.find(f".//{{{spreadsheet_ns}}}t")
                        value = inline.text if inline is not None else (value_el.text if value_el is not None else "")
                        if cell_type == "s" and value:
                            try:
                                value = shared[int(value)]
                            except (ValueError, IndexError):
                                pass
                        values.append((value or "").strip())
                    text = " | ".join(values).strip(" |")
                    if text:
                        result.append(SourceParagraph(
                            text=text, index=index, style="table-row", source_part=title,
                        ))
                        index += 1
        return result

    def _load_pdf(self, path: Path) -> Tuple[List[SourceParagraph], List[str]]:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Для PDF установите локальный пакет pypdf. DOCX/TXT/MD/XLSX работают без него."
            ) from exc
        reader = PdfReader(str(path))
        result: List[SourceParagraph] = []
        warnings: List[str] = []
        index = 0
        for page_no, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                warnings.append(f"Страница {page_no}: текстовый слой отсутствует, нужен OCR")
                continue
            for line in text.splitlines():
                if line.strip():
                    result.append(SourceParagraph(
                        text=line.strip(), index=index, source_part=f"page-{page_no}",
                    ))
                    index += 1
        return result, warnings


class TaskGraphDocument:
    """Персональный граф одного ТЗ; не смешивается со статическим графом ERP."""

    def __init__(self, task_id: str, source: ParsedRequirementDocument,
                 graph: Optional[nx.MultiDiGraph] = None):
        self.task_id = task_id
        self.source = source
        self.graph = graph or nx.MultiDiGraph()

    @classmethod
    def from_parsed(cls, source: ParsedRequirementDocument,
                    task_id: Optional[str] = None) -> "TaskGraphDocument":
        task_id = task_id or f"{_slug(Path(source.source_path).stem, 48)}-{source.sha256[:10]}"
        instance = cls(task_id, source)
        instance._build()
        return instance

    def _build(self) -> None:
        root_id = f"task:{self.task_id}"
        self.graph.add_node(
            root_id, node_type="task", title=Path(self.source.source_path).name,
            source_path=self.source.source_path, sha256=self.source.sha256,
        )
        block_pattern = re.compile(r"^\s*БЛОК\s+(\d+)[\.\s:-]*(.*)$", re.IGNORECASE)
        section_pattern = re.compile(r"^\s*([^:]{2,80})\s*:\s*$")
        blocks: List[str] = []
        current_block: Optional[str] = None
        current_section: Optional[str] = None
        previous_item: Optional[str] = None
        synthetic_no = 0

        for para in self.source.paragraphs:
            text = para.text.strip()
            block_match = block_pattern.match(text)
            is_heading = bool(para.style and (
                "heading" in para.style.lower() or "заголов" in para.style.lower()
                or para.style.startswith("heading-")
            ))
            if block_match or (is_heading and not text.startswith("ЛИСТ:")):
                if block_match:
                    number = block_match.group(1)
                    title = block_match.group(2).strip() or text
                else:
                    synthetic_no += 1
                    number = f"h{synthetic_no}"
                    title = text
                current_block = f"{root_id}:block:{number}"
                self.graph.add_node(
                    current_block, node_type="process_block", title=title,
                    original_text=text, source_index=para.index, order=len(blocks),
                )
                self.graph.add_edge(root_id, current_block, relation="contains", weight=1.0)
                if blocks:
                    self.graph.add_edge(blocks[-1], current_block, relation="precedes", weight=1.0)
                blocks.append(current_block)
                current_section = None
                previous_item = None
                continue
            if current_block is None:
                current_block = f"{root_id}:block:0"
                self.graph.add_node(
                    current_block, node_type="process_block", title="Общее описание",
                    original_text="", source_index=para.index, order=0,
                )
                self.graph.add_edge(root_id, current_block, relation="contains", weight=1.0)
                blocks.append(current_block)
            section_match = section_pattern.match(text)
            if section_match:
                normalized = section_match.group(1).strip().lower().replace("ё", "е")
                normalized = re.sub(r"\s+", " ", normalized)
                section_type = SECTION_ALIASES.get(normalized)
                if section_type:
                    current_section = section_type
                    previous_item = None
                    continue
            section_type = current_section or "statement"
            item_id = f"{current_block}:{section_type}:{para.index}"
            self.graph.add_node(
                item_id, node_type=section_type, title=text, original_text=text,
                source_index=para.index, source_part=para.source_part,
            )
            self.graph.add_edge(current_block, item_id, relation=f"has_{section_type}", weight=1.0)
            if previous_item:
                self.graph.add_edge(previous_item, item_id, relation="precedes", weight=0.8)
            previous_item = item_id

        self._link_block_flows(blocks)

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {"данные", "система", "документы", "документ", "результат", "входные"}
        return {
            token for token in re.findall(r"[а-яёa-z0-9]{4,}", text.lower())
            if token not in stop
        }

    def _link_block_flows(self, blocks: Sequence[str]) -> None:
        for left, right in zip(blocks, blocks[1:]):
            outputs = [
                node for _, node, data in self.graph.out_edges(left, data=True)
                if data.get("relation") == "has_output"
            ]
            inputs = [
                node for _, node, data in self.graph.out_edges(right, data=True)
                if data.get("relation") == "has_input"
            ]
            for output in outputs:
                out_terms = self._terms(self.graph.nodes[output].get("title", ""))
                for input_node in inputs:
                    in_terms = self._terms(self.graph.nodes[input_node].get("title", ""))
                    union = out_terms | in_terms
                    score = len(out_terms & in_terms) / len(union) if union else 0.0
                    if score >= 0.12:
                        self.graph.add_edge(
                            output, input_node, relation="feeds", weight=round(0.5 + score, 3),
                        )

    def save(self, base_dir: Path | str = TASK_DATA_DIR) -> Path:
        task_dir = Path(base_dir) / self.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        target = task_dir / "task_graph.json"
        payload = {
            "schema_version": 1,
            "task_id": self.task_id,
            "source": {
                "source_path": self.source.source_path,
                "sha256": self.source.sha256,
                "size_bytes": self.source.size_bytes,
                "mime_type": self.source.mime_type,
                "warnings": self.source.warnings,
            },
            "nodes": [
                {"id": node, **_safe_json_value(data)}
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target_node, "key": key, **_safe_json_value(data)}
                for source, target_node, key, data in self.graph.edges(keys=True, data=True)
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str) -> "TaskGraphDocument":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_data = payload["source"]
        source = ParsedRequirementDocument(
            source_path=source_data["source_path"], sha256=source_data["sha256"],
            size_bytes=source_data["size_bytes"], mime_type=source_data["mime_type"],
            paragraphs=[], warnings=source_data.get("warnings", []),
        )
        graph = nx.MultiDiGraph()
        for node in payload.get("nodes", []):
            data = dict(node)
            node_id = data.pop("id")
            graph.add_node(node_id, **data)
        for edge in payload.get("edges", []):
            data = dict(edge)
            source_id = data.pop("source")
            target_id = data.pop("target")
            key = data.pop("key", None)
            graph.add_edge(source_id, target_id, key=key, **data)
        return cls(payload["task_id"], source, graph)

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            counts[node_type] = counts.get(node_type, 0) + 1
        return {
            "task_id": self.task_id,
            "source_path": self.source.source_path,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": counts,
            "warnings": self.source.warnings,
        }

    def searchable_texts(self) -> List[Tuple[str, str]]:
        return [
            (node, str(data.get("title", "")))
            for node, data in self.graph.nodes(data=True)
            if data.get("node_type") not in {"task"} and data.get("title")
        ]


class TzNormalizer(ABC):
    """Контракт субагента «ТЗ → язык ERP».

    Реализация не привязана к PI/herdr. Внешний агент должен вернуть только ID,
    уже существующие в ERP-графе; validate_mappings отбрасывает галлюцинации.
    """

    @abstractmethod
    def normalize(self, task: TaskGraphDocument, erp_graph: nx.Graph) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def validate_mappings(payload: Dict[str, Any], erp_graph: nx.Graph) -> Dict[str, Any]:
        valid: List[Dict[str, Any]] = []
        gaps: List[Dict[str, Any]] = list(payload.get("mapping_gaps", []))
        for mapping in payload.get("mappings", []):
            node_id = mapping.get("erp_node_id")
            if node_id in erp_graph:
                valid.append(mapping)
            else:
                gaps.append({
                    "task_node_id": mapping.get("task_node_id"),
                    "proposed_erp_node_id": node_id,
                    "reason": "ERP node ID отсутствует в статическом графе",
                })
        return {**payload, "mappings": valid, "mapping_gaps": gaps}


class GraphOnlyTzNormalizer(TzNormalizer):
    """Детерминированный офлайн-нормализатор на основе названий узлов."""

    ERP_LEXICON: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        (r"(?:заявк\w*|заказ\w*).{0,60}клиент\w*|клиент\w*.{0,60}(?:заявк\w*|заказ\w*)",
         ("ERPcode/Documents/ЗаказКлиента",)),
        (r"план\w*.{0,40}производ\w*",
         ("ERPcode/Documents/ПланПроизводства",)),
        (r"заказ\w*.{0,40}поставщик\w*",
         ("ERPcode/Documents/ЗаказПоставщику",)),
        (r"заказ\w*.{0,40}производ\w*",
         ("ERPcode/Documents/ЗаказНаПроизводство2_2",)),
        (r"этап\w*.{0,40}производ\w*|производ\w*.{0,40}этап\w*",
         ("ERPcode/Documents/ЭтапПроизводства2_2",)),
        (r"перемещ\w*.{0,40}(?:товар\w*|сырь\w*|материал\w*|продукц\w*)",
         ("ERPcode/Documents/ПеремещениеТоваров",)),
        (r"реализац\w*.{0,40}(?:товар\w*|услуг\w*|продукц\w*)",
         ("ERPcode/Documents/РеализацияТоваровУслуг",)),
        (r"возврат\w*.{0,40}клиент\w*|клиент\w*.{0,40}возврат\w*",
         ("ERPcode/Documents/ВозвратТоваровОтКлиента",)),
        (r"ресурсн\w*.{0,20}спецификац\w*|рецептур\w*",
         ("ERPcode/Catalogs/РесурсныеСпецификации",)),
        (r"номенклатур\w*",
         ("ERPcode/Catalogs/Номенклатура",)),
    )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {
            "который", "данные", "система", "документ", "документы", "пользователь",
            "обработка", "формирование", "проверка", "контроль", "создание", "получение",
            "значение", "текущий", "автоматически", "справочник", "перечисление", "регистр",
        }
        return {
            token for token in re.findall(r"[а-яёa-z0-9]{4,}", text.lower())
            if token not in stop
        }

    @staticmethod
    def _family(node: str, layer: int) -> str:
        if layer == 1:
            return "scenario"
        parts = str(node).split("/")
        return parts[1] if len(parts) > 2 and parts[0] == "ERPcode" else "metadata"

    @staticmethod
    def _allowed_families(task_type: str) -> set[str]:
        return {
            "process_block": {"scenario", "Subsystems"},
            "input": {"Documents", "Catalogs"},
            "action": {"Documents"},
            "output": {"Documents"},
            "control": {"scenario", "Reports"},
            "constraint": {"scenario", "Enums"},
            "exception": {"scenario", "Documents"},
        }.get(task_type, {"scenario", "Documents", "Catalogs"})

    def normalize(self, task: TaskGraphDocument, erp_graph: nx.Graph) -> Dict[str, Any]:
        erp_candidates: List[Tuple[str, set[str], str, int, str]] = []
        for node, data in erp_graph.nodes(data=True):
            layer = int(data.get("layer", 4) or 4)
            if layer not in {1, 3}:
                continue
            # Поля, команды и группы UI нужны после выбора ERP-объекта, но
            # слишком шумны для первичного перевода формулировок ТЗ.
            node_type = str(data.get("node_type", ""))
            if layer == 3 and node_type != "metadata":
                continue
            title = str(data.get("title", ""))
            tokens = self._tokens(title)
            if tokens:
                erp_candidates.append((node, tokens, title, layer, self._family(node, layer)))
        mappings: List[Dict[str, Any]] = []
        gaps: List[Dict[str, Any]] = []
        for task_node, text in task.searchable_texts():
            task_tokens = self._tokens(text)
            if not task_tokens:
                continue
            task_type = str(task.graph.nodes[task_node].get("node_type", ""))
            allowed_families = self._allowed_families(task_type)

            lexicon_nodes: List[str] = []
            lowered = text.lower().replace("ё", "е")
            for pattern, candidate_ids in self.ERP_LEXICON:
                if not re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
                    continue
                for erp_node in candidate_ids:
                    if erp_node not in erp_graph:
                        continue
                    family = self._family(erp_node, int(erp_graph.nodes[erp_node].get("layer", 3) or 3))
                    if family in allowed_families and erp_node not in lexicon_nodes:
                        lexicon_nodes.append(erp_node)
            if lexicon_nodes:
                for erp_node in lexicon_nodes[:2]:
                    data = erp_graph.nodes[erp_node]
                    mappings.append({
                        "task_node_id": task_node, "source_text": text,
                        "erp_node_id": erp_node, "erp_title": data.get("title", erp_node),
                        "erp_layer": int(data.get("layer", 3) or 3), "confidence": 0.95,
                        "status": "confirmed_by_erp_lexicon",
                        "derivation": "graph-only ERP terminology rule",
                    })
                continue

            scored: List[Tuple[float, str, str, int, str]] = []
            for erp_node, erp_tokens, title, layer, family in erp_candidates:
                if family not in allowed_families:
                    continue
                overlap = task_tokens & erp_tokens
                if not overlap:
                    continue
                # Dice не позволяет одному общему слову дать ложную уверенность
                # 100%, как это происходило с «заявка» и десятками документов.
                score = 2 * len(overlap) / max(1, len(task_tokens) + len(erp_tokens))
                if score >= 0.30:
                    scored.append((score, erp_node, title, layer, family))
            scored.sort(key=lambda item: (-item[0], str(item[1])))
            top = scored[0] if scored else None
            margin = top[0] - scored[1][0] if top and len(scored) > 1 else (top[0] if top else 0.0)
            if top and top[0] >= 0.42 and margin >= 0.08:
                score, erp_node, title, layer, _family = top
                mappings.append({
                    "task_node_id": task_node, "source_text": text,
                    "erp_node_id": erp_node, "erp_title": title,
                    "erp_layer": layer, "confidence": round(score, 3),
                    "status": "high_confidence_candidate",
                    "derivation": "graph-only unique lexical candidate",
                })
            elif task_type in {"process_block", "input", "action", "output", "control", "constraint", "exception"}:
                gaps.append({
                    "task_node_id": task_node, "source_text": text,
                    "reason": (
                        "Неоднозначное соответствие ERP — требуется подтверждение"
                        if scored else "Не найдено уверенного соответствия в L1/L3"
                    ),
                    "candidates": [
                        {"erp_node_id": item[1], "erp_title": item[2], "score": round(item[0], 3)}
                        for item in scored[:5]
                    ],
                })
        return self.validate_mappings({
            "normalizer": "graph-only", "task_id": task.task_id,
            "mappings": mappings, "mapping_gaps": gaps,
        }, erp_graph)


class CallableAgentTzNormalizer(TzNormalizer):
    """Адаптер для будущего PI/herdr без встроенного сетевого клиента.

    caller получает сериализуемый Task Graph и список допустимых ERP ID.
    Создание/настройка внешнего клиента остаётся за интеграционным слоем.
    """

    def __init__(self, caller):
        self.caller = caller

    def normalize(self, task: TaskGraphDocument, erp_graph: nx.Graph) -> Dict[str, Any]:
        request = {
            "task_id": task.task_id,
            "task_nodes": [
                {"id": node, **_safe_json_value(data)}
                for node, data in task.graph.nodes(data=True)
            ],
            "allowed_erp_nodes": [
                {"id": node, "title": data.get("title", ""), "layer": data.get("layer")}
                for node, data in erp_graph.nodes(data=True)
                if int(data.get("layer", 4) or 4) in {1, 3}
            ],
        }
        response = self.caller(request)
        if not isinstance(response, dict):
            raise TypeError("Субагент должен вернуть dict по контракту TzNormalizer")
        return self.validate_mappings(response, erp_graph)


def mapping_gap_question_id(task_id: str, gap: Dict[str, Any]) -> str:
    source_text = str(gap.get("source_text", "")).strip()
    return "question:gap:" + hashlib.sha1(
        f"{task_id}|{gap.get('task_node_id')}|{source_text}".encode("utf-8")
    ).hexdigest()[:16]


def apply_mapping_answers(payload: Dict[str, Any], answers: Optional[Dict[str, Any]],
                          erp_graph: nx.Graph) -> Dict[str, Any]:
    """Превращает выбранные пользователем ERP-кандидаты в подтверждённые связи."""
    if not answers:
        return payload
    answer_map = answers.get("answers", answers) if isinstance(answers, dict) else {}
    if not isinstance(answer_map, dict):
        return payload
    mappings = list(payload.get("mappings", []))
    remaining_gaps: List[Dict[str, Any]] = []
    task_id = str(payload.get("task_id", ""))
    for gap in payload.get("mapping_gaps", []):
        qid = mapping_gap_question_id(task_id, gap)
        raw = answer_map.get(qid)
        if raw is None:
            remaining_gaps.append(gap)
            continue
        if isinstance(raw, dict):
            selected = raw.get("erp_node_id") or raw.get("value") or raw.get("answer")
        else:
            selected = raw
        selected_text = str(selected or "").strip()
        candidate_id = selected_text if selected_text in erp_graph else None
        if not candidate_id:
            for candidate in gap.get("candidates", []):
                if selected_text in {
                    str(candidate.get("erp_node_id", "")),
                    str(candidate.get("erp_title", "")),
                }:
                    candidate_id = candidate.get("erp_node_id")
                    break
        if not candidate_id or candidate_id not in erp_graph:
            remaining_gaps.append({
                **gap, "answer_error": "Ответ не соответствует допустимому ERP node ID",
            })
            continue
        data = erp_graph.nodes[candidate_id]
        mappings.append({
            "task_node_id": gap.get("task_node_id"),
            "source_text": gap.get("source_text", ""),
            "erp_node_id": candidate_id,
            "erp_title": data.get("title", candidate_id),
            "erp_layer": int(data.get("layer", 3) or 3),
            "confidence": 1.0,
            "status": "user_confirmed",
            "derivation": "answer to ERP mapping question",
        })
    return {**payload, "mappings": mappings, "mapping_gaps": remaining_gaps}


QUESTION_TEMPLATES: Tuple[Dict[str, Any], ...] = (
    {"category": "scope", "text": "Какие юридические организации и контуры учёта входят в моделируемый процесс?", "keywords": (), "priority": 85, "blocking": True},
    {"category": "scope", "text": "Какие подразделения отвечают за каждый этап и где проходит граница ответственности?", "keywords": ("подразделен", "цех", "склад"), "priority": 78},
    {"category": "sales", "text": "Чем отличаются процессы для собственного магазина, оптового покупателя и сетевой розницы?", "keywords": ("магазин", "оптов", "рознич", "клиент"), "priority": 92, "blocking": True},
    {"category": "sales", "text": "Что является подтверждённой потребностью клиента и кто её подтверждает?", "keywords": ("заявк", "заказ", "подтверж"), "priority": 95, "blocking": True},
    {"category": "sales", "text": "Как обрабатываются заявки, поступившие после установленного времени отсечения?", "keywords": ("12:00", "срочн", "после"), "priority": 88},
    {"category": "sales", "text": "Нужны ли индивидуальные соглашения, цены, скидки и условия оплаты по каналам продаж?", "keywords": ("клиент", "продаж", "цена", "оплат"), "priority": 75},
    {"category": "production", "text": "Производство всегда работает под заказ или допускается выпуск на склад?", "keywords": ("производ", "заказ", "запас"), "priority": 96, "blocking": True},
    {"category": "production", "text": "Каким объектом должен фиксироваться производственный план и с каким горизонтом?", "keywords": ("план", "производ"), "priority": 91, "blocking": True},
    {"category": "production", "text": "Полуфабрикат учитывается как отдельная номенклатура и отдельный выпуск?", "keywords": ("полуфабрикат", "тесто", "холодн"), "priority": 94, "blocking": True},
    {"category": "production", "text": "Нужен ли пооперационный учёт или достаточно этапов производства?", "keywords": ("операц", "этап", "цех", "технолог"), "priority": 80},
    {"category": "production", "text": "Как оформляется передача полуфабриката между цехами и кладовыми?", "keywords": ("переда", "цех", "кладов", "полуфабрикат"), "priority": 90},
    {"category": "production", "text": "Как фиксируются отклонения фактического расхода от ресурсной спецификации?", "keywords": ("рецепт", "спецификац", "расход", "сырь"), "priority": 79},
    {"category": "procurement", "text": "Минимальный запас задаётся отдельно по каждой позиции, складу и периоду?", "keywords": ("минимальн", "запас", "лимит", "закуп"), "priority": 89, "blocking": True},
    {"category": "procurement", "text": "Как рассчитывается количество заказа поставщику: средний расход, min/max или прогноз?", "keywords": ("средн", "расход", "заказ постав", "закуп"), "priority": 90, "blocking": True},
    {"category": "procurement", "text": "По каким правилам выбирается быстрый или долгий поставщик?", "keywords": ("поставщик", "быстр", "долг", "поставка"), "priority": 87},
    {"category": "procurement", "text": "Нужен ли контроль графика поставок и просроченных заказов поставщикам?", "keywords": ("поставк", "отслеж", "срок"), "priority": 72},
    {"category": "warehouse", "text": "Какие склады, помещения и кладовые должны быть заведены как отдельные объекты ERP?", "keywords": ("склад", "кладов", "помещен"), "priority": 93, "blocking": True},
    {"category": "warehouse", "text": "Нужна ли ордерная схема при поступлении, перемещении и отгрузке?", "keywords": ("приемк", "отгруз", "склад", "перемещ"), "priority": 91, "blocking": True},
    {"category": "warehouse", "text": "FIFO должен обеспечиваться по партиям, сериям или только организационным правилом?", "keywords": ("fifo", "фифо", "парт", "сер"), "priority": 96, "blocking": True},
    {"category": "warehouse", "text": "Нужен ли адресный склад и учёт по ячейкам?", "keywords": ("склад", "хранен", "размещ"), "priority": 68},
    {"category": "quality", "text": "Где и кем фиксируются результаты входного контроля качества?", "keywords": ("контрол", "качеств", "лаборатор", "осмотр"), "priority": 94, "blocking": True},
    {"category": "quality", "text": "Какие решения возможны по результатам контроля: принять, изолировать, вернуть, списать?", "keywords": ("брак", "возврат", "контрол", "качеств"), "priority": 92, "blocking": True},
    {"category": "quality", "text": "Как отражается производственный брак и кто разрешает дальнейшее использование?", "keywords": ("брак", "забрак", "производ"), "priority": 88},
    {"category": "traceability", "text": "Нужно ли вести серии сырья и продукции с датами производства и сроками годности?", "keywords": ("срок годност", "сер", "парт", "хранен"), "priority": 97, "blocking": True},
    {"category": "traceability", "text": "Требуется ли прослеживание готовой продукции до партии использованного сырья?", "keywords": ("прослеж", "сырь", "продукц", "парт"), "priority": 82},
    {"category": "units", "text": "Все ли операции выполняются в штуках или нужны упаковки и коэффициенты пересчёта?", "keywords": ("штук", "единиц", "упаков"), "priority": 91, "blocking": True},
    {"category": "packaging", "text": "Упаковочные материалы списываются по норме или по фактическому расходу?", "keywords": ("упаков", "плен", "пакет", "материал"), "priority": 80},
    {"category": "packaging", "text": "Многооборотная тара учитывается по владельцам, местам хранения и возвратам?", "keywords": ("тара", "лотк", "тележ"), "priority": 87},
    {"category": "logistics", "text": "Доставка планируется внутри ERP или передаётся внешней системе/перевозчику?", "keywords": ("достав", "логист", "маршрут", "перевоз"), "priority": 89, "blocking": True},
    {"category": "logistics", "text": "Нужно ли планировать рейсы, транспорт, водителей и временные окна?", "keywords": ("рейс", "маршрут", "транспорт", "достав"), "priority": 80},
    {"category": "returns", "text": "Какие финансовые последствия возврата: возврат денег, замена, уменьшение долга?", "keywords": ("возврат", "клиент", "деньг", "пересчет"), "priority": 93, "blocking": True},
    {"category": "returns", "text": "Почему утилизация возврата не отражается в учёте и допустимо ли это для контроля остатков?", "keywords": ("утилиз", "не отраж", "возврат"), "priority": 97, "blocking": True},
    {"category": "finance", "text": "Какой контур нужен: только оперативный, управленческий, регламентированный или несколько?", "keywords": ("оператив", "бухгалтер", "учет", "контур"), "priority": 95, "blocking": True},
    {"category": "finance", "text": "Как должна рассчитываться материальная себестоимость при выбранном контуре учёта?", "keywords": ("себестоим", "материал", "затрат"), "priority": 94, "blocking": True},
    {"category": "integration", "text": "Какие заявки и документы поступают автоматически и из каких систем?", "keywords": ("автомат", "интеграц", "заявк", "обмен"), "priority": 82},
    {"category": "roles", "text": "Какие роли создают, согласуют, проводят и контролируют документы на каждом этапе?", "keywords": ("соглас", "подтверж", "контрол", "ответствен"), "priority": 84},
    {"category": "exceptions", "text": "Какие исключительные ветки должны быть разрешены и кто их согласует?", "keywords": ("исключ", "срочн", "отклон", "брак"), "priority": 83},
    {"category": "acceptance", "text": "Какие отчёты и измеримые показатели считаются критериями успешного внедрения?", "keywords": ("контрол", "отчет", "результат"), "priority": 76},
)


class QuestionPlanner:
    """Создаёт много кандидатов и выдаёт их приоритетными раундами."""

    def __init__(self, templates: Sequence[Dict[str, Any]] = QUESTION_TEMPLATES):
        self.templates = tuple(templates)

    @staticmethod
    def _text(task: TaskGraphDocument) -> str:
        return "\n".join(text for _, text in task.searchable_texts()).lower().replace("ё", "е")

    def plan(self, task: TaskGraphDocument, mappings: Optional[Dict[str, Any]] = None,
             answered_ids: Optional[Iterable[str]] = None,
             limit: Optional[int] = None) -> List[PlannedQuestion]:
        corpus = self._text(task)
        answered = set(answered_ids or ())
        questions: List[PlannedQuestion] = []
        for template in self.templates:
            keywords = tuple(str(k).lower().replace("ё", "е") for k in template.get("keywords", ()))
            matched = [keyword for keyword in keywords if keyword in corpus]
            if keywords and not matched:
                continue
            qid = "question:" + hashlib.sha1(
                f"{task.task_id}|{template['category']}|{template['text']}".encode("utf-8")
            ).hexdigest()[:16]
            if qid in answered:
                continue
            score = float(template.get("priority", 50))
            score += min(8, len(matched) * 2)
            if template.get("blocking"):
                score += 8
            questions.append(PlannedQuestion(
                id=qid, category=template["category"], text=template["text"],
                priority=round(score, 2), blocking=bool(template.get("blocking")),
                reason=("Совпали признаки: " + ", ".join(matched)) if matched else "Базовое моделирующее решение",
                answer_type=template.get("answer_type", "text"),
                options=list(template.get("options", [])),
            ))
        if mappings:
            # Не превращаем большое ТЗ в сотни вопросов за один раунд:
            # максимум три нерешённых сопоставления на процессный блок. После
            # ответов answered_ids следующий вызов поднимет следующие пробелы.
            gap_counts: Dict[str, int] = {}
            for gap in mappings.get("mapping_gaps", []):
                source_text = str(gap.get("source_text", "")).strip()
                if not source_text:
                    continue
                qid = mapping_gap_question_id(task.task_id, gap)
                if qid in answered:
                    continue
                block_match = re.search(r":block:\d+", str(gap.get("task_node_id", "")))
                group = block_match.group(0) if block_match else "task"
                if gap_counts.get(group, 0) >= 3:
                    continue
                gap_counts[group] = gap_counts.get(group, 0) + 1
                candidates = gap.get("candidates", [])
                options = [
                    str(candidate.get("erp_title", candidate.get("erp_node_id", "")))
                    for candidate in candidates[:4]
                    if candidate.get("erp_title") or candidate.get("erp_node_id")
                ]
                questions.append(PlannedQuestion(
                    id=qid, category="erp_mapping",
                    text=f"Какой результат в ERP должен соответствовать требованию: «{source_text[:220]}»?",
                    priority=96.0, blocking=True,
                    source_node=gap.get("task_node_id"),
                    reason="Для требования не найдено подтверждённого объекта ERP",
                    answer_type="choice" if options else "text",
                    options=options,
                ))
        # Дедупликация по нормализованному тексту.
        unique: Dict[str, PlannedQuestion] = {}
        for question in questions:
            key = re.sub(r"\W+", "", question.text.lower(), flags=re.UNICODE)
            existing = unique.get(key)
            if existing is None or question.priority > existing.priority:
                unique[key] = question
        result = sorted(unique.values(), key=lambda q: (-q.priority, q.category, q.text))
        return result[:limit] if limit is not None else result

    @staticmethod
    def materialize(task: TaskGraphDocument, questions: Sequence[PlannedQuestion]) -> None:
        root_id = f"task:{task.task_id}"
        for question in questions:
            task.graph.add_node(
                question.id, node_type="question", title=question.text,
                category=question.category, priority=question.priority,
                blocking=question.blocking, reason=question.reason,
                answer_type=question.answer_type, options=question.options,
            )
            task.graph.add_edge(root_id, question.id, relation="needs_answer", weight=1.0)
            if question.source_node and question.source_node in task.graph:
                task.graph.add_edge(question.id, question.source_node, relation="clarifies", weight=1.0)


def ingest_requirement_file(path: Path | str, task_id: Optional[str] = None,
                            output_dir: Path | str = TASK_DATA_DIR,
                            max_file_mb: int = DEFAULT_MAX_FILE_MB) -> Tuple[TaskGraphDocument, Path]:
    loader = RequirementFileLoader(max_file_mb=max_file_mb)
    parsed = loader.load(path)
    task = TaskGraphDocument.from_parsed(parsed, task_id=task_id)
    saved = task.save(output_dir)
    return task, saved


def stream_copy_with_limit(chunks: Iterable[bytes], target: Path,
                           max_bytes: int) -> Tuple[int, str]:
    """Сохраняет поток запроса на диск без удержания файла в памяти."""
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    digest = hashlib.sha256()
    try:
        with target.open("wb") as handle:
            for chunk in chunks:
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Файл превышает лимит {max_bytes / 1024 / 1024:.0f} МБ")
                digest.update(chunk)
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return total, digest.hexdigest()
