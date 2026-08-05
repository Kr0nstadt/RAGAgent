#!/usr/bin/env python3
"""Типизированные связи ERP, планировщики и проверка целостности графа."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import networkx as nx


TYPE_TO_DIR = {
    "Catalog": "Catalogs",
    "Document": "Documents",
    "Enum": "Enums",
    "InformationRegister": "InformationRegisters",
    "AccumulationRegister": "AccumulationRegisters",
    "AccountingRegister": "AccountingRegisters",
    "CalculationRegister": "CalculationRegisters",
    "ChartOfAccounts": "ChartsOfAccounts",
    "ChartOfCharacteristicTypes": "ChartsOfCharacteristicTypes",
    "ChartOfCalculationTypes": "ChartsOfCalculationTypes",
    "BusinessProcess": "BusinessProcesses",
    "Task": "Tasks",
    "Report": "Reports",
    "DataProcessor": "DataProcessors",
    "Constant": "Constants",
    "Subsystem": "Subsystems",
    "ExchangePlan": "ExchangePlans",
    "FunctionalOption": "FunctionalOptions",
    "Role": "Roles",
    "DocumentJournal": "DocumentJournals",
    "Filter": "Filters",
    "HTTPService": "HTTPServices",
    "WebService": "WebServices",
    "ScheduledJob": "ScheduledJobs",
    "SessionParameter": "SessionParameters",
    "SettingsStorage": "SettingsStorages",
    "CommonCommand": "CommonCommands",
    "CommonForm": "CommonForms",
    "CommonModule": "CommonModules",
}

REF_PREFIX_TO_TYPE = {
    "CatalogRef": "Catalog",
    "DocumentRef": "Document",
    "EnumRef": "Enum",
    "InformationRegisterRef": "InformationRegister",
    "AccumulationRegisterRef": "AccumulationRegister",
    "AccountingRegisterRef": "AccountingRegister",
    "CalculationRegisterRef": "CalculationRegister",
    "ChartOfAccountsRef": "ChartOfAccounts",
    "ChartOfCharacteristicTypesRef": "ChartOfCharacteristicTypes",
    "ChartOfCalculationTypesRef": "ChartOfCalculationTypes",
    "BusinessProcessRef": "BusinessProcess",
    "TaskRef": "Task",
    "ExchangePlanRef": "ExchangePlan",
}

OBJECT_TAGS = set(TYPE_TO_DIR)
FIELD_TAGS = {
    "Attribute", "Dimension", "Resource", "AccountingFlag",
    "ExtDimensionAccountingFlag", "Recalculation", "AddressingAttribute",
}
REGISTER_TYPES = {
    "InformationRegister", "AccumulationRegister", "AccountingRegister", "CalculationRegister"
}

OPERATIONAL_FIELD_PARTS = {
    "количество", "цена", "сумма", "номенклатура", "характеристика",
    "серия", "склад", "подразделение", "организация", "контрагент",
    "партнер", "соглашение", "договор", "статус", "назначение",
    "дата", "валюта", "единицаизмерения", "видноменклатуры",
}

TYPED_RELATION_WEIGHTS = {
    "has_field": 1.0,
    "field_type": 0.95,
    "required_reference": 1.0,
    "requires": 1.0,
    "can_create_inline": 0.9,
    "has_tabular_section": 1.0,
    "can_be_created_on_basis_of": 0.9,
    "creates_on_basis": 0.9,
    "may_write_register": 0.95,
    "has_registrator": 0.95,
    "contains_object": 0.85,
    "in_subsystem": 0.85,
    "has_command": 0.9,
    "opens_object": 0.95,
    "opened_by_command": 0.95,
    "has_form": 0.85,
    "form_of": 0.85,
    "form_contains_field": 0.8,
    "shown_on_form": 0.8,
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def direct_child(element: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if element is None:
        return None
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def child_text(element: Optional[ET.Element], name: str, default: str = "") -> str:
    child = direct_child(element, name)
    return (child.text or "").strip() if child is not None else default


def russian_presentation(element: Optional[ET.Element], default: str = "") -> str:
    if element is None:
        return default
    fallback = ""
    for item in element.iter():
        if local_name(item.tag) != "item":
            continue
        language = ""
        content = ""
        for child in list(item):
            if local_name(child.tag) == "lang":
                language = (child.text or "").strip().lower()
            elif local_name(child.tag) == "content":
                content = (child.text or "").strip()
        if content and not fallback:
            fallback = content
        if language == "ru" and content:
            return content
    for child in element.iter():
        if local_name(child.tag) == "content" and (child.text or "").strip():
            return (child.text or "").strip()
    return fallback or default


def canonical_metadata_id(reference: str) -> Optional[str]:
    """Преобразует cfg/xr ссылку 1С в ID, совместимый с текущим графом."""
    reference = (reference or "").strip()
    if not reference:
        return None
    if reference.startswith("ERPcode/"):
        return reference
    reference = reference.removeprefix("cfg:")
    head = reference.split(".", 1)[0]
    tail = reference.split(".", 1)[1] if "." in reference else ""
    if not tail:
        return None
    object_name = tail.split(".", 1)[0]
    object_type = REF_PREFIX_TO_TYPE.get(head, head)
    directory = TYPE_TO_DIR.get(object_type)
    if not directory:
        return None
    return f"ERPcode/{directory}/{object_name}"


def reference_object(reference: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    reference = (reference or "").strip().removeprefix("cfg:")
    if "." not in reference:
        return None, None, None
    head, tail = reference.split(".", 1)
    object_type = REF_PREFIX_TO_TYPE.get(head, head)
    object_name = tail.split(".", 1)[0]
    return object_type, object_name, canonical_metadata_id(reference)


def relation(target: str, relation_type: str, *, reverse_type: Optional[str] = None,
             weight: Optional[float] = None, evidence: str = "",
             properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "target": target,
        "type": relation_type,
        "weight": float(weight if weight is not None else TYPED_RELATION_WEIGHTS.get(relation_type, 0.7)),
    }
    if reverse_type:
        payload["reverse_type"] = reverse_type
    if evidence:
        payload["evidence"] = evidence
    if properties:
        payload["properties"] = properties
    return payload


def _append_relation(chunk: Any, payload: Dict[str, Any]) -> None:
    current = getattr(chunk, "relations", None)
    if current is None:
        chunk.relations = []
        current = chunk.relations
    signature = (payload.get("target"), payload.get("type"), json.dumps(payload.get("properties", {}), sort_keys=True, ensure_ascii=False))
    existing = {
        (item.get("target"), item.get("type"), json.dumps(item.get("properties", {}), sort_keys=True, ensure_ascii=False))
        for item in current
    }
    if signature not in existing:
        current.append(payload)


def _metadata_object(root: ET.Element, expected_type: str) -> Optional[ET.Element]:
    for element in root.iter():
        if local_name(element.tag) == expected_type:
            return element
    return None


def _type_references(properties: Optional[ET.Element]) -> List[str]:
    type_node = direct_child(properties, "Type")
    if type_node is None:
        return []
    refs: List[str] = []
    for element in type_node.iter():
        if local_name(element.tag) != "Type":
            continue
        text = (element.text or "").strip()
        if canonical_metadata_id(text) and text not in refs:
            refs.append(text)
    return refs


def _tooltip(properties: Optional[ET.Element]) -> str:
    return russian_presentation(direct_child(properties, "ToolTip"), "")


def _field_chunk(object_chunk: Any, field_element: ET.Element, kind: str,
                 source_path: Path, make_chunk: Callable[..., Any],
                 prefix: str = "Fields") -> Tuple[Any, List[str]]:
    properties = direct_child(field_element, "Properties")
    name = child_text(properties, "Name")
    if not name:
        raise ValueError("Поле без Name")
    synonym = russian_presentation(direct_child(properties, "Synonym"), name)
    required = child_text(properties, "FillChecking") == "ShowError"
    create_on_input = child_text(properties, "CreateOnInput") or "Auto"
    type_refs = _type_references(properties)
    target_ids = [canonical_metadata_id(ref) for ref in type_refs]
    target_ids = [target for target in target_ids if target]
    field_id = f"{object_chunk.id}/{prefix}/{name}"
    content = [
        f"Поле метаданных: {synonym} ({name})",
        f"Объект: {object_chunk.title}",
        f"Вид поля: {kind}",
        f"Обязательное: {'да' if required else 'нет'}",
        f"Создание при вводе: {create_on_input}",
    ]
    tooltip = _tooltip(properties)
    if target_ids:
        content.append("Типы: " + ", ".join(target_ids))
    if tooltip:
        content.append("Подсказка: " + tooltip)
    metadata = {
        "object_id": object_chunk.id,
        "field_name": name,
        "field_kind": kind,
        "required": required,
        # FillChecking=ShowError подтверждает проверку заполнения в метаданных,
        # но не доказывает, что реквизит безусловно доступен во всех операциях,
        # функциональных опциях и формах.
        "required_by_metadata": required,
        "applicability": "conditional_or_unknown" if required else "optional",
        "create_on_input": create_on_input,
        "type_references": target_ids,
        "tooltip": tooltip,
        "source_xml": str(source_path),
    }
    field_chunk = make_chunk(
        id=field_id, title=f"Поле {object_chunk.title}: {synonym}",
        content="\n".join(content), path=field_id, layer=3,
        node_type="metadata_field", parent_id=object_chunk.id,
        level=int(getattr(object_chunk, "level", 2) or 2) + 1,
        metadata=metadata,
    )
    for target in target_ids:
        _append_relation(field_chunk, relation(
            target, "field_type", reverse_type="referenced_by_field",
            evidence=str(source_path), properties={"field": name, "required": required},
        ))
        if required:
            _append_relation(field_chunk, relation(
                target, "required_reference", reverse_type="required_by_field",
                evidence=str(source_path), properties={
                    "field": name, "required_by_metadata": True,
                    "applicability": "conditional_or_unknown",
                },
            ))
            _append_relation(object_chunk, relation(
                target, "requires", reverse_type="required_by",
                evidence=str(source_path), properties={
                    "field_id": field_id, "field": name,
                    "create_on_input": create_on_input,
                    "required_by_metadata": True,
                    "applicability": "conditional_or_unknown",
                },
            ))
        if create_on_input in {"Auto", "Use", "true", "True"}:
            _append_relation(field_chunk, relation(
                target, "can_create_inline", reverse_type="inline_created_from",
                evidence=str(source_path), properties={"field": name},
            ))
    _append_relation(object_chunk, relation(
        field_id, "has_field", reverse_type="field_of", evidence=str(source_path),
        properties={"required": required, "field": name},
    ))
    return field_chunk, target_ids


def _field_is_relevant(field_element: ET.Element) -> bool:
    properties = direct_child(field_element, "Properties")
    name = child_text(properties, "Name").lower().replace("ё", "е")
    required = child_text(properties, "FillChecking") == "ShowError"
    has_reference = bool(_type_references(properties))
    operational = any(part in name for part in OPERATIONAL_FIELD_PARTS)
    return required or has_reference or operational


def _parse_tabular_section(object_chunk: Any, element: ET.Element, source_path: Path,
                           make_chunk: Callable[..., Any],
                           relevant_fields_only: bool = True) -> List[Any]:
    properties = direct_child(element, "Properties")
    name = child_text(properties, "Name")
    if not name:
        return []
    synonym = russian_presentation(direct_child(properties, "Synonym"), name)
    section_id = f"{object_chunk.id}/TabularSections/{name}"
    section = make_chunk(
        id=section_id, title=f"Табличная часть {object_chunk.title}: {synonym}",
        content=f"Табличная часть {synonym} ({name}) объекта {object_chunk.title}",
        path=section_id, layer=3, node_type="tabular_section",
        parent_id=object_chunk.id, level=int(getattr(object_chunk, "level", 2) or 2) + 1,
        metadata={"object_id": object_chunk.id, "name": name, "source_xml": str(source_path)},
    )
    _append_relation(object_chunk, relation(
        section_id, "has_tabular_section", reverse_type="tabular_section_of",
        evidence=str(source_path),
    ))
    result = [section]
    children = direct_child(element, "ChildObjects")
    if children is not None:
        for field_element in list(children):
            kind = local_name(field_element.tag)
            if kind not in FIELD_TAGS:
                continue
            if relevant_fields_only and not _field_is_relevant(field_element):
                continue
            try:
                field_chunk, _ = _field_chunk(
                    section, field_element, kind, source_path, make_chunk, prefix="Fields",
                )
            except ValueError:
                continue
            result.append(field_chunk)
    return result


def enrich_erp_metadata(erp_root: Path, metadata_chunks: Sequence[Any],
                        make_chunk: Callable[..., Any],
                        include_fields: bool = True,
                        include_ui: bool = True,
                        include_forms: bool = True,
                        relevant_fields_only: bool = True,
                        only_objects: Optional[set[str]] = None) -> Tuple[List[Any], Dict[str, int]]:
    """Дополняет существующие L3-объекты точными отношениями из XML."""
    chunk_map = {chunk.id: chunk for chunk in metadata_chunks}
    additions: List[Any] = []
    stats = Counter()
    for object_type, directory in TYPE_TO_DIR.items():
        base = erp_root / directory
        if not base.is_dir() or object_type in {"Subsystem", "CommonCommand", "CommonForm", "CommonModule"}:
            continue
        for source_path in sorted(base.glob("*.xml")):
            object_id = f"ERPcode/{directory}/{source_path.stem}"
            if only_objects and object_id not in only_objects:
                continue
            object_chunk = chunk_map.get(object_id)
            if object_chunk is None:
                continue
            if getattr(object_chunk, "metadata", None) is None:
                object_chunk.metadata = {}
            object_chunk.metadata.update({"metadata_type": object_type, "source_xml": str(source_path)})
            try:
                root = ET.parse(source_path).getroot()
            except (ET.ParseError, OSError) as exc:
                stats["parse_errors"] += 1
                object_chunk.metadata.setdefault("parse_warning", str(exc))
                continue
            metadata_object = _metadata_object(root, object_type)
            if metadata_object is None:
                continue
            properties = direct_child(metadata_object, "Properties")
            if properties is None:
                continue
            stats["objects_enriched"] += 1
            default_forms = []
            for key in ("DefaultObjectForm", "DefaultListForm", "DefaultChoiceForm"):
                value = child_text(properties, key)
                if value:
                    default_forms.append(value)
            object_chunk.metadata["default_forms"] = default_forms
            if object_type == "Document":
                based_on = direct_child(properties, "BasedOn")
                if based_on is not None:
                    for item in list(based_on):
                        target = canonical_metadata_id((item.text or "").strip())
                        if target:
                            _append_relation(object_chunk, relation(
                                target, "can_be_created_on_basis_of", reverse_type="creates_on_basis",
                                evidence=str(source_path),
                            ))
                            stats["based_on"] += 1
                register_records = direct_child(properties, "RegisterRecords")
                if register_records is not None:
                    for item in list(register_records):
                        target = canonical_metadata_id((item.text or "").strip())
                        if target:
                            _append_relation(object_chunk, relation(
                                target, "may_write_register", reverse_type="has_registrator",
                                evidence=str(source_path),
                                properties={"conditional": True, "source": "Document.RegisterRecords"},
                            ))
                            stats["document_register"] += 1
            if not include_fields:
                continue
            children = direct_child(metadata_object, "ChildObjects")
            if children is None:
                continue
            for element in list(children):
                kind = local_name(element.tag)
                if kind in FIELD_TAGS:
                    if relevant_fields_only and not _field_is_relevant(element):
                        continue
                    try:
                        field_chunk, target_ids = _field_chunk(
                            object_chunk, element, kind, source_path, make_chunk,
                        )
                    except ValueError:
                        continue
                    additions.append(field_chunk)
                    stats["fields"] += 1
                    stats["field_type_refs"] += len(target_ids)
                    if field_chunk.metadata.get("required"):
                        stats["required_fields"] += 1
                elif kind == "TabularSection":
                    section_chunks = _parse_tabular_section(
                        object_chunk, element, source_path, make_chunk,
                        relevant_fields_only=relevant_fields_only,
                    )
                    additions.extend(section_chunks)
                    stats["tabular_sections"] += 1
                    stats["fields"] += max(0, len(section_chunks) - 1)
            if include_forms and object_type in {"Catalog", "Document"}:
                additions.extend(_parse_default_forms(
                    erp_root, object_chunk, object_type, source_path.stem,
                    default_forms, chunk_map, make_chunk, stats,
                ))
    if include_ui:
        additions.extend(parse_subsystems_and_commands(erp_root, make_chunk, stats))
    return additions, dict(stats)


def _parse_default_forms(erp_root: Path, object_chunk: Any, object_type: str,
                         object_name: str, default_forms: Sequence[str],
                         chunk_map: Dict[str, Any], make_chunk: Callable[..., Any],
                         stats: Counter) -> List[Any]:
    directory = TYPE_TO_DIR[object_type]
    additions: List[Any] = []
    seen_forms: set[str] = set()
    for reference in default_forms:
        match = re.match(r"[^.]+\.([^.]+)\.Form\.([^.]+)", reference)
        if not match:
            continue
        form_name = match.group(2)
        if form_name in seen_forms:
            continue
        seen_forms.add(form_name)
        form_path = erp_root / directory / object_name / "Forms" / form_name / "Ext" / "Form.xml"
        if not form_path.is_file():
            continue
        # Очень большие формы всё равно разбираются потоковым iterparse ниже.
        form_id = f"ui:form:{object_type}.{object_name}.{form_name}"
        form_chunk = make_chunk(
            id=form_id, title=f"Форма {form_name}: {object_chunk.title}",
            content=f"Форма {form_name} объекта {object_chunk.title}",
            path=str(form_path), layer=3, node_type="ui_form", level=3,
            metadata={"object_id": object_chunk.id, "form_name": form_name, "source_xml": str(form_path)},
        )
        _append_relation(object_chunk, relation(
            form_id, "has_form", reverse_type="form_of", evidence=str(form_path),
        ))
        additions.append(form_chunk)
        stats["forms"] += 1
        try:
            iterator = ET.iterparse(form_path, events=("end",))
            for _, element in iterator:
                if local_name(element.tag) != "InputField":
                    continue
                data_path = child_text(element, "DataPath")
                if not data_path.startswith("Объект."):
                    element.clear()
                    continue
                field_name = data_path.split(".", 1)[1].split(".", 1)[0]
                metadata_field_id = f"{object_chunk.id}/Fields/{field_name}"
                form_field_name = element.get("name", field_name)
                form_field_id = f"{form_id}/Fields/{form_field_name}"
                create_button = child_text(element, "CreateButton") or "Auto"
                form_field = make_chunk(
                    id=form_field_id,
                    title=f"Поле формы {form_field_name}: {object_chunk.title}",
                    content=f"Поле формы {form_field_name}; путь данных {data_path}; кнопка создания {create_button}",
                    path=f"{form_path}#{form_field_name}", layer=3,
                    node_type="ui_form_field", parent_id=form_id, level=4,
                    metadata={
                        "form_id": form_id, "data_path": data_path,
                        "create_button": create_button, "source_xml": str(form_path),
                    },
                )
                _append_relation(form_chunk, relation(
                    form_field_id, "form_contains_field", reverse_type="form_field_of",
                    evidence=str(form_path),
                ))
                if metadata_field_id in chunk_map or metadata_field_id.startswith(object_chunk.id):
                    _append_relation(form_field, relation(
                        metadata_field_id, "shown_on_form", reverse_type="shown_as_form_field",
                        evidence=str(form_path), properties={"create_button": create_button},
                    ))
                additions.append(form_field)
                stats["form_fields"] += 1
                element.clear()
        except (ET.ParseError, OSError):
            stats["form_parse_errors"] += 1
    return additions


def _subsystem_names(relative_path: Path) -> List[str]:
    names: List[str] = []
    for part in relative_path.parts:
        if part == "Subsystems":
            continue
        name = Path(part).stem if part.endswith(".xml") else part
        if name not in {"Ext", "Forms"}:
            names.append(name)
    return names


def _command_target(command_name: str) -> Optional[str]:
    match = re.match(r"([A-Za-z]+)\.([^.]+)\.", command_name)
    if not match:
        return None
    return canonical_metadata_id(f"{match.group(1)}.{match.group(2)}")


def _parse_command_interface(path: Path) -> Dict[str, Dict[str, Any]]:
    commands: Dict[str, Dict[str, Any]] = {}
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return commands
    for section in list(root):
        section_name = local_name(section.tag)
        if section_name not in {"CommandsVisibility", "CommandsPlacement", "CommandsOrder"}:
            continue
        for command in list(section):
            if local_name(command.tag) != "Command":
                continue
            name = command.get("name", "")
            if not name:
                continue
            data = commands.setdefault(name, {"name": name, "groups": [], "roles": []})
            group = child_text(command, "CommandGroup")
            if group and group not in data["groups"]:
                data["groups"].append(group)
            visibility = direct_child(command, "Visibility")
            if visibility is not None:
                for value in list(visibility):
                    if local_name(value.tag) == "Value" and value.get("name"):
                        data["roles"].append(value.get("name"))
                    elif local_name(value.tag) == "Common":
                        data["common_visibility"] = (value.text or "").strip().lower() == "true"
    return commands


def parse_subsystems_and_commands(erp_root: Path, make_chunk: Callable[..., Any],
                                  stats: Optional[Counter] = None) -> List[Any]:
    stats = stats if stats is not None else Counter()
    base = erp_root / "Subsystems"
    additions: List[Any] = []
    if not base.is_dir():
        return additions
    for source_path in sorted(base.rglob("*.xml")):
        if "Ext" in source_path.parts or source_path.parent.name != "Subsystems":
            continue
        relative = source_path.relative_to(base)
        names = _subsystem_names(relative)
        if not names:
            continue
        subsystem_id = "ui:subsystem:" + "/".join(names)
        parent_id = "ui:subsystem:" + "/".join(names[:-1]) if len(names) > 1 else None
        try:
            root = ET.parse(source_path).getroot()
        except (ET.ParseError, OSError):
            stats["subsystem_parse_errors"] += 1
            continue
        subsystem = _metadata_object(root, "Subsystem")
        if subsystem is None:
            continue
        properties = direct_child(subsystem, "Properties")
        synonym = russian_presentation(direct_child(properties, "Synonym"), names[-1])
        include = child_text(properties, "IncludeInCommandInterface")
        chunk = make_chunk(
            id=subsystem_id, title=f"Подсистема {synonym}",
            content=f"Подсистема: {' → '.join(names)}; в командном интерфейсе: {include or 'Auto'}",
            path=str(source_path), layer=3, node_type="ui_subsystem",
            parent_id=parent_id, level=len(names),
            metadata={
                "subsystem_path": names, "include_in_command_interface": include,
                "source_xml": str(source_path),
            },
        )
        content = direct_child(properties, "Content")
        if content is not None:
            for item in list(content):
                target = canonical_metadata_id((item.text or "").strip())
                if target:
                    _append_relation(chunk, relation(
                        target, "contains_object", reverse_type="in_subsystem",
                        evidence=str(source_path),
                    ))
                    stats["subsystem_objects"] += 1
        additions.append(chunk)
        stats["subsystems"] += 1
        command_path = source_path.with_suffix("") / "Ext" / "CommandInterface.xml"
        if not command_path.is_file():
            continue
        for command_name, command_data in _parse_command_interface(command_path).items():
            target = _command_target(command_name)
            command_hash = hashlib.sha1(command_name.encode("utf-8")).hexdigest()[:12]
            command_id = f"ui:command:{'/'.join(names)}:{command_hash}"
            group = command_data.get("groups", [])
            ui_path = " → ".join(names + [target.rsplit("/", 1)[-1] if target else command_name])
            command_chunk = make_chunk(
                id=command_id, title=f"Команда {command_name}",
                content=f"Команда: {command_name}\nПуть: {ui_path}\nГруппы: {', '.join(group)}",
                path=f"{command_path}#{command_name}", layer=3,
                node_type="ui_command", parent_id=subsystem_id,
                level=len(names) + 1,
                metadata={
                    **command_data, "subsystem_id": subsystem_id,
                    "ui_path": ui_path, "source_xml": str(command_path),
                },
            )
            _append_relation(chunk, relation(
                command_id, "has_command", reverse_type="command_of_subsystem",
                evidence=str(command_path),
            ))
            if target:
                _append_relation(command_chunk, relation(
                    target, "opens_object", reverse_type="opened_by_command",
                    evidence=str(command_path), properties={"ui_path": ui_path, "groups": group},
                ))
                stats["commands_with_target"] += 1
            additions.append(command_chunk)
            stats["commands"] += 1
    return additions


def iter_edges_with_relation(graph: nx.Graph, node: str,
                             direction: str = "out") -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    if node not in graph:
        return
    if graph.is_multigraph():
        if direction == "out":
            iterator = graph.out_edges(node, keys=True, data=True)  # type: ignore[attr-defined]
            for source, target, _, data in iterator:
                yield source, target, data
        else:
            iterator = graph.in_edges(node, keys=True, data=True)  # type: ignore[attr-defined]
            for source, target, _, data in iterator:
                yield source, target, data
    else:
        iterator = graph.out_edges(node, data=True) if direction == "out" else graph.in_edges(node, data=True)  # type: ignore[attr-defined]
        for source, target, data in iterator:
            yield source, target, data


def resolve_graph_node(graph: nx.Graph, query: str,
                       allowed_types: Optional[set[str]] = None) -> Optional[str]:
    if query in graph:
        return query
    normalized = re.sub(r"\W+", "", query.lower(), flags=re.UNICODE)
    title_exact: List[str] = []
    basename_exact: List[str] = []
    partial: List[Tuple[int, str]] = []
    for node, data in graph.nodes(data=True):
        if allowed_types and data.get("node_type") not in allowed_types:
            continue
        title = str(data.get("title", ""))
        candidate = re.sub(r"\W+", "", title.lower(), flags=re.UNICODE)
        presentation = title
        for prefix in (
            "Документ ", "Справочник ", "Перечисление ",
            "Регистр накопления ", "Регистр сведений ",
            "Регистр бухгалтерии ", "Регистр расчёта ", "Регистр расчета ",
        ):
            if presentation.lower().startswith(prefix.lower()):
                presentation = presentation[len(prefix):]
                break
        presentation_candidate = re.sub(
            r"\W+", "", presentation.lower(), flags=re.UNICODE,
        )
        basename = re.sub(r"\W+", "", str(node).rsplit("/", 1)[-1].lower(), flags=re.UNICODE)
        if normalized in {candidate, presentation_candidate}:
            title_exact.append(node)
        elif normalized == basename:
            basename_exact.append(node)
        elif normalized and (normalized in candidate or normalized in basename):
            partial.append((abs(len(candidate) - len(normalized)), node))
    # Синоним текущего объекта приоритетнее технического Name старой версии.
    # Например, запрос без суффикса версии должен выбрать объект, у которого
    # синоним тоже не содержит «(2.1)».
    if title_exact:
        return sorted(title_exact, key=str)[0]
    if basename_exact:
        return sorted(basename_exact, key=str)[0]
    if partial:
        partial.sort(key=lambda item: (item[0], str(item[1])))
        return partial[0][1]
    return None


class DependencyPlanner:
    RELATIONS = {"requires", "required_reference"}

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def plan(self, target: str, max_depth: int = 5,
             include_optional: bool = False) -> Dict[str, Any]:
        target_id = resolve_graph_node(self.graph, target, allowed_types={"metadata"})
        if not target_id:
            return {"error": f"Объект не найден: {target}", "target": target}
        visited_path: set[str] = set()

        def build(node: str, depth: int) -> Dict[str, Any]:
            data = self.graph.nodes[node]
            result: Dict[str, Any] = {
                "node_id": node, "title": data.get("title", node),
                "node_type": data.get("node_type"), "dependencies": [],
            }
            if depth >= max_depth:
                result["truncated"] = True
                return result
            if node in visited_path:
                result["cycle"] = True
                return result
            visited_path.add(node)
            dependencies: Dict[str, Dict[str, Any]] = {}
            for _, child, edge in iter_edges_with_relation(self.graph, node, "out"):
                relation_type = edge.get("relation")
                if relation_type not in self.RELATIONS and not (
                    include_optional and relation_type == "field_type"
                ):
                    continue
                properties = edge.get("properties", {})
                if isinstance(properties, str):
                    try:
                        properties = json.loads(properties)
                    except json.JSONDecodeError:
                        properties = {}
                dep = dependencies.setdefault(child, {
                    "node_id": child, "relation": relation_type,
                    "field": properties.get("field"),
                    "field_id": properties.get("field_id"),
                    "create_on_input": properties.get("create_on_input"),
                    "required_by_metadata": properties.get(
                        "required_by_metadata", relation_type in self.RELATIONS,
                    ),
                    "applicability": properties.get(
                        "applicability", "conditional_or_unknown",
                    ),
                    "evidence": edge.get("evidence", ""),
                })
                dep["required"] = relation_type in self.RELATIONS
            for child, dep in sorted(dependencies.items(), key=lambda item: str(item[0])):
                dep["inline"] = dep.get("create_on_input") in {"Auto", "Use", "true", True}
                # CreateOnInput=Auto/Use — это кандидат на создание из поля,
                # а не гарантия: итог зависит от формы и прав пользователя.
                dep["inline_candidate"] = dep["inline"]
                dep["inline_guaranteed"] = False
                dep["classification"] = (
                    "metadata-required-inline-candidate"
                    if dep["required_by_metadata"] and dep["inline_candidate"]
                    else "metadata-required-precreate-candidate"
                    if dep["required_by_metadata"]
                    else "optional-reference"
                )
                dep["dependency"] = build(child, depth + 1)
                result["dependencies"].append(dep)
            visited_path.remove(node)
            return result

        return {"target": target_id, "max_depth": max_depth, "tree": build(target_id, 0)}


class DocumentChainPlanner:
    FORWARD_RELATIONS = {"creates_on_basis", "precedes"}

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def _neighbors(self, node: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        for _, target, edge in iter_edges_with_relation(self.graph, node, "out"):
            if edge.get("relation") in self.FORWARD_RELATIONS:
                yield target, edge
        # Документ B с обязательной ссылкой на документ A процессно следует
        # после A, даже если B создаётся фоновым механизмом и отсутствует в
        # коллекции BasedOn.
        if "/Documents/" in str(node):
            for source, _, edge in iter_edges_with_relation(self.graph, node, "in"):
                if edge.get("relation") != "requires":
                    continue
                source_data = self.graph.nodes[source]
                if source_data.get("node_type") != "metadata" or "/Documents/" not in str(source):
                    continue
                inferred = dict(edge)
                inferred["relation"] = "precedes_via_required_document_reference"
                yield source, inferred

    def plan(self, start: str, end: Optional[str] = None,
             max_depth: int = 10) -> Dict[str, Any]:
        allowed = {"metadata", "scenario"}
        start_id = resolve_graph_node(self.graph, start, allowed_types=allowed)
        end_id = resolve_graph_node(self.graph, end, allowed_types=allowed) if end else None
        if not start_id:
            return {"error": f"Начальный документ не найден: {start}"}
        queue = deque([(start_id, [start_id], [])])
        seen_depth = {start_id: 0}
        found_paths: List[Tuple[List[str], List[Dict[str, Any]]]] = []
        while queue:
            node, path, edges = queue.popleft()
            if end_id and node == end_id:
                found_paths.append((path, edges))
                break
            if len(path) - 1 >= max_depth:
                if not end_id:
                    found_paths.append((path, edges))
                continue
            expanded = False
            for target, edge in self._neighbors(node):
                if target in path:
                    continue
                expanded = True
                depth = len(path)
                if depth <= seen_depth.get(target, 10**9):
                    seen_depth[target] = depth
                    queue.append((target, path + [target], edges + [edge]))
            if not end_id and not expanded:
                found_paths.append((path, edges))
            if len(found_paths) >= 20:
                break
        if end_id and not found_paths:
            return {"error": f"Цепочка не найдена: {start_id} → {end_id}", "start": start_id, "end": end_id}
        rendered = [self._render_path(path, edges) for path, edges in found_paths]
        rendered.sort(key=lambda item: (len(item["steps"]), [step["node_id"] for step in item["steps"]]))
        return {"start": start_id, "end": end_id, "paths": rendered}

    def _render_path(self, path: Sequence[str], edges: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        steps = []
        for index, node in enumerate(path):
            data = self.graph.nodes[node]
            registers = []
            ui_paths = []
            for _, target, edge in iter_edges_with_relation(self.graph, node, "out"):
                rel = edge.get("relation")
                if rel == "may_write_register":
                    registers.append({
                        "node_id": target, "title": self.graph.nodes[target].get("title", target),
                        "conditional": True,
                    })
                elif rel == "opened_by_command":
                    command_data = self.graph.nodes[target]
                    command_metadata = command_data.get("metadata", {})
                    if isinstance(command_metadata, str):
                        try:
                            command_metadata = json.loads(command_metadata)
                        except json.JSONDecodeError:
                            command_metadata = {}
                    ui_path = command_metadata.get("ui_path") if isinstance(command_metadata, dict) else None
                    ui_paths.append(ui_path or command_data.get("title", target))
            step = {
                "order": index + 1, "node_id": node,
                "title": data.get("title", node), "registers": registers,
                "ui_paths": sorted(set(ui_paths)),
            }
            if index > 0:
                step["incoming_relation"] = edges[index - 1].get("relation")
                step["evidence"] = edges[index - 1].get("evidence", "")
            steps.append(step)
        return {"steps": steps}


class EndToEndProcessPlanner:
    """Собирает сквозной офлайн-план из Task Graph и подтверждённых ERP ID."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.dependencies = DependencyPlanner(graph)
        self.documents = DocumentChainPlanner(graph)

    def plan(self, task_graph: nx.Graph, normalization: Dict[str, Any],
             dependency_depth: int = 3) -> Dict[str, Any]:
        mappings = []
        for mapping in normalization.get("mappings", []):
            task_node = mapping.get("task_node_id")
            erp_node = mapping.get("erp_node_id")
            if task_node not in task_graph or erp_node not in self.graph:
                continue
            task_data = task_graph.nodes[task_node]
            mappings.append({
                **mapping,
                "source_index": int(task_data.get("source_index", 10**9) or 10**9),
                "task_node_type": task_data.get("node_type"),
            })
        mappings.sort(key=lambda item: (
            item["source_index"], -float(item.get("confidence", 0)), str(item.get("erp_node_id")),
        ))

        # Для каждого фрагмента оставляем лучший кандидат, затем сохраняем
        # порядок появления требований в исходном ТЗ.
        best_by_task: Dict[str, Dict[str, Any]] = {}
        for mapping in mappings:
            current = best_by_task.get(mapping["task_node_id"])
            if current is None or float(mapping.get("confidence", 0)) > float(current.get("confidence", 0)):
                best_by_task[mapping["task_node_id"]] = mapping
        ordered = sorted(best_by_task.values(), key=lambda item: item["source_index"])
        document_ids: List[str] = []
        object_ids: List[str] = []
        candidate_object_ids: List[str] = []
        for mapping in ordered:
            erp_id = mapping["erp_node_id"]
            status = str(mapping.get("status", "confirmed"))
            confirmed = status not in {"candidate", "proposed", "high_confidence_candidate"}
            if confirmed:
                if erp_id not in object_ids:
                    object_ids.append(erp_id)
                if "/Documents/" in erp_id and erp_id not in document_ids:
                    document_ids.append(erp_id)
            elif erp_id not in candidate_object_ids:
                candidate_object_ids.append(erp_id)

        operational_steps = []
        for index, document_id in enumerate(document_ids):
            rendered = self.documents._render_path([document_id], [])["steps"][0]
            rendered["source_mappings"] = [
                mapping for mapping in ordered if mapping["erp_node_id"] == document_id
            ]
            operational_steps.append(rendered)

        chains = []
        chain_gaps = []
        for left, right in zip(document_ids, document_ids[1:]):
            chain = self.documents.plan(left, right, max_depth=8)
            if chain.get("error"):
                chain_gaps.append({"from": left, "to": right, "reason": chain["error"]})
            else:
                chains.append({"from": left, "to": right, "paths": chain.get("paths", [])[:3]})

        dependency_plans = []
        # НСИ и документы получают отдельный «точно вовремя» план зависимостей.
        for object_id in object_ids[:100]:
            plan = self.dependencies.plan(object_id, max_depth=dependency_depth)
            tree = plan.get("tree", {})
            if tree.get("dependencies"):
                dependency_plans.append(plan)

        return {
            "task_nodes_mapped": len(best_by_task),
            "erp_objects": object_ids,
            "candidate_erp_objects": candidate_object_ids,
            "documents_in_requirement_order": document_ids,
            "operational_steps": operational_steps,
            "document_chains": chains,
            "chain_gaps": chain_gaps,
            "dependency_plans": dependency_plans,
            "mapping_gaps": normalization.get("mapping_gaps", []),
            "ready": not chain_gaps and not normalization.get("mapping_gaps"),
        }


def render_offline_instruction(task_summary: Dict[str, Any], process: Dict[str, Any],
                               questions: Sequence[Any]) -> str:
    """Рендерит проверяемый черновик инструкции без LLM."""
    lines = [
        f"# Инструкция по задаче {task_summary.get('task_id', '')}",
        "",
        f"Источник ТЗ: {task_summary.get('source_path', '')}",
        "",
    ]
    blocking = [question for question in questions if getattr(question, "blocking", False)]
    if blocking:
        lines.extend([
            "## Блокирующие уточнения",
            "",
            "До фиксации этих решений инструкция остаётся черновиком:",
            "",
        ])
        for number, question in enumerate(blocking, 1):
            lines.append(f"{number}. {getattr(question, 'text', str(question))}")
        lines.append("")

    dependencies = process.get("dependency_plans", [])
    if dependencies:
        lines.extend(["## Предварительная нормативно-справочная информация", ""])

        def emit_tree(tree: Dict[str, Any], indent: int = 0) -> None:
            for item in tree.get("dependencies", []):
                dependency = item.get("dependency", {})
                title = dependency.get("title", dependency.get("node_id", ""))
                field = f"; поле: {item.get('field')}" if item.get("field") else ""
                if item.get("inline_candidate"):
                    creation = "проверить создание из поля (зависит от формы и прав)"
                else:
                    creation = "кандидат на предварительное создание"
                applicability = (
                    "; обязательность задана метаданными, применимость проверить по операции и настройкам"
                    if item.get("required_by_metadata") else ""
                )
                lines.append(
                    f"{'  ' * indent}- {title}{field}; {creation}{applicability}"
                )
                emit_tree(dependency, indent + 1)

        for plan in dependencies:
            tree = plan.get("tree", {})
            lines.append(f"### {tree.get('title', plan.get('target', 'Объект'))}")
            lines.append("")
            emit_tree(tree)
            lines.append("")

    steps = process.get("operational_steps", [])
    lines.extend(["## Операционная цепочка документов", ""])
    if not steps:
        lines.append("Документы ERP пока не определены — требуется ответить на вопросы сопоставления.")
        lines.append("")
    for step in steps:
        lines.append(f"### Шаг {step.get('order')}. {step.get('title')}")
        lines.append("")
        ui_paths = step.get("ui_paths", [])
        if ui_paths:
            lines.append("Где открыть:")
            for ui_path in ui_paths:
                lines.append(f"- {ui_path}")
        else:
            lines.append("Где открыть: путь в CommandInterface не найден; не подставлять предположение.")
        source_mappings = step.get("source_mappings", [])
        if source_mappings:
            lines.append("Основание из ТЗ:")
            for mapping in source_mappings[:5]:
                lines.append(f"- {mapping.get('source_text', '')}")
        registers = step.get("registers", [])
        if registers:
            lines.append("Проверка после проведения (движения могут зависеть от операции и статуса):")
            for register in registers:
                lines.append(f"- {register.get('title', register.get('node_id'))}")
        lines.append("")

    if process.get("document_chains"):
        lines.extend(["## Подтверждённые переходы документов", ""])
        for chain in process["document_chains"]:
            paths = chain.get("paths", [])
            if not paths:
                continue
            titles = [step.get("title", "") for step in paths[0].get("steps", [])]
            lines.append("- " + " → ".join(titles))
        lines.append("")
    if process.get("chain_gaps"):
        lines.extend(["## Неподтверждённые переходы", ""])
        for gap in process["chain_gaps"]:
            lines.append(f"- {gap.get('from')} → {gap.get('to')}: {gap.get('reason')}")
        lines.append("")
    lines.extend([
        "## Правило достоверности",
        "",
        "В черновик включены только объекты и связи, существующие в статическом графе ERP. "
        "Отсутствующие пути и переходы помечены как пробелы, а не дополнены предположениями.",
        "",
    ])
    return "\n".join(lines)


def graph_relation_counts(graph: nx.Graph) -> Counter:
    if graph.is_multigraph():
        return Counter(data.get("relation", "unknown") for *_, data in graph.edges(keys=True, data=True))
    return Counter(data.get("relation", "unknown") for *_, data in graph.edges(data=True))


def validate_graph(chunks: Sequence[Any], graph: nx.Graph,
                   vectors: Any = None, node_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    ids = [chunk.id for chunk in chunks]
    duplicates = sorted(node for node, count in Counter(ids).items() if count > 1)
    relations = graph_relation_counts(graph)
    isolated_by_layer = Counter()
    for node in nx.isolates(graph):
        isolated_by_layer[int(graph.nodes[node].get("layer", 4) or 4)] += 1
    missing_relation_targets = []
    for chunk in chunks:
        for item in getattr(chunk, "relations", []) or []:
            if item.get("target") not in graph:
                missing_relation_targets.append({
                    "source": chunk.id, "target": item.get("target"), "relation": item.get("type"),
                })
    vector_rows = int(vectors.shape[0]) if vectors is not None and hasattr(vectors, "shape") else None
    searchable_ids = len(node_ids) if node_ids is not None else None
    errors = []
    if duplicates:
        errors.append(f"Дубли ID: {len(duplicates)}")
    if node_ids is not None and vector_rows != searchable_ids:
        errors.append(f"Строк векторов {vector_rows} != node_ids {searchable_ids}")
    if node_ids is not None and len(set(node_ids)) != len(node_ids):
        errors.append(f"Дубли в node_ids: {len(node_ids) - len(set(node_ids))}")
    return {
        "ok": not errors,
        "errors": errors,
        "chunks": len(chunks), "unique_chunk_ids": len(set(ids)),
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "vector_rows": vector_rows, "node_ids": searchable_ids,
        "duplicate_ids": duplicates[:100],
        "missing_relation_targets": len(missing_relation_targets),
        "missing_relation_target_samples": missing_relation_targets[:100],
        "isolated_by_layer": dict(sorted(isolated_by_layer.items())),
        "relations": dict(relations.most_common()),
    }


def ensure_unique_chunks(chunks: Sequence[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Детерминированно устраняет коллизии ID до векторизации.

    Полные дубли удаляются, различные узлы получают суффикс по пути/содержимому.
    """
    result: List[Any] = []
    by_id: Dict[str, Any] = {}
    renamed: Dict[str, List[str]] = {}
    dropped_identical = 0
    for chunk in chunks:
        existing = by_id.get(chunk.id)
        if existing is None:
            by_id[chunk.id] = chunk
            result.append(chunk)
            continue
        signature = (chunk.title, chunk.path, chunk.content, chunk.layer, chunk.node_type)
        existing_signature = (existing.title, existing.path, existing.content, existing.layer, existing.node_type)
        if signature == existing_signature:
            dropped_identical += 1
            continue
        old_id = chunk.id
        digest = hashlib.sha1(
            f"{chunk.layer}|{chunk.node_type}|{chunk.path}|{chunk.title}|{chunk.content[:500]}".encode("utf-8")
        ).hexdigest()[:12]
        candidate = f"{old_id}#{digest}"
        counter = 2
        while candidate in by_id:
            candidate = f"{old_id}#{digest}-{counter}"
            counter += 1
        chunk.id = candidate
        # Родитель остаётся исходным узлом; собственные дочерние/relations уже
        # используют стабильные канонические ID целей.
        renamed.setdefault(old_id, []).append(candidate)
        by_id[candidate] = chunk
        result.append(chunk)
    return result, {
        "input": len(chunks), "output": len(result),
        "dropped_identical": dropped_identical,
        "renamed_collisions": sum(len(values) for values in renamed.values()),
        "renamed_samples": dict(list(renamed.items())[:20]),
    }
