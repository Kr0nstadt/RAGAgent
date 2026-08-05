#!/usr/bin/env python3
"""Persistent, API-free coordinator state for the Herdr/Pi ERP workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT / "task_data"
ENGINE = ROOT / "graph_rag_1c_erp.py"
STATES = {
    "NEW", "INGESTED", "NEEDS_CLARIFICATION", "READY_TO_PLAN",
    "PLANNED", "DRAFTED", "VERIFIED", "COMPLETE",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str, limit: int = 48) -> str:
    value = re.sub(r"[^\w-]+", "_", value.strip(), flags=re.UNICODE).strip("_")
    return value[:limit] or "task"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temp_name)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def run_engine(*args: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(ENGINE), *args], cwd=ROOT,
        text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}")
    output = proc.stdout.strip()
    start = output.find("{")
    if start < 0:
        raise RuntimeError(f"Команда не вернула JSON: {output[-1000:]}")
    try:
        return json.loads(output[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Некорректный JSON команды: {output[-2000:]}") from exc


def task_dir(task_id: str) -> Path:
    if not re.fullmatch(r"[\w.-]+", task_id, flags=re.UNICODE):
        raise ValueError("Недопустимый task ID")
    return TASK_ROOT / task_id


def load_state(task_id: str) -> tuple[Path, dict[str, Any]]:
    directory = task_dir(task_id)
    path = directory / "workspace_state.json"
    if path.is_file():
        return directory, json.loads(path.read_text(encoding="utf-8"))
    graph_path = directory / "task_graph.json"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Не найден Task Graph: {graph_path}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    state = {
        "schema_version": 1, "task_id": task_id, "status": "INGESTED",
        "source": graph.get("source", {}), "question_round": 0,
        "artifacts": {"task_graph": str(graph_path)}, "created_at": now(), "updated_at": now(),
    }
    atomic_json(path, state)
    return directory, state


def save_state(directory: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    atomic_json(directory / "workspace_state.json", state)


def cmd_start(path_text: str, explicit_id: str | None) -> dict[str, Any]:
    source = Path(path_text).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Файл ТЗ не найден: {source}")
    digest = sha256(source)
    task_id = explicit_id or f"{slug(source.stem)}-{digest[:10]}"
    directory = task_dir(task_id)
    graph_path = directory / "task_graph.json"
    reused = False
    if graph_path.is_file():
        existing = json.loads(graph_path.read_text(encoding="utf-8"))
        old_hash = existing.get("source", {}).get("sha256")
        if old_hash != digest:
            raise RuntimeError(
                f"Task ID {task_id} уже относится к другому содержимому. "
                "Укажите новый --task-id; существующий Task Graph не перезаписан."
            )
        reused = True
    else:
        run_engine("ingest-tz", str(source), "--task-id", task_id)
    directory, state = load_state(task_id)
    state["source"] = {"path": str(source), "sha256": digest}
    state["status"] = state.get("status", "INGESTED") if reused else "INGESTED"
    state.setdefault("artifacts", {})["task_graph"] = str(graph_path)
    save_state(directory, state)
    return {"task_id": task_id, "status": state["status"], "reused": reused,
            "state_file": str(directory / "workspace_state.json")}


def cmd_prepare(task_id: str, limit: int) -> dict[str, Any]:
    directory, state = load_state(task_id)
    mapping = run_engine("normalize-tz", task_id)
    questions = run_engine("task-questions", task_id, "--limit", str(limit))
    atomic_json(directory / "questions.json", questions)
    blocking = [q for q in questions.get("questions", []) if q.get("blocking")]
    state["question_round"] = int(state.get("question_round", 0)) + 1
    state["status"] = "NEEDS_CLARIFICATION" if blocking else "READY_TO_PLAN"
    state.setdefault("artifacts", {}).update({
        "erp_mapping": str(directory / "erp_mapping.json"),
        "questions": str(directory / "questions.json"),
    })
    save_state(directory, state)
    return {"task_id": task_id, "status": state["status"],
            "round": state["question_round"], "returned": questions.get("returned", 0),
            "candidate_count": questions.get("candidate_count", 0),
            "blocking_returned": len(blocking), "mapping": mapping}


def cmd_answer(task_id: str, question_id: str, answer: str) -> dict[str, Any]:
    directory, state = load_state(task_id)
    result = run_engine("answer-task", task_id, question_id, answer)
    state["status"] = "INGESTED"
    state.setdefault("artifacts", {})["answers"] = str(directory / "answers.json")
    save_state(directory, state)
    return {**result, "status": state["status"]}


def cmd_plan(task_id: str, force: bool) -> dict[str, Any]:
    directory, state = load_state(task_id)
    questions = run_engine("task-questions", task_id, "--all")
    blocking = [q for q in questions.get("questions", []) if q.get("blocking")]
    if blocking and not force:
        raise RuntimeError(
            f"Осталось блокирующих вопросов: {len(blocking)}. "
            "Ответьте на них или явно используйте --force, зафиксировав допущения."
        )
    result = run_engine("plan-task", task_id)
    state["status"] = "PLANNED"
    state["forced_with_blocking_questions"] = len(blocking) if force else 0
    state.setdefault("artifacts", {}).update({
        "process_plan": str(directory / "process_plan.json"),
        "offline_draft": str(directory / "instruction_draft.md"),
    })
    save_state(directory, state)
    return {**result, "status": state["status"], "blocking_questions": len(blocking)}


def cmd_context(task_id: str) -> dict[str, Any]:
    directory, state = load_state(task_id)
    names = ["task_graph.json", "answers.json", "erp_mapping.json", "questions.json", "process_plan.json"]
    context: dict[str, Any] = {"task_id": task_id, "workflow": state, "artifacts": {}}
    for name in names:
        path = directory / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "task_graph.json":
            payload = {**payload, "nodes": payload.get("nodes", [])[:300], "edges": payload.get("edges", [])[:600]}
        elif name == "erp_mapping.json":
            payload = {**payload, "mappings": payload.get("mappings", [])[:200],
                       "mapping_gaps": payload.get("mapping_gaps", [])[:100]}
        context["artifacts"][name] = payload
    target = directory / "agent_context.json"
    atomic_json(target, context)
    state.setdefault("artifacts", {})["agent_context"] = str(target)
    save_state(directory, state)
    return {"task_id": task_id, "saved_to": str(target), "included": list(context["artifacts"])}


def cmd_finalize(task_id: str) -> dict[str, Any]:
    directory, state = load_state(task_id)
    draft = directory / "instruction_candidate.md"
    verification_path = directory / "verification.json"
    if not draft.is_file():
        raise FileNotFoundError(f"Нет кандидатной инструкции: {draft}")
    if not verification_path.is_file():
        raise FileNotFoundError(f"Нет результата проверки: {verification_path}")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("approved") is not True:
        defects = verification.get("defects", [])
        raise RuntimeError(f"Проверка не пройдена; дефектов: {len(defects)}")
    target = directory / "final_instruction.md"
    atomic_copy(draft, target)
    state["status"] = "COMPLETE"
    state.setdefault("artifacts", {}).update({
        "candidate": str(draft), "verification": str(verification_path), "final": str(target),
    })
    save_state(directory, state)
    return {"task_id": task_id, "status": "COMPLETE", "saved_to": str(target),
            "checked_claims": verification.get("checked_claims", 0)}


def cmd_status(task_id: str) -> dict[str, Any]:
    directory, state = load_state(task_id)
    state["existing_files"] = sorted(p.name for p in directory.iterdir() if p.is_file())
    return state


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start"); start.add_argument("file"); start.add_argument("--task-id")
    prepare = sub.add_parser("prepare"); prepare.add_argument("task_id"); prepare.add_argument("--limit", type=int, default=10)
    answer = sub.add_parser("answer"); answer.add_argument("task_id"); answer.add_argument("question_id"); answer.add_argument("answer")
    plan = sub.add_parser("plan"); plan.add_argument("task_id"); plan.add_argument("--force", action="store_true")
    context = sub.add_parser("context"); context.add_argument("task_id")
    finalize = sub.add_parser("finalize"); finalize.add_argument("task_id")
    status = sub.add_parser("status"); status.add_argument("task_id")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "start": result = cmd_start(args.file, args.task_id)
        elif args.command == "prepare": result = cmd_prepare(args.task_id, args.limit)
        elif args.command == "answer": result = cmd_answer(args.task_id, args.question_id, args.answer)
        elif args.command == "plan": result = cmd_plan(args.task_id, args.force)
        elif args.command == "context": result = cmd_context(args.task_id)
        elif args.command == "finalize": result = cmd_finalize(args.task_id)
        else: result = cmd_status(args.task_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "command": args.command}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
