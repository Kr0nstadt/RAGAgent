import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import workspace_task as wt


class WorkspaceTaskTests(unittest.TestCase):
    def test_slug_and_task_id_reject_path_traversal(self):
        self.assertEqual(wt.slug("Новое ТЗ / ERP"), "Новое_ТЗ_ERP")
        with self.assertRaises(ValueError):
            wt.task_dir("../outside")

    def test_start_is_idempotent_by_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source = temp_path / "requirements.md"
            source.write_text("Заказ клиента", encoding="utf-8")
            root = temp_path / "task_data"

            def fake_engine(*args):
                task_id = args[args.index("--task-id") + 1]
                directory = root / task_id
                directory.mkdir(parents=True)
                wt.atomic_json(directory / "task_graph.json", {
                    "task_id": task_id,
                    "source": {"sha256": wt.sha256(source), "source_path": str(source)},
                    "nodes": [], "edges": [],
                })
                return {"task_id": task_id}

            with mock.patch.object(wt, "TASK_ROOT", root), mock.patch.object(wt, "run_engine", fake_engine):
                first = wt.cmd_start(str(source), None)
                second = wt.cmd_start(str(source), None)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["task_id"], second["task_id"])

    def test_finalize_requires_approved_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "t1"
            directory.mkdir()
            wt.atomic_json(directory / "task_graph.json", {"task_id": "t1", "source": {}})
            (directory / "instruction_candidate.md").write_text("draft", encoding="utf-8")
            wt.atomic_json(directory / "verification.json", {
                "approved": False, "defects": [{"severity": "blocking"}],
            })
            with mock.patch.object(wt, "TASK_ROOT", root):
                with self.assertRaises(RuntimeError):
                    wt.cmd_finalize("t1")
                wt.atomic_json(directory / "verification.json", {
                    "approved": True, "checked_claims": 3, "defects": [],
                })
                result = wt.cmd_finalize("t1")
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual((directory / "final_instruction.md").read_text(encoding="utf-8"), "draft")

    def test_agent_files_do_not_contain_literal_api_key(self):
        project = Path(__file__).resolve().parents[1]
        for path in (project / ".pi").rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", text))
                self.assertNotIn("DEEPSEEK_API_KEY=", text)


if __name__ == "__main__":
    unittest.main()
