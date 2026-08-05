import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse as sp

from erp_graph_enhancements import (
    DependencyPlanner,
    DocumentChainPlanner,
    _parse_command_interface,
    enrich_erp_metadata,
    ensure_unique_chunks,
    graph_relation_counts,
    validate_graph,
)
from graph_rag_1c_erp import (
    DATA_DIR,
    ERPCODE_DIR,
    DocChunk,
    Embedder,
    GraphRAG,
    LlmClient,
    build_knowledge_graph,
    compact_search_index,
)
from tz_pipeline import (
    GraphOnlyTzNormalizer,
    ParsedRequirementDocument,
    QuestionPlanner,
    RequirementFileLoader,
    SourceParagraph,
    TaskGraphDocument,
    apply_mapping_answers,
    mapping_gap_question_id,
)


class RequirementPipelineTests(unittest.TestCase):
    def test_docx_without_heading_styles_becomes_structured_task_graph(self):
        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>БЛОК 1. ПРИЕМ ЗАКАЗА</w:t></w:r></w:p>
            <w:p><w:r><w:t>Входные данные:</w:t></w:r></w:p>
            <w:p><w:r><w:t>Заявка клиента.</w:t></w:r></w:p>
            <w:p><w:r><w:t>Действия:</w:t></w:r></w:p>
            <w:p><w:r><w:t>Подтвердить заявку.</w:t></w:r></w:p>
            <w:p><w:r><w:t>Результат:</w:t></w:r></w:p>
            <w:p><w:r><w:t>Подтвержденный заказ.</w:t></w:r></w:p>
            <w:p><w:r><w:t>БЛОК 2. ПРОИЗВОДСТВО</w:t></w:r></w:p>
            <w:p><w:r><w:t>Входные данные:</w:t></w:r></w:p>
            <w:p><w:r><w:t>Подтвержденный заказ.</w:t></w:r></w:p>
          </w:body>
        </w:document>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            parsed = RequirementFileLoader().load(path)
            task = TaskGraphDocument.from_parsed(parsed, task_id="sample")
        counts = task.summary()["node_types"]
        self.assertEqual(counts["process_block"], 2)
        self.assertEqual(counts["input"], 2)
        self.assertEqual(counts["action"], 1)
        self.assertEqual(counts["output"], 1)
        self.assertTrue(any(data.get("relation") == "feeds" for *_, data in task.graph.edges(data=True)))

    def test_question_planner_returns_more_than_five_relevant_questions(self):
        source = ParsedRequirementDocument(
            source_path="memory.txt", sha256="x", size_bytes=1, mime_type="text/plain",
            paragraphs=[
                SourceParagraph(
                    "Производство под заказ, склады FIFO, серии и срок годности, "
                    "контроль качества, возврат и утилизация не отражается в учете",
                    0,
                )
            ],
        )
        task = TaskGraphDocument.from_parsed(source, task_id="questions")
        questions = QuestionPlanner().plan(task)
        self.assertGreater(len(questions), 5)
        self.assertTrue(any(q.category == "quality" for q in questions))
        self.assertTrue(any(q.category == "warehouse" for q in questions))

    def test_erp_lexicon_maps_client_request_without_random_application_documents(self):
        source = ParsedRequirementDocument(
            source_path="memory.txt", sha256="x", size_bytes=1, mime_type="text/plain",
            paragraphs=[SourceParagraph("Заявка от клиента должна быть подтверждена.", 0)],
        )
        task = TaskGraphDocument.from_parsed(source, task_id="mapping")
        graph = nx.MultiDiGraph()
        graph.add_node(
            "ERPcode/Documents/ЗаказКлиента", title="Документ Заказ клиента",
            layer=3, node_type="metadata",
        )
        graph.add_node(
            "ERPcode/Documents/ЗаявкаНаКомандировку", title="Документ Заявка на командировку",
            layer=3, node_type="metadata",
        )
        result = GraphOnlyTzNormalizer().normalize(task, graph)
        mapped = {item["erp_node_id"] for item in result["mappings"]}
        self.assertEqual(mapped, {"ERPcode/Documents/ЗаказКлиента"})
        self.assertTrue(all(item["status"] == "confirmed_by_erp_lexicon" for item in result["mappings"]))

    def test_mapping_answer_confirms_graph_id(self):
        graph = nx.MultiDiGraph()
        graph.add_node("erp:one", title="Заказ клиента", layer=3, node_type="metadata")
        gap = {
            "task_node_id": "task:t:block:1:action:1", "source_text": "Оформить заказ",
            "candidates": [{"erp_node_id": "erp:one", "erp_title": "Заказ клиента"}],
        }
        payload = {"task_id": "t", "mappings": [], "mapping_gaps": [gap]}
        qid = mapping_gap_question_id("t", gap)
        result = apply_mapping_answers(payload, {qid: "Заказ клиента"}, graph)
        self.assertFalse(result["mapping_gaps"])
        self.assertEqual(result["mappings"][0]["status"], "user_confirmed")


class TypedErpGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = [
            "ERPcode/Catalogs/Номенклатура",
            "ERPcode/Catalogs/ВидыНоменклатуры",
            "ERPcode/Catalogs/УпаковкиЕдиницыИзмерения",
            "ERPcode/Documents/РеализацияТоваровУслуг",
            "ERPcode/Documents/ЗаказКлиента",
            "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
        ]
        base = [
            DocChunk(
                id=node_id, title=node_id.rsplit("/", 1)[-1], content="", path=node_id,
                layer=3, node_type="metadata", level=2,
            )
            for node_id in cls.ids
        ]
        additions, cls.stats = enrich_erp_metadata(
            ERPCODE_DIR, base, DocChunk, include_fields=True,
            include_ui=False, include_forms=False, only_objects=set(cls.ids),
        )
        cls.chunks = base + additions
        cls.graph = build_knowledge_graph(cls.chunks)

    def test_nomenclature_has_required_reverse_dependencies(self):
        plan = DependencyPlanner(self.graph).plan(self.ids[0], max_depth=2)
        dependencies = plan["tree"]["dependencies"]
        dependency_ids = {item["node_id"] for item in dependencies}
        self.assertIn(self.ids[1], dependency_ids)
        self.assertIn(self.ids[2], dependency_ids)
        self.assertTrue(all(item["inline"] for item in dependencies if item["node_id"] in {self.ids[1], self.ids[2]}))

    def test_based_on_document_chain_is_directional(self):
        result = DocumentChainPlanner(self.graph).plan(self.ids[4], self.ids[3])
        self.assertNotIn("error", result)
        self.assertEqual(result["paths"][0]["steps"][0]["node_id"], self.ids[4])
        self.assertEqual(result["paths"][0]["steps"][-1]["node_id"], self.ids[3])

    def test_register_has_reverse_registrator(self):
        counts = graph_relation_counts(self.graph)
        self.assertGreaterEqual(counts["may_write_register"], 1)
        self.assertGreaterEqual(counts["has_registrator"], 1)

    def test_command_interface_contains_real_menu_group(self):
        path = (
            ERPCODE_DIR / "Subsystems" / "Продажи" / "Subsystems" / "ОптовыеПродажи"
            / "Ext" / "CommandInterface.xml"
        )
        commands = _parse_command_interface(path)
        command = commands["Document.ЗаказКлиента.StandardCommand.OpenList"]
        self.assertIn("CommandGroup.РазделГлавное030_Продажи", command["groups"])


class SearchAndIntegrityTests(unittest.TestCase):
    def test_dense_search_uses_dot_product_and_groups_real_layer(self):
        chunks = [
            DocChunk(id="a", title="Номенклатура", content="", path="a", layer=3, node_type="metadata"),
            DocChunk(id="b", title="Производство", content="", path="b", layer=1, node_type="scenario"),
        ]
        embedder = Embedder(vectorizer_path=None)
        vectors = embedder.encode(["Номенклатура", "Производство"], fit=True).toarray().astype(np.float32)
        rag = GraphRAG(chunks, nx.MultiDiGraph(), vectors, ["a", "b"], embedder)
        result = rag.search("Номенклатура", top_k=2)
        self.assertEqual(result["vector_results"][0]["node_id"], "a")
        self.assertEqual(result["by_layer"]["metadata"][0]["node_id"], "a")

    def test_unique_ids_and_validator(self):
        chunks = [
            DocChunk(id="x", title="A", content="1", path="one"),
            DocChunk(id="x", title="B", content="2", path="two"),
        ]
        unique, report = ensure_unique_chunks(chunks)
        self.assertEqual(len({chunk.id for chunk in unique}), 2)
        graph = build_knowledge_graph(unique)
        vectors = sp.csr_matrix(np.eye(2, dtype=np.float32))
        validation = validate_graph(unique, graph, vectors, [chunk.id for chunk in unique])
        self.assertTrue(validation["ok"])
        self.assertEqual(report["renamed_collisions"], 1)

    def test_compact_search_index_keeps_last_duplicate_row(self):
        dense = np.asarray([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)
        matrix, node_ids = compact_search_index(dense, ["a", "b", "a"], {"a", "b"}, batch_size=1)
        self.assertEqual(node_ids, ["b", "a"])
        np.testing.assert_allclose(matrix.toarray(), [[0, 1], [0.5, 0.5]])

    def test_llm_is_disabled_without_explicit_opt_in(self):
        old = os.environ.pop("RAG_ENABLE_LLM", None)
        try:
            response = LlmClient(provider="wormsoft").prompt("test")
        finally:
            if old is not None:
                os.environ["RAG_ENABLE_LLM"] = old
        self.assertIn("LLM-вызовы отключены", response)


if __name__ == "__main__":
    unittest.main()
