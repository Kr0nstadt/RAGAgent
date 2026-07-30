#!/usr/bin/env python3
"""
query_graph.py — быстрый поиск по 4-слойному графу 1С ERP без LLM.
Вывод в JSON для использования opencode.

Использование:
  python query_graph.py "запрос"
  python query_graph.py "запрос" --top-k 20
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from graph_rag_1c_erp import load_data, Embedder, GraphRAG, TFIDF_FILE

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Укажите запрос"}, ensure_ascii=False))
        sys.exit(1)
    
    query = sys.argv[1]
    top_k = 25
    
    if "--top-k" in sys.argv:
        idx = sys.argv.index("--top-k")
        if idx + 1 < len(sys.argv):
            top_k = int(sys.argv[idx + 1])
    
    t0 = time.time()
    chunks, graph, vectors, node_ids = load_data(lightweight=True)
    if not chunks:
        print(json.dumps({"error": "Нет данных. Сначала: python graph_rag_1c_erp.py build"}, ensure_ascii=False))
        sys.exit(1)
    
    embedder = Embedder(vectorizer_path=TFIDF_FILE)
    rag = GraphRAG(chunks, graph, vectors, node_ids, embedder)
    result = rag.search(query, top_k=top_k)
    
    by_layer = result.get("by_layer", {})
    
    output = {
        "query": query,
        "load_time_s": round(time.time() - t0, 1),
        "total_found": len(result["vector_results"]),
        "layers": {
            "scenarios": [{"title": r["title"], "score": round(r["score"], 3)} 
                         for r in by_layer.get("scenarios", [])],
            "clarifications": [{"title": r["title"], "score": round(r["score"], 3)} 
                              for r in by_layer.get("clarifications", [])],
            "metadata": [{"title": r["title"], "path": r["path"], "score": round(r["score"], 3)} 
                        for r in by_layer.get("metadata", [])[:10]],
            "knowledge": [{"title": r["title"], "path": r["path"], "score": round(r["score"], 3)} 
                         for r in by_layer.get("knowledge", [])[:10]],
        },
        "workflow": result.get("workflow", [])[:15],
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
