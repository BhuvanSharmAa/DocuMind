import json, time, os, argparse
from src.ingestion import ingest_pdf
from src.retriever import Retriever
from src.chain import answer_question, ask_llm
from src.config import LOG_DIR

def precision_at_k(results, expected_keywords):
    if not expected_keywords:
        return None
    hits = sum(1 for r in results if any(kw.lower() in r["chunk"].lower() for kw in expected_keywords))
    return hits / len(results) if results else 0.0

def get_queries_interactively():
    print("\nEnter 10 test queries about your PDF.")
    print("For each query, optionally enter expected keywords (comma-separated) for Precision@k.")
    print("Leave keywords blank to mark as 'manual review'.\n")
    queries = []
    for i in range(1, 11):
        query = input(f"Query {i:02d}: ").strip()
        if not query:
            print("Skipping empty query.")
            continue
        keywords_raw = input(f"  Expected keywords (or press Enter to skip): ").strip()
        keywords = [k.strip() for k in keywords_raw.split(",")] if keywords_raw else []
        queries.append({"query": query, "expected_keywords": keywords})
    return queries

def run_evaluation(pdf_path, queries=None):
    print(f"\n{'='*60}\nDocuMind Evaluation — {os.path.basename(pdf_path)}\n{'='*60}")

    chunks = ingest_pdf(pdf_path)
    retriever = Retriever()
    retriever.load_or_build(chunks)

    if queries is None:
        queries = get_queries_interactively()

    results_log = []
    print()
    for i, item in enumerate(queries, 1):
        query = item["query"]
        keywords = item.get("expected_keywords", [])
        print(f"[{i:02d}] {query}")

        t0 = time.time()
        rag_result = answer_question(query, retriever)
        rag_latency = (time.time() - t0) * 1000

        direct_answer = ask_llm(f"Answer briefly: {query}")
        p_at_k = precision_at_k(rag_result["sources"], keywords)

        print(f"  RAG:    {rag_result['answer'][:100]}")
        print(f"  Direct: {direct_answer[:100]}")
        print(f"  P@k: {p_at_k if p_at_k is not None else 'manual'}  |  latency: {rag_result['latency_ms']}ms\n")

        results_log.append({
            "query": query,
            "rag_answer": rag_result["answer"],
            "direct_answer": direct_answer,
            "precision_at_k": p_at_k,
            "retrieval_latency_ms": rag_result["latency_ms"],
            "rag_total_latency_ms": round(rag_latency, 1),
        })

    scored = [r for r in results_log if r["precision_at_k"] is not None]
    avg_p = sum(r["precision_at_k"] for r in scored) / len(scored) if scored else 0
    avg_lat = sum(r["retrieval_latency_ms"] for r in results_log) / len(results_log)

    summary = {
        "pdf": os.path.basename(pdf_path),
        "avg_precision_at_k": round(avg_p, 3),
        "avg_retrieval_latency_ms": round(avg_lat, 1),
        "results": results_log
    }
    out = os.path.join(LOG_DIR, "eval_results.json")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'='*60}")
    print(f"Avg Precision@k:       {avg_p:.1%}  (on {len(scored)} scored queries)")
    print(f"Avg retrieval latency: {avg_lat:.1f} ms")
    print(f"Results saved to:      {out}")
    print(f"\n★ Resume metric: 'RAG achieved {avg_p:.0%} Precision@4 with {avg_lat:.0f}ms avg retrieval latency'")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF to evaluate")
    args = parser.parse_args()
    run_evaluation(args.pdf)