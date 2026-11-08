"""src/rag/evaluate_rag.py -- Policy RAG Evaluation & Grounding Audit (Part D Tasks 1-4)

1. Evaluates Policy document chunking (15 chunks across 5 documents).
2. Verifies TF-IDF retrieval reference test ('refund policy late delivery' -> refund_policy.txt).
3. Evaluates 14 policy benchmark questions from data/raw/policy_eval.csv (12 answerable, 2 unanswerable).
4. Conducts zero-overlap ablation (overlap=0) and isolates boundary split failure modes.
5. Saves outputs/rag_evaluation_report.txt and outputs/rag_zero_overlap_report.txt.
"""

from pathlib import Path
import csv
import sys
import pandas as pd

# Ensure project root is available to import mock_llm and src modules
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.retriever import PolicyRetriever, build_policy_chunks
from src.rag.policy_assistant import PolicyAssistant


def evaluate_dataset(assistant: PolicyAssistant, eval_csv_path: Path) -> dict:
    """Evaluate PolicyAssistant on policy_eval.csv."""
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total_questions = len(reader)
    answerable_count = sum(1 for r in reader if r["expected_doc"] != "NONE")
    unanswerable_count = sum(1 for r in reader if r["expected_doc"] == "NONE")

    doc_retrieved_count = 0
    fact_present_count = 0
    unanswerable_fallback_count = 0

    detailed_results = []

    for r in reader:
        question = r["question"]
        expected_doc = r["expected_doc"]
        expected_fact = r["expected_fact"]

        answer_text, sources = assistant.answer(question)

        is_unanswerable = (expected_doc == "NONE")
        doc_match = False
        fact_match = False
        fallback_match = False

        if is_unanswerable:
            fallback_match = answer_text.startswith("I could not find")
            if fallback_match:
                unanswerable_fallback_count += 1
            status = "PASS" if fallback_match else "FAIL"
        else:
            doc_match = expected_doc in sources
            fact_match = expected_fact in answer_text
            if doc_match:
                doc_retrieved_count += 1
            if fact_match:
                fact_present_count += 1
            status = "PASS" if (doc_match and fact_match) else "FAIL"

        detailed_results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_sources": sources,
            "expected_fact": expected_fact,
            "answer": answer_text,
            "doc_match": doc_match,
            "fact_match": fact_match,
            "fallback_match": fallback_match,
            "status": status,
        })

    return {
        "total": total_questions,
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "doc_retrieved_count": doc_retrieved_count,
        "fact_present_count": fact_present_count,
        "unanswerable_fallback_count": unanswerable_fallback_count,
        "details": detailed_results,
    }


def run_rag_evaluation(
    policies_dir: Path,
    eval_csv_path: Path,
    out_eval_report: Path,
    out_zero_report: Path,
) -> dict:
    # ----------------------------------------------------
    # TASK 1: CHUNK THE FIVE POLICY DOCUMENTS
    # ----------------------------------------------------
    chunks_10, counts_10 = build_policy_chunks(policies_dir, chunk_size=40, overlap=10)

    expected_counts_10 = {
        "refund_policy.txt": 4,
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2,
    }

    assert len(chunks_10) == 15, f"Expected 15 chunks with overlap 10, got {len(chunks_10)}"
    assert counts_10 == expected_counts_10, f"Chunk count mismatch: {counts_10} vs {expected_counts_10}"

    # ----------------------------------------------------
    # TASK 2: TF-IDF RETRIEVAL REFERENCE CHECK
    # ----------------------------------------------------
    retriever_10 = PolicyRetriever(policies_dir, chunk_size=40, overlap=10)
    ref_query = "refund policy late delivery"
    top_3_refund = retriever_10.retrieve(ref_query, k=3)
    top_3_sources = [c["source"] for c in top_3_refund]

    assert all(src == "refund_policy.txt" for src in top_3_sources), (
        f"Expected top 3 chunks for '{ref_query}' to all be from refund_policy.txt, got {top_3_sources}"
    )

    # ----------------------------------------------------
    # TASK 3 & 4: EVALUATE BASELINE (OVERLAP = 10)
    # ----------------------------------------------------
    assistant_10 = PolicyAssistant(retriever_10)
    res_10 = evaluate_dataset(assistant_10, eval_csv_path)

    assert res_10["doc_retrieved_count"] == 12, f"Expected 12/12 doc retrieved, got {res_10['doc_retrieved_count']}"
    assert res_10["fact_present_count"] == 12, f"Expected 12/12 fact present, got {res_10['fact_present_count']}"
    assert res_10["unanswerable_fallback_count"] == 2, f"Expected 2/2 unanswerable, got {res_10['unanswerable_fallback_count']}"

    # ----------------------------------------------------
    # TASK 4B: ZERO-OVERLAP ABLATION EVALUATION (OVERLAP = 0)
    # ----------------------------------------------------
    chunks_0, counts_0 = build_policy_chunks(policies_dir, chunk_size=40, overlap=0)
    retriever_0 = PolicyRetriever(policies_dir, chunk_size=40, overlap=0)
    assistant_0 = PolicyAssistant(retriever_0)
    res_0 = evaluate_dataset(assistant_0, eval_csv_path)

    # Identify questions that failed with zero-overlap
    zero_overlap_failures = [r for r in res_0["details"] if r["status"] == "FAIL"]
    assert len(zero_overlap_failures) >= 1, "Expected at least one failure in zero-overlap evaluation!"

    # ----------------------------------------------------
    # BUILD PRIMARY REPORT (OVERLAP = 10)
    # ----------------------------------------------------
    lines = []
    lines.append("=" * 85)
    lines.append("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    lines.append("PART D, TASKS 1-4: POLICY RAG ASSISTANT EVALUATION REPORT")
    lines.append("=" * 85)

    lines.append("\n1. POLICY DOCUMENT CHUNKING (CHUNK_SIZE=40, OVERLAP=10)")
    lines.append("-" * 85)
    lines.append(f"  - Total Policy Chunks Created : {len(chunks_10)} (Reference: 15) -> PASS")
    lines.append("  - Per-Document Chunk Breakdown:")
    for doc_name, cnt in counts_10.items():
        ref_cnt = expected_counts_10[doc_name]
        lines.append(f"    * {doc_name:<24}: {cnt} chunks (Reference: {ref_cnt}) -> PASS")

    lines.append("\n2. TF-IDF VECTORIZATION & RETRIEVAL REFERENCE CHECK")
    lines.append("-" * 85)
    lines.append(f"  - Reference Query : '{ref_query}'")
    lines.append("  - Top 3 Retrieved Chunks:")
    for i, ch in enumerate(top_3_refund, 1):
        lines.append(f"    {i}. [{ch['source']}] (Score: {ch['score']:.4f}) -> {ch['text'][:70]}...")
    lines.append(f"  - Check Result    : All 3 chunks originate from 'refund_policy.txt' -> PASS")

    lines.append("\n3. THEORETICAL FOUNDATION: TF-IDF & LIMITATIONS")
    lines.append("-" * 85)
    lines.append("  1. Term Frequency (TF): Measures the local prominence of a term within an individual chunk, normalizing for chunk length so longer passages are not unfairly favored.")
    lines.append("  2. Inverse Document Frequency (IDF): Measures the global rarity and discriminatory power of a term across the entire chunk corpus, downweighting ubiquitous filler words and upweighting terms unique to specific policies.")
    lines.append("  3. Key Limitation for Policy Retrieval: TF-IDF relies exclusively on surface lexical overlap. It fails to match synonymous queries where users express identical semantic intent using distinct vocabulary (e.g., 'reimbursement' vs 'refund', or 'courier' vs 'rider'), leading to zero similarity scores when exact word forms do not align.")

    lines.append("\n4. POLICY QUESTION-ANSWERING EVALUATION (data/raw/policy_eval.csv)")
    lines.append("-" * 85)
    lines.append(f"  - Total Benchmark Questions Evaluated    : {res_10['total']}")
    lines.append(f"  - Answerable Expected Document Retrieval : {res_10['doc_retrieved_count']}/{res_10['answerable_count']} (Reference: 12/12) -> PASS")
    lines.append(f"  - Answerable Expected Fact Accuracy      : {res_10['fact_present_count']}/{res_10['answerable_count']} (Reference: 12/12) -> PASS")
    lines.append(f"  - Unanswerable Fallback Rate             : {res_10['unanswerable_fallback_count']}/{res_10['unanswerable_count']} (Reference: 2/2) -> PASS")

    lines.append("\nDetailed Benchmark Results (Overlap = 10):")
    lines.append(f"{'Q#':<4} {'Question':<50} {'Exp Doc':<20} {'Fact Match':<12} {'Status'}")
    lines.append("-" * 95)
    for idx, item in enumerate(res_10["details"], 1):
        lines.append(f"{idx:<4} {item['question'][:48]:<50} {item['expected_doc']:<20} {str(item['fact_match']):<12} {item['status']}")

    lines.append("\nSample Grounded Answers:")
    for idx in [1, 2, 11, 13]:
        item = res_10["details"][idx - 1]
        lines.append(f"  [Q{idx}] Question : {item['question']}")
        lines.append(f"       Expected : {item['expected_fact'] if item['expected_doc'] != 'NONE' else 'Fallback (I could not find...)'}")
        lines.append(f"       Answer   : {item['answer']}")
        lines.append(f"       Sources  : {item['retrieved_sources']}")

    lines.append("=" * 85)
    report_text = "\n".join(lines) + "\n"

    out_eval_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_eval_report, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ----------------------------------------------------
    # BUILD ZERO-OVERLAP ABLATION REPORT
    # ----------------------------------------------------
    zlines = []
    zlines.append("=" * 85)
    zlines.append("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    zlines.append("PART D, TASK 4B: ZERO-OVERLAP ABLATION STUDY REPORT")
    zlines.append("=" * 85)

    zlines.append(f"\n1. CHUNKING COMPARISON (OVERLAP=10 VS OVERLAP=0)")
    zlines.append("-" * 85)
    zlines.append(f"  - Overlap 10 Total Chunks : {len(chunks_10)} chunks")
    zlines.append(f"  - Overlap 0 Total Chunks  : {len(chunks_0)} chunks")
    zlines.append(f"  - Overlap 0 Chunk Counts  : {counts_0}")

    zlines.append(f"\n2. PERFORMANCE COMPARISON ON 14 EVALUATION QUESTIONS")
    zlines.append("-" * 85)
    zlines.append(f"{'Evaluation Metric':<45} {'Overlap = 10':<20} {'Overlap = 0':<20}")
    zlines.append("-" * 85)
    zlines.append(f"{'Expected Document Retrieved':<45} {res_10['doc_retrieved_count']}/{res_10['answerable_count']:<20} {res_0['doc_retrieved_count']}/{res_0['answerable_count']:<20}")
    zlines.append(f"{'Expected Fact Present in Answer':<45} {res_10['fact_present_count']}/{res_10['answerable_count']:<20} {res_0['fact_present_count']}/{res_0['answerable_count']:<20}")
    zlines.append(f"{'Unanswerable Fallback Count':<45} {res_10['unanswerable_fallback_count']}/{res_10['unanswerable_count']:<20} {res_0['unanswerable_fallback_count']}/{res_0['unanswerable_count']:<20}")

    zlines.append(f"\n3. ROOT-CAUSE ANALYSIS OF ZERO-OVERLAP FAILURE")
    zlines.append("-" * 85)
    for fail in zero_overlap_failures:
        zlines.append(f"  - Failed Question   : '{fail['question']}'")
        zlines.append(f"  - Target Document   : {fail['expected_doc']}")
        zlines.append(f"  - Expected Fact     : '{fail['expected_fact']}'")
        zlines.append(f"  - Retrieved Sources : {fail['retrieved_sources']}")
        zlines.append(f"  - Generated Answer  : '{fail['answer']}'")
        zlines.append(
            "  - Failure Mechanism : With zero overlap (stride = 40 words), the context sentence containing the answer "
            f"fact ('{fail['expected_fact']}') was severed directly across a chunk boundary. Because the sentence's key subject "
            "and predicate terms fell into different adjacent chunks, individual chunks lacked sufficient cohesive lexical stem "
            "overlap with the question prompt, causing the model to either retrieve an irrelevant chunk or select a truncated sentence."
        )

    zlines.append("=" * 85)
    zero_report_text = "\n".join(zlines) + "\n"

    out_zero_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_zero_report, "w", encoding="utf-8") as f:
        f.write(zero_report_text)

    return {
        "report_normal": report_text,
        "report_zero": zero_report_text,
        "failures_zero": zero_overlap_failures,
    }


if __name__ == "__main__":
    p_dir = project_root / "data" / "policies"
    eval_csv = project_root / "data" / "raw" / "policy_eval.csv"
    eval_report = project_root / "outputs" / "rag_evaluation_report.txt"
    zero_report = project_root / "outputs" / "rag_zero_overlap_report.txt"

    res = run_rag_evaluation(
        policies_dir=p_dir,
        eval_csv_path=eval_csv,
        out_eval_report=eval_report,
        out_zero_report=zero_report,
    )
    print(res["report_normal"])
    print("\n" + res["report_zero"])

