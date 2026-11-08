"""src/rag/retriever.py -- Policy Document Chunking & TF-IDF Retrieval (Part D Tasks 1 & 2)

1. Splits policy text files into 40-word chunks with configurable overlap (default 10 words, step 30).
2. Fits scikit-learn TfidfVectorizer on all chunk texts.
3. Implements retrieve(query, k=3) returning ranked chunks by cosine similarity.
"""

from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_document(text: str, source: str, chunk_size: int = 40, overlap: int = 10) -> list[dict]:
    """Split document into words and chunk with overlap until reaching or passing end of text."""
    words = text.strip().split()
    n_words = len(words)
    step = chunk_size - overlap
    assert step > 0, "Chunk size must be strictly greater than overlap"

    chunks = []
    start = 0
    while start < n_words:
        end = min(start + chunk_size, n_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "source": source,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_idx": start,
            "end_idx": end,
        })
        if end >= n_words:
            break
        start += step

    return chunks


def build_policy_chunks(policies_dir: Path | str, chunk_size: int = 40, overlap: int = 10) -> tuple[list[dict], dict[str, int]]:
    """Read all .txt policy files in policies_dir and chunk them."""
    policies_dir = Path(policies_dir)
    policy_files = sorted(policies_dir.glob("*.txt"))
    all_chunks = []
    counts_by_doc = {}

    for pfile in policy_files:
        with open(pfile, "r", encoding="utf-8") as f:
            text = f.read()
        doc_chunks = chunk_document(text=text, source=pfile.name, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)
        counts_by_doc[pfile.name] = len(doc_chunks)

    return all_chunks, counts_by_doc


class PolicyRetriever:
    """TF-IDF vectorizer and cosine similarity retriever over policy document chunks."""

    def __init__(self, policies_dir: Path, chunk_size: int = 40, overlap: int = 10):
        self.policies_dir = policies_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks, self.counts = build_policy_chunks(policies_dir, chunk_size=chunk_size, overlap=overlap)

        self.vectorizer = TfidfVectorizer()
        self.chunk_texts = [c["text"] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Rank chunks by cosine similarity against query in descending order."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = similarities.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": float(similarities[idx]),
                "text": self.chunks[idx]["text"],
            })
        return results

