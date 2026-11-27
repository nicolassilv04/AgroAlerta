import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, List, Sequence

import numpy as np
from rank_bm25 import BM25Okapi


EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATE_LIMIT = 50
SEMANTIC_WEIGHT = 0.5
BM25_WEIGHT = 0.5

# Raw reranker cutoff calibrated against the labeled in-scope/out-of-scope
# evaluation examples; it is not a probability.
MIN_RERANKER_SCORE = -6.0

_QUERY_SYNONYMS = {
    "mobilidade": "redistribuicao",
}


def tokenize(text: str) -> List[str]:
    """Normalize Portuguese text and remove punctuation for BM25."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = unicodedata.normalize("NFKD", normalized)
    unaccented = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.findall(r"[a-z0-9]+", unaccented)


def expand_query_tokens(question: str) -> List[str]:
    """Add explicit domain synonyms without duplicating query terms."""
    tokens = tokenize(question)
    expanded = list(tokens)
    for token in tokens:
        synonym = _QUERY_SYNONYMS.get(token)
        if synonym and synonym not in expanded:
            expanded.append(synonym)
    return expanded


def expand_reranker_query(question: str) -> str:
    """Expose the known mobility/redistribution synonym to the reranker."""
    tokens = tokenize(question)
    if "mobilidade" not in tokens or "redistribuicao" in tokens:
        return question
    return re.sub(
        r"\bmobilidade\b",
        "mobilidade (redistribuição)",
        question,
        count=1,
        flags=re.IGNORECASE,
    )


def normalize_minmax(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Scores must be a one-dimensional sequence.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Scores contain non-finite values.")
    if values.size == 0:
        return values

    score_range = float(values.max() - values.min())
    if score_range == 0:
        return np.zeros_like(values)
    return (values - values.min()) / score_range


def validate_embeddings(
    chunks: Sequence[Dict[str, Any]], embeddings: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")
    if matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Chunk/embedding count mismatch: {len(chunks)} chunks, "
            f"{matrix.shape[0]} embedding rows."
        )
    if matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("Embeddings must have non-empty, finite vectors.")

    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Embeddings are not normalized; regenerate them with "
            "normalize_embeddings=True."
        )
    return matrix


def build_bm25(chunks: Sequence[Dict[str, Any]]) -> BM25Okapi:
    corpus = [tokenize(str(chunk.get("texto", ""))) for chunk in chunks]
    if not corpus or any(not tokens for tokens in corpus):
        raise ValueError("Each chunk must contain tokenizable text.")
    return BM25Okapi(corpus)


def hybrid_search(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    model: Any,
    bm25: BM25Okapi,
    limit: int = CANDIDATE_LIMIT,
) -> List[Dict[str, Any]]:
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if limit <= 0:
        raise ValueError("Candidate limit must be greater than zero.")

    matrix = validate_embeddings(chunks, embeddings)
    query_vector = np.asarray(
        model.encode([question], normalize_embeddings=True)[0]
    )
    if query_vector.ndim != 1 or query_vector.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Query embedding dimension {query_vector.shape} does not match "
            f"document vectors {matrix.shape[1]}."
        )
    if not np.all(np.isfinite(query_vector)):
        raise ValueError("Query embedding contains non-finite values.")

    semantic_raw = matrix @ query_vector
    bm25_raw = np.asarray(
        bm25.get_scores(expand_query_tokens(question)), dtype=np.float64
    )
    if semantic_raw.shape != (len(chunks),) or bm25_raw.shape != (len(chunks),):
        raise ValueError("A retrieval score row does not match the chunk list.")

    semantic_normalized = normalize_minmax(semantic_raw)
    bm25_normalized = normalize_minmax(bm25_raw)
    hybrid_scores = (
        SEMANTIC_WEIGHT * semantic_normalized
        + BM25_WEIGHT * bm25_normalized
    )
    indices = np.argsort(-hybrid_scores, kind="stable")[:limit]

    return [
        {
            "indice": int(index),
            "score_hibrido": float(hybrid_scores[index]),
            "score_semantico": float(semantic_normalized[index]),
            "score_semantico_raw": float(semantic_raw[index]),
            "score_bm25": float(bm25_normalized[index]),
            "score_bm25_raw": float(bm25_raw[index]),
        }
        for index in indices
    ]


def is_in_scope(
    raw_reranker_scores: Sequence[float],
    threshold: float = MIN_RERANKER_SCORE,
) -> bool:
    scores = np.asarray(raw_reranker_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        return False
    if not np.all(np.isfinite(scores)):
        raise ValueError("Reranker scores contain non-finite values.")
    return float(scores.max()) >= threshold


def rerank_candidates(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    reranker: Any,
    batch_size: int = 8,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("Reranker batch size must be greater than zero.")

    indices = [candidate["indice"] for candidate in candidates]
    if any(index < 0 or index >= len(chunks) for index in indices):
        raise ValueError("A candidate index is outside the chunk list.")

    expanded_question = expand_reranker_query(question)
    if candidates:
        pairs = [
            [expanded_question, chunks[index]["texto"]]
            for index in indices
        ]
        raw_scores = np.asarray(
            reranker.compute_score(
                pairs,
                batch_size=batch_size,
                normalize=False,
            ),
            dtype=np.float64,
        )
        if raw_scores.shape != (len(candidates),):
            raise ValueError("Reranker score count does not match candidates.")
    else:
        raw_scores = np.empty(0, dtype=np.float64)

    supported = is_in_scope(raw_scores)
    order = np.argsort(-raw_scores, kind="stable")
    ranked_candidates = [
        {
            **candidates[position],
            "score_reranker_raw": float(raw_scores[position]),
            "documento": chunks[indices[position]].get("documento"),
            "pagina": chunks[indices[position]].get("pagina"),
        }
        for position in order
    ]

    return {
        "pergunta_expandida_reranker": expanded_question,
        "fora_do_escopo": not supported,
        "threshold_score_bruto": MIN_RERANKER_SCORE,
        "score_reranker_bruto_max": (
            float(raw_scores.max()) if raw_scores.size else None
        ),
        "quantidade_candidatos_avaliados": len(candidates),
        "candidatos_reranqueados": ranked_candidates if supported else [],
    }


def chunks_fingerprint(chunks: Sequence[Dict[str, Any]]) -> str:
    serialized = json.dumps(
        chunks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
