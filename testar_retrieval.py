
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_retrieval import (
    CANDIDATE_LIMIT,
    EMBEDDING_MODEL,
    build_bm25,
    chunks_fingerprint,
    hybrid_search,
    validate_embeddings,
)


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documentos"
QUESTIONS = [
    "Quais nutrientes apresentam alta mobilidade na planta?",
    "Quais nutrientes apresentam alta redistribuição dentro da planta?",
    "Como combater uma praga que não aparece nos documentos?",
]
EXPECTED_PAGES = {
    "Quais nutrientes apresentam alta mobilidade na planta?": 201,
    "Quais nutrientes apresentam alta redistribuição dentro da planta?": 201,
}


def find_expected_chunk(chunks, page):
    return next(
        (
            index
            for index, chunk in enumerate(chunks)
            if chunk.get("pagina") == page
            and "alta redistribuição" in chunk.get("texto", "").casefold()
        ),
        None,
    )


def main():
    with (DOCUMENTS_DIR / "chunks.json").open(encoding="utf-8") as arquivo:
        chunks = json.load(arquivo)
    embeddings = np.load(DOCUMENTS_DIR / "embeddings.npy")
    validate_embeddings(chunks, embeddings)

    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    bm25 = build_bm25(chunks)
    results = []

    for question in QUESTIONS:
        candidates = hybrid_search(
            question,
            chunks,
            embeddings,
            model,
            bm25,
            limit=CANDIDATE_LIMIT,
        )
        expected_page = EXPECTED_PAGES.get(question)
        expected_index = (
            find_expected_chunk(chunks, expected_page)
            if expected_page is not None
            else None
        )
        candidate_indices = [candidate["indice"] for candidate in candidates]

        if expected_page is not None:
            if expected_index is None:
                raise AssertionError(
                    f"Expected passage on page {expected_page} is absent from chunks."
                )
            if expected_index not in candidate_indices:
                raise AssertionError(
                    f"Page {expected_page} is not in the top {CANDIDATE_LIMIT} "
                    f"candidates for: {question}"
                )
            print(
                f"Página {expected_page}: candidata "
                f"{candidate_indices.index(expected_index) + 1}/{len(candidates)}"
            )

        results.append(
            {
                "pergunta": question,
                "pagina_relevante_esperada": expected_page,
                "candidatos": candidates,
            }
        )
        print(f"{question} -> {len(candidates)} candidatos")

    output = {
        "chunks_sha256": chunks_fingerprint(chunks),
        "embedding_shape": list(embeddings.shape),
        "resultados": results,
    }
    with (BASE_DIR / "resultados_retrieval.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(output, arquivo, ensure_ascii=False, indent=2)

    print("Retrieval concluído. Resultados salvos em resultados_retrieval.json")


if __name__ == "__main__":
    main()
