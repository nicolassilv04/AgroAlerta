
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from FlagEmbedding import FlagReranker

from rag_retrieval import (
    RERANKER_MODEL,
    chunks_fingerprint,
    rerank_candidates,
)


DOCUMENTS_DIR = BASE_DIR / "documentos"
RERANK_BATCH_SIZE = 8


def main():
    with (DOCUMENTS_DIR / "chunks.json").open(encoding="utf-8") as arquivo:
        chunks = json.load(arquivo)
    with (BASE_DIR / "resultados_retrieval.json").open(
        encoding="utf-8"
    ) as arquivo:
        retrieval_data = json.load(arquivo)

    if retrieval_data.get("chunks_sha256") != chunks_fingerprint(chunks):
        raise ValueError(
            "Os candidatos foram gerados com outro chunks.json. "
            "Execute tests/testar_retrieval.py novamente."
        )

    reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
    reranked_results = []

    for result in retrieval_data["resultados"]:
        question = result["pergunta"]
        candidates = result["candidatos"]
        reranking = rerank_candidates(
            question,
            chunks,
            candidates,
            reranker,
            batch_size=RERANK_BATCH_SIZE,
        )
        reranked_results.append({"pergunta": question, **reranking})

        supported = not reranking["fora_do_escopo"]
        ranked_candidates = reranking["candidatos_reranqueados"]
        status = "FORA DO ESCOPO" if not supported else "evidência recuperada"
        print(f"\nPergunta: {question}\nStatus: {status}")
        if supported:
            for position, candidate in enumerate(ranked_candidates[:5], 1):
                print(
                    f"{position}. página {candidate['pagina']} "
                    f"(score bruto {candidate['score_reranker_raw']:.4f})"
                )

    output = {
        "chunks_sha256": retrieval_data["chunks_sha256"],
        "reranker_model": RERANKER_MODEL,
        "reranker_score_is_probability": False,
        "resultados": reranked_results,
    }
    with (BASE_DIR / "resultados_reranker.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(output, arquivo, ensure_ascii=False, indent=2)

    print("\nReranking concluído. Resultados salvos em resultados_reranker.json")


if __name__ == "__main__":
    main()
