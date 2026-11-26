import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_retrieval import EMBEDDING_MODEL, validate_embeddings


DOCUMENTS_DIR = Path(__file__).resolve().parent / "documentos"


def main():
    with (DOCUMENTS_DIR / "chunks.json").open(encoding="utf-8") as arquivo:
        chunks = json.load(arquivo)

    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    texts = [chunk["texto"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    validate_embeddings(chunks, embeddings)
    np.save(DOCUMENTS_DIR / "embeddings.npy", embeddings)
    print("Embeddings gerados:", embeddings.shape)


if __name__ == "__main__":
    main()