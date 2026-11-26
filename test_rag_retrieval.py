import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer

from rag_retrieval import (
    CANDIDATE_LIMIT,
    EMBEDDING_MODEL,
    MIN_RERANKER_SCORE,
    RERANKER_MODEL,
    build_bm25,
    expand_reranker_query,
    hybrid_search,
    is_in_scope,
    tokenize,
    validate_embeddings,
)


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documentos"
MOBILITY_QUESTION = "Quais nutrientes apresentam alta mobilidade na planta?"
REDISTRIBUTION_QUESTION = (
    "Quais nutrientes apresentam alta redistribuição dentro da planta?"
)
OUT_OF_SCOPE_QUESTION = "Como combater uma praga que não aparece nos documentos?"


class RetrievalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (DOCUMENTS_DIR / "chunks.json").open(encoding="utf-8") as arquivo:
            cls.chunks = json.load(arquivo)
        cls.embeddings = np.load(DOCUMENTS_DIR / "embeddings.npy")
        validate_embeddings(cls.chunks, cls.embeddings)
        cls.model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        cls.bm25 = build_bm25(cls.chunks)
        cls.reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
        cls.target_index = next(
            index
            for index, chunk in enumerate(cls.chunks)
            if chunk.get("pagina") == 201
            and "alta redistribuição" in chunk.get("texto", "").casefold()
        )

    def retrieve(self, question):
        return hybrid_search(
            question,
            self.chunks,
            self.embeddings,
            self.model,
            self.bm25,
            limit=CANDIDATE_LIMIT,
        )

    def rerank(self, question, candidates):
        pairs = [
            [
                expand_reranker_query(question),
                self.chunks[candidate["indice"]]["texto"],
            ]
            for candidate in candidates
        ]
        return np.asarray(
            self.reranker.compute_score(
                pairs,
                batch_size=8,
                normalize=False,
            ),
            dtype=np.float64,
        )

    def test_embedding_rows_match_chunks_and_page_201(self):
        self.assertEqual(self.embeddings.shape[0], len(self.chunks))
        target_text = self.chunks[self.target_index]["texto"]
        current_vector = self.model.encode(
            [target_text],
            normalize_embeddings=True,
        )[0]
        cosine = float(self.embeddings[self.target_index] @ current_vector)
        nearest_row = int(np.argmax(self.embeddings @ current_vector))

        self.assertGreater(cosine, 0.999)
        self.assertEqual(nearest_row, self.target_index)

    def test_tokenizer_normalizes_accents_and_punctuation(self):
        self.assertEqual(
            tokenize("Redistribuição, na planta?"),
            ["redistribuicao", "na", "planta"],
        )
        self.assertEqual(
            tokenize("redistribuicao na planta"),
            ["redistribuicao", "na", "planta"],
        )

    def test_relevant_page_is_in_top_50_for_both_phrasings(self):
        for question in (MOBILITY_QUESTION, REDISTRIBUTION_QUESTION):
            with self.subTest(question=question):
                candidates = self.retrieve(question)
                indices = [candidate["indice"] for candidate in candidates]
                self.assertLessEqual(len(candidates), CANDIDATE_LIMIT)
                self.assertIn(self.target_index, indices)

    def test_reranker_promotes_page_201_for_mobility_paraphrase(self):
        candidates = self.retrieve(MOBILITY_QUESTION)
        indices = [candidate["indice"] for candidate in candidates]
        self.assertIn(self.target_index, indices)

        scores = self.rerank(MOBILITY_QUESTION, candidates)
        ranked_indices = [
            indices[index] for index in np.argsort(-scores, kind="stable")
        ]

        self.assertEqual(ranked_indices[0], self.target_index)
        self.assertTrue(is_in_scope(scores))
        self.assertGreater(float(scores.max()), MIN_RERANKER_SCORE)

    def test_unanswerable_question_is_flagged_out_of_scope(self):
        candidates = self.retrieve(OUT_OF_SCOPE_QUESTION)
        scores = self.rerank(OUT_OF_SCOPE_QUESTION, candidates)

        self.assertFalse(is_in_scope(scores))
        self.assertLess(float(scores.max()), MIN_RERANKER_SCORE)


if __name__ == "__main__":
    unittest.main()
