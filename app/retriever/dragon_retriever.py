"""Dragon-style retriever: dense pre-retrieval + cross-encoder re-ranking.

This approximates a "Dragon Retriever" setup by first doing a fast dense
retrieval (using the existing stored embeddings/FAISS), then re-ranking the
top candidates with a strong cross-encoder similar to Dragon-style
re-rankers. It integrates seamlessly with the current pipeline without
requiring re-embedding the corpus.

If the cross-encoder model cannot be loaded, the retriever falls back to
returning dense results only.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .dense_retriever import DenseRetriever
from app.vector_db.base_vector_store import BaseVectorStore  # type: ignore

logger = logging.getLogger(__name__)


class DragonRetriever:
    """Dense retrieval + cross-encoder re-ranking retriever.

    Parameters
    ----------
    candidate_multiplier : int
        How many initial candidates to fetch for re-ranking (multiple of top_k).
    reranker_model : str
        Hugging Face cross-encoder name. Defaults to a strong MS MARCO reranker.
    """

    def __init__(
        self,
        candidate_multiplier: int = 5,
        reranker_model: str | None = None,
    ) -> None:
        self.base = DenseRetriever()
        self.candidate_multiplier = max(2, candidate_multiplier)
        # Allow env override
        import os
        self.reranker_model = (
            reranker_model
            or os.environ.get("DRAGON_RERANKER_MODEL")
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._cross_encoder = CrossEncoder(self.reranker_model)
            logger.info(
                f"DragonRetriever loaded cross-encoder reranker: {self.reranker_model}"
            )
        except Exception as e:
            logger.warning(
                f"DragonRetriever could not load reranker '{self.reranker_model}': {e}. "
                f"Will fallback to dense-only scores."
            )

    def retrieve(self, vector_store: BaseVectorStore, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> List[str]:
        pairs = self.retrieve_with_scores(vector_store, query_embedding, query_text, top_k)
        return [c for c, _ in pairs]

    def retrieve_with_scores(
        self,
        vector_store: BaseVectorStore,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        try:
            candidate_k = min(top_k * self.candidate_multiplier, 100)
            base_pairs = self.base.retrieve_with_scores(vector_store, query_embedding, candidate_k)
            if not base_pairs:
                return []

            # If no cross-encoder, return base results
            if self._cross_encoder is None:
                return base_pairs[:top_k]

            # Prepare pairs for reranker: (query, passage)
            passages = [chunk for chunk, _ in base_pairs]
            inputs = [(query_text, passage) for passage in passages]

            try:
                scores = self._cross_encoder.predict(inputs)
            except Exception as e:
                logger.warning(f"Dragon reranker failed, falling back to dense: {e}")
                return base_pairs[:top_k]

            # Combine scores: weighted sum of dense score and reranker score
            # Normalize reranker scores to 0..1 via min-max on this batch
            scores = np.array(scores, dtype=float)
            r_min, r_max = float(scores.min()), float(scores.max())
            if r_max > r_min:
                rerank_norm = (scores - r_min) / (r_max - r_min)
            else:
                rerank_norm = np.zeros_like(scores)

            dense_scores = np.array([s for _, s in base_pairs], dtype=float)
            # Normalize dense scores to 0..1 similarly
            d_min, d_max = float(dense_scores.min()), float(dense_scores.max())
            if d_max > d_min:
                dense_norm = (dense_scores - d_min) / (d_max - d_min)
            else:
                dense_norm = np.zeros_like(dense_scores)

            final = 0.4 * dense_norm + 0.6 * rerank_norm
            ranked = sorted(
                zip(passages, final.tolist()), key=lambda x: x[1], reverse=True
            )
            return ranked[:top_k]
        except Exception as e:
            logger.error(f"DragonRetriever failed: {e}")
            try:
                return self.base.retrieve_with_scores(vector_store, query_embedding, top_k)
            except Exception:
                return []

