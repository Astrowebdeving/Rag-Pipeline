"""Retriever that augments candidate scoring using Google's langextract.

This retriever first retrieves candidates using dense similarity, then
computes metadata for each candidate chunk via `langextract` if available:
- keywords
- descriptors
- tone words

It then re-ranks candidates using overlap between query terms and the
extracted metadata. Other retrievers are unaffected; only this retriever
relies on langextract. If langextract is unavailable, it falls back to a
lightweight heuristic keyword extractor.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from .dense_retriever import DenseRetriever
from app.vector_db.base_vector_store import BaseVectorStore  # type: ignore

logger = logging.getLogger(__name__)


def _safe_langextract(text: str) -> Dict[str, List[str]]:
    """Use langextract via lx.extract() per README; gracefully fallback if unavailable.

    Returns dict with keys: keywords, descriptors, tone (lists of strings).
    """
    try:
        import os
        import textwrap
        import langextract as lx  # type: ignore

        # Build a small extraction task: keywords, descriptors, tone
        prompt = textwrap.dedent(
            """
            Extract three classes from the text using exact spans from the source:
            - keyword: important domain-specific or topical term
            - descriptor: descriptive phrase or adjective that characterizes content
            - tone: style or affect words indicating emotion or tone (e.g., formal, urgent, optimistic)
            Use exact text for extractions. Do not paraphrase or overlap entities.
            """
        )

        # Minimal example to anchor classes
        examples = [
            lx.data.ExampleData(
                text="The urgent memo outlines a formal procedure for data migration to the cloud.",
                extractions=[
                    lx.data.Extraction(extraction_class="keyword", extraction_text="data migration"),
                    lx.data.Extraction(extraction_class="descriptor", extraction_text="formal procedure"),
                    lx.data.Extraction(extraction_class="tone", extraction_text="urgent"),
                ],
            )
        ]

        model_id = os.environ.get("LANGEXTRACT_MODEL_ID", "deepseek-r1:8b")
        model_url = os.environ.get("LANGEXTRACT_MODEL_URL", "http://localhost:11434")

        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id=model_id,
            model_url=model_url,
            fence_output=False,
            use_schema_constraints=False,
        )

        # Parse extractions
        keywords: List[str] = []
        descriptors: List[str] = []
        tones: List[str] = []

        exs = None
        try:
            # Object API
            exs = getattr(result, "extractions", None)
        except Exception:
            exs = None
        if exs is None and isinstance(result, dict):
            exs = result.get("extractions")

        if exs:
            for ex in exs:
                try:
                    ex_class = getattr(ex, "extraction_class", None) or ex.get("extraction_class")
                    ex_text = getattr(ex, "extraction_text", None) or ex.get("extraction_text")
                    if not ex_class or not ex_text:
                        continue
                    ex_class_l = str(ex_class).lower()
                    if ex_class_l == "keyword":
                        keywords.append(str(ex_text))
                    elif ex_class_l == "descriptor":
                        descriptors.append(str(ex_text))
                    elif ex_class_l == "tone":
                        tones.append(str(ex_text))
                except Exception:
                    continue

        return {
            "keywords": keywords[:12],
            "descriptors": descriptors[:12],
            "tone": tones[:8],
        }
    except Exception as e:
        logger.debug(f"langextract unavailable or failed; using fallback: {e}")
        return {"keywords": [], "descriptors": [], "tone": []}


def _fallback_keywords(text: str, top_k: int = 12) -> List[str]:
    # Simple fallback: split and filter short/common tokens
    import re
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    stop = {
        "the","and","for","with","that","from","this","have","into","your","about",
        "are","was","were","will","shall","can","could","would","should","their","there",
        "been","being","than","then","them","they","you","our","but","not","also","such",
    }
    freq: Dict[str, int] = {}
    for t in tokens:
        if t in stop:
            continue
        freq[t] = freq.get(t, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]


class LangExtractRetriever:
    """Retriever that re-ranks using langextract-derived metadata."""

    def __init__(self, base: Optional[DenseRetriever] = None) -> None:
        self.base = base or DenseRetriever()
        # Store metadata for last retrieval for optional use downstream
        self.last_metadata: List[Dict[str, Any]] = []

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        meta = _safe_langextract(text)
        if not meta["keywords"] and not meta["descriptors"] and not meta["tone"]:
            meta["keywords"] = _fallback_keywords(text)
        return {
            "lex_keywords": meta["keywords"],
            "lex_descriptors": meta["descriptors"],
            "lex_tone": meta["tone"],
        }

    def _extract_query_terms(self, query_text: str) -> List[str]:
        # Use same lightweight approach for query
        return _fallback_keywords(query_text, top_k=15)

    def _score_with_metadata(
        self,
        base_score: float,
        query_terms: List[str],
        metadata: Dict[str, Any],
    ) -> float:
        terms = set(t.lower() for t in query_terms)
        meta_terms: List[str] = []
        for key in ("lex_keywords", "lex_descriptors", "lex_tone"):
            meta_terms.extend([str(x).lower() for x in metadata.get(key, [])])
        meta_set = set(meta_terms)

        overlap = len(terms & meta_set)
        if terms:
            overlap_score = overlap / max(len(terms), 5)
        else:
            overlap_score = 0.0

        # Combine with a small weight to avoid overpowering vector score
        return float(base_score) * 0.8 + overlap_score * 0.2

    def retrieve(
        self,
        vector_store: BaseVectorStore,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
    ) -> List[str]:
        pairs = self.retrieve_with_scores(vector_store, query_embedding, query_text, top_k)
        return [chunk for chunk, _ in pairs]

    def retrieve_with_scores(
        self,
        vector_store: BaseVectorStore,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        try:
            # Base candidates
            base_pairs = self.base.retrieve_with_scores(vector_store, query_embedding, top_k * 3)
            query_terms = self._extract_query_terms(query_text)

            enriched: List[Tuple[str, float, Dict[str, Any]]] = []
            self.last_metadata = []
            for chunk, base_score in base_pairs:
                md = self._extract_metadata(chunk)
                score = self._score_with_metadata(base_score, query_terms, md)
                enriched.append((chunk, score, md))
                self.last_metadata.append(md)

            # Sort by new score
            enriched.sort(key=lambda x: x[1], reverse=True)
            return [(c, s) for (c, s, _md) in enriched[:top_k]]
        except Exception as e:
            logger.error(f"LangExtractRetriever failed, falling back to dense: {e}")
            try:
                base_pairs = self.base.retrieve_with_scores(vector_store, query_embedding, top_k)
                # add empty metadata to maintain signature
                self.last_metadata = [{} for _ in base_pairs]
                return base_pairs
            except Exception:
                return []

