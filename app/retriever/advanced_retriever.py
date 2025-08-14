"""Advanced retriever with re-ranking, query expansion, and LLM enhancements.

This retriever provides enhanced search capabilities including:
- Query expansion for better recall
- Result re-ranking based on multiple factors
- Contextual filtering for improved precision
- Diversity-aware selection to avoid redundant results
- LLM-powered query enhancement and metadata scoring
"""

import re  # Regular expressions for text processing
import logging  # Logging system for debugging
from typing import List, Tuple, Dict, Any, Set, Optional  # Type hints for clarity
from collections import Counter  # For frequency counting
import numpy as np  # Numerical operations
from .dense_retriever import DenseRetriever  # Base dense retrieval
from .hybrid_retriever import HybridRetriever  # Hybrid retrieval
from app.vector_db.base_vector_store import BaseVectorStore  # Vector store interface

# Setup logging for this module
logger = logging.getLogger(__name__)  # Create logger instance

class AdvancedRetriever:
    """Advanced retriever with re-ranking, query enhancement, and LLM capabilities."""
    
    def __init__(
        self,
        base_retriever_type: str = 'hybrid',  # Base retriever to use
        rerank_results: bool = True,  # Whether to re-rank results
        expand_queries: bool = True,  # Whether to expand queries
        diversity_factor: float = 0.3,  # Factor for result diversification
        context_window: int = 2,  # Context window for relevance scoring
        min_relevance_score: float = 0.1,  # Minimum relevance threshold
        use_llm_query_enhancement: bool = True  # Whether to use LLM for query enhancement
    ):
        """Initialize advanced retriever.
        
        Args:
            base_retriever_type: Base retriever to use ('dense' or 'hybrid')
            rerank_results: Whether to rerank results
            expand_queries: Whether to expand queries
            diversity_factor: Factor for diversity in results (0.0-1.0)
            context_window: Context window for chunk relationships
            min_relevance_score: Minimum relevance score threshold
            use_llm_query_enhancement: Whether to use LLM for query enhancement
        """
        self.base_retriever_type = base_retriever_type
        self.rerank_results = rerank_results
        self.expand_queries = expand_queries
        self.diversity_factor = diversity_factor
        self.context_window = context_window
        self.min_relevance_score = min_relevance_score
        self.use_llm_query_enhancement = use_llm_query_enhancement
        
        # Initialize base retriever
        if base_retriever_type == 'hybrid':
            self.base_retriever = HybridRetriever()
        else:
            self.base_retriever = DenseRetriever()
        
        # LLM query enhancer (lazy loading)
        self._query_enhancer = None
        # Persist last metadata for UI/generator
        self.last_metadata: List[Dict[str, Any]] = []
        
        logger.info(f"AdvancedRetriever initialized with {base_retriever_type} base, rerank={rerank_results}, expand={expand_queries}, llm_enhance={use_llm_query_enhancement}")
    
    def _get_query_enhancer(self):
        """Get or create query enhancer for LLM-based query expansion.
        
        Returns:
            Generator instance or None if failed
        """
        if self._query_enhancer is None and self.use_llm_query_enhancement:
            try:
                from app.augmented_generation.ollama_generator import OllamaGenerator
                from app.services.state_service import rag_state
                self._query_enhancer = OllamaGenerator(
                    model_name="deepseek-r1:8b",
                    max_tokens=50,
                    temperature=0.2,
                    timeout=rag_state.config.get('llm_timeout', 30)
                )
                logger.info("Ollama query enhancer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama query enhancer: {e}")
                self._query_enhancer = None
        return self._query_enhancer
    
    def _enhance_query_with_llm(self, original_query: str) -> str:
        """Use LLM to enhance the original query with related terms.
        
        Args:
            original_query: Original user query
            
        Returns:
            str: Enhanced query with additional terms
        """
        enhancer = self._get_query_enhancer()
        if enhancer is None:
            return original_query
        
        try:
            # Create a focused prompt for query enhancement
            enhancement_prompt = f"""Expand this search query with related keywords and synonyms. Return a compact list of comma-separated terms only.

Query: {original_query}

Related terms:"""
            
            response_data = enhancer.generate_response(
                query=enhancement_prompt,
                retrieved_chunks=[],  # Empty for query enhancement
                max_tokens_override=80,  # keep short to avoid long think loops
                suppress_info_log=True,
            )
            
            enhanced_portion = response_data.get('response', '') if response_data.get('success', False) else ''
            
            if enhanced_portion and enhanced_portion.strip():
                enhanced_portion = enhanced_portion.strip()
                
                # Clean up common prefixes that LLMs add
                prefixes_to_remove = [
                    "Enhanced query:", "Improved query:", "Better search:",
                    "Search for:", "Find:"
                ]
                
                for prefix in prefixes_to_remove:
                    if enhanced_portion.lower().startswith(prefix.lower()):
                        enhanced_portion = enhanced_portion[len(prefix):].strip()
                
                # Combine original query with enhanced terms
                combined_query = f"{original_query} {enhanced_portion}"
                logger.debug(f"Query enhanced: '{original_query}' -> '{combined_query}'")
                return combined_query
                
        except Exception as e:
            logger.warning(f"Failed to enhance query with LLM: {e}")
        
        return original_query
    
    def _generate_chunk_metadata_on_demand(self, chunk: str, chunk_metadata: dict) -> dict:
        """Generate metadata for a chunk on-demand during query processing."""
        # If metadata already has LLM analysis, return it
        if chunk_metadata.get('topics_analyzed', False) and chunk_metadata.get('semantic_description'):
            return chunk_metadata
        
        # Generate metadata using LLM
        try:
            enhancer = self._get_query_enhancer()  # Reuse the same LLM instance
            if enhancer is None:
                return chunk_metadata
            
            analysis_prompt = f"""Analyze this text and provide a brief summary and key topics. Keep under 3 sentences.

Text: {chunk[:400]}

Summary and keywords:"""

            response_data = enhancer.generate_response(
                query=analysis_prompt,
                retrieved_chunks=[],
                max_tokens_override=120,
                suppress_info_log=True,
            )
            if response_data.get('success', False):
                response = response_data.get('response', '').strip()
                
                if response and len(response) > 10:
                    # Update metadata
                    updated_metadata = chunk_metadata.copy()
                    updated_metadata['semantic_description'] = response[:200] if response else ''
                    updated_metadata['keywords'] = response.split()[:5]  # Simple keyword extraction
                    updated_metadata['topics_analyzed'] = True
                    
                    logger.debug(f"Generated on-demand metadata for chunk: {response[:50]}...")
                    return updated_metadata
            
        except Exception as e:
            logger.debug(f"Failed to generate on-demand metadata: {e}")
        
        return chunk_metadata
    
    def retrieve(
        self,
        vector_store: BaseVectorStore,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        candidate_multiplier: int = 4
    ) -> List[str]:
        """Retrieve relevant chunks using advanced strategies.
        
        Args:
            vector_store: Vector store to search
            query_embedding: Query embedding vector
            query_text: Original query text
            top_k: Number of final results to return
            candidate_multiplier: Multiplier for initial candidate retrieval
            
        Returns:
            List[str]: Retrieved text chunks
        """
        try:
            # Step 1: Enhance query with LLM if enabled
            enhanced_query = self._enhance_query_with_llm(query_text) if self.use_llm_query_enhancement else query_text
            
            # Step 2: Expand queries if enabled
            expanded_queries = self._expand_query(enhanced_query) if self.expand_queries else [enhanced_query]
            
            # Step 3: Retrieve candidates from all expanded queries
            all_candidates = []
            candidate_k = min(top_k * candidate_multiplier, 50)  # Limit to prevent excessive retrieval
            
            for expanded_query in expanded_queries:
                try:
                    # Try to retrieve with metadata if vector store supports it
                    chunks, scores, metadata_list = vector_store.search(query_embedding, candidate_k)
                    
                    if chunks and scores:
                        for i, (chunk, score) in enumerate(zip(chunks, scores)):
                            metadata = metadata_list[i] if i < len(metadata_list) else {}
                            # Generate metadata on-demand if needed for advanced retrieval
                            enhanced_metadata = self._generate_chunk_metadata_on_demand(chunk, metadata)
                            all_candidates.append({
                                'chunk': chunk,
                                'base_score': score,
                                'query': expanded_query,
                                'embedding': query_embedding,
                                'metadata': enhanced_metadata
                            })
                except Exception as e:
                    logger.warning(f"Failed to retrieve with metadata, falling back to base retriever: {e}")
                    
                    # Fallback to base retriever without metadata
                    if self.base_retriever_type == 'hybrid':
                        candidates = self.base_retriever.retrieve_with_scores(
                            vector_store, query_embedding, expanded_query, candidate_k
                        )
                    else:
                        candidates = self.base_retriever.retrieve_with_scores(
                            vector_store, query_embedding, candidate_k
                        )
                    
                    if candidates:
                        for chunk, score in candidates:
                            # Generate metadata on-demand even for fallback candidates
                            basic_metadata = {'chunk_index': len(all_candidates), 'topics_analyzed': False}
                            enhanced_metadata = self._generate_chunk_metadata_on_demand(chunk, basic_metadata)
                            all_candidates.append({
                                'chunk': chunk,
                                'base_score': score,
                                'query': expanded_query,
                                'embedding': query_embedding,
                                'metadata': enhanced_metadata
                            })
            
            # Remove duplicates while preserving best scores
            unique_candidates = self._deduplicate_candidates(all_candidates)
            
            # Step 4: Rerank candidates if enabled
            if self.rerank_results and unique_candidates:
                reranked_results = self._rerank_candidates(unique_candidates, query_text, vector_store)
                top_pairs = reranked_results[:top_k]
                final_chunks = [chunk for chunk, score in top_pairs]
                # Persist metadata for selected chunks
                selected = set(final_chunks)
                self.last_metadata = [cand.get('metadata', {}) for cand in unique_candidates if cand['chunk'] in selected]
            else:
                # Sort by base score and take top_k
                unique_candidates.sort(key=lambda x: x['base_score'], reverse=True)
                selected = unique_candidates[:top_k]
                final_chunks = [candidate['chunk'] for candidate in selected]
                self.last_metadata = [candidate.get('metadata', {}) for candidate in selected]
            
            # Step 5: Apply diversity if needed
            if self.diversity_factor > 0:
                final_chunks = self._apply_diversity(final_chunks, self.diversity_factor)
            
            logger.info(f"Advanced retrieval: {len(expanded_queries)} queries → {len(unique_candidates)} candidates → {len(final_chunks)} final results")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Advanced retrieval failed: {e}")
            # Fallback to simple dense retrieval
            try:
                fallback_retriever = DenseRetriever()
                self.last_metadata = []
                return fallback_retriever.retrieve(vector_store, query_embedding, top_k)
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {fallback_error}")
                return []
    
    def retrieve_with_scores(
        self,
        vector_store: BaseVectorStore,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Retrieve documents with similarity scores using advanced methods.
        
        Args:
            vector_store: Vector store to search
            query_embedding: Query embedding vector
            query_text: Original query text
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples ordered by relevance
        """
        try:
            # Step 1: Enhance query with LLM if enabled
            enhanced_query = self._enhance_query_with_llm(query_text) if self.use_llm_query_enhancement else query_text
            
            # Step 2: Expand queries if enabled
            expanded_queries = self._expand_query(enhanced_query) if self.expand_queries else [enhanced_query]
            
            # Step 3: Retrieve candidates from all expanded queries
            all_candidates = []
            candidate_k = min(top_k * 4, 50)  # Limit to prevent excessive retrieval
            
            for expanded_query in expanded_queries:
                try:
                    # Try to retrieve with metadata if vector store supports it
                    chunks, scores, metadata_list = vector_store.search(query_embedding, candidate_k)
                    
                    if chunks and scores:
                        for i, (chunk, score) in enumerate(zip(chunks, scores)):
                            metadata = metadata_list[i] if i < len(metadata_list) else {}
                            # Generate metadata on-demand if needed for advanced retrieval
                            enhanced_metadata = self._generate_chunk_metadata_on_demand(chunk, metadata)
                            all_candidates.append({
                                'chunk': chunk,
                                'base_score': score,
                                'query': expanded_query,
                                'embedding': query_embedding,
                                'metadata': enhanced_metadata
                            })
                except Exception as e:
                    logger.warning(f"Failed to retrieve with metadata, falling back to base retriever: {e}")
                    
                    # Fallback to base retriever without metadata
                    if self.base_retriever_type == 'hybrid':
                        candidates = self.base_retriever.retrieve_with_scores(
                            vector_store, query_embedding, expanded_query, candidate_k
                        )
                    else:
                        candidates = self.base_retriever.retrieve_with_scores(
                            vector_store, query_embedding, candidate_k
                        )
                    
                    if candidates:
                        for chunk, score in candidates:
                            # Generate metadata on-demand even for fallback candidates
                            basic_metadata = {'chunk_index': len(all_candidates), 'topics_analyzed': False}
                            enhanced_metadata = self._generate_chunk_metadata_on_demand(chunk, basic_metadata)
                            all_candidates.append({
                                'chunk': chunk,
                                'base_score': score,
                                'query': expanded_query,
                                'embedding': query_embedding,
                                'metadata': enhanced_metadata
                            })
            
            # Remove duplicates while preserving best scores
            unique_candidates = self._deduplicate_candidates(all_candidates)
            
            # Step 4: Rerank candidates if enabled
            if self.rerank_results and unique_candidates:
                reranked_results = self._rerank_candidates(unique_candidates, query_text, vector_store)
                return reranked_results[:top_k]
            else:
                # Sort by base score and return top_k
                unique_candidates.sort(key=lambda x: x['base_score'], reverse=True)
                return [(candidate['chunk'], candidate['base_score']) for candidate in unique_candidates[:top_k]]
            
        except Exception as e:
            logger.error(f"Advanced retrieval with scores failed: {e}")
            # Fallback to simple dense retrieval
            try:
                fallback_retriever = DenseRetriever()
                return fallback_retriever.retrieve_with_scores(vector_store, query_embedding, top_k)
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {fallback_error}")
                return []
    
    def _expand_query(self, query_text: str) -> List[str]:
        """Expand query with synonyms and related terms, optimized for procedural content.
        
        Parameters
        ----------
        query_text : str
            Original query text
            
        Returns
        -------
        List[str]
            List of expanded queries including original
        """
        expanded_queries = [query_text]  # Start with original query
        
        # Detect if this is a procedural query
        is_procedural = self._is_procedural_query(query_text)
        
        # Technical domain expansions
        expansions = {
            'protocol': ['communication protocol', 'network protocol', 'data protocol'],
            'security': ['cybersecurity', 'data security', 'network security'],
            'communication': ['data transmission', 'messaging', 'networking'],
            'configuration': ['setup', 'installation', 'settings'],
            'authentication': ['login', 'access control', 'verification'],
            'encryption': ['cryptography', 'data protection', 'secure transmission'],
            'network': ['networking', 'connectivity', 'infrastructure'],
            'system': ['platform', 'architecture', 'framework'],
            'data': ['information', 'content', 'payload'],
            'control': ['management', 'administration', 'governance']
        }
        
        # Procedural/installation specific expansions
        procedural_expansions = {
            'install': ['installation', 'setup', 'mount', 'connect', 'configure'],
            'setup': ['installation', 'configure', 'prepare', 'initialize'],
            'steps': ['procedure', 'instructions', 'process', 'method'],
            'how to': ['procedure for', 'steps to', 'method to', 'process to'],
            'power supply': ['PSU', 'power unit', 'power source'],
            'controllogix': ['control logix', 'allen bradley', 'plc', 'controller'],
            'chassis': ['rack', 'enclosure', 'mounting frame'],
            'procedure': ['steps', 'instructions', 'process', 'method'],
            'connect': ['attach', 'fasten', 'secure', 'join'],
            'mount': ['install', 'attach', 'secure', 'fasten']
        }
        
        # Check for expandable terms in query
        query_lower = query_text.lower()  # Convert to lowercase for matching
        
        # Apply technical expansions
        for term, synonyms in expansions.items():
            if term in query_lower:
                # Add queries with synonyms
                for synonym in synonyms:
                    expanded_query = query_text.replace(term, synonym)  # Replace term with synonym
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)  # Add unique expansion
        
        # Apply procedural expansions if this is a procedural query
        if is_procedural:
            for term, synonyms in procedural_expansions.items():
                if term in query_lower:
                    # Add queries with synonyms
                    for synonym in synonyms:
                        expanded_query = query_text.replace(term, synonym)  # Replace term with synonym
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)  # Add unique expansion
            
            # Add procedural keywords to help find step-by-step content
            procedural_keywords = [
                f"step {query_text}",
                f"procedure {query_text}",
                f"instructions {query_text}",
                f"{query_text} steps",
                f"{query_text} procedure"
            ]
            for keyword_query in procedural_keywords:
                if keyword_query not in expanded_queries:
                    expanded_queries.append(keyword_query)
        
        # Limit number of expansions to avoid overwhelming retrieval
        max_expansions = 6 if is_procedural else 3  # More expansions for procedural queries
        if len(expanded_queries) > max_expansions:
            expanded_queries = expanded_queries[:max_expansions]  # Limit expansions
        
        logger.debug(f"Query expansion: '{query_text}' → {len(expanded_queries)} variants (procedural: {is_procedural})")
        
        return expanded_queries  # Return expanded query list
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates while preserving the best scores.
        
        Args:
            candidates: List of candidate dictionaries
            
        Returns:
            List[Dict[str, Any]]: Deduplicated candidates
        """
        seen_chunks = {}
        
        for candidate in candidates:
            chunk = candidate['chunk']
            score = candidate['base_score']
            
            if chunk not in seen_chunks or score > seen_chunks[chunk]['base_score']:
                seen_chunks[chunk] = candidate
        
        return list(seen_chunks.values())
    
    def _rerank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        original_query: str,
        vector_store: BaseVectorStore
    ) -> List[Tuple[str, float]]:
        """Rerank candidates using multiple scoring factors.
        
        Args:
            candidates: List of candidate dictionaries
            original_query: Original query text
            vector_store: Vector store for additional context
            
        Returns:
            List[Tuple[str, float]]: Reranked (chunk, score) pairs
        """
        reranked_results = []
        is_procedural = self._is_procedural_query(original_query)
        
        for candidate in candidates:
            chunk = candidate['chunk']
            base_score = candidate['base_score']
            metadata = candidate.get('metadata', {})  # Get chunk metadata
            
            # Calculate various scoring factors
            query_match_score = self._calculate_query_match_score(chunk, original_query)
            length_score = self._calculate_length_score(chunk)
            context_score = self._calculate_context_score(chunk, candidates)
            metadata_score = self._calculate_metadata_score(metadata, original_query)  # Metadata-based scoring
            
            # Procedural query bonus
            procedural_score = 0.0
            if is_procedural:
                procedural_score = self._calculate_procedural_score(chunk)
            
            # Combine scores with weights
            if is_procedural:
                final_score = (
                    0.25 * base_score +
                    0.2 * query_match_score +
                    0.2 * procedural_score +
                    0.15 * metadata_score +
                    0.15 * context_score +
                    0.05 * length_score
                )
            else:
                final_score = (
                    0.35 * base_score +
                    0.25 * query_match_score +
                    0.2 * metadata_score +
                    0.15 * context_score +
                    0.05 * length_score
                )
            
            reranked_results.append((chunk, final_score))
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Re-ranking completed for {len(candidates)} candidates")
        return reranked_results
    
    def _calculate_metadata_score(self, metadata: Dict[str, Any], query_text: str) -> float:
        """Calculate score based on metadata semantic description.
        
        Args:
            metadata: Chunk metadata dictionary
            query_text: Original query text
            
        Returns:
            float: Metadata-based relevance score (0.0-1.0)
        """
        # Handle missing or invalid metadata gracefully
        if not metadata or not isinstance(metadata, dict):
            return 0.0

        query_lower = query_text.lower()
        query_terms = set(w for w in query_lower.split() if len(w) > 2)
        score_components = []

        # 1) Semantic description overlap (if present)
        semantic_desc = metadata.get('semantic_description', '')
        if isinstance(semantic_desc, str) and semantic_desc:
            desc_words = set(w for w in semantic_desc.lower().split() if len(w) > 2)
            if query_terms and desc_words:
                overlap = len(query_terms & desc_words)
                total_unique = len(query_terms | desc_words)
                score_components.append(overlap / total_unique if total_unique > 0 else 0.0)

        # 2) LangExtract metadata overlap (if present)
        for key, weight in (("lex_keywords", 1.0), ("lex_descriptors", 0.8), ("lex_tone", 0.5)):
            vals = metadata.get(key)
            if isinstance(vals, list) and vals:
                meta_terms = set(str(v).lower() for v in vals if isinstance(v, (str, int, float)))
                if meta_terms and query_terms:
                    overlap = len(query_terms & meta_terms)
                    denom = max(len(query_terms), 6)
                    score_components.append((overlap / denom) * weight)

        # 3) Bonus for LLM-analyzed chunks
        analysis_bonus = 0.05 if metadata.get('topics_analyzed', False) else 0.0

        total = sum(score_components) + analysis_bonus
        # Normalize roughly to 0..1 range
        return float(min(total, 1.0))
    
    def _calculate_query_match_score(self, chunk: str, query: str) -> float:
        """Calculate direct query-chunk matching score."""
        chunk_lower = chunk.lower()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Count exact word matches
        matches = sum(1 for word in query_words if word in chunk_lower)
        match_ratio = matches / len(query_words) if query_words else 0
        
        # Bonus for phrase matches
        phrase_bonus = 0.2 if query_lower in chunk_lower else 0
        
        return min(match_ratio + phrase_bonus, 1.0)
    
    def _calculate_length_score(self, chunk: str) -> float:
        """Calculate score based on chunk length (prefer moderate length)."""
        length = len(chunk)
        
        # Optimal length range
        if 100 <= length <= 500:
            return 1.0
        elif length < 100:
            return length / 100.0  # Penalty for very short chunks
        else:
            return max(0.5, 1000.0 / length)  # Penalty for very long chunks
    
    def _calculate_context_score(self, chunk: str, all_candidates: List[Dict[str, Any]]) -> float:
        """Calculate score based on context and relationships."""
        # Simple implementation: score based on similarity to other candidates
        similarities = []
        
        for candidate in all_candidates[:10]:  # Limit comparison
            other_chunk = candidate['chunk']
            if other_chunk != chunk:
                # Simple word overlap similarity
                words1 = set(chunk.lower().split())
                words2 = set(other_chunk.lower().split())
                if words1 and words2:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_procedural_score(self, chunk: str) -> float:
        """Calculate score for procedural/instructional content."""
        chunk_lower = chunk.lower()
        
        # Look for procedural indicators
        procedural_indicators = [
            r'\d+\.\s',  # Numbered lists
            r'step\s+\d+',  # Step numbers
            r'first|second|third|next|then|finally',  # Sequence words
            r'click|press|select|choose|enter',  # Action words
            r'install|configure|setup|run'  # Setup words
        ]
        
        score = 0.0
        for pattern in procedural_indicators:
            matches = len(re.findall(pattern, chunk_lower))
            score += matches * 0.1
        
        return min(score, 1.0)
    
    def _is_procedural_query(self, query: str) -> bool:
        """Check if query is asking for procedural information."""
        procedural_keywords = [
            r'\bhow\s+to\b', r'\bsteps?\b', r'\binstall\b', r'\bsetup\b',
            r'\bconfigure\b', r'\bprocedure\b', r'\binstructions?\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in procedural_keywords)
    
    def _apply_diversity(self, chunks: List[str], diversity_factor: float) -> List[str]:
        """Apply diversity to reduce redundancy in results."""
        if len(chunks) <= 1:
            return chunks
        
        diverse_chunks = [chunks[0]]  # Always include the top result
        
        for chunk in chunks[1:]:
            # Check similarity with already selected chunks
            is_diverse = True
            for selected in diverse_chunks:
                # Simple word overlap check
                words1 = set(chunk.lower().split())
                words2 = set(selected.lower().split())
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                
                if overlap > (1.0 - diversity_factor):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_chunks.append(chunk)
        
        return diverse_chunks
    
    def _extract_important_terms(self, text: str) -> Set[str]:
        """Extract important terms from text (excluding stop words)."""
        # Basic stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'must'
        }
        
        # Extract words using regex
        words = re.findall(r'\b\w+\b', text.lower())  # Extract words in lowercase
        
        # Filter important terms
        important_terms = {
            word for word in words 
            if len(word) > 2 and word not in stop_words  # Keep words longer than 2 chars, not stop words
        }
        
        return important_terms  # Return set of important terms
    
    def _apply_diversity_filtering(self, scored_candidates: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
        """Apply diversity filtering to avoid redundant results."""
        if self.diversity_factor == 0 or len(scored_candidates) <= top_k:
            return scored_candidates[:top_k]  # No diversity filtering needed
        
        diverse_results = []  # List for diverse results
        remaining_candidates = scored_candidates.copy()  # Copy candidates list
        
        # Always include the top result
        if remaining_candidates:
            diverse_results.append(remaining_candidates.pop(0))  # Add top result
        
        # Add diverse results
        while len(diverse_results) < top_k and remaining_candidates:
            best_candidate = None  # Best candidate for diversity
            best_diversity_score = -1  # Best diversity score
            
            for i, (candidate_chunk, candidate_score) in enumerate(remaining_candidates):
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(
                    candidate_chunk, [chunk for chunk, _ in diverse_results]
                )
                
                # Combine relevance and diversity
                combined_score = (
                    (1 - self.diversity_factor) * candidate_score +  # Relevance component
                    self.diversity_factor * diversity_score  # Diversity component
                )
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score  # Update best score
                    best_candidate = (candidate_chunk, candidate_score)  # Update best candidate
                    best_index = i  # Track index for removal
            
            if best_candidate:
                diverse_results.append(best_candidate)  # Add diverse candidate
                remaining_candidates.pop(best_index)  # Remove from remaining
        
        logger.debug(f"Diversity filtering: applied factor {self.diversity_factor} to select {len(diverse_results)} diverse results")
        
        return diverse_results  # Return diverse results
    
    def _calculate_diversity_score(self, candidate_chunk: str, selected_chunks: List[str]) -> float:
        """Calculate diversity score for a candidate relative to selected chunks."""
        if not selected_chunks:
            return 1.0  # Maximum diversity if no selections yet
        
        candidate_terms = self._extract_important_terms(candidate_chunk)  # Get candidate terms
        
        # Calculate minimum similarity to any selected chunk
        min_similarity = float('inf')  # Initialize minimum similarity
        
        for selected_chunk in selected_chunks:
            selected_terms = self._extract_important_terms(selected_chunk)  # Get selected chunk terms
            
            # Calculate Jaccard similarity
            if candidate_terms or selected_terms:
                union_size = len(candidate_terms.union(selected_terms))  # Size of union
                intersection_size = len(candidate_terms.intersection(selected_terms))  # Size of intersection
                similarity = intersection_size / union_size if union_size > 0 else 0  # Jaccard similarity
                min_similarity = min(min_similarity, similarity)  # Track minimum similarity
        
        # Diversity is inverse of similarity
        diversity_score = 1.0 - min_similarity if min_similarity != float('inf') else 1.0
        
        return diversity_score  # Return diversity score


def get_retriever_info() -> Dict[str, Any]:
    """Get information about the advanced retriever.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing retriever information and capabilities
    """
    return {
        'method': 'advanced',  # Retriever method identifier
        'base_retrievers': ['dense', 'hybrid'],  # Available base retrievers
        'features': [
            'query_expansion',  # Query expansion capability
            'result_reranking',  # Result re-ranking capability
            'diversity_filtering',  # Diversity filtering capability
            'contextual_scoring',  # Contextual scoring capability
            'llm_query_enhancement',  # LLM query enhancement capability
            'metadata_scoring'  # Metadata-based scoring capability
        ],
        'supports_scoring': True,  # Whether retriever supports scoring
        'configurable': True,  # Whether retriever is configurable
        'description': 'Advanced retriever with query expansion, re-ranking, diversity filtering, and LLM enhancements'
    } 