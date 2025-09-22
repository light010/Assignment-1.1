"""Weighted OpenSearch retriever for semantic document similarity search.

This module provides specialized components for semantic retrieval from OpenSearch indices
using a weighted vector similarity search approach with configurable filtering capabilities
and comprehensive error handling.

The implementation supports:
- Weighted scoring between content and query vector similarity
- Bundle-based filtering for document hierarchies
- Flexible result formatting with metadata extraction
- Robust retry mechanisms for search reliability
- Score thresholding to filter low-quality matches

Classes:
    WeightedOpensearchRetriever: Implementation of weighted vector similarity search from OpenSearch with embedded configuration
"""

import time
from typing import Any, Literal

from loguru import logger
from oneailib.core.documents.base import Document
from oneailib.core.retrievers.base import Retriever, RetrievedContextData
from oneailib.core.utils.opensearch_base import OpensearchBase
from pydantic import Field, model_validator


class WeightedOpensearchRetriever(OpensearchBase, Retriever):
    """Weighted OpenSearch retriever for semantic document similarity search.

    This retriever implements semantic search using a dual vector similarity approach
    for enhanced accuracy and relevance scoring.

    **Dual Vector Similarity Scoring:**
    The retriever combines two types of embedded vectors to compute relevance:

    1. **Content Vector** (content_vector_weight): Measures semantic similarity between the query
       and document content. Higher weight emphasizes content relevance.

    2. **Query Vector** (query_vector_weight): Measures similarity between the query
       and query-style embeddings. Higher weight emphasizes query-specific matching.

    The final similarity score = (content_score × content_vector_weight) + (query_score × query_vector_weight)

    **Weight Configuration:**
    Both weights must be specified and sum to exactly 1.0:
    - content_vector_weight + query_vector_weight = 1.0
    - Default: content_vector_weight=0.7, query_vector_weight=0.3
    - For content-focused search: increase content_vector_weight (e.g., 0.8/0.2)
    - For query-focused search: increase query_vector_weight (e.g., 0.4/0.6)

    **Additional Features:**
    - Configurable retry logic for reliability
    - Comprehensive metadata in results
    - Score thresholding for quality control
    """

    content_vector_weight: float = Field(
        default=0.7,
        description="Weight for content vector similarity (0-1). Higher values prioritize document content similarity over query-specific matching.",
        ge=0.0,
        le=1.0,
    )
    query_vector_weight: float = Field(
        default=0.3,
        description="Weight for query vector similarity (0-1). Higher values prioritize query-specific matching over general content similarity.",
        ge=0.0,
        le=1.0,
    )
    content_field: str = Field(
        default="text_chunk",
        description="Field to use as the main content in search results. Will fall back to 'text_chunk' then 'page_content' if not found.",
    )
    score_threshold: float | None = Field(
        default=None, description="Minimum score threshold for results. Results with lower scores will be filtered out."
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for search operations.")
    retry_delay: float = Field(
        default=0.5, description="Initial delay between retries in seconds. Subsequent retries use exponential backoff."
    )
    content_vector_field: str = Field(
        default="metadata.embedding.text_chunk",
        description="Path to the content embedding vector field in OpenSearch documents for similarity scoring.",
    )
    query_vector_field: str = Field(
        default="metadata.embedding.question",
        description="Path to the query embedding vector field in both input document metadata and OpenSearch documents for similarity scoring.",
    )
    retriever_returns_type: Literal["string", "list"] = Field(default="string")

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self):
        """Ensure content_vector_weight + query_vector_weight = 1.0.

        This validator ensures that the two vector weights are properly normalized
        and sum to exactly 1.0, which is required for proper weighted scoring.

        Returns:
            The validated model instance

        Raises:
            ValueError: If the weights don't sum to 1.0 (within floating point tolerance)
        """
        total_weight = self.content_vector_weight + self.query_vector_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Vector weights must sum to 1.0. Got content_vector_weight={self.content_vector_weight} + "
                f"query_vector_weight={self.query_vector_weight} = {total_weight}"
            )
        return self

    def similarity_search_document(self, document: Document) -> RetrievedContextData:
        """Perform semantic similarity search based on the provided document.

        This method serves as the primary retriever interface, executing a search against
        OpenSearch and enriching the input document with retrieved context. It handles:
        - Vector similarity search using document embeddings
        - Content extraction and formatting
        - Metadata preservation

        Args:
            document: Input document containing query information (embeddings)

        Returns:
            RetrievedContextData: The retrieved context data for the document.

        Note:
            The document metadata will be updated with:
            - "context": Combined text content from matched documents
            - "context_metadata": List of metadata from matched documents
            - "search_telemetry": Performance and result metrics
        """
        start_time = time.time()

        response = self.execute_search(document)
        results = self._extract_results(response)
        telemetry = self._get_telemetry(start_time, results, response)
        context_list = self._get_search_context(results)
        context_metadata = self._get_context_metadata(results)

        return RetrievedContextData(
            retrieved_context=context_list,
            retrieved_context_metadata=context_metadata,
            retrieved_additional_metadata={"search_telemetry": telemetry},
        )

    def execute_search(self, document: Document) -> dict[str, Any]:
        """Execute a search query against OpenSearch.

        This method builds a search query based on the provided document and executes it
        against the configured OpenSearch index. It handles retries with exponential backoff
        for transient errors.

        Args:
            document: Document containing query information (embeddings)

        Returns:
            dict: OpenSearch response or empty dict if search failed

        Note:
            The retry behavior is controlled by max_retries and retry_delay configuration.
            Network errors and OpenSearch service errors will trigger retry attempts.
        """
        query = self._build_query(document)

        retries = 0
        start_time = time.time()

        search_metrics = {"query_size": len(str(query)), "retry_count": 0, "success": False}

        while True:
            try:
                logger.debug(f"OpenSearch query: {query}")

                result = self.client.search(index=self.search_index_name, body=query)

                duration_ms = self._calculate_duration_ms(start_time)
                hit_count = len(result.get("hits", {}).get("hits", []))

                logger.debug(f"Search completed in {duration_ms}ms, found {hit_count} hits")

                search_metrics.update({"duration_ms": duration_ms, "hit_count": hit_count, "success": True})

                result["_search_metrics"] = search_metrics

                return result

            except Exception as e:
                retries += 1
                search_metrics["retry_count"] = retries

                if retries > self.max_retries:
                    duration_ms = self._calculate_duration_ms(start_time)
                    search_metrics["duration_ms"] = duration_ms

                    logger.error(f"Search failed after {retries} attempts ({duration_ms}ms): {e}")
                    return {"_search_metrics": search_metrics}

                delay = self.retry_delay * (2 ** (retries - 1))

                logger.warning(f"Retry {retries}/{self.max_retries} in {delay:.2f}s: {e}")

                time.sleep(delay)

    def _build_query(self, doc: Document) -> dict[str, Any]:
        """Build OpenSearch query based on document and configuration.

        Constructs a script_score query with weighted cosine similarity calculations.
        The query will include weighted scoring between content and query vectors.

        Args:
            doc: Document containing embeddings

        Returns:
            dict containing an OpenSearch query or empty dict if query could not be built
        """
        query_vector = self._get_query_embedding_from_path(doc.metadata, self.query_vector_field)

        if not query_vector:
            logger.warning(
                f"No query vector found for script_score search. Document metadata keys: {list(doc.metadata.keys())}"
            )
            return {}

        query = {
            "size": self.n_responses_to_retrieve,
            "query": self._script_score_query(query_vector),
        }

        return query

    def _script_score_query(self, query_vector: list[float]) -> dict[str, Any]:
        """Create a script_score query with weighted cosine similarity.

        Creates an OpenSearch script_score query that computes:
        score = content_vector_weight * cosineSimilarity(query_vector, content_vector) +
                query_vector_weight * cosineSimilarity(query_vector, query_vector)

        This approach allows weighting similarity scores between different vector fields
        to achieve optimal retrieval performance based on both text content and questions.

        Args:
            query_vector: Query embedding vector to search with

        Returns:
            dict: OpenSearch script_score query component
        """
        content_vector_field = self.content_vector_field
        query_vector_field = self.query_vector_field

        base_query = {"match_all": {}}

        return {
            "script_score": {
                "query": base_query,
                "script": {
                    "source": f"""
                        double contentScore = cosineSimilarity(params.query_vector, doc['{content_vector_field}']) * params.content_weight;
                        double queryScore = cosineSimilarity(params.query_vector, doc['{query_vector_field}']) * params.query_weight;
                        return contentScore + queryScore;
                    """,
                    "params": {
                        "query_vector": query_vector,
                        "content_weight": self.content_vector_weight,
                        "query_weight": self.query_vector_weight,
                    },
                },
            }
        }

    def _get_telemetry(
        self, start_time: float, results: list[dict[str, Any]], response: dict[str, Any]
    ) -> dict[str, Any]:
        """Create telemetry data for search performance tracking.

        Args:
            start_time: The start time of the search operation
            results: List of processed search results
            response: Raw OpenSearch response

        Returns:
            dict: Telemetry data including search metrics
        """
        telemetry = {
            "search_mode": "weighted_vector",
            "index": self.search_index_name,
            "result_count": len(results),
        }

        if response and isinstance(response, dict) and "_search_metrics" in response:
            telemetry.update(response["_search_metrics"])

        return telemetry

    def _get_search_context(self, results: list[dict[str, Any]]) -> list[str]:
        """Extract and join content from search results into a list of context strings.

        Args:
            results: List of processed search results

        Returns:
            list[str]: List of context content from all results
        """
        content_pieces = []
        for r in results:
            content = self._get_content_from_result(r)
            if content:
                content_pieces.append(content)

        return content_pieces

    def _get_context_metadata(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract and format metadata from search results.

        Args:
            results: List of processed search results

        Returns:
            list: List of metadata dictionaries for each result
        """
        context_metadata = []
        for r in results:
            meta = {field: r[field] for field in (self.metadata_keys or []) if field in r}
            
            meta["_id"] = r.get("_id", "")
            meta["_score"] = r.get("_score", 0)
            
            if "normalized_score" in r:
                meta["normalized_score"] = r["normalized_score"]
                
            context_metadata.append(meta)
        
        return context_metadata

    def _extract_results(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract and process search results from OpenSearch response.

        This method transforms OpenSearch response hits into a simplified structure,
        extracting document content and metadata fields specified in configuration.
        It also handles score normalization and optional threshold filtering.

        Args:
            response: Raw OpenSearch response dictionary

        Returns:
            list of processed result dictionaries containing content and metadata
        """
        if not response or "hits" not in response:
            logger.warning("Empty or invalid OpenSearch response")
            return []

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            logger.debug("No hits found in search results")
            return []

        results = self._process_hits(hits)
        return self._apply_score_threshold(results)

    def _process_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process individual hits from OpenSearch response."""
        results = []
        max_score = max([hit.get("_score", 0) for hit in hits], default=1.0)
        min_score = min([hit.get("_score", 0) for hit in hits], default=0.0)
        score_range = max_score - min_score
        
        for hit in hits:
            result = self._create_result_from_hit(hit, min_score, score_range)
            results.append(result)
        
        return results

    def _create_result_from_hit(self, hit: dict[str, Any], min_score: float, score_range: float) -> dict[str, Any]:
        """Create a result dictionary from a single OpenSearch hit."""
        src = hit.get("_source", {})
        meta = src.get("metadata", {})
        raw_score = hit.get("_score", 0)

        result = {
            "_id": hit.get("_id", ""),
            "_score": raw_score,
            "page_content": src.get("page_content", ""),
        }
        
        # Calculate normalized score
        if "normalized_score" in src:
            result["normalized_score"] = src["normalized_score"]
        elif score_range > 0:
            result["normalized_score"] = (raw_score - min_score) / score_range
        else:
            result["normalized_score"] = 1.0 if raw_score > 0 else 0.0
        
        # Extract specified metadata fields
        for field in self.metadata_keys or []:
            if field in src:
                result[field] = src[field]
            elif field in meta:
                result[field] = meta[field]

        return result

    def _apply_score_threshold(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply score threshold filtering if configured."""
        if self.score_threshold is not None and results:
            original_count = len(results)
            results = [r for r in results if r.get("normalized_score", 0) >= self.score_threshold]
            if len(results) < original_count:
                logger.debug(
                    f"Filtered {original_count - len(results)} results below score threshold {self.score_threshold}"
                )

        return results

    def _get_content_from_result(self, result: dict[str, Any]) -> str:
        """Extract content using configured field with fallbacks.

        Tries to get content from the configured content field first,
        then falls back to text_chunk and finally page_content.
        This provides flexibility to work with different document structures
        and ensures that the best available content is always used.

        Args:
            result: Result dictionary from search response

        Returns:
            str: Extracted content or empty string if no content found
        """
        content = result.get(self.content_field)
        if content is None:
            content = result.get("text_chunk")
        if content is None:
            content = result.get("page_content")
        return str(content) if content else ""

    def _calculate_duration_ms(self, start_time: float) -> int:
        """Calculate duration in milliseconds from start time.

        Args:
            start_time: The start time from time.time()

        Returns:
            int: Duration in milliseconds
        """
        return int((time.time() - start_time) * 1000)

    def _get_query_embedding_from_path(self, metadata: dict[str, Any], path: str) -> list[float] | None:
        """Extract embedding vector from metadata using dot-notation path.

        Args:
            metadata: Document metadata dictionary
            path: Dot notation path to the embedding vector (format: "metadata.nestedfield.field")

        Returns:
            list of floats (embedding vector) or None if not found
        """
        if not metadata:
            logger.debug(f"No metadata provided for path '{path}'")
            return None

        path_parts = path.replace("metadata.", "", 1).split(".")

        current_data = metadata
        for part in path_parts:
            if not isinstance(current_data, dict) or part not in current_data:
                logger.debug(f"Failed to navigate path '{path}' at part '{part}'")
                return None
            current_data = current_data[part]

        if isinstance(current_data, list) and current_data and all(isinstance(x, (int, float)) for x in current_data):
            logger.debug(f"Successfully extracted embedding vector from '{path}' with {len(current_data)} dimensions")
            return current_data

        logger.debug(f"Path '{path}' does not contain a valid embedding vector. Found: {type(current_data).__name__}")
        return None
