# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
NVIDIA Rerank API Client.

Supports NVIDIA's rerank API with automatic format conversion:
- Input: OpenViking standard format (query: str, documents: List[str])
- Output: NVIDIA API format (query: {text}, passages: [{text}])
- Response: NVIDIA rankings -> OpenViking scores
"""

import logging
from typing import Dict, List, Optional

import requests

from openviking.models.rerank.base import RerankBase

logger = logging.getLogger(__name__)


class NvidiaRerankClient(RerankBase):
    """
    NVIDIA rerank API client with format conversion.

    Handles the special NVIDIA API format:
    - Request: query={text}, passages=[{text}]
    - Response: rankings[] with logit field
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        provider: str = "nvidia",
        api_base: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        threshold: float = 0.0,
    ) -> None:
        """
        Initialize NVIDIA rerank client.

        Args:
            api_key: NVIDIA API key (nvapi-...)
            model_name: Model name like "nvidia/llama-nemotron-rerank-1b-v2"
            provider: Provider name for metrics (default: "nvidia")
            api_base: Full endpoint URL (configurable via config, falls back to NVIDIA default)
            extra_headers: Optional extra headers for API requests
            threshold: Score threshold (0.0-1.0); results below this are filtered (default: 0.0)
        """
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.extra_headers = extra_headers or {}
        self.threshold = threshold

        # Parse model name to build API endpoint
        # Format: provider/model-id -> e.g., "nvidia/llama-nemotron-rerank-1b-v2"
        if "/" in model_name:
            self._provider, self._model_id = model_name.split("/", 1)
        else:
            self._provider = "nvidia"
            self._model_id = model_name

        # Use configured api_base if provided, otherwise build from provider/model
        if api_base is not None:
            self.api_base = api_base
        else:
            self.api_base = f"https://ai.api.nvidia.com/v1/retrieval/{self._provider}/{self._model_id}/reranking"

    def rerank_batch(self, query: str, documents: List[str]) -> Optional[List[float]]:
        """
        Batch rerank documents against a query.

        Handles format conversion:
        - OpenViking: query=str, documents=[str]
        - NVIDIA API: query={text}, passages=[{text}]

        Args:
            query: Query text
            documents: List of document texts to rank

        Returns:
            List of rerank scores for each document (same order as input),
            or None when rerank fails and the caller should fall back
        """
        if not documents:
            return []

        # Filter out empty/whitespace-only documents (NVIDIA API requires at least 1 character)
        # Track original indices for proper score mapping
        valid_indices = [i for i, doc in enumerate(documents) if doc and doc.strip()]
        filtered_docs = [documents[i] for i in valid_indices]

        if not filtered_docs:
            logger.debug("[NvidiaRerankClient] All documents are empty after filtering")
            # Return zeros for all documents instead of None (graceful degradation)
            return [0.0] * len(documents)

        # Format conversion: OpenViking -> NVIDIA
        # OpenViking: {query: str, documents: [str]}
        # NVIDIA: {query: {text}, passages: [{text}]}
        req_body = {
            "model": self.model_name,
            "query": {"text": query},
            "passages": [{"text": doc} for doc in filtered_docs],
        }

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.extra_headers:
                headers.update(self.extra_headers)

            response = requests.post(
                url=self.api_base,
                headers=headers,
                json=req_body,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Update token usage tracking
            self._extract_and_update_token_usage(result, query, documents)

            # Format conversion: NVIDIA -> OpenViking
            # NVIDIA returns: {rankings: [{index, logit}], usage: {...}}
            # We need: scores list in original document order
            rankings = result.get("rankings")
            if not rankings:
                logger.warning(f"[NvidiaRerankClient] No rankings in response: {result}")
                return None

            if len(rankings) != len(filtered_docs):
                logger.warning(
                    "[NvidiaRerankClient] Unexpected result length: expected=%s actual=%s",
                    len(filtered_docs),
                    len(rankings),
                )
                return None

            # Convert logit to score (0-1 range using sigmoid-like transformation)
            # NVIDIA logit can be negative, convert to bounded score
            scores = [0.0] * len(documents)  # Full size for original documents

            for item in rankings:
                api_idx = item.get("index")
                if api_idx is None or not (0 <= api_idx < len(filtered_docs)):
                    logger.warning(
                        "[NvidiaRerankClient] Out-of-bounds index in result: %s", item
                    )
                    return None
                # Map API's filtered index back to original document index
                original_idx = valid_indices[api_idx]
                scores[original_idx] = item.get("logit", 0.0)

            # Apply threshold filtering: set scores below threshold to 0
            if self.threshold > 0.0:
                filtered_scores = []
                for score in scores:
                    if score >= self.threshold:
                        filtered_scores.append(score)
                    else:
                        filtered_scores.append(0.0)
                        logger.debug(
                            f"[NvidiaRerankClient] Filtered score {score:.4f} below threshold {self.threshold}"
                        )
                scores = filtered_scores

            logger.debug(f"[NvidiaRerankClient] Reranked {len(documents)} documents")
            return scores

        except requests.exceptions.HTTPError as e:
            logger.error(f"[NvidiaRerankClient] HTTP error: {e}, response: {getattr(e, 'response', None)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"[NvidiaRerankClient] Response text: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"[NvidiaRerankClient] Rerank failed: {e}")
            return None

    @classmethod
    def from_config(cls, config) -> Optional["NvidiaRerankClient"]:
        """
        Create NvidiaRerankClient from RerankConfig.

        Args:
            config: RerankConfig instance with provider='nvidia'

        Returns:
            NvidiaRerankClient instance or None if config is not available
        """
        if not config or not config.is_available():
            return None

        # Extract provider from model name or use default
        model_name = config.model or "nvidia/llama-nemotron-rerank-1b-v2"
        provider = model_name.split("/")[0] if "/" in model_name else "nvidia"

        return cls(
            api_key=config.api_key,
            model_name=model_name,
            provider=provider,
            api_base=config.api_base,
            extra_headers=config.extra_headers,
            threshold=config.threshold,
        )
