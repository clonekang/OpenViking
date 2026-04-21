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
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize NVIDIA rerank client.

        Args:
            api_key: NVIDIA API key (nvapi-...)
            model_name: Model name like "nvidia/llama-nemotron-rerank-1b-v2"
            provider: Provider name for metrics (default: "nvidia")
            extra_headers: Optional extra headers for API requests
        """
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.extra_headers = extra_headers or {}

        # Parse model name to build API endpoint
        # Format: provider/model-id -> e.g., "nvidia/llama-nemotron-rerank-1b-v2"
        if "/" in model_name:
            self._provider, self._model_id = model_name.split("/", 1)
        else:
            self._provider = "nvidia"
            self._model_id = model_name

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

        # Format conversion: OpenViking -> NVIDIA
        # OpenViking: {query: str, documents: [str]}
        # NVIDIA: {query: {text}, passages: [{text}]}
        req_body = {
            "model": self.model_name,
            "query": {"text": query},
            "passages": [{"text": doc} for doc in documents],
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

            if len(rankings) != len(documents):
                logger.warning(
                    "[NvidiaRerankClient] Unexpected result length: expected=%s actual=%s",
                    len(documents),
                    len(rankings),
                )
                return None

            # Convert logit to score (0-1 range using sigmoid-like transformation)
            # NVIDIA logit can be negative, convert to bounded score
            scores = [0.0] * len(documents)
            for item in rankings:
                idx = item.get("index")
                if idx is None or not (0 <= idx < len(documents)):
                    logger.warning(
                        "[NvidiaRerankClient] Out-of-bounds index in result: %s", item
                    )
                    return None
                # Use logit directly as score (higher = more relevant)
                scores[idx] = item.get("logit", 0.0)

            logger.debug(f"[NvidiaRerankClient] Reranked {len(documents)} documents")
            return scores

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
            extra_headers=config.extra_headers,
        )
