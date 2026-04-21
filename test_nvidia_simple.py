#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试 NVIDIA Rerank Client - 不依赖完整项目
"""

import sys
from pathlib import Path

# Direct import of the nvidia rerank module
sys.path.insert(0, str(Path(__file__).parent / "openviking/models/rerank"))

import requests
import logging
logging.basicConfig(level=logging.INFO)

class SimpleNvidiaRerank:
    def __init__(self, api_key: str, model_name: str = "nvidia/llama-nemotron-rerank-1b-v2"):
        self.api_key = api_key
        self.model_name = model_name

        if "/" in model_name:
            self._provider, self._model_id = model_name.split("/", 1)
        else:
            self._provider = "nvidia"
            self._model_id = model_name

        self.api_base = f"https://ai.api.nvidia.com/v1/retrieval/{self._provider}/{self._model_id}/reranking"

    def rerank_batch(self, query: str, documents):
        req_body = {
            "model": self.model_name,
            "query": {"text": query},
            "passages": [{"text": doc} for doc in documents],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(url=self.api_base, headers=headers, json=req_body, timeout=30)
        response.raise_for_status()
        result = response.json()

        rankings = result.get("rankings")
        if not rankings:
            return None

        scores = [0.0] * len(documents)
        for item in rankings:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(documents):
                scores[idx] = item.get("logit", 0.0)

        return scores


def main():
    print("=== NVIDIA Rerank API 测试 ===")
    print()

    import os
    api_key = os.environ.get("NVIDIA_API_KEY", "your-api-key-here")
    if api_key == "your-api-key-here":
        print("⚠️  NVIDIA_API_KEY not set, using placeholder")
        print("Set NVIDIA_API_KEY environment variable to test API")
        return
    client = SimpleNvidiaRerank(api_key=api_key)

    print(f"API Base: {client.api_base}")
    print(f"Model: {client.model_name}")
    print()

    query = "什么是人工智能？"
    documents = [
        "人工智能是模拟人类智能的技术",
        "今天天气不错",
        "深度学习是机器学习的一个子集"
    ]

    print(f"Query: {query}")
    print("Documents:")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")
    print()

    scores = client.rerank_batch(query, documents)

    if scores:
        print("✅ API 调用成功！")
        print("Results:")
        for i, (doc, score) in enumerate(zip(documents, scores)):
            print(f"  [{i}] logit={score:.4f}: {doc}")
    else:
        print("❌ API 调用失败")


if __name__ == "__main__":
    main()
