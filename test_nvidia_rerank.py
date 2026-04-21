#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 NVIDIA Rerank 配置
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from openviking_cli.utils.config.rerank_config import RerankConfig
from openviking.models.rerank.volcengine_rerank import RerankClient


def test_nvidia_config():
    """测试 NVIDIA rerank 配置解析和客户端创建"""

    print("=== 测试 NVIDIA Rerank 配置 ===")
    print()

    # Test 1: 使用 model name 自动检测 provider
    print("Test 1: 自动检测 provider (从 model name)")
    config1 = RerankConfig(
        api_key="nvapi-test123",
        model="nvidia/llama-nemotron-rerank-1b-v2"
    )
    print(f"  provider: {config1._effective_provider()}")
    print(f"  is_available: {config1.is_available()}")
    assert config1._effective_provider() == "nvidia"
    assert config1.is_available()
    print("  ✅ Pass")
    print()

    # Test 2: 显式指定 provider
    print("Test 2: 显式指定 provider='nvidia'")
    config2 = RerankConfig(
        provider="nvidia",
        api_key="nvapi-test456",
        model="nvidia/llama-nemotron-rerank-1b-v2"
    )
    print(f"  provider: {config2._effective_provider()}")
    print(f"  is_available: {config2.is_available()}")
    assert config2._effective_provider() == "nvidia"
    assert config2.is_available()
    print("  ✅ Pass")
    print()

    # Test 3: 创建客户端
    print("Test 3: 创建 NvidiaRerankClient")
    client = RerankClient.from_config(config2)
    print(f"  client type: {type(client).__name__}")
    print(f"  api_base: {client.api_base}")
    print(f"  model_name: {client.model_name}")
    assert "nvidia" in str(type(client).__module__.lower())
    assert "reranking" in client.api_base
    print("  ✅ Pass")
    print()

    # Test 4: API 调用测试
    print("Test 4: API 调用测试 (需要有效 API Key)")
    import os
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "your-api-key-here")
    if nvidia_api_key == "your-api-key-here":
        print("  ⚠️  NVIDIA_API_KEY not set, skipping API test")
        print()
        return
    config4 = RerankConfig(
        provider="nvidia",
        api_key=nvidia_api_key,
        model="nvidia/llama-nemotron-rerank-1b-v2"
    )
    client4 = RerankClient.from_config(config4)

    query = "什么是人工智能？"
    documents = [
        "人工智能是模拟人类智能的技术",
        "今天天气不错",
        "深度学习是机器学习的一个子集"
    ]

    scores = client4.rerank_batch(query, documents)
    if scores:
        print(f"  query: {query}")
        for i, (doc, score) in enumerate(zip(documents, scores)):
            print(f"  [{i}] score={score:.4f}: {doc}")
        print("  ✅ Pass - API 调用成功")
    else:
        print("  ⚠️  API 调用失败 (可能网络问题或 Key 过期)")
    print()

    print("=== 所有测试完成 ===")


if __name__ == "__main__":
    test_nvidia_config()
