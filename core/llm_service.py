#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM服务集成模块
支持多个国内LLM提供商（DeepSeek、通义千问、智谱AI）

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
"""

import os
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import requests

# 加载.env文件
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    provider: str
    response_time_ms: float
    tokens_used: int = 0


class LLMProvider:
    """LLM提供商基类"""

    def __init__(self, api_key: str, model: str, temperature: float = 0.2, max_tokens: int = 2048):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """聊天接口"""
        raise NotImplementedError


class DeepSeekProvider(LLMProvider):
    """DeepSeek提供商（国内首选）"""

    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 temperature: float = 0.2, max_tokens: int = 2048):
        super().__init__(api_key, model, temperature, max_tokens)
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.provider_name = "DeepSeek"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        response_time = (time.time() - start_time) * 1000

        return LLMResponse(
            content=result["choices"][0]["message"]["content"],
            model=self.model,
            provider="DeepSeek",
            response_time_ms=response_time,
            tokens_used=result.get("usage", {}).get("total_tokens", 0)
        )


class DashScopeProvider(LLMProvider):
    """通义千问提供商（阿里云）"""

    def __init__(self, api_key: str, model: str = "qwen-plus",
                 temperature: float = 0.2, max_tokens: int = 2048):
        super().__init__(api_key, model, temperature, max_tokens)
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.provider_name = "DashScope"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 通义千问的格式略有不同
        data = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "result_format": "message"
            }
        }

        response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        response_time = (time.time() - start_time) * 1000

        return LLMResponse(
            content=result["output"]["choices"][0]["message"]["content"],
            model=self.model,
            provider="DashScope",
            response_time_ms=response_time,
            tokens_used=result.get("usage", {}).get("total_tokens", 0)
        )


class ZhipuProvider(LLMProvider):
    """智谱AI提供商"""

    def __init__(self, api_key: str, model: str = "glm-4-flash",
                 temperature: float = 0.2, max_tokens: int = 2048):
        super().__init__(api_key, model, temperature, max_tokens)
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.provider_name = "ZhipuAI"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        response_time = (time.time() - start_time) * 1000

        return LLMResponse(
            content=result["choices"][0]["message"]["content"],
            model=self.model,
            provider="ZhipuAI",
            response_time_ms=response_time,
            tokens_used=result.get("usage", {}).get("total_tokens", 0)
        )


class UnifiedLLMService:
    """统一LLM服务（支持多提供商）"""

    def __init__(self):
        self.providers: List[LLMProvider] = []
        self.current_provider_index = 0
        self._init_providers()

    def _init_providers(self):
        """初始化LLM提供商（按优先级）"""

        # 1. DeepSeek（国内首选）
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            try:
                provider = DeepSeekProvider(
                    api_key=deepseek_key,
                    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                    temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2")),
                    max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048"))
                )
                self.providers.append(provider)
                print(f"[LLM] [OK] DeepSeek已初始化")
            except Exception as e:
                print(f"[LLM] [!] DeepSeek初始化失败: {e}")

        # 2. 通义千问 DashScope
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            try:
                provider = DashScopeProvider(
                    api_key=dashscope_key,
                    model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
                    temperature=float(os.getenv("DASHSCOPE_TEMPERATURE", "0.2")),
                    max_tokens=int(os.getenv("DASHSCOPE_MAX_TOKENS", "2048"))
                )
                self.providers.append(provider)
                print(f"[LLM] [OK] DashScope已初始化")
            except Exception as e:
                print(f"[LLM] [!] DashScope初始化失败: {e}")

        # 3. 智谱AI
        zhipu_key = os.getenv("ZHIPU_API_KEY")
        if zhipu_key:
            try:
                provider = ZhipuProvider(
                    api_key=zhipu_key,
                    model=os.getenv("ZHIPU_MODEL", "glm-4-flash"),
                    temperature=float(os.getenv("ZHIPU_TEMPERATURE", "0.2")),
                    max_tokens=int(os.getenv("ZHIPU_MAX_TOKENS", "2048"))
                )
                self.providers.append(provider)
                print(f"[LLM] [OK] ZhipuAI已初始化")
            except Exception as e:
                print(f"[LLM] [!] ZhipuAI初始化失败: {e}")

        if not self.providers:
            raise ValueError("[X] 没有可用的LLM提供商！请检查.env配置")

        print(f"[LLM] 共初始化 {len(self.providers)} 个提供商")

    def chat(self, user_message: str, system_prompt: Optional[str] = None,
             conversation_history: Optional[List[Dict[str, str]]] = None) -> LLMResponse:
        """
        聊天接口

        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            conversation_history: 对话历史

        Returns:
            LLMResponse: LLM响应
        """

        # 构建消息列表
        messages = []

        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 添加对话历史
        if conversation_history:
            messages.extend(conversation_history)

        # 添加用户消息
        messages.append({"role": "user", "content": user_message})

        # 尝试所有提供商（按优先级）
        last_error = None
        for i in range(len(self.providers)):
            provider_index = (self.current_provider_index + i) % len(self.providers)
            provider = self.providers[provider_index]

            try:
                response = provider.chat(messages)
                print(f"[LLM] 使用{provider.provider_name} ({response.response_time_ms:.0f}ms)")
                return response

            except Exception as e:
                print(f"[LLM] {provider.provider_name}调用失败: {e}")
                last_error = e
                continue

        # 如果所有提供商都失败
        raise RuntimeError(f"所有LLM提供商都不可用: {last_error}")

    def switch_provider(self, index: int):
        """切换提供商"""
        if 0 <= index < len(self.providers):
            self.current_provider_index = index
            print(f"[LLM] 切换到提供商 {index}: {self.providers[index].provider_name}")
        else:
            raise IndexError(f"提供商索引超出范围: {index}")

    def get_current_provider(self) -> str:
        """获取当前提供商"""
        return self.providers[self.current_provider_index].provider_name


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "LLM服务测试")
    print("="*70)

    try:
        # 初始化LLM服务
        llm = UnifiedLLMService()

        # 测试1：简单对话
        print("\n[测试1] 简单对话")
        response = llm.chat("你好，请用一句话介绍你自己")
        print(f"\n[回复] {response.content}")
        print(f"[提供商] {response.provider}")
        print(f"[模型] {response.model}")
        print(f"[响应时间] {response.response_time_ms:.0f}ms")
        print(f"[Token使用] {response.tokens_used}")

        # 测试2：带系统提示
        print("\n" + "="*70)
        print("[测试2] 带系统提示")
        response = llm.chat(
            user_message="什么是强化学习？",
            system_prompt="你是一个AI研究助手，擅长用简洁的语言解释复杂概念。"
        )
        print(f"\n[回复] {response.content}")
        print(f"[提供商] {response.provider}")
        print(f"[响应时间] {response.response_time_ms:.0f}ms")

        # 测试3：多轮对话
        print("\n" + "="*70)
        print("[测试3] 多轮对话")
        history = [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好小明！很高兴认识你。"}
        ]
        response = llm.chat(
            user_message="你还记得我的名字吗？",
            conversation_history=history
        )
        print(f"\n[回复] {response.content}")
        print(f"[提供商] {response.provider}")

        print("\n" + "="*70)
        print("[OK] LLM服务测试完成")

    except Exception as e:
        print(f"\n[X] 测试失败: {e}")
        import traceback
        traceback.print_exc()
