from __future__ import annotations

import asyncio
import json
import random
import re

import httpx

from .config import Config, ModelConfig

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMError(Exception):
    pass


class BaseLLM:
    cacheable = True

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    async def chat(self, system: str, user: str) -> str:
        raise NotImplementedError


class HTTPLLM(BaseLLM):
    def __init__(self, cfg: ModelConfig, http: httpx.AsyncClient, max_retries: int):
        super().__init__(cfg)
        self.http = http
        self.max_retries = max_retries
        self.api_key = cfg.resolve_api_key()
        if not self.api_key:
            hint = f"环境变量 {cfg.api_key_env}" if cfg.api_key_env else "api_key / api_key_env"
            raise LLMError(f"模型 {cfg.name} 没有配置 API key（请设置 {hint}），或使用 --mock 测试流程")

    def _build_request(self, system: str, user: str) -> tuple[str, dict, dict]:
        raise NotImplementedError

    def _parse_response(self, data: dict) -> str:
        raise NotImplementedError

    async def chat(self, system: str, user: str) -> str:
        url, headers, body = self._build_request(system, user)
        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.http.post(url, headers=headers, json=body)
            except httpx.TransportError as e:
                last_err = f"网络错误: {e!r}"
            else:
                if resp.status_code == 200:
                    try:
                        return self._parse_response(resp.json())
                    except ValueError as e:
                        raise LLMError(f"[{self.cfg.name}] 响应不是合法 JSON: {e}") from e
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRY_STATUS:
                    break
            if attempt < self.max_retries:
                await asyncio.sleep(min(2**attempt, 20) + random.random())
        raise LLMError(f"[{self.cfg.name}] 调用失败: {last_err}")


class OpenAICompatLLM(HTTPLLM):
    def _build_request(self, system, user):
        base = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.cfg.extra_body)
        headers = {"Authorization": f"Bearer {self.api_key}", **self.cfg.extra_headers}
        return f"{base}/chat/completions", headers, body

    def _parse_response(self, data):
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"[{self.cfg.name}] 响应格式异常: {str(data)[:300]}") from e


class AnthropicLLM(HTTPLLM):
    def _build_request(self, system, user):
        base = (self.cfg.base_url or "https://api.anthropic.com").rstrip("/")
        url = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
        body = {
            "model": self.cfg.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        body.update(self.cfg.extra_body)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            **self.cfg.extra_headers,
        }
        return url, headers, body

    def _parse_response(self, data):
        blocks = data.get("content") if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            raise LLMError(f"[{self.cfg.name}] 响应格式异常: {str(data)[:300]}")
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


_TEXT_RE = re.compile(r"<text>\n?(.*?)\n?</text>", re.S)


class MockLLM(BaseLLM):
    """假模型：按关键词确定“正确答案”，再按 mock_accuracy 随机犯错；复核轮次准确率会提高。"""

    cacheable = False

    def __init__(self, cfg: ModelConfig, labels: list[str], keywords: dict[str, list[str]]):
        super().__init__(cfg)
        self.labels = labels
        self.keywords = keywords

    def _truth(self, text: str) -> str:
        for label, words in self.keywords.items():
            if any(w in text for w in words):
                return label
        return self.labels[-1]

    async def chat(self, system: str, user: str) -> str:
        await asyncio.sleep(random.uniform(0.01, 0.05))
        m = _TEXT_RE.search(user)
        text = m.group(1) if m else user
        reviewing = "<peer_opinions>" in user
        rng = random.Random(f"{self.cfg.name}|{text}|{reviewing}")

        acc = self.cfg.mock_accuracy
        if reviewing:
            acc += (1 - acc) * 0.5
        truth = self._truth(text)
        if rng.random() < acc:
            label, conf = truth, rng.uniform(0.7, 0.95)
        else:
            label = rng.choice([lab for lab in self.labels if lab != truth] or [truth])
            conf = rng.uniform(0.4, 0.8)
        return json.dumps(
            {"label": label, "confidence": round(conf, 2), "reason": f"mock 判断为{label}"},
            ensure_ascii=False,
        )


class LocalLLM(BaseLLM):
    """在训练集上训练的 TF-IDF + 逻辑回归分类器，伪装成聊天模型参与投票。

    它不阅读提示词里的分类标准和他人意见，只看 <text> 中的原文，所以复核轮次会坚持原判。
    """

    cacheable = False

    def __init__(self, cfg: ModelConfig, config: Config):
        super().__init__(cfg)
        from .local_model import get_classifier

        fs = config.fewshot
        path = cfg.train_path or fs.path
        self.clf = get_classifier(str(path), tuple(config.task.label_names), fs.text_col, fs.label_col)

    async def chat(self, system: str, user: str) -> str:
        m = _TEXT_RE.search(user)
        label, prob = self.clf.predict(m.group(1) if m else user)
        return json.dumps(
            {"label": label, "confidence": round(prob, 3), "reason": f"训练集分类器预测概率 {prob:.2f}"},
            ensure_ascii=False,
        )


def create_llm(cfg: ModelConfig, http: httpx.AsyncClient, config: Config) -> BaseLLM:
    if cfg.provider == "mock":
        return MockLLM(cfg, config.task.label_names, config.mock_keywords)
    if cfg.provider == "local":
        return LocalLLM(cfg, config)
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg, http, config.pipeline.max_retries)
    return OpenAICompatLLM(cfg, http, config.pipeline.max_retries)
