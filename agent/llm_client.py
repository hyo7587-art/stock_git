"""
agent/llm_client.py
===================
LLM 래퍼. Gemini API 우선 지원 및 Ollama 로컬 폴백.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Generator

import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("DEFAULT_GEMINI_API_KEY", "")
OLLAMA_BASE    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL  = os.getenv("REPORT_LLM_MODEL", "qwen2.5:7b")
FALLBACK_MODEL = "llama3.2:latest"

DEFAULT_SYSTEM = (
    "당신은 한국 금융 분석가입니다. 사용자가 다른 언어를 명시적으로 요청하지 않는 한 "
    "반드시 자연스러운 한국어로만 작성하고, 하나의 답변 안에서 다른 언어 문자를 섞지 마세요."
)

_ollama_available: bool | None = None


def _check_gemini() -> bool:
    return bool(GEMINI_API_KEY)


def _check_ollama() -> bool:
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.ok:
            names = {m["name"] for m in resp.json().get("models", [])}
            _ollama_available = bool(names)
        else:
            _ollama_available = False
    except Exception:
        _ollama_available = False
    return _ollama_available


def _pick_ollama_model() -> str:
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.ok:
            names = {m["name"] for m in resp.json().get("models", [])}
            for candidate in (DEFAULT_MODEL, "qwen2.5:7b", FALLBACK_MODEL):
                if candidate in names:
                    return candidate
    except Exception:
        pass
    return DEFAULT_MODEL


def _generate_gemini(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    # Google AI Studio Gemini API 직접 호출
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        logger.error(f"[llm_client] Gemini generate 실패: {e}")
        return ""


def _generate_ollama(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    model = _pick_ollama_model()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if system:
        payload["system"] = system
    try:
        resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"[llm_client] Ollama generate 실패 ({model}): {e}")
        return ""


def generate(
    prompt: str,
    *,
    system: str = "",
    _model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    if not system:
        system = DEFAULT_SYSTEM

    if _check_gemini():
        logger.info("[llm_client] Gemini API 사용")
        res = _generate_gemini(prompt, system, temperature, max_tokens)
        if res:
            return res

    if _check_ollama():
        logger.info("[llm_client] Ollama 사용")
        return _generate_ollama(prompt, system, temperature, max_tokens)

    logger.warning("[llm_client] LLM 미가용 — 빈 응답 반환")
    return ""


def stream_generate(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    # 스트리밍 시에도 텍스트 통째 반환 지원
    text = generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    if text:
        yield text


def is_available() -> bool:
    return _check_gemini() or _check_ollama()


def reset_cache():
    global _ollama_available
    _ollama_available = None
