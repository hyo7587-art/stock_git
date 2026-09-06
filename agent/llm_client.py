"""
agent/llm_client.py
===================
LLM 래퍼. Google Gemini API 전용 (최신 모델 자동 호환).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Generator

import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("DEFAULT_GEMINI_API_KEY", "")

DEFAULT_SYSTEM = (
    "당신은 한국 금융 분석가입니다. 사용자가 다른 언어를 명시적으로 요청하지 않는 한 "
    "반드시 자연스러운 한국어로만 작성하고, 하나의 답변 안에서 다른 언어 문자를 섞지 마세요."
)


def _check_gemini() -> bool:
    return bool(GEMINI_API_KEY)


def _generate_gemini(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    # 2026 권장 모델 및 검증된 안정 텍스트 모델 목록
    models = [
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
    ]

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    headers = {"Content-Type": "application/json"}

    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            # 404(모델 지원 종료/미지원), 400(모달리티 불일치), 503(일시적 과부하) 발생 시 다음 안정 모델로 즉시 전환
            logger.info(f"[llm_client] {model_name} (상태: {resp.status_code}) -> 다음 모델 시도")
        except Exception:
            continue

    logger.error("[llm_client] 모든 지원 모델 호출 실패")
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

    logger.warning("[llm_client] LLM 미가용 또는 응답 생성 실패")
    return ""


def stream_generate(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    text = generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    if text:
        yield text


def is_available() -> bool:
    return _check_gemini()


def reset_cache():
    pass
