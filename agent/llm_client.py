def _generate_gemini(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    # Google AI Studio Gemini API 호출 (안정 엔드포인트 및 최신/호환 모델 순차 폴백)
    models = ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-pro"]
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"[llm_client] {model_name} 호출 실패: {e}")
            continue

    logger.error("[llm_client] 모든 Gemini 모델 호출 실패")
    return ""
