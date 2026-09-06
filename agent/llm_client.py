def _generate_gemini(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    # Google AI Studio Gemini API 호출 (최신 2.0 및 1.5 flash 대응)
    models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    headers = {
        "Content-Type": "application/json"
    }

    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.error(f"[llm_client] {model_name} HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"[llm_client] {model_name} 요청 예외: {e}")
            continue

    logger.error("[llm_client] 모든 Gemini 모델 호출 실패")
    return ""
