from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.config import get_settings

settings = get_settings()

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(filename: str) -> str:
    prompt_path = _PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


ORCHESTRATOR_PROMPT = load_prompt("MarketLogicOrchestrator_prompt.txt")
DATA_AGENT_PROMPT = load_prompt("DataAgent_prompt.txt")
DOCUMENT_RETRIEVAL_PROMPT = load_prompt("DocumentRetrievalAgent_prompt.txt")
VALUATION_AGENT_PROMPT = load_prompt("ValuationAgent_prompt.txt")
RISK_AGENT_PROMPT = load_prompt("RiskAgent_prompt.txt")
STRATEGY_AGENT_PROMPT = load_prompt("StrategyAgent_prompt.txt")


def provider_runtime_enabled(provider_enabled: bool) -> bool:
    if not provider_enabled:
        return False
    return bool(settings.google_api_key or settings.google_genai_use_vertexai)


def _extract_json(raw: str) -> dict[str, Any] | list[Any] | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        try:
            return json.loads(text[start_obj : end_obj + 1])
        except Exception:
            pass

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        try:
            return json.loads(text[start_arr : end_arr + 1])
        except Exception:
            pass
    return None


async def run_prompt_json(
    *,
    prompt: str,
    input_payload: dict[str, Any],
    model: str,
) -> dict[str, Any] | list[Any] | None:
    try:
        from google import genai
    except Exception:
        logger.warning("prompt_runtime_genai_import_failed")
        return None

    try:
        if settings.google_api_key:
            client = genai.Client(api_key=settings.google_api_key)
        else:
            client = genai.Client()

        compiled_prompt = (
            f"{prompt}\n\n"
            "Return only valid JSON. Do not include markdown, code fences, or extra narration.\n\n"
            f"INPUT:\n{json.dumps(input_payload, ensure_ascii=True)}"
        )
        response = client.models.generate_content(
            model=model,
            contents=compiled_prompt,
        )
        raw_text = getattr(response, "text", "") or ""
        return _extract_json(raw_text)
    except Exception as exc:
        logger.warning("prompt_runtime_generation_failed error={}", str(exc))
        return None
