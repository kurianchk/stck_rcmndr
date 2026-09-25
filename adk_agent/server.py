"""HTTP sidecar that runs the official Google ADK agent for price forecasts."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

# google-adk Runner API (0.x / 1.x compatible import paths)
try:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Install google-adk and google-genai: pip install -r adk_agent/requirements.txt"
    ) from exc

try:
    from .agent import root_agent
except ImportError:  # running as `uvicorn server:app` inside adk_agent/
    from agent import root_agent

APP_NAME = "stock_recommender"
USER_ID = "rust_client"
SESSION_ID = "predict"


class PredictRequest(BaseModel):
    ticker: str
    horizon_days: int = Field(ge=1, le=365)
    recent_closes: list[dict[str, Any]]


app = FastAPI(title="Google ADK stock price forecaster")
_session_service = InMemorySessionService()
_runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=_session_service)


@app.on_event("startup")
async def _startup() -> None:
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
        print("WARNING: set GOOGLE_API_KEY for the ADK agent to call Gemini")
    await _session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": root_agent.name}


@app.post("/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    closes = ", ".join(
        f"{c.get('date')}:{float(c.get('close')):.2f}" for c in req.recent_closes[-60:]
    )
    prompt = (
        f"Forecast the next {req.horizon_days} trading days for {req.ticker}.\n"
        f"recent_closes=[{closes}]\n"
        f"Return JSON with ticker, horizon_days={req.horizon_days}, "
        "hit_probability, rationale, points[]."
    )

    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_text = ""
    async for event in _runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text += part.text

    payload = _extract_json(final_text)
    if payload is None:
        return {
            "ticker": req.ticker,
            "horizon_days": req.horizon_days,
            "hit_probability": 0.0,
            "rationale": f"ADK agent returned unparseable output: {final_text[:200]}",
            "points": [],
        }
    return payload


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
