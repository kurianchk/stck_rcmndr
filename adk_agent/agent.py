"""Official Google ADK stock price forecaster.

Run:
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  export GOOGLE_API_KEY=...
  uvicorn server:app --host 127.0.0.1 --port 8081

Rust client:
  PREDICTOR=python_adk cargo run -- --predictor python-adk
"""

from google.adk.agents.llm_agent import Agent


def describe_forecast_schema(ticker: str) -> dict:
    """Return the JSON schema the forecaster must emit."""
    return {
        "ticker": ticker,
        "required_json_shape": {
            "ticker": "SYM",
            "horizon_days": 30,
            "hit_probability": 0.0,
            "rationale": "short reason",
            "points": [
                {
                    "date": "YYYY-MM-DD",
                    "expected_price": 0.0,
                    "low": 0.0,
                    "high": 0.0,
                }
            ],
        },
        "rules": [
            "One point per trading day for the full horizon",
            "Skip weekends",
            "hit_probability in [0,1]",
            "Respond with ONLY a single JSON object",
        ],
    }


root_agent = Agent(
    model="gemini-2.5-flash",
    name="stock_price_forecaster",
    description="Predicts future daily stock closes as structured JSON for a buy-date recommender.",
    instruction=(
        "You are a quantitative price-path forecaster inside a stock recommender. "
        "Always call describe_forecast_schema first, then output ONLY valid JSON matching "
        "that schema. Base the path on the provided recent closes (drift + mean-reversion). "
        "Do not give investment advice beyond the numeric forecast. No markdown fences."
    ),
    tools=[describe_forecast_schema],
)
