# Stock Recommender (Rust + Google ADK)

Predicts **when** a stock may fall to your buy threshold. If nothing in the universe qualifies, returns `NO_OPPORTUNITY` instead of forcing a pick.

## Architecture

```
universe + buy_below thresholds
        │
        ▼
  price history (demo or Yahoo)
        │
        ▼
  ADK price forecaster ──┬── Gemini REST (ADK-style agent, default)
                         ├── Python google-adk sidecar (official ADK)
                         └── heuristic fallback (offline)
        │
        ▼
  find first date price ≤ buy_below
        │
        ▼
  rank by confidence / date  OR  NO_OPPORTUNITY
```

Official [Google ADK](https://google.github.io/adk-docs/) supports Python, TypeScript, Go, and Java — not Rust. This project:

1. Implements the recommender in **Rust**
2. Forecasts with **Gemini** using an ADK-style agent prompt (default)
3. Optionally runs the **official `google-adk` Python agent** under `adk_agent/`

> Note: the community [ADK-Rust](https://github.com/zavora-ai/adk-rust) crate needs Rust **1.95+**. This repo targets **1.94** and uses Gemini + the official Python ADK instead.

## Quick start (offline heuristic)

```bash
cargo run -- --predictor heuristic
```

Example with custom thresholds:

```bash
cargo run -- --predictor heuristic \
  --buy-below AAPL:180,MSFT:400,GOOG:160,NVDA:100,AMZN:170 \
  --horizon 60 --min-confidence 0.5
```

## Gemini ADK-style predictor

```bash
cp .env.example .env   # set GOOGLE_API_KEY from Google AI Studio
cargo run -- --predictor adk --live --buy-below AAPL:180,MSFT:400
```

## Official Google ADK (Python) sidecar

```bash
cd adk_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=...
uvicorn server:app --host 127.0.0.1 --port 8081
```

Then:

```bash
PREDICTOR=python_adk cargo run -- --predictor python-adk
```

## CLI flags

| Flag | Meaning |
|------|---------|
| `--tickers AAPL,MSFT` | Universe (default demo set) |
| `--buy-below AAPL:180,...` | Buy when predicted price ≤ value (default: 8% below last close) |
| `--horizon 60` | Trading days to search |
| `--min-confidence 0.55` | Reject weak forecasts |
| `--live` | Fetch Yahoo Finance history |
| `--predictor adk\|python-adk\|heuristic` | Forecast backend |
| `--json` | Machine-readable output |

## `NO_OPPORTUNITY`

When no ticker is expected to hit its buy level with enough confidence, the system returns reasons and suggestions (relax confidence, extend horizon, raise thresholds) — it does **not** invent a “least bad” stock.
