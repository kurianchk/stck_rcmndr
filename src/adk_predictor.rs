//! Price predictors: Google Gemini (ADK-style) + official Python Google ADK + heuristic.
//!
//! Official [Google ADK](https://google.github.io/adk-docs/) is Python/TS/Go/Java.
//! This Rust crate:
//! 1. Calls **Gemini** with an ADK-style agent prompt + tool schema (default when
//!    `GOOGLE_API_KEY` is set).
//! 2. Optionally calls the **Python `google-adk`** sidecar in `adk_agent/`
//!    (`PREDICTOR=python_adk`).
//! 3. Falls back to a local heuristic model (offline demos).

use anyhow::{anyhow, Context, Result};
use chrono::NaiveDate;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::forecast::{heuristic_forecast, PredictedPoint, PriceForecast};
use crate::market::PriceHistory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictorKind {
    /// Gemini via REST using an ADK-style agent instruction (needs `GOOGLE_API_KEY`).
    AdkGemini,
    /// Official Google ADK Python sidecar over HTTP.
    PythonAdk,
    /// Local drift/vol model (offline).
    Heuristic,
}

impl PredictorKind {
    pub fn from_env() -> Self {
        match std::env::var("PREDICTOR")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "python" | "python_adk" | "google_adk" => Self::PythonAdk,
            "heuristic" | "local" => Self::Heuristic,
            "adk" | "gemini" | "adk_gemini" => Self::AdkGemini,
            _ => {
                if std::env::var("GOOGLE_API_KEY").is_ok() {
                    Self::AdkGemini
                } else {
                    Self::Heuristic
                }
            }
        }
    }
}

#[async_trait::async_trait]
pub trait Predictor: Send + Sync {
    async fn predict(&self, history: &PriceHistory, horizon_days: u32) -> Result<PriceForecast>;
}

pub struct AdkPricePredictor {
    kind: PredictorKind,
    python_url: String,
}

impl AdkPricePredictor {
    pub fn from_env() -> Self {
        Self {
            kind: PredictorKind::from_env(),
            python_url: std::env::var("ADK_PYTHON_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8081/predict".into()),
        }
    }

    pub fn kind(&self) -> PredictorKind {
        self.kind
    }
}

#[async_trait::async_trait]
impl Predictor for AdkPricePredictor {
    async fn predict(&self, history: &PriceHistory, horizon_days: u32) -> Result<PriceForecast> {
        match self.kind {
            PredictorKind::Heuristic => Ok(heuristic_forecast(history, horizon_days)),
            PredictorKind::PythonAdk => {
                predict_via_python_adk(&self.python_url, history, horizon_days).await
            }
            PredictorKind::AdkGemini => predict_via_gemini_adk_agent(history, horizon_days).await,
        }
    }
}

const AGENT_SYSTEM: &str = r#"You are stock_price_forecaster, an ADK-style agent tool for a stock recommender.
Your only job: emit a future daily price path as JSON.

Tool contract (describe_forecast_schema):
{
  "ticker": "SYM",
  "horizon_days": 30,
  "hit_probability": 0.0,
  "rationale": "short reason",
  "points": [
    {"date": "YYYY-MM-DD", "expected_price": 0.0, "low": 0.0, "high": 0.0}
  ]
}

Rules:
- One point per trading day for the full horizon; skip weekends.
- hit_probability = chance price reaches a meaningful buy dip within the horizon (0..1).
- Base the path on provided recent closes (drift + mild mean-reversion + uncertainty bands).
- Respond with ONLY a single JSON object. No markdown fences. No investment advice prose.
"#;

async fn predict_via_gemini_adk_agent(
    history: &PriceHistory,
    horizon_days: u32,
) -> Result<PriceForecast> {
    let api_key = std::env::var("GOOGLE_API_KEY")
        .context("GOOGLE_API_KEY required for Gemini ADK-style predictor")?;
    let model =
        std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());

    let user = format!(
        "Call describe_forecast_schema mentally, then forecast the next {horizon_days} trading days.\n{}\n\
         Return JSON with ticker, horizon_days={horizon_days}, hit_probability, rationale, points[].",
        history.prompt_summary(60)
    );

    info!(ticker = %history.ticker, model = %model, "running Gemini ADK-style forecast agent");

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    );
    let body = json!({
        "system_instruction": { "parts": [{ "text": AGENT_SYSTEM }] },
        "contents": [{ "role": "user", "parts": [{ "text": user }] }],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    });

    let client = reqwest::Client::new();
    let resp: Value = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Gemini HTTP request failed")?
        .error_for_status()
        .context("Gemini returned an error status")?
        .json()
        .await
        .context("Gemini response was not JSON")?;

    let text = extract_gemini_text(&resp)
        .ok_or_else(|| anyhow!("Gemini response missing text: {resp}"))?;

    parse_forecast_json(&text, history, horizon_days, "google-gemini-adk-agent")
}

fn extract_gemini_text(resp: &Value) -> Option<String> {
    let parts = resp
        .pointer("/candidates/0/content/parts")?
        .as_array()?;
    let mut out = String::new();
    for p in parts {
        if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
            out.push_str(t);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

async fn predict_via_python_adk(
    url: &str,
    history: &PriceHistory,
    horizon_days: u32,
) -> Result<PriceForecast> {
    let client = reqwest::Client::new();
    let body = json!({
        "ticker": history.ticker,
        "horizon_days": horizon_days,
        "recent_closes": history.bars.iter().rev().take(60).rev().map(|b| {
            json!({"date": b.date.to_string(), "close": b.close})
        }).collect::<Vec<_>>(),
    });
    let resp = client
        .post(url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("Python ADK sidecar unreachable at {url}"))?
        .error_for_status()
        .context("Python ADK HTTP error")?
        .text()
        .await?;
    parse_forecast_json(&resp, history, horizon_days, "google-adk-python")
}

#[derive(Debug, Deserialize)]
struct RawForecast {
    ticker: Option<String>,
    horizon_days: Option<u32>,
    hit_probability: Option<f64>,
    rationale: Option<String>,
    points: Vec<RawPoint>,
}

#[derive(Debug, Deserialize)]
struct RawPoint {
    date: String,
    expected_price: f64,
    low: Option<f64>,
    high: Option<f64>,
}

fn parse_forecast_json(
    text: &str,
    history: &PriceHistory,
    horizon_days: u32,
    source: &str,
) -> Result<PriceForecast> {
    let json_str = extract_json(text).ok_or_else(|| anyhow!("no JSON object in model output"))?;
    match serde_json::from_str::<RawForecast>(json_str) {
        Ok(raw) if !raw.points.is_empty() => {
            let mut points = Vec::new();
            for p in raw.points {
                let date = NaiveDate::parse_from_str(&p.date, "%Y-%m-%d")
                    .with_context(|| format!("bad date {}", p.date))?;
                points.push(PredictedPoint {
                    date,
                    expected_price: p.expected_price,
                    low: p.low,
                    high: p.high,
                });
            }
            Ok(PriceForecast {
                ticker: raw.ticker.unwrap_or_else(|| history.ticker.clone()),
                horizon_days: raw.horizon_days.unwrap_or(horizon_days),
                points,
                hit_probability: raw.hit_probability.unwrap_or(0.5).clamp(0.0, 1.0),
                rationale: raw.rationale.unwrap_or_else(|| "ADK forecast".into()),
                source: source.into(),
            })
        }
        Ok(_) | Err(_) => {
            warn!("failed to parse ADK JSON; falling back to heuristic");
            let mut f = heuristic_forecast(history, horizon_days);
            f.source = format!("{source}+heuristic-fallback");
            f.rationale = format!(
                "ADK parse failed; used heuristic. Raw snippet: {}",
                text.chars().take(180).collect::<String>()
            );
            Ok(f)
        }
    }
}

fn extract_json(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return Some(&trimmed[start..=end]);
        }
    }
    None
}
