//! Market price history (demo series + optional Yahoo Finance fetch).

use chrono::{Datelike, Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceBar {
    pub date: NaiveDate,
    pub close: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceHistory {
    pub ticker: String,
    pub bars: Vec<PriceBar>,
}

#[derive(Debug, Error)]
pub enum MarketError {
    #[error("no price data for {0}")]
    Empty(String),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("parse error: {0}")]
    Parse(String),
}

impl PriceHistory {
    pub fn latest_close(&self) -> Option<f64> {
        self.bars.last().map(|b| b.close)
    }

    pub fn closes(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.close).collect()
    }

    /// Compact summary for LLM / ADK agent prompts.
    pub fn prompt_summary(&self, max_points: usize) -> String {
        let n = self.bars.len();
        let start = n.saturating_sub(max_points);
        let slice = &self.bars[start..];
        let lines: Vec<String> = slice
            .iter()
            .map(|b| format!("{}:{:.2}", b.date, b.close))
            .collect();
        format!(
            "ticker={} last_close={:.2} recent_closes=[{}]",
            self.ticker,
            self.latest_close().unwrap_or(0.0),
            lines.join(", ")
        )
    }
}

/// Deterministic synthetic history for offline demos (no network).
pub fn demo_history(ticker: &str, days: i64, start_price: f64, daily_drift: f64) -> PriceHistory {
    let mut bars = Vec::new();
    let mut price = start_price;
    let mut day = Utc::now().date_naive() - Duration::days(days);
    for i in 0..days {
        // Mild oscillation so forecasts are interesting.
        let wave = ((i as f64) * 0.15).sin() * start_price * 0.01;
        price = (price * (1.0 + daily_drift) + wave).max(1.0);
        // Skip weekends roughly.
        if day.weekday().number_from_monday() <= 5 {
            bars.push(PriceBar {
                date: day,
                close: (price * 100.0).round() / 100.0,
            });
        }
        day += Duration::days(1);
    }
    PriceHistory {
        ticker: ticker.to_uppercase(),
        bars,
    }
}

/// Built-in demo universe with mixed trajectories.
pub fn demo_universe() -> Vec<PriceHistory> {
    vec![
        demo_history("AAPL", 120, 195.0, -0.0012), // drifting down → may hit buy
        demo_history("MSFT", 120, 420.0, 0.0008),  // drifting up → unlikely
        demo_history("GOOG", 120, 175.0, -0.0003), // mild down
        demo_history("NVDA", 120, 130.0, 0.0015),  // strong up
        demo_history("AMZN", 120, 190.0, -0.0009), // mild down
    ]
}

#[derive(Debug, Deserialize)]
struct YahooChart {
    chart: YahooChartInner,
}

#[derive(Debug, Deserialize)]
struct YahooChartInner {
    result: Option<Vec<YahooResult>>,
}

#[derive(Debug, Deserialize)]
struct YahooResult {
    timestamp: Option<Vec<i64>>,
    indicators: YahooIndicators,
}

#[derive(Debug, Deserialize)]
struct YahooIndicators {
    quote: Vec<YahooQuote>,
}

#[derive(Debug, Deserialize)]
struct YahooQuote {
    close: Option<Vec<Option<f64>>>,
}

/// Fetch recent daily closes from Yahoo Finance (public chart API).
pub async fn fetch_yahoo_history(ticker: &str, range: &str) -> Result<PriceHistory, MarketError> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range={}",
        ticker, range
    );
    let client = reqwest::Client::builder()
        .user_agent("stock_recommender/0.1")
        .build()?;
    let chart: YahooChart = client.get(&url).send().await?.error_for_status()?.json().await?;
    let result = chart
        .chart
        .result
        .and_then(|mut r| r.pop())
        .ok_or_else(|| MarketError::Empty(ticker.to_string()))?;
    let timestamps = result.timestamp.unwrap_or_default();
    let closes = result
        .indicators
        .quote
        .first()
        .and_then(|q| q.close.clone())
        .unwrap_or_default();

    let mut bars = Vec::new();
    for (ts, close) in timestamps.into_iter().zip(closes.into_iter()) {
        if let Some(c) = close {
            let date = chrono::DateTime::from_timestamp(ts, 0)
                .map(|dt| dt.date_naive())
                .ok_or_else(|| MarketError::Parse(format!("bad timestamp {ts}")))?;
            bars.push(PriceBar { date, close: c });
        }
    }
    if bars.is_empty() {
        return Err(MarketError::Empty(ticker.to_string()));
    }
    Ok(PriceHistory {
        ticker: ticker.to_uppercase(),
        bars,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_has_weekday_bars() {
        let h = demo_history("TEST", 30, 100.0, 0.0);
        assert!(!h.bars.is_empty());
        assert!(h.latest_close().is_some());
    }
}
