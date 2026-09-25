//! Forecast types and buy-window detection.

use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};

use crate::market::PriceHistory;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedPoint {
    pub date: NaiveDate,
    pub expected_price: f64,
    /// Optional lower confidence bound.
    pub low: Option<f64>,
    /// Optional upper confidence bound.
    pub high: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceForecast {
    pub ticker: String,
    pub horizon_days: u32,
    pub points: Vec<PredictedPoint>,
    /// Model-estimated probability of hitting the buy threshold within horizon (0..1).
    pub hit_probability: f64,
    pub rationale: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuyWindow {
    pub ticker: String,
    pub buy_threshold: f64,
    pub earliest_date: NaiveDate,
    pub expected_price: f64,
    pub hit_probability: f64,
    pub rationale: String,
}

impl PriceForecast {
    /// First day where expected (or low) price is at/below the buy threshold.
    pub fn first_buy_date(&self, buy_threshold: f64) -> Option<BuyWindow> {
        for p in &self.points {
            let probe = p.low.unwrap_or(p.expected_price);
            if probe <= buy_threshold || p.expected_price <= buy_threshold {
                return Some(BuyWindow {
                    ticker: self.ticker.clone(),
                    buy_threshold,
                    earliest_date: p.date,
                    expected_price: p.expected_price,
                    hit_probability: self.hit_probability,
                    rationale: self.rationale.clone(),
                });
            }
        }
        None
    }
}

/// Simple local fallback: drift + volatility from recent returns (no LLM).
pub fn heuristic_forecast(history: &PriceHistory, horizon_days: u32) -> PriceForecast {
    let closes = history.closes();
    let last = *closes.last().unwrap_or(&100.0);
    let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
    let n = returns.len().max(1) as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(0.002);

    let mut points = Vec::new();
    let mut price = last;
    let mut date = history
        .bars
        .last()
        .map(|b| b.date)
        .unwrap_or_else(|| chrono::Utc::now().date_naive());

    for day in 1..=horizon_days {
        // Advance calendar, skip weekends.
        loop {
            date = date + chrono::Duration::days(1);
            if date.weekday().number_from_monday() <= 5 {
                break;
            }
        }
        price *= 1.0 + mean;
        let band = price * std * (day as f64).sqrt() * 1.65;
        points.push(PredictedPoint {
            date,
            expected_price: (price * 100.0).round() / 100.0,
            low: Some(((price - band) * 100.0).round() / 100.0),
            high: Some(((price + band) * 100.0).round() / 100.0),
        });
    }

    PriceForecast {
        ticker: history.ticker.clone(),
        horizon_days,
        points,
        hit_probability: 0.5,
        rationale: format!(
            "Heuristic drift={mean:.5}/day vol≈{std:.4} from {} bars",
            closes.len()
        ),
        source: "heuristic".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::demo_history;

    #[test]
    fn declining_series_finds_buy() {
        let h = demo_history("X", 60, 100.0, -0.01);
        let f = heuristic_forecast(&h, 40);
        let last = h.latest_close().unwrap();
        let window = f.first_buy_date(last * 0.95);
        assert!(window.is_some());
    }
}
