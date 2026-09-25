//! Universe scanning and NO_OPPORTUNITY handling.

use serde::{Deserialize, Serialize};

use crate::forecast::{BuyWindow, PriceForecast};
use crate::market::PriceHistory;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendRequest {
    /// Maximum days ahead to search for a buy date.
    pub horizon_days: u32,
    /// Minimum hit probability to accept a candidate (0..1).
    pub min_confidence: f64,
    /// Per-ticker buy thresholds. Missing tickers are skipped.
    pub buy_thresholds: Vec<TickerThreshold>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerThreshold {
    pub ticker: String,
    pub buy_below: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseFilter {
    pub rejected_never_crosses: Vec<String>,
    pub rejected_low_confidence: Vec<String>,
    pub rejected_missing_threshold: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub window: BuyWindow,
    pub forecast: PriceForecast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoOpportunityReport {
    pub message: String,
    pub filter: UniverseFilter,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Recommendation {
    BuyCandidates { ranked: Vec<Candidate> },
    NoOpportunity(NoOpportunityReport),
}

pub struct Recommender;

impl Recommender {
    /// Rank stocks that are predicted to hit their buy threshold in-horizon.
    pub fn recommend(
        forecasts: &[(PriceHistory, PriceForecast)],
        req: &RecommendRequest,
    ) -> Recommendation {
        let mut filter = UniverseFilter {
            rejected_never_crosses: vec![],
            rejected_low_confidence: vec![],
            rejected_missing_threshold: vec![],
        };
        let mut candidates = Vec::new();

        for (history, forecast) in forecasts {
            let threshold = req
                .buy_thresholds
                .iter()
                .find(|t| tickers_match(&t.ticker, &history.ticker))
                .map(|t| t.buy_below);

            let Some(buy_below) = threshold else {
                filter
                    .rejected_missing_threshold
                    .push(history.ticker.clone());
                continue;
            };

            // Already at/below target → buy now (use forecast day 0 = today via first point or today).
            if let Some(last) = history.latest_close() {
                if last <= buy_below {
                    let today = history.bars.last().map(|b| b.date).unwrap();
                    candidates.push(Candidate {
                        window: BuyWindow {
                            ticker: history.ticker.clone(),
                            buy_threshold: buy_below,
                            earliest_date: today,
                            expected_price: last,
                            hit_probability: 1.0,
                            rationale: "Already at or below buy threshold".into(),
                        },
                        forecast: forecast.clone(),
                    });
                    continue;
                }
            }

            match forecast.first_buy_date(buy_below) {
                Some(window) if forecast.hit_probability >= req.min_confidence => {
                    candidates.push(Candidate {
                        window,
                        forecast: forecast.clone(),
                    });
                }
                Some(_) => filter
                    .rejected_low_confidence
                    .push(history.ticker.clone()),
                None => filter.rejected_never_crosses.push(history.ticker.clone()),
            }
        }

        if candidates.is_empty() {
            let suggestions = build_suggestions(&filter, req);
            return Recommendation::NoOpportunity(NoOpportunityReport {
                message: "No good stock in the recommendation universe".into(),
                filter,
                suggestions,
            });
        }

        candidates.sort_by(|a, b| {
            b.window
                .hit_probability
                .partial_cmp(&a.window.hit_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.window.earliest_date.cmp(&b.window.earliest_date))
        });

        Recommendation::BuyCandidates {
            ranked: candidates,
        }
    }
}

/// Match `SBIN` to `NSE:SBIN-EQ`, and exact / case-insensitive equality.
fn tickers_match(a: &str, b: &str) -> bool {
    if a.eq_ignore_ascii_case(b) {
        return true;
    }
    let norm = |s: &str| {
        let u = s.to_ascii_uppercase();
        u.rsplit(':')
            .next()
            .unwrap_or(&u)
            .trim_end_matches("-EQ")
            .trim_end_matches("-INDEX")
            .to_string()
    };
    norm(a) == norm(b)
}

fn build_suggestions(filter: &UniverseFilter, req: &RecommendRequest) -> Vec<String> {
    let mut s = Vec::new();
    if !filter.rejected_low_confidence.is_empty() {
        s.push(format!(
            "Lower min_confidence below {:.0}% to reconsider: {}",
            req.min_confidence * 100.0,
            filter.rejected_low_confidence.join(", ")
        ));
    }
    if !filter.rejected_never_crosses.is_empty() {
        s.push(format!(
            "Extend horizon beyond {} days or raise buy thresholds for: {}",
            req.horizon_days,
            filter.rejected_never_crosses.join(", ")
        ));
    }
    if !filter.rejected_missing_threshold.is_empty() {
        s.push(format!(
            "Set buy_below thresholds for: {}",
            filter.rejected_missing_threshold.join(", ")
        ));
    }
    if s.is_empty() {
        s.push("Hold cash / wait; re-scan when market conditions change.".into());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::heuristic_forecast;
    use crate::market::demo_history;

    #[test]
    fn empty_universe_is_no_opportunity() {
        let h = demo_history("UP", 60, 100.0, 0.02);
        let f = heuristic_forecast(&h, 20);
        let req = RecommendRequest {
            horizon_days: 20,
            min_confidence: 0.9,
            buy_thresholds: vec![TickerThreshold {
                ticker: "UP".into(),
                buy_below: 50.0, // unreachable
            }],
        };
        match Recommender::recommend(&[(h, f)], &req) {
            Recommendation::NoOpportunity(r) => {
                assert!(!r.filter.rejected_never_crosses.is_empty() || !r.suggestions.is_empty());
            }
            other => panic!("expected NO_OPPORTUNITY, got {other:?}"),
        }
    }
}
