//! Threshold-based stock recommender.
//!
//! Predicts *when* a ticker is likely to trade at or below a buy threshold,
//! using a Google ADK (Gemini) agent for price-path forecasts. If the whole
//! universe has no viable candidate, returns [`Recommendation::NoOpportunity`].

pub mod market;
pub mod forecast;
pub mod recommend;
pub mod adk_predictor;
pub mod fyers;

pub use forecast::{BuyWindow, PriceForecast, PredictedPoint};
pub use market::{PriceBar, PriceHistory};
pub use recommend::{
    Candidate, NoOpportunityReport, Recommendation, RecommendRequest, Recommender, UniverseFilter,
};
pub use adk_predictor::{AdkPricePredictor, Predictor, PredictorKind};
pub use fyers::{FyersClient, FyersConfig};
