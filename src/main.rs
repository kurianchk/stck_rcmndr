//! CLI for the stock buy-date recommender.

use std::collections::HashMap;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use tracing_subscriber::EnvFilter;

use stock_recommender::adk_predictor::{AdkPricePredictor, Predictor};
use stock_recommender::market::{demo_universe, fetch_yahoo_history, PriceHistory};
use stock_recommender::recommend::{
    RecommendRequest, Recommendation, Recommender, TickerThreshold,
};

#[derive(Parser, Debug)]
#[command(
    name = "stock_recommender",
    about = "Predict when stocks may hit your buy threshold (Google ADK / Gemini)"
)]
struct Cli {
    /// Comma-separated tickers (default: demo universe AAPL,MSFT,GOOG,NVDA,AMZN)
    #[arg(long)]
    tickers: Option<String>,

    /// Buy thresholds as TICKER:PRICE pairs, e.g. AAPL:180,MSFT:400
    #[arg(long, value_delimiter = ',')]
    buy_below: Vec<String>,

    /// Forecast horizon in trading days
    #[arg(long, default_value_t = 60)]
    horizon: u32,

    /// Minimum hit probability (0..1) to accept a candidate
    #[arg(long, default_value_t = 0.55)]
    min_confidence: f64,

    /// Use live Yahoo Finance history instead of demo series
    #[arg(long)]
    live: bool,

    /// Predictor backend
    #[arg(long, value_enum, default_value_t = PredictorArg::Auto)]
    predictor: PredictorArg,

    /// Pretty-print JSON only
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Debug, ValueEnum)]
enum PredictorArg {
    Auto,
    Adk,
    PythonAdk,
    Heuristic,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    apply_predictor_env(&cli.predictor);

    let histories = load_histories(&cli).await?;
    let thresholds = resolve_thresholds(&cli, &histories)?;
    let predictor = AdkPricePredictor::from_env();

    eprintln!(
        "predictor={:?}  universe={}  horizon={}d  min_confidence={:.0}%",
        predictor.kind(),
        histories.len(),
        cli.horizon,
        cli.min_confidence * 100.0
    );

    let mut pairs = Vec::new();
    for h in &histories {
        let forecast = predictor.predict(h, cli.horizon).await?;
        pairs.push((h.clone(), forecast));
    }

    let req = RecommendRequest {
        horizon_days: cli.horizon,
        min_confidence: cli.min_confidence,
        buy_thresholds: thresholds,
    };
    let recommendation = Recommender::recommend(&pairs, &req);

    if cli.json {
        println!("{}", serde_json::to_string_pretty(&recommendation)?);
    } else {
        print_human(&recommendation);
    }

    Ok(())
}

fn apply_predictor_env(arg: &PredictorArg) {
    match arg {
        PredictorArg::Auto => {}
        PredictorArg::Adk => std::env::set_var("PREDICTOR", "adk"),
        PredictorArg::PythonAdk => std::env::set_var("PREDICTOR", "python_adk"),
        PredictorArg::Heuristic => std::env::set_var("PREDICTOR", "heuristic"),
    }
}

async fn load_histories(cli: &Cli) -> Result<Vec<PriceHistory>> {
    if let Some(list) = &cli.tickers {
        let tickers: Vec<&str> = list.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        let mut out = Vec::new();
        for t in tickers {
            if cli.live {
                out.push(
                    fetch_yahoo_history(t, "6mo")
                        .await
                        .with_context(|| format!("fetch {t}"))?,
                );
            } else {
                // Demo series keyed by ticker hash-ish defaults.
                let start = match t.to_ascii_uppercase().as_str() {
                    "AAPL" => 195.0,
                    "MSFT" => 420.0,
                    "GOOG" | "GOOGL" => 175.0,
                    "NVDA" => 130.0,
                    "AMZN" => 190.0,
                    _ => 100.0,
                };
                let drift = match t.to_ascii_uppercase().as_str() {
                    "NVDA" | "MSFT" => 0.001,
                    _ => -0.0008,
                };
                out.push(stock_recommender::market::demo_history(t, 120, start, drift));
            }
        }
        Ok(out)
    } else if cli.live {
        let mut out = Vec::new();
        for t in ["AAPL", "MSFT", "GOOG", "NVDA", "AMZN"] {
            out.push(fetch_yahoo_history(t, "6mo").await.with_context(|| format!("fetch {t}"))?);
        }
        Ok(out)
    } else {
        Ok(demo_universe())
    }
}

fn resolve_thresholds(cli: &Cli, histories: &[PriceHistory]) -> Result<Vec<TickerThreshold>> {
    let mut map: HashMap<String, f64> = HashMap::new();
    for item in &cli.buy_below {
        let (ticker, price) = item
            .split_once(':')
            .with_context(|| format!("expected TICKER:PRICE, got {item}"))?;
        map.insert(
            ticker.trim().to_ascii_uppercase(),
            price.trim().parse().context("price must be a number")?,
        );
    }

    // Default: buy 8% below latest close when not specified.
    let mut out = Vec::new();
    for h in histories {
        let buy_below = map
            .get(&h.ticker)
            .copied()
            .or_else(|| h.latest_close().map(|c| (c * 0.92 * 100.0).round() / 100.0))
            .unwrap_or(0.0);
        out.push(TickerThreshold {
            ticker: h.ticker.clone(),
            buy_below,
        });
    }
    Ok(out)
}

fn print_human(rec: &Recommendation) {
    match rec {
        Recommendation::BuyCandidates { ranked } => {
            println!("BUY CANDIDATES ({})\n", ranked.len());
            for (i, c) in ranked.iter().enumerate() {
                let w = &c.window;
                println!(
                    "{}. {} — buy on or around {} (expected ~${:.2}, threshold ≤ ${:.2})",
                    i + 1,
                    w.ticker,
                    w.earliest_date,
                    w.expected_price,
                    w.buy_threshold
                );
                println!(
                    "   confidence {:.0}% | source={} | {}",
                    w.hit_probability * 100.0,
                    c.forecast.source,
                    w.rationale
                );
            }
        }
        Recommendation::NoOpportunity(report) => {
            println!("NO_OPPORTUNITY");
            println!("{}\n", report.message);
            println!("Filter breakdown:");
            println!(
                "  never crosses horizon: {}",
                join_or_none(&report.filter.rejected_never_crosses)
            );
            println!(
                "  low confidence:        {}",
                join_or_none(&report.filter.rejected_low_confidence)
            );
            println!(
                "  missing threshold:     {}",
                join_or_none(&report.filter.rejected_missing_threshold)
            );
            println!("\nSuggestions:");
            for s in &report.suggestions {
                println!("  - {s}");
            }
        }
    }
}

fn join_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "—".into()
    } else {
        items.join(", ")
    }
}
