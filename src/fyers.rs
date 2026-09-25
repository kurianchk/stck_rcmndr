//! Fyers API v3 client — auth + historical candles.
//!
//! Auth flow mirrors `/Users/kurian/code/trading`:
//! 1. Open `generate-authcode` URL (redirect → local `/token`)
//! 2. Exchange `auth_code` via `POST /api/v3/validate-authcode`
//! 3. Call data APIs with `Authorization: {app_id}:{access_token}`
//!
//! Env (same names as the trading project):
//! - `FYR_APP_ID` — app id / client_id (e.g. `XXXXXX-100`)
//! - `FYR_APP_N_SCRT` — SHA-256 hex of `{app_id}:{app_secret}`
//! - `FYERS_REDIRECT_URI` — must match the app (default `http://kurianck.fyers/token`)
//! - `FYERS_ACCESS_TOKEN` — optional pre-set token (skips login)
//!
//! Docs: https://myapi.fyers.in/docsv3 (App Creation / Authentication / Historical Data)

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use chrono::{Duration, NaiveDate, Utc};
use reqwest::header::{AUTHORIZATION, USER_AGENT};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{oneshot, Mutex};
use tracing::{info, warn};

use crate::market::{PriceBar, PriceHistory};

const API_BASE: &str = "https://api-t1.fyers.in";
const USER_AGENT_VALUE: &str = "kurians_rust_api_call";
const DEFAULT_REDIRECT: &str = "http://kurianck.fyers/token";
const TOKEN_CACHE_FILE: &str = ".fyers_token";

#[derive(Debug, Clone)]
pub struct FyersConfig {
    pub app_id: String,
    /// Precomputed SHA-256(`app_id:secret`) hex — env `FYR_APP_N_SCRT`.
    pub app_id_hash: String,
    pub redirect_uri: String,
}

impl FyersConfig {
    pub fn from_env() -> Result<Self> {
        let app_id = std::env::var("FYR_APP_ID")
            .context("FYR_APP_ID missing (same as trading project)")?;
        let app_id_hash = std::env::var("FYR_APP_N_SCRT")
            .context("FYR_APP_N_SCRT missing (SHA-256 of app_id:secret)")?;
        let redirect_uri =
            std::env::var("FYERS_REDIRECT_URI").unwrap_or_else(|_| DEFAULT_REDIRECT.into());
        Ok(Self {
            app_id,
            app_id_hash,
            redirect_uri,
        })
    }

    pub fn login_url(&self, state: &str) -> String {
        format!(
            "{API_BASE}/api/v3/generate-authcode?client_id={}&redirect_uri={}&response_type=code&state={}",
            urlencoding_encode(&self.app_id),
            urlencoding_encode(&self.redirect_uri),
            urlencoding_encode(state),
        )
    }
}

#[derive(Debug, Clone)]
pub struct FyersClient {
    config: FyersConfig,
    access_token: String,
    http: Client,
}

impl FyersClient {
    pub fn new(config: FyersConfig, access_token: impl Into<String>) -> Self {
        Self {
            config,
            access_token: access_token.into(),
            http: Client::new(),
        }
    }

    /// Resolve token from env, cache file, or interactive login callback.
    pub async fn connect(interactive_login: bool) -> Result<Self> {
        let config = FyersConfig::from_env()?;

        if let Ok(token) = std::env::var("FYERS_ACCESS_TOKEN") {
            if !token.is_empty() {
                info!("using FYERS_ACCESS_TOKEN from environment");
                return Ok(Self::new(config, token));
            }
        }

        if let Some(token) = load_cached_token()? {
            info!("using cached Fyers access token from {TOKEN_CACHE_FILE}");
            return Ok(Self::new(config, token));
        }

        if !interactive_login {
            bail!(
                "No Fyers token. Run `cargo run -- login` first, or set FYERS_ACCESS_TOKEN.\n\
                 Login URL would be:\n{}",
                config.login_url("stock_recommender")
            );
        }

        let token = run_login_callback(&config).await?;
        save_cached_token(&token)?;
        Ok(Self::new(config, token))
    }

    fn auth_header(&self) -> String {
        format!("{}:{}", self.config.app_id, self.access_token)
    }

    /// Daily candles via `GET /data/history` (Fyers Market Data permission required).
    pub async fn fetch_history(
        &self,
        symbol: &str,
        range_from: NaiveDate,
        range_to: NaiveDate,
    ) -> Result<PriceHistory> {
        let symbol = normalize_symbol(symbol);
        let url = format!(
            "{API_BASE}/data/history?symbol={}&resolution=D&date_format=1&range_from={}&range_to={}&cont_flag=1",
            urlencoding_encode(&symbol),
            range_from,
            range_to
        );

        let resp = self
            .http
            .get(&url)
            .header(AUTHORIZATION, self.auth_header())
            .header(USER_AGENT, USER_AGENT_VALUE)
            .send()
            .await
            .context("Fyers history request failed")?;

        let status = resp.status();
        let body: Value = resp.json().await.context("Fyers history JSON decode")?;

        if !status.is_success() || body.get("s").and_then(|s| s.as_str()) != Some("ok") {
            bail!(
                "Fyers history error for {symbol}: HTTP {status} body={}",
                body
            );
        }

        let candles = body
            .get("candles")
            .and_then(|c| c.as_array())
            .ok_or_else(|| anyhow!("Fyers history missing candles for {symbol}"))?;

        let mut bars = Vec::with_capacity(candles.len());
        for c in candles {
            // [epoch, open, high, low, close, volume]
            let arr = c
                .as_array()
                .ok_or_else(|| anyhow!("bad candle row: {c}"))?;
            if arr.len() < 5 {
                continue;
            }
            let epoch = arr[0]
                .as_i64()
                .or_else(|| arr[0].as_f64().map(|f| f as i64))
                .ok_or_else(|| anyhow!("bad candle epoch: {c}"))?;
            let close = arr[4]
                .as_f64()
                .ok_or_else(|| anyhow!("bad candle close: {c}"))?;
            let date = chrono::DateTime::from_timestamp(epoch, 0)
                .map(|dt| dt.date_naive())
                .ok_or_else(|| anyhow!("bad timestamp {epoch}"))?;
            bars.push(PriceBar { date, close });
        }

        if bars.is_empty() {
            bail!("no candles returned for {symbol}");
        }

        Ok(PriceHistory {
            ticker: symbol,
            bars,
        })
    }

    /// Last ~180 calendar days of daily history.
    pub async fn fetch_recent_history(&self, symbol: &str) -> Result<PriceHistory> {
        let to = Utc::now().date_naive();
        let from = to - Duration::days(180);
        self.fetch_history(symbol, from, to).await
    }

    pub async fn fetch_quotes(&self, symbols: &[&str]) -> Result<Value> {
        let joined = symbols
            .iter()
            .map(|s| normalize_symbol(s))
            .collect::<Vec<_>>()
            .join(",");
        let url = format!(
            "{API_BASE}/data/quotes?symbols={}",
            urlencoding_encode(&joined)
        );
        let resp = self
            .http
            .get(&url)
            .header(AUTHORIZATION, self.auth_header())
            .header(USER_AGENT, USER_AGENT_VALUE)
            .send()
            .await?;
        let status = resp.status();
        let body: Value = resp.json().await?;
        if !status.is_success() {
            bail!("Fyers quotes error HTTP {status}: {body}");
        }
        Ok(body)
    }
}

/// Default NSE equity universe for the recommender.
pub fn default_universe() -> Vec<&'static str> {
    vec![
        "NSE:SBIN-EQ",
        "NSE:RELIANCE-EQ",
        "NSE:TCS-EQ",
        "NSE:INFY-EQ",
        "NSE:HDFCBANK-EQ",
    ]
}

/// Map short names like `SBIN` → `NSE:SBIN-EQ`. Pass through if already `EXCHANGE:SYM`.
pub fn normalize_symbol(raw: &str) -> String {
    let s = raw.trim().to_ascii_uppercase();
    if s.contains(':') {
        return s;
    }
    // Already has -EQ / -INDEX suffix
    if s.contains('-') {
        return format!("NSE:{s}");
    }
    format!("NSE:{s}-EQ")
}

#[derive(Clone)]
struct LoginState {
    config: FyersConfig,
    tx: Arc<Mutex<Option<oneshot::Sender<String>>>>,
}

/// Local callback server (same pattern as trading project's `/token` on :3000).
pub async fn run_login_callback(config: &FyersConfig) -> Result<String> {
    let (tx, rx) = oneshot::channel::<String>();
    let state = LoginState {
        config: config.clone(),
        tx: Arc::new(Mutex::new(Some(tx))),
    };

    let app = Router::new()
        .route("/token", get(token_callback))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("bind :3000 for Fyers redirect (same as trading project)")?;

    let login_url = config.login_url("stock_recommender");
    println!("\n=== Fyers login ===");
    println!("1. Ensure redirect URI on the app is: {}", config.redirect_uri);
    println!("2. Open this URL, log in, and allow the app:\n{login_url}\n");
    println!("Waiting for callback on http://0.0.0.0:3000/token ...");
    if config.redirect_uri.contains("kurianck.fyers") {
        println!("(Uses your fyers.conf nginx proxy → localhost:3000)");
    }

    let server = axum::serve(listener, app.into_make_service());
    tokio::select! {
        token = rx => {
            let token = token.context("login channel closed")?;
            // Give the HTTP response a moment to flush, then exit.
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok(token)
        }
        res = server => {
            res.context("login server error")?;
            bail!("login server exited before receiving a token");
        }
    }
}

async fn token_callback(
    State(state): State<LoginState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, StatusCode> {
    let code = params
        .get("auth_code")
        .cloned()
        .ok_or(StatusCode::BAD_REQUEST)?;

    let token = exchange_auth_code(&state.config, &code)
        .await
        .map_err(|e| {
            warn!("validate-authcode failed: {e:#}");
            StatusCode::BAD_GATEWAY
        })?;

    if let Some(tx) = state.tx.lock().await.take() {
        let _ = tx.send(token.clone());
    }

    Ok(Json(json!({
        "s": "ok",
        "message": "Fyers access token saved. You can close this tab and return to the terminal.",
    })))
}

pub async fn exchange_auth_code(config: &FyersConfig, auth_code: &str) -> Result<String> {
    let client = Client::new();
    let body = json!({
        "grant_type": "authorization_code",
        "appIdHash": config.app_id_hash,
        "code": auth_code,
    });

    let resp = client
        .post(format!("{API_BASE}/api/v3/validate-authcode"))
        .header(USER_AGENT, USER_AGENT_VALUE)
        .json(&body)
        .send()
        .await
        .context("validate-authcode request")?;

    let status = resp.status();
    let json: Value = resp.json().await.context("validate-authcode JSON")?;
    if !status.is_success() {
        bail!("validate-authcode HTTP {status}: {json}");
    }

    json.get("access_token")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("validate-authcode missing access_token: {json}"))
}

/// Exchange an auth_code from CLI without starting the server (paste redirect URL or code).
pub async fn login_with_auth_code(auth_code_or_url: &str) -> Result<String> {
    let config = FyersConfig::from_env()?;
    let code = extract_auth_code(auth_code_or_url);
    let token = exchange_auth_code(&config, &code).await?;
    save_cached_token(&token)?;
    Ok(token)
}

fn extract_auth_code(input: &str) -> String {
    let trimmed = input.trim();
    if let Some(q) = trimmed.split('?').nth(1) {
        for pair in q.split('&') {
            if let Some((k, v)) = pair.split_once('=') {
                if k == "auth_code" {
                    return v.to_string();
                }
            }
        }
    }
    trimmed.to_string()
}

fn token_cache_path() -> PathBuf {
    PathBuf::from(TOKEN_CACHE_FILE)
}

#[derive(Serialize, Deserialize)]
struct TokenCache {
    access_token: String,
}

fn load_cached_token() -> Result<Option<String>> {
    let path = token_cache_path();
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let cache: TokenCache = serde_json::from_str(&raw).context("parse token cache")?;
    if cache.access_token.is_empty() {
        Ok(None)
    } else {
        Ok(Some(cache.access_token))
    }
}

fn save_cached_token(token: &str) -> Result<()> {
    let path = token_cache_path();
    let cache = TokenCache {
        access_token: token.to_string(),
    };
    std::fs::write(&path, serde_json::to_string_pretty(&cache)?)
        .with_context(|| format!("write {}", path.display()))?;
    // Best-effort restrict permissions on Unix.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    info!("saved Fyers token to {}", path.display());
    let _ = Path::new(&path);
    Ok(())
}

fn urlencoding_encode(s: &str) -> String {
    // Minimal encoder for query values (app ids / URIs are simple ASCII).
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b':' => out.push_str("%3A"),
            b'/' => out.push_str("%2F"),
            b'?' => out.push_str("%3F"),
            b'&' => out.push_str("%26"),
            b'=' => out.push_str("%3D"),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_short_and_full() {
        assert_eq!(normalize_symbol("sbin"), "NSE:SBIN-EQ");
        assert_eq!(normalize_symbol("NSE:SBIN-EQ"), "NSE:SBIN-EQ");
        assert_eq!(normalize_symbol("NIFTY50-INDEX"), "NSE:NIFTY50-INDEX");
    }

    #[test]
    fn extract_code_from_url() {
        let u = "http://kurianck.fyers/token?auth_code=ABC123&state=x";
        assert_eq!(extract_auth_code(u), "ABC123");
        assert_eq!(extract_auth_code("ABC123"), "ABC123");
    }
}
