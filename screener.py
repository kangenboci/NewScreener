"""
Pang Pang Manis - Multi-Timeframe Stock Screener (yfinance version)

Talks to a Google Apps Script Web App instead of the Google Sheets API,
so NO Google Cloud service account / credentials JSON is needed.

Writes:
  - Screener!A:D  ticker + signal per timeframe (15m/30m/1d)
  - Screener!G    last close price
  - Bandar!A:B    ticker + turnover-based accumulation proxy label
                  (NOT real broker/bandar-flow data - see note below)

Env vars (set as GitHub Actions secrets):
  WEBAPP_URL  - the Apps Script Web App /exec URL (see Code.gs)
  API_SECRET  - shared secret string, must match Script Properties in Apps Script
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

WEBAPP_URL = os.environ["WEBAPP_URL"]
API_SECRET = os.environ["API_SECRET"]

EMA_LEN     = 10
SMA1_LEN    = 20
SMA2_LEN    = 50
SMA3_LEN    = 100
SMA4_LEN    = 200
ADX_LEN     = 14
ADX_TH      = 20
VOL_LEN     = 20
DRY_FACTOR  = 0.50
LOOKBACK    = 30
VOL_SPIKE   = 1.10

# order here = order written to columns B, C, D
TIMEFRAMES = {
    "15m": {"interval": "15m", "period": "60d"},
    "30m": {"interval": "30m", "period": "60d"},
    "1d":  {"interval": "1d",  "period": "2y"},
}

SIG_TEXT = {3: "STRONG BUY", 2: "BUY", 1: "DIP BUY", 0: "Bullish", -1: "-"}

# --- turnover-based "bandar" proxy (public price*volume data only) ---
BANDAR_MA_SHORT = 10
BANDAR_MA_LONG  = 20
BANDAR_VALUE_MIN = 1_000_000_000  # IDR, same order of magnitude as the Stockbit rule


def wilder_adx(df: pd.DataFrame, length: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1).fillna(0).values

    up_move = (high - prev_high).values
    down_move = (prev_low - low).values

    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    dm_plus = np.nan_to_num(dm_plus)
    dm_minus = np.nan_to_num(dm_minus)

    n = len(tr)
    str_ = np.zeros(n)
    sdmp = np.zeros(n)
    sdmm = np.zeros(n)

    for i in range(1, n):
        str_[i] = str_[i - 1] - str_[i - 1] / length + tr[i]
        sdmp[i] = sdmp[i - 1] - sdmp[i - 1] / length + dm_plus[i]
        sdmm[i] = sdmm[i - 1] - sdmm[i - 1] / length + dm_minus[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = np.where(str_ != 0, sdmp / str_ * 100, 0.0)
        di_minus = np.where(str_ != 0, sdmm / str_ * 100, 0.0)
        denom = di_plus + di_minus
        dx = np.where(denom != 0, np.abs(di_plus - di_minus) / denom * 100, 0.0)

    adx = pd.Series(dx, index=df.index).rolling(length).mean()
    return pd.Series(di_plus, index=df.index), pd.Series(di_minus, index=df.index), adx


def barssince(cond: pd.Series) -> pd.Series:
    out = np.full(len(cond), np.inf)
    last = np.inf
    for i, v in enumerate(cond.values):
        last = 0 if v else (last + 1 if np.isfinite(last) else np.inf)
        out[i] = last
    return pd.Series(out, index=cond.index)


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)


def compute_signal(df: pd.DataFrame) -> int:
    if df is None or len(df) < max(SMA4_LEN, LOOKBACK) + 5:
        return -1  # not enough history yet

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    ema = close.ewm(span=EMA_LEN, adjust=False).mean()
    sma1 = close.rolling(SMA1_LEN).mean()
    sma2 = close.rolling(SMA2_LEN).mean()
    sma3 = close.rolling(SMA3_LEN).mean()
    sma4 = close.rolling(SMA4_LEN).mean()

    trend_bull = (ema > sma1) & (sma1 > sma2)
    trend_strong = trend_bull & (sma2 > sma3) & (sma3 > sma4)

    di_plus, di_minus, adx = wilder_adx(df, ADX_LEN)
    momentum_bull = (adx > ADX_TH) & (di_plus > di_minus)

    vol_ma = vol.rolling(VOL_LEN).mean()
    range_high = high.rolling(LOOKBACK).max()
    is_dry = vol < vol_ma * DRY_FACTOR
    was_dry_recently = barssince(is_dry) <= LOOKBACK
    broke_out_recent = barssince(crossover(close, range_high)) <= 3
    vol_confirm = vol > vol_ma * VOL_SPIKE

    buy = trend_bull & momentum_bull & broke_out_recent & vol_confirm
    strong_buy = buy & trend_strong & was_dry_recently

    pullback_touch = (low <= sma1) & (close > sma1)
    pullback_bounce = crossover(close, ema)
    dip_buy = trend_bull & momentum_bull & (pullback_touch | pullback_bounce) & (close > sma2)

    if bool(strong_buy.iloc[-1]):
        return 3
    if bool(buy.iloc[-1]):
        return 2
    if bool(dip_buy.iloc[-1]):
        return 1
    if bool(trend_bull.iloc[-1]):
        return 0
    return -1

def normalize_ticker(t: str) -> str:
    """IDX stocks need a .JK suffix for yfinance (e.g. BMRI -> BMRI.JK)."""
    t = t.strip().upper()
    if "." not in t:
        t = f"{t}.JK"
    return t


def fetch(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=False)
        if df.empty:
            return None
        # newer yfinance returns MultiIndex columns (e.g. ('Close', 'BMRI.JK'))
        # even for a single ticker - flatten so df["Close"] is a plain Series
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        print(f"[WARN] fetch failed {ticker} {interval}: {e}")
        return None


def get_tickers():
    r = requests.get(WEBAPP_URL, params={"secret": API_SECRET}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    raw = data.get("tickers", [])
    return [normalize_ticker(t) for t in raw if t and t.strip()]


def push_results(screener_rows, bandar_rows):
    payload = {"secret": API_SECRET, "screener_rows": screener_rows, "bandar_rows": bandar_rows}
    r = requests.post(WEBAPP_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    print(f"Updated screener={data.get('screener_updated', 0)} bandar={data.get('bandar_updated', 0)}")


def main():
    tickers = get_tickers()

    screener_rows = []
    bandar_rows = []

    for t in tickers:
        row = [t]
        dfs = {}
        for tf, cfg in TIMEFRAMES.items():
            df = fetch(t, cfg["interval"], cfg["period"])
            dfs[tf] = df
            code = compute_signal(df)
            row.append(SIG_TEXT[code])
            time.sleep(0.3)  # be gentle with Yahoo's rate limits

        df_1d = dfs.get("1d")
        last_price = float(df_1d["Close"].iloc[-1]) if df_1d is not None and len(df_1d) else ""
        row.append(last_price)
        screener_rows.append(row)

        bandar_rows.append([t, bandar_proxy(df_1d)])

        print(row)

    if screener_rows or bandar_rows:
        push_results(screener_rows, bandar_rows)


if __name__ == "__main__":
    main()
