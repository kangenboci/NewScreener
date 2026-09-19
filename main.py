import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import gspread

from datetime import datetime
from google.oauth2.service_account import Credentials


# ============================================================
# CONFIGURATION
# ============================================================

SPREADSHEET_ID = "ISI_SPREADSHEET_ID_ANDA"

SHEET_SCREENER = "Screener"
SHEET_BANDAR = "Bandar"

SERVICE_ACCOUNT_FILE = "service_account.json"

TELEGRAM_BOT_TOKEN = "ISI_BOT_TOKEN"
TELEGRAM_CHAT_ID = "ISI_CHAT_ID"


# ============================================================
# PANG PANG MANIS PARAMETERS
# ============================================================

EMA_LENGTH = 10

SMA1_LENGTH = 20
SMA2_LENGTH = 50
SMA3_LENGTH = 100
SMA4_LENGTH = 200

ADX_LENGTH = 14
ADX_THRESHOLD = 20

VOL_LENGTH = 20
DRY_FACTOR = 0.50

LOOKBACK = 30
VOL_SPIKE = 1.10


# ============================================================
# GOOGLE SHEETS
# ============================================================

def connect_google_sheet():

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=scopes
    )

    client = gspread.authorize(credentials)

    spreadsheet = client.open_by_key(SPREADSHEET_ID)

    return spreadsheet


# ============================================================
# READ TICKERS
# ============================================================

def get_tickers(sheet):

    values = sheet.col_values(1)

    tickers = []

    for value in values[1:]:

        ticker = value.strip().upper()

        if not ticker:
            continue

        # Header protection
        if ticker in ["TICKER", "SAHAM"]:
            continue

        tickers.append(ticker)

    return tickers


# ============================================================
# NORMALIZE IDX TICKER
# ============================================================

def normalize_ticker(ticker):

    ticker = ticker.upper().strip()

    if ticker.endswith(".JK"):
        return ticker

    return ticker + ".JK"


# ============================================================
# ADX
# ============================================================

def calculate_adx(df, length=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (high - previous_close).abs()

    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    # Wilder smoothing
    atr = true_range.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    plus_dm = pd.Series(
        plus_dm,
        index=df.index
    ).ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    minus_dm = pd.Series(
        minus_dm,
        index=df.index
    ).ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    plus_di = 100 * plus_dm / atr

    minus_di = 100 * minus_dm / atr

    dx = (
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di)
    ) * 100

    adx = dx.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    return adx, plus_di, minus_di


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data(ticker, interval):

    yf_ticker = normalize_ticker(ticker)

    if interval in ["15m", "30m"]:

        # Intraday Yahoo limitation
        period = "60d"

    else:

        # Need enough history for SMA200
        period = "2y"

    try:

        df = yf.download(
            yf_ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        # Handle MultiIndex returned by newer yfinance
        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        df = df[required].copy()

        df.dropna(inplace=True)

        return df

    except Exception as e:

        print(
            f"ERROR download {ticker} "
            f"{interval}: {e}"
        )

        return None


# ============================================================
# PANG PANG MANIS SCANNER
# ============================================================

def scan_pang_pang(df):

    if df is None:
        return {
            "signal": "-",
            "close": None,
            "adx": None,
            "strong": False
        }

    minimum_bars = max(
        SMA4_LENGTH + 10,
        LOOKBACK + 10,
        100
    )

    if len(df) < minimum_bars:

        return {
            "signal": "-",
            "close": None,
            "adx": None,
            "strong": False
        }

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    ema = close.ewm(
        span=EMA_LENGTH,
        adjust=False
    ).mean()

    sma1 = close.rolling(
        SMA1_LENGTH
    ).mean()

    sma2 = close.rolling(
        SMA2_LENGTH
    ).mean()

    sma3 = close.rolling(
        SMA3_LENGTH
    ).mean()

    sma4 = close.rolling(
        SMA4_LENGTH
    ).mean()

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend_bullish = (
        (ema > sma1)
        &
        (sma1 > sma2)
    )

    trend_strong_bullish = (
        trend_bullish
        &
        (sma2 > sma3)
        &
        (sma3 > sma4)
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    adx, di_plus, di_minus = calculate_adx(
        df,
        ADX_LENGTH
    )

    momentum_bullish = (
        (adx > ADX_THRESHOLD)
        &
        (di_plus > di_minus)
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    vol_ma = volume.rolling(
        VOL_LENGTH
    ).mean()

    is_dry = (
        volume
        <
        vol_ma * DRY_FACTOR
    )

    was_dry_recently = (
        is_dry
        .rolling(LOOKBACK)
        .max()
        .fillna(0)
        .astype(bool)
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    # IMPORTANT:
    # gunakan resistance dari candle sebelumnya
    # agar close benar-benar bisa breakout

    previous_range_high = (
        high.shift(1)
        .rolling(LOOKBACK)
        .max()
    )

    breakout = (
        (close > previous_range_high)
        &
        (close.shift(1) <= previous_range_high.shift(1))
    )

    # breakout dalam 3 candle terakhir

    broke_out_recent = (
        breakout
        .rolling(4)
        .max()
        .fillna(0)
        .astype(bool)
    )

    volume_confirm = (
        volume > vol_ma * VOL_SPIKE
    )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    buy_signal = (
        trend_bullish
        &
        momentum_bullish
        &
        broke_out_recent
        &
        volume_confirm
    )

    # --------------------------------------------------------
    # STRONG BUY
    # --------------------------------------------------------

    strong_buy_signal = (
        buy_signal
        &
        trend_strong_bullish
        &
        was_dry_recently
    )

    # --------------------------------------------------------
    # DIP BUY
    # --------------------------------------------------------

    pullback_touch = (
        (low <= sma1)
        &
        (close > sma1)
    )

    pullback_bounce = (
        (close > ema)
        &
        (close.shift(1) <= ema.shift(1))
    )

    pullback_buy_signal = (
        trend_bullish
        &
        momentum_bullish
        &
        (pullback_touch | pullback_bounce)
        &
        (close > sma2)
    )

    # --------------------------------------------------------
    # CURRENT BAR
    # --------------------------------------------------------

    last = -1

    if bool(strong_buy_signal.iloc[last]):

        signal = "STRONG BUY"

    elif bool(buy_signal.iloc[last]):

        signal = "BUY"

    elif bool(pullback_buy_signal.iloc[last]):

        signal = "DIP BUY"

    elif bool(trend_bullish.iloc[last]):

        signal = "BULLISH"

    else:

        signal = "-"

    return {

        "signal": signal,

        "close": float(close.iloc[last]),

        "adx": float(adx.iloc[last])
        if not pd.isna(adx.iloc[last])
        else None,

        "strong": bool(
            trend_strong_bullish.iloc[last]
        )

    }


# ============================================================
# BANDAR LOOKUP
# ============================================================

def get_bandar_data(bandar_sheet):

    rows = bandar_sheet.get_all_values()

    bandar = {}

    for row in rows[1:]:

        if len(row) < 2:
            continue

        ticker = row[0].strip().upper()

        result = row[1].strip()

        if ticker:

            bandar[ticker] = result

    return bandar


# ============================================================
# SIGNAL SCORE
# ============================================================

def signal_score(signal):

    scores = {

        "STRONG BUY": 4,

        "BUY": 3,

        "DIP BUY": 2,

        "BULLISH": 1,

        "-": 0

    }

    return scores.get(signal, 0)


# ============================================================
# FINAL RESULT
# ============================================================

def determine_result(
    signal_15m,
    signal_30m,
    signal_1d,
    bandar
):

    s15 = signal_score(signal_15m)

    s30 = signal_score(signal_30m)

    s1d = signal_score(signal_1d)

    bandar_text = bandar.upper()

    is_accumulation = (
        "ACCUMULATION" in bandar_text
    )

    is_uptrend = (
        "UPTREND" in bandar_text
    )

    # --------------------------------------------------------
    # STRONG ALERT
    # --------------------------------------------------------

    if (
        is_accumulation
        and
        s15 >= 4
        and
        s30 >= 3
        and
        s1d >= 1
    ):

        return "STRONG ALERT"

    # --------------------------------------------------------
    # BUY ALERT
    # --------------------------------------------------------

    buy_count = sum(
        x >= 3
        for x in [s15, s30, s1d]
    )

    if (
        is_accumulation
        and
        buy_count >= 2
        and
        s1d >= 1
    ):

        return "BUY ALERT"

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    if (
        (is_accumulation or is_uptrend)
        and
        max(s15, s30, s1d) >= 2
    ):

        return "WATCH"

    return "-"


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {

        "chat_id": TELEGRAM_CHAT_ID,

        "text": message,

        "parse_mode": "HTML"

    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=15
        )

        response.raise_for_status()

        print("Telegram sent")

    except Exception as e:

        print(
            f"Telegram ERROR: {e}"
        )


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    print("=" * 70)

    print(
        "PANG PANG MANIS "
        "MULTI TIMEFRAME SCREENER"
    )

    print("=" * 70)

    spreadsheet = connect_google_sheet()

    screener_sheet = spreadsheet.worksheet(
        SHEET_SCREENER
    )

    bandar_sheet = spreadsheet.worksheet(
        SHEET_BANDAR
    )

    tickers = get_tickers(
        screener_sheet
    )

    bandar_data = get_bandar_data(
        bandar_sheet
    )

    print(
        f"Total ticker: {len(tickers)}"
    )

    alerts = []

    results = []

    for ticker in tickers:

        print(
            f"\nScanning {ticker}..."
        )

        # ----------------------------------------------------
        # 15 MIN
        # ----------------------------------------------------

        df15 = download_data(
            ticker,
            "15m"
        )

        result15 = scan_pang_pang(
            df15
        )

        print(
            f"  15m  : "
            f"{result15['signal']}"
        )

        # ----------------------------------------------------
        # 30 MIN
        # ----------------------------------------------------

        df30 = download_data(
            ticker,
            "30m"
        )

        result30 = scan_pang_pang(
            df30
        )

        print(
            f"  30m  : "
            f"{result30['signal']}"
        )

        # ----------------------------------------------------
        # DAILY
        # ----------------------------------------------------

        df1d = download_data(
            ticker,
            "1d"
        )

        result1d = scan_pang_pang(
            df1d
        )

        print(
            f"  1D   : "
            f"{result1d['signal']}"
        )

        # ----------------------------------------------------
        # BANDAR
        # ----------------------------------------------------

        bandar = bandar_data.get(
            ticker.upper(),
            "-"
        )

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        final_result = determine_result(

            result15["signal"],

            result30["signal"],

            result1d["signal"],

            bandar
        )

        print(
            f"  Bandar: {bandar}"
        )

        print(
            f"  RESULT: {final_result}"
        )

        results.append({

            "ticker": ticker,

            "15m": result15["signal"],

            "30m": result30["signal"],

            "1D": result1d["signal"],

            "Bandar": bandar,

            "Result": final_result

        })

        # ----------------------------------------------------
        # ALERT
        # ----------------------------------------------------

        if final_result in [
            "STRONG ALERT",
            "BUY ALERT"
        ]:

            alerts.append({

                "ticker": ticker,

                "15m": result15["signal"],

                "30m": result30["signal"],

                "1D": result1d["signal"],

                "Bandar": bandar,

                "Result": final_result

            })

    # ========================================================
    # WRITE TO GOOGLE SHEETS
    # ========================================================

    output = []

    for row in results:

        output.append([

            row["ticker"],

            row["15m"],

            row["30m"],

            row["1D"],

            row["Bandar"],

            row["Result"]

        ])

    if output:

        screener_sheet.update(
            f"A2:F{len(output) + 1}",
            output
        )

    # ========================================================
    # TELEGRAM ALERT
    # ========================================================

    if alerts:

        message = (
            "<b>🚨 PANG PANG MANIS ALERT</b>\n\n"
        )

        for item in alerts:

            emoji = (
                "🔥"
                if item["Result"]
                == "STRONG ALERT"
                else "🟢"
            )

            message += (
                f"{emoji} "
                f"<b>{item['ticker']}</b>\n"
                f"15m : {item['15m']}\n"
                f"30m : {item['30m']}\n"
                f"1D  : {item['1D']}\n"
                f"Bandar : {item['Bandar']}\n"
                f"Result : <b>{item['Result']}</b>\n\n"
            )

        message += (
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        send_telegram(message)

    else:

        print(
            "\nNo BUY ALERT."
        )

    print(
        "\nSCAN COMPLETE."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
