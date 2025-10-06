"""
signal_engine.py
=================

Core logic for computing event-driven signals for a set of highly liquid
tickers. Wraps Yahoo Finance via ``yfinance`` to retrieve news, option chain
metrics and price data. Outputs features (sentiment, PCR, ΔIV, VolSpike)
and combines them into an EventScore.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Weights for EventScore (tunable)
W1_SENTIMENT = 0.4
W2_PCR = 0.3
W3_DELTA_IV = 0.2
W4_VOL_SPIKE = 0.1


@dataclass
class TickerSignal:
    """Container for all computed data for a ticker on a single day."""

    ticker: str
    date: datetime.date
    headline: Optional[str]
    sentiment: float
    pcr: float
    delta_iv: float
    vol_spike: float
    event_score: float
    signal: str
    magnitude: str
    next_day_return: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "ticker": self.ticker,
            "headline": self.headline,
            "sentiment": round(self.sentiment, 4),
            "PCR": round(self.pcr, 4),
            "ΔIV": round(self.delta_iv, 4),
            "VolSpike": round(self.vol_spike, 4),
            "EventScore": round(self.event_score, 4),
            "Signal": self.signal,
            "Magnitude": self.magnitude,
            "NextDayReturn": None if self.next_day_return is None else round(self.next_day_return, 4),
        }


# -----------------------
# News utilities
# -----------------------

def get_latest_news_and_sentiment(ticker: str) -> Tuple[Optional[str], float]:
    """Fetch the most recent headline and sentiment score for a ticker."""
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        news = []
    headline = None
    sentiment_score = 0.0
    if news:
        headline = news[0].get("title", None)
        if headline:
            analyzer = SentimentIntensityAnalyzer()
            sentiment_score = analyzer.polarity_scores(headline)["compound"]
    return headline, sentiment_score


def get_news_feed(ticker: str) -> List[dict]:
    """
    Fetch a list of recent news items for a ticker.
    Each item contains title, link, and publisher.
    """
    try:
        news_items = yf.Ticker(ticker).news
        if not news_items:
            return []
        return news_items[:5]  # top 5 only
    except Exception as e:
        return [{"title": f"Failed to fetch news: {e}", "link": None, "publisher": "N/A"}]


# -----------------------
# Options + volume metrics
# -----------------------

def get_option_chain_metrics(ticker: str) -> Tuple[float, float, float]:
    """
    Retrieve option chain and compute:
      - PCR (put/call ratio)
      - ΔIV (IV change, approximated)
      - total volume
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        expiry = expirations[0] if expirations else None
        if not expiry:
            return 1.0, 0.0, 0.0
        chain = t.option_chain(expiry)
        calls_df, puts_df = chain.calls, chain.puts
    except Exception:
        return 1.0, 0.0, 0.0

    call_volume = calls_df["volume"].fillna(0).sum()
    put_volume = puts_df["volume"].fillna(0).sum()
    total_volume = call_volume + put_volume
    pcr = (put_volume / call_volume) if call_volume else 1.0

    iv_series = pd.concat([
        calls_df["impliedVolatility"].dropna(),
        puts_df["impliedVolatility"].dropna()
    ])
    mean_iv_today = iv_series.mean() if not iv_series.empty else 0.0

    # Placeholder ΔIV (no historical IV available in yfinance)
    mean_iv_past = mean_iv_today
    delta_iv = mean_iv_today - mean_iv_past if mean_iv_past else 0.0

    return pcr, delta_iv, total_volume


def get_volume_spike_ratio(ticker: str, total_option_volume: float) -> float:
    """Approximate unusual options activity by comparing to avg stock volume."""
    try:
        hist = yf.Ticker(ticker).history(period="1mo")
        avg_stock_volume = hist["Volume"].mean()
        return total_option_volume / avg_stock_volume if avg_stock_volume else 0.0
    except Exception:
        return 0.0


# -----------------------
# EventScore + classification
# -----------------------

def compute_event_score(sentiment: float, pcr: float, delta_iv: float, vol_spike: float) -> float:
    """Weighted sum of features into a single EventScore."""
    return (
        W1_SENTIMENT * sentiment
        - W2_PCR * pcr  # higher PCR → bearish
        + W3_DELTA_IV * delta_iv
        + W4_VOL_SPIKE * vol_spike
    )


def classify_signal(event_score: float) -> Tuple[str, str]:
    """Turn EventScore into directional signal + magnitude bucket."""
    direction = "RISE" if event_score > 0 else "FALL"
    abs_score = abs(event_score)
    if abs_score < 0.3:
        magnitude = "weak"
    elif abs_score < 0.7:
        magnitude = "moderate"
    else:
        magnitude = "strong"
    return direction, magnitude


# -----------------------
# Backtesting
# -----------------------

def compute_next_day_return(ticker: str, date: datetime.date) -> Optional[float]:
    """Fetch price and compute next day return (close-to-close)."""
    try:
        start = date - datetime.timedelta(days=1)
        end = date + datetime.timedelta(days=2)
        hist = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())
        hist = hist.sort_index()
        dates = [idx.date() for idx in hist.index]
        if date in dates:
            idx = dates.index(date)
            if idx + 1 < len(hist):
                today_close = hist.iloc[idx]["Close"]
                next_close = hist.iloc[idx + 1]["Close"]
                if today_close:
                    return (next_close / today_close) - 1.0
    except Exception:
        pass
    return None


# -----------------------
# Orchestration
# -----------------------

def generate_signal_for_ticker(ticker: str, date: datetime.date) -> TickerSignal:
    """Master function to generate one ticker’s signal."""
    headline, sentiment = get_latest_news_and_sentiment(ticker)
    pcr, delta_iv, total_vol = get_option_chain_metrics(ticker)
    vol_spike = get_volume_spike_ratio(ticker, total_vol)
    event_score = compute_event_score(sentiment, pcr, delta_iv, vol_spike)
    direction, magnitude = classify_signal(event_score)
    next_ret = compute_next_day_return(ticker, date)
    return TickerSignal(
        ticker, date, headline, sentiment, pcr, delta_iv, vol_spike,
        event_score, direction, magnitude, next_ret
    )


def generate_signals_for_tickers(tickers: List[str]) -> pd.DataFrame:
    """Generate signals for a list of tickers (today’s date)."""
    today = datetime.date.today()
    signals = [generate_signal_for_ticker(t, today).to_dict() for t in tickers]
    return pd.DataFrame(signals)


def append_signals_to_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Append signals to CSV, creating it if missing."""
    try:
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        combined = df.copy()
    combined.to_csv(csv_path, index=False)
