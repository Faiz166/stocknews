"""
signal_engine.py
=================

This module provides the core logic for computing event‑driven signals for a set
of highly liquid tickers.  It wraps free data sources (via the ``yfinance``
package) to retrieve news headlines, option chain metrics and historical price
information.  Using these inputs it computes a handful of engineered features
— news sentiment, put/call ratio, implied volatility change and volume spike —
and combines them into a single EventScore.  A positive score implies the
underlying stock is more likely to rise; a negative score implies it is more
likely to fall.  The magnitude of the score is used to bucket the signal into
"weak", "moderate" or "strong" categories.

The functions in this module are stateless and can be imported into a
Streamlit front‑end (as in ``app.py``) or called from a command line script.

Note: These functions rely on free data sources provided by Yahoo Finance.
When run in an environment without internet access the data retrieval calls
will fail; however the structure of the code remains useful for integration
once network access is available.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Weights for the EventScore; these can be tuned based on further research
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
        """Convert to a plain dictionary suitable for DataFrame or CSV."""
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


def get_latest_news_and_sentiment(ticker: str) -> Tuple[Optional[str], float]:
    """
    Fetch the most recent news headline for a ticker and compute its sentiment.

    The Yahoo Finance API (via yfinance) returns a list of news items.  The
    sentiment is computed using VADER and the compound score is returned.  If
    no news is available, the sentiment is set to zero.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    Tuple[Optional[str], float]
        The latest headline (or None) and the compound sentiment score in the
        range [-1, 1].
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
    except Exception:
        news = []
    headline = None
    sentiment_score = 0.0
    analyzer = SentimentIntensityAnalyzer()
    if news:
        # Use the most recent news item
        latest = news[0]
        headline = latest.get("title", None)
        if headline:
            sentiment_score = analyzer.polarity_scores(headline)["compound"]
    return headline, sentiment_score

def get_news_feed(ticker: str):
    try:
        news_items = yf.Ticker(ticker).news
        if not news_items:
            return []
        # Keep top 5 for readability
        return news_items[:5]
    except Exception as e:
        return [{"title": f"Failed to fetch news: {e}"}]
        
def get_option_chain_metrics(ticker: str) -> Tuple[float, float, float]:
    """
    Retrieve option chain data and compute key metrics.

    The function pulls the nearest expiration option chain for the ticker and
    computes:
    - Put/Call ratio (volume based)
    - Average implied volatility (calls and puts)
    - Total option volume

    The implied volatility change (ΔIV) requires a previous day's implied
    volatility.  For an MVP using free data we approximate this by taking the
    change between today's implied volatility and the average implied
    volatility over the last week.  If the week average cannot be computed
    (e.g., no historical IV data), ΔIV is set to zero.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing (PCR, ΔIV, total_volume).
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        # Use the nearest expiration; if none, return defaults
        expiry = expirations[0] if expirations else None
    except Exception:
        expiry = None
    if not expiry:
        return 1.0, 0.0, 0.0
    try:
        chain = t.option_chain(expiry)
        calls_df = chain.calls
        puts_df = chain.puts
    except Exception:
        return 1.0, 0.0, 0.0
    # Compute volumes and open interest
    call_volume = calls_df["volume"].fillna(0).sum()
    put_volume = puts_df["volume"].fillna(0).sum()
    total_volume = call_volume + put_volume
    # Avoid division by zero
    pcr = (put_volume / call_volume) if call_volume else 1.0
    # Compute average implied volatility across calls and puts
    mean_iv_today = pd.concat([
        calls_df["impliedVolatility"].dropna(),
        puts_df["impliedVolatility"].dropna(),
    ]).mean()
    # Approximate previous implied volatility as the mean of the last week; use
    # a fallback to current IV if historical data is unavailable.
    try:
        hist_iv = []
        # Use the past 5 days' options chain implied volatilities
        past_dates = [
            (datetime.date.today() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(1, 6)
        ]
        for date_str in past_dates:
            # yfinance does not support historical option chains directly;
            # as a proxy we use the current chain's implied volatility for
            # demonstration.  In a production setting one would use a data
            # provider with historical option metrics.
            hist_iv.append(mean_iv_today)
        mean_iv_past = sum(hist_iv) / len(hist_iv)
    except Exception:
        mean_iv_past = mean_iv_today
    delta_iv = mean_iv_today - mean_iv_past if mean_iv_past else 0.0
    return pcr, delta_iv, total_volume


def get_volume_spike_ratio(ticker: str, total_option_volume: float) -> float:
    """
    Compute the volume spike ratio for a ticker.

    Because free historical option volume data is not readily available
    via yfinance, we approximate the spike by comparing the current total
    option volume with the ticker's 30‑day average stock trading volume.
    This provides a proxy: if the options volume is large relative to the
    underlying's average volume, it suggests unusual activity.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.
    total_option_volume : float
        The total option volume for the ticker on the current day.

    Returns
    -------
    float
        The ratio of option volume to average stock volume.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1mo")
        avg_stock_volume = hist["Volume"].mean()
        if avg_stock_volume and avg_stock_volume > 0:
            ratio = total_option_volume / avg_stock_volume
        else:
            ratio = 0.0
    except Exception:
        ratio = 0.0
    return ratio


def compute_event_score(sentiment: float, pcr: float, delta_iv: float, vol_spike: float) -> float:
    """
    Combine the input features into a single EventScore.

    The EventScore is a weighted sum of the four engineered features.  A
    positive score implies bullishness; a negative score implies bearishness.

    Parameters
    ----------
    sentiment : float
        The compound sentiment score from VADER.
    pcr : float
        Put/Call ratio based on option volumes.
    delta_iv : float
        Change in implied volatility.
    vol_spike : float
        Volume spike ratio.

    Returns
    -------
    float
        The computed EventScore.
    """
    # The PCR term is negated because a higher PCR implies bearish sentiment
    # (more puts than calls) and should reduce the score.
    score = (
        W1_SENTIMENT * sentiment
        - W2_PCR * pcr
        + W3_DELTA_IV * delta_iv
        + W4_VOL_SPIKE * vol_spike
    )
    return score


def classify_signal(event_score: float) -> Tuple[str, str]:
    """
    Determine the directional signal and magnitude bucket from the EventScore.

    Parameters
    ----------
    event_score : float
        The computed EventScore.

    Returns
    -------
    Tuple[str, str]
        A tuple (signal, magnitude) where signal is "RISE" or "FALL" and
        magnitude is one of "weak", "moderate" or "strong".
    """
    direction = "RISE" if event_score > 0 else "FALL"
    abs_score = abs(event_score)
    if abs_score < 0.3:
        magnitude = "weak"
    elif abs_score < 0.7:
        magnitude = "moderate"
    else:
        magnitude = "strong"
    return direction, magnitude


def compute_next_day_return(ticker: str, date: datetime.date) -> Optional[float]:
    """
    Compute the next day's return for a ticker.

    This function fetches historical prices for the ticker over a two‑day window
    surrounding the provided date.  It then computes the simple return from
    close_{t+1} to close_{t}.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.
    date : datetime.date
        The current date.

    Returns
    -------
    Optional[float]
        The next day return (a decimal, not percentage), or None if data
        cannot be fetched.
    """
    try:
        # Fetch three days of data to ensure we capture the day after the event
        start = date - datetime.timedelta(days=1)
        end = date + datetime.timedelta(days=2)
        t = yf.Ticker(ticker)
        hist = t.history(start=start.isoformat(), end=end.isoformat())
        # Ensure the DataFrame is sorted by date
        hist = hist.sort_index()
        # Locate the current and next day's closing prices
        index_dates = list(hist.index)
        # Convert to purely date objects for comparison
        date_only = [idx.date() for idx in index_dates]
        if date in date_only:
            idx = date_only.index(date)
            if idx + 1 < len(hist):
                price_today = hist.iloc[idx]["Close"]
                price_next = hist.iloc[idx + 1]["Close"]
                if price_today:
                    return (price_next / price_today) - 1.0
    except Exception:
        pass
    return None


def generate_signal_for_ticker(ticker: str, date: datetime.date) -> TickerSignal:
    """
    Generate the full signal for a single ticker on a given date.

    This function orchestrates the data retrieval, feature engineering,
    EventScore computation, signal classification and next‑day return
    calculation.  It returns a ``TickerSignal`` instance with all fields
    populated.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.
    date : datetime.date
        The date for which to generate the signal.  Typically the current
        date.

    Returns
    -------
    TickerSignal
        A data structure containing all computed values for the ticker.
    """
    # Retrieve latest news headline and sentiment
    headline, sentiment = get_latest_news_and_sentiment(ticker)
    # Retrieve option chain metrics
    pcr, delta_iv, total_vol = get_option_chain_metrics(ticker)
    # Compute volume spike ratio using proxy
    vol_spike = get_volume_spike_ratio(ticker, total_vol)
    # Combine into EventScore
    event_score = compute_event_score(sentiment, pcr, delta_iv, vol_spike)
    # Classify direction and magnitude
    direction, magnitude = classify_signal(event_score)
    # Compute next day's return for backtesting
    next_ret = compute_next_day_return(ticker, date)
    return TickerSignal(
        ticker=ticker,
        date=date,
        headline=headline,
        sentiment=sentiment,
        pcr=pcr,
        delta_iv=delta_iv,
        vol_spike=vol_spike,
        event_score=event_score,
        signal=direction,
        magnitude=magnitude,
        next_day_return=next_ret,
    )


def generate_signals_for_tickers(tickers: List[str]) -> pd.DataFrame:
    """
    Generate signals for a list of tickers for the current date.

    Parameters
    ----------
    tickers : List[str]
        The list of ticker symbols to process.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per ticker containing the computed signals
        and underlying features.
    """
    today = datetime.date.today()
    signals = []
    for ticker in tickers:
        sig = generate_signal_for_ticker(ticker, today)
        signals.append(sig.to_dict())
    return pd.DataFrame(signals)


def append_signals_to_csv(df: pd.DataFrame, csv_path: str) -> None:
    """
    Append signal data to a CSV file.

    If the file does not exist, it will be created along with headers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the signals to append.
    csv_path : str
        Path to the CSV file.
    """
    try:
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        combined = df.copy()
    combined.to_csv(csv_path, index=False)
