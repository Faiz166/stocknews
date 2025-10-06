"""
Streamlit application for event-driven stock signal generation.

This app runs daily across a predefined set of highly liquid tickers.  It
retrieves the latest news headlines and option chain data from Yahoo Finance,
computes an EventScore for each ticker using sentiment, put/call ratio,
implied volatility change and volume spike, and classifies the expected
direction of the underlying price ("RISE" or "FALL") along with a
confidence magnitude.  All signals and their underlying metrics are logged
to a CSV file for later analysis.
"""

# -----------------------
# Monkey-patch yfinance requests with modern headers + retry
# -----------------------
import requests
from requests.adapters import HTTPAdapter, Retry

# Build a global session that looks like Chrome
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
    "Connection": "keep-alive",
})

# Add retry for 429/500s
retries = Retry(total=3, backoff_factor=0.6,
                status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# Monkey-patch yfinance to use this session
import yfinance as yf
yf.shared._requests = session

# -----------------------
# Standard imports
# -----------------------
import os
import time
import datetime
import pandas as pd
import streamlit as st

from signal_engine import (
    generate_signals_for_tickers,
    append_signals_to_csv,
    get_news_feed,
)


# -----------------------
# Config
# -----------------------
WATCHLIST = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "trades.csv")
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------
# YFinance Tester
# -----------------------
def yfinance_tester():
    with st.expander("🔍 YFinance Tester"):
        ticker_input = st.text_input("Enter a ticker (default: AAPL)", "AAPL")

        if st.button("Run Test"):
            try:
                t = yf.Ticker(ticker_input)

                # Basic info (safe)
                info = getattr(t, "info", {})
                st.write("**Company Info (summary):**")
                st.json({k: info.get(k) for k in ["symbol", "shortName", "sector", "industry", "marketCap"]})

                # Recent history
                hist = t.history(period="5d")
                if not hist.empty:
                    st.write("**Last 5 days history:**")
                    st.dataframe(hist)
                else:
                    st.warning("No history returned.")

                # News
                try:
                    news = getattr(t, "news", [])
                    if news:
                        st.write("**Latest News Headlines:**")
                        for n in news[:5]:
                            st.markdown(f"- [{n['title']}]({n.get('link','')}) ({n.get('publisher','Unknown')})")
                    else:
                        st.warning("No news returned.")
                except Exception as e:
                    st.error(f"News fetch failed: {e}")

            except Exception as e:
                st.error(f"Ticker fetch failed: {e}")


# -----------------------
# Cached Data Fetch
# -----------------------
@st.cache_data(ttl=600)  # cache for 10 minutes
def fetch_signals():
    """Fetch signals for all tickers with minimal Yahoo requests."""
    results = generate_signals_for_tickers(WATCHLIST)
    time.sleep(1)  # small backoff
    return results


@st.cache_data(ttl=600)
def fetch_news():
    """Fetch news for watchlist with staggered requests."""
    news_dict = {}
    for t in WATCHLIST:
        try:
            news_dict[t] = get_news_feed(t)
        except Exception:
            news_dict[t] = []
        time.sleep(1)  # stagger requests
    return news_dict


# -----------------------
# Main App
# -----------------------
def main():
    st.set_page_config(page_title="Event-Driven Signal Dashboard", layout="wide")

    st.title("Event-Driven Signal Dashboard")
    st.markdown(
        """
        This dashboard predicts whether the underlying price of a stock will **rise**
        or **fall** after the latest news event, using free data from Yahoo
        Finance. Each run processes a small set of liquid tickers, computes a
        suite of features from news and options data, and logs the predictions
        alongside the next day's return for backtesting.
        """
    )

    # Manual reload button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

    # Signals
    with st.spinner("Fetching signals..."):
        signals_df = fetch_signals()

    st.subheader("Today's Signals")
    st.dataframe(signals_df)

    append_signals_to_csv(signals_df, CSV_PATH)

    # Trade history
    st.subheader("Trade Log (historic predictions)")
    try:
        history_df = pd.read_csv(CSV_PATH)
        st.dataframe(history_df.tail(50))
    except Exception:
        st.info("No trade history found yet.")

    # News
    st.subheader("Latest News Feed")
    news_data = fetch_news()
    for t, items in news_data.items():
        st.markdown(f"### {t}")
        if not items:
            st.write("No news available.")
        else:
            for item in items:
                title = item.get("title", "No title")
                link = item.get("link")
                publisher = item.get("publisher", "Unknown")
                if link:
                    st.markdown(f"- [{title}]({link}) ({publisher})")
                else:
                    st.markdown(f"- {title} ({publisher})")

    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Debug tester
    yfinance_tester()


if __name__ == "__main__":
    main()
