"""
Streamlit application for event‑driven stock signal generation.

This app runs daily across a predefined set of highly liquid tickers.  It
retrieves the latest news headlines and option chain data from Yahoo Finance,
computes an EventScore for each ticker using sentiment, put/call ratio,
implied volatility change and volume spike, and classifies the expected
direction of the underlying price ("RISE" or "FALL") along with a
confidence magnitude.  All signals and their underlying metrics are logged
to a CSV file for later analysis.

To run this app locally:

    streamlit run app.py

To deploy for free on Streamlit Cloud:
1. Push this repository to GitHub.
2. Sign in to https://streamlit.io and create a new app using the GitHub repo.
3. Choose ``app.py`` as the entry point.
4. The app will fetch data and log signals on each run.

Note: Data retrieval relies on external network access.  In offline
environments the app will display zeros or fallback values instead of
live data.
"""

import os
import datetime

import pandas as pd
import streamlit as st

from signal_engine import (
    generate_signals_for_tickers,
    append_signals_to_csv,
)


# Define the list of tickers to track
WATCHLIST = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA"]

# Define the path for logging signals; ensure the data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "trades.csv")
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    st.title("Event‑Driven Signal Dashboard")
    st.markdown(
        """
        This dashboard predicts whether the underlying price of a stock will **rise**
        or **fall** after the latest news event, using free data from Yahoo
        Finance.  Each run processes a small set of liquid tickers, computes a
        suite of features from news and options data, and logs the predictions
        alongside the next day's return for backtesting.
        """
    )
    # Generate signals and display them
    with st.spinner("Fetching data and computing signals..."):
        signals_df = generate_signals_for_tickers(WATCHLIST)
    st.subheader("Today's Signals")
    st.dataframe(signals_df)
    # Append to CSV
    append_signals_to_csv(signals_df, CSV_PATH)
    # Display trade history
    st.subheader("Trade Log (historic predictions)")
    try:
        history_df = pd.read_csv(CSV_PATH)
        st.dataframe(history_df)
    except Exception:
        st.info("No trade history found yet.")

    # Show the date/time of last update
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.subheader("Latest News Feed")

    for t in tickers:
        st.markdown(f"### {t}")
        news_items = se.get_news_feed(t)
        if not news_items:
            st.write("No news available.")
        else:
            for item in news_items:
                title = item.get("title", "No title")
                link = item.get("link", None)
                publisher = item.get("publisher", "Unknown")
                if link:
                    st.markdown(f"- [{title}]({link}) ({publisher})")
                else:
                    st.markdown(f"- {title} ({publisher})")


if __name__ == "__main__":
    main()
