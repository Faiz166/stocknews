"""
Streamlit application for event-driven stock signal generation.

This app runs daily across a predefined set of highly liquid tickers. It
retrieves the latest news headlines and option chain data from Yahoo Finance,
computes an EventScore for each ticker, and classifies the expected direction
("RISE" or "FALL") along with a confidence magnitude. All signals and their
underlying metrics are logged to a CSV file for later analysis.
"""

import os
import datetime

import pandas as pd
import streamlit as st

from signal_engine import (
    generate_signals_for_tickers,
    append_signals_to_csv,
    get_news_feed,
)

# -----------------------
# Configuration
# -----------------------
WATCHLIST = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "trades.csv")
os.makedirs(DATA_DIR, exist_ok=True)


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

    # -----------------------
    # Generate signals
    # -----------------------
    with st.spinner("Fetching data and computing signals..."):
        signals_df = generate_signals_for_tickers(WATCHLIST)

    st.subheader("Today's Signals")
    st.dataframe(signals_df)

    # Append to CSV for trade log
    append_signals_to_csv(signals_df, CSV_PATH)

    # -----------------------
    # Trade history
    # -----------------------
    st.subheader("Trade Log (historic predictions)")
    try:
        history_df = pd.read_csv(CSV_PATH)
        st.dataframe(history_df.tail(50))  # show last 50
    except FileNotFoundError:
        st.info("No trade history found yet.")

    # -----------------------
    # News Feed
    # -----------------------
    st.subheader("Latest News Feed")
    for ticker in WATCHLIST:
        st.markdown(f"### {ticker}")
        news_items = get_news_feed(ticker)
        if not news_items:
            st.write("No news available.")
        else:
            for item in news_items:
                title = item.get("title", "No title")
                link = item.get("link")
                publisher = item.get("publisher", "Unknown")
                if link:
                    st.markdown(f"- [{title}]({link}) ({publisher})")
                else:
                    st.markdown(f"- {title} ({publisher})")

    # -----------------------
    # Footer
    # -----------------------
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
