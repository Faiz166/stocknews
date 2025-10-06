"""
Streamlit application for event-driven stock signal generation.

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
import yfinance as yf

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

                # Basic info
                info = t.info
                st.write("**Company Info (summary):**")
                st.json({k: info.get(k) for k in ["symbol", "shortName", "sector", "industry", "marketCap"]})

                # Recent history
                hist = t.history(period="5d")
                st.write("**Last 5 days history:**")
                st.dataframe(hist)

                # News
                try:
                    news = t.news
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

    # Generate signals and display them
    with st.spinner("Fetching data and computing signals..."):
        signals_df = generate_signals_for_tickers(WATCHLIST)

    st.subheader("Today's Signals")
    st.dataframe(signals_df)

    # Append to CSV
    append_signals_to_csv(signals_df, CSV_PATH)

    # Trade history
    st.subheader("Trade Log (historic predictions)")
    try:
        history_df = pd.read_csv(CSV_PATH)
        st.dataframe(history_df.tail(50))
    except Exception:
        st.info("No trade history found yet.")

    # News feed
    st.subheader("Latest News Feed")
    for t in WATCHLIST:
        st.markdown(f"### {t}")
        news_items = get_news_feed(t)
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

    # Footer
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Debug tester
    yfinance_tester()


if __name__ == "__main__":
    main()
