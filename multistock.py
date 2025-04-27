import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import date, timedelta
import socket
from io import BytesIO

# === Simulated Data for Demo ===
# === Simulated Data for Demo ===
@st.cache_data
def load_sample_data():
    dates = pd.date_range(start="2024-01-01", periods=100, freq='B')
    stock_list = [
        {"Ticker": "AAPL", "Industry": "Technology"},
        {"Ticker": "MSFT", "Industry": "Technology"},
        {"Ticker": "JPM", "Industry": "Financial Services"},
        {"Ticker": "TSLA", "Industry": "Automotive"}
    ]
    data = pd.DataFrame()
    for stock in stock_list:
        prices = pd.Series(150 + (pd.Series(range(100)) * 0.3).values + (pd.Series(range(100)).rolling(5).mean().fillna(0)))
        sentiments = pd.Series([1 if i % 3 == 0 else -1 if i % 5 == 0 else 0 for i in range(100)])
        temp = pd.DataFrame({
            "Date": dates,
            "Ticker": stock["Ticker"],
            "Industry": stock["Industry"],
            "Adj Close": prices,
            "avg_sentiment": sentiments,
            "interest_rate": 3 + pd.Series(range(100)) * 0.01
        })
        temp["return_1d"] = temp["Adj Close"].pct_change()
        temp["predicted_change"] = temp["return_1d"].shift(-1).fillna(0)
        data = pd.concat([data, temp])
    return data.dropna()

# === Ollama Connection Check ===
def is_ollama_running(host="localhost", port=11434):
    try:
        socket.create_connection((host, port), timeout=1)
        return True
    except OSError:
        return False

# === GenAI Insight Function Using Ollama ===
def generate_insight_ollama(prompt_text):
    if not is_ollama_running():
        return "[Ollama is not running. Please open a terminal and run: ollama run mistral]"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt_text, "stream": False}
        )
        content = response.text.strip()
        return content if content else "[No response content returned from Ollama]"
    except Exception as e:
        return f"[Error generating insight: {str(e)}]"

# === Fetch News Headlines using NewsAPI ===
def fetch_news_headlines(stock_symbol, api_key):
    try:
        url = f"https://newsapi.org/v2/everything?q={stock_symbol}&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}"
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [article["title"] for article in articles if "title" in article]
    except Exception as e:
        return [f"Error fetching headlines: {str(e)}"]

# === Prompt Template Builder ===
def build_prompt(date, sentiment, predicted_change, headlines):
    return f"""
    Date: {date}
    Sentiment Score: {sentiment}
    Model Prediction: {predicted_change:.2%}

    Headlines:
    - {headlines[0] if len(headlines) > 0 else 'No headline'}
    - {headlines[1] if len(headlines) > 1 else ''}

    Based on the sentiment, market conditions, and the model prediction,
    provide a financial analyst-style explanation of what might influence the stock performance.
    """

# === Streamlit UI ===
st.set_page_config(page_title="Stock Insight Assistant", layout="wide")

st.title("📈 Intelligent Stock Prediction & Insight Assistant")

# Load data
data = load_sample_data()
st.sidebar.header("Configuration")
stocks = data["Ticker"].unique().tolist()
selected_stocks = st.sidebar.multiselect("Select Stocks", stocks, default=["AAPL"])
selected_industry = st.sidebar.selectbox("Select Industry", data["Industry"].unique())
st.sidebar.header("Select Date")
selected_date = st.sidebar.date_input("Date", value=date(2024, 3, 15), min_value=data["Date"].min().date(), max_value=data["Date"].max().date())
newsapi_key = st.sidebar.text_input("NewsAPI Key", type="password")

filtered_data = data[(data["Ticker"].isin(selected_stocks)) & (data["Industry"] == selected_industry)]
selected_row = filtered_data[filtered_data["Date"] == pd.to_datetime(selected_date)]

if not selected_row.empty:
    st.subheader("📋 Summary Table")
    summary_table = selected_row[["Ticker", "predicted_change", "avg_sentiment", "interest_rate"]].copy()
    summary_table.rename(columns={
        "predicted_change": "Predicted % Change",
        "avg_sentiment": "Sentiment Score",
        "interest_rate": "Interest Rate (%)"
    }, inplace=True)
    summary_table["Predicted % Change"] = summary_table["Predicted % Change"].apply(lambda x: f"{x:.2%}")
    summary_table["Interest Rate (%)"] = summary_table["Interest Rate (%)"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(summary_table, use_container_width=True)

    csv = summary_table.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Summary Table as CSV", data=csv, file_name="summary_table.csv", mime="text/csv")

    for ticker in selected_row["Ticker"].unique():
        row = selected_row[selected_row["Ticker"] == ticker].iloc[0]
        st.subheader(f"Market Details for {ticker} on {selected_date} ({row['Industry']})")

        if newsapi_key:
            headlines = fetch_news_headlines(ticker, newsapi_key)
        else:
            headlines = ["Apple launches new AI chip for edge computing", "Fed expected to maintain interest rates"]

        st.markdown("### 🧠 AI Insight")
        with st.spinner(f"Generating GenAI insight from Ollama for {ticker}..."):
            prompt = build_prompt(selected_date, row['avg_sentiment'], row['predicted_change'], headlines)
            insight = generate_insight_ollama(prompt)
            st.success(insight)

        st.markdown("### 📰 News Headlines")
        for headline in headlines:
            st.write(f"- {headline}")
else:
    st.warning("No data available for selected date, stocks, and industry.")

# === Stock Price & Prediction Trend Chart with Legend and Hover ===
st.markdown("### 📊 Stock Price & Prediction Trend")
if not filtered_data.empty:
    pivot_data = filtered_data.pivot(index="Date", columns="Ticker", values="Adj Close")
    fig_price = px.line(pivot_data, labels={"value": "Adj Close", "Date": "Date"}, title="Stock Prices")
    st.plotly_chart(fig_price, use_container_width=True)

    fig_price_buffer = BytesIO()
    fig_price.write_image(fig_price_buffer, format="png")
    st.download_button("📥 Download Stock Price Chart", data=fig_price_buffer.getvalue(), file_name="stock_price_chart.png", mime="image/png")

# === Chat-like Q&A Interface ===
st.markdown("### 💬 Ask a Question About the Market")
user_question = st.text_input("Ask something like 'Why is AAPL predicted to go up?' or 'What does the sentiment look like this week?'")

if user_question and not filtered_data.empty:
    with st.spinner("Generating response from Ollama..."):
        chat_prompt = f"""
        Based on recent stock data and sentiment for selected stocks in {selected_industry}, answer this question:
        {user_question}
        """
        chat_response = generate_insight_ollama(chat_prompt)
        st.markdown(f"**🧠 GenAI:** {chat_response}")

# === 7-Day Sentiment Trend Chart with Legend and Hover ===
st.markdown("### 📈 7-Day Sentiment Trend")
recent_data = filtered_data[filtered_data["Date"] >= pd.to_datetime(selected_date) - timedelta(days=7)]
if not recent_data.empty:
    sentiment_chart = recent_data.pivot_table(index="Date", columns="Ticker", values="avg_sentiment", aggfunc="mean")
    fig_sentiment = px.line(sentiment_chart, labels={"value": "Avg Sentiment", "Date": "Date"}, title="7-Day Sentiment Trend")
    st.plotly_chart(fig_sentiment, use_container_width=True)

    fig_sentiment_buffer = BytesIO()
    fig_sentiment.write_image(fig_sentiment_buffer, format="png")
    st.download_button("📥 Download Sentiment Trend Chart", data=fig_sentiment_buffer.getvalue(), file_name="sentiment_trend_chart.png", mime="image/png")
else:
    st.info("Not enough recent data to show sentiment trend.")
