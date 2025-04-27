import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import date, timedelta
import socket

# === Simulated Data for Demo ===
@st.cache_data
def load_sample_data():
    dates = pd.date_range(start="2024-01-01", periods=100, freq='B')
    prices = pd.Series(150 + (pd.Series(range(100)) * 0.3).values + (pd.Series(range(100)).rolling(5).mean().fillna(0)))
    sentiments = pd.Series([1 if i % 3 == 0 else -1 if i % 5 == 0 else 0 for i in range(100)])
    data = pd.DataFrame({
        "Date": dates,
        "Ticker": ["AAPL"] * 100,  # Default ticker
        "Adj Close": prices,
        "avg_sentiment": sentiments,
        "interest_rate": 3 + pd.Series(range(100)) * 0.01
    })
    data["return_1d"] = data["Adj Close"].pct_change()
    data["predicted_change"] = data["return_1d"].shift(-1).fillna(0)
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
selected_stock = st.sidebar.selectbox("Select Stock", stocks)
st.sidebar.header("Select Date")
selected_date = st.sidebar.date_input("Date", value=date(2024, 3, 15), min_value=data["Date"].min().date(), max_value=data["Date"].max().date())
newsapi_key = st.sidebar.text_input("NewsAPI Key", type="password")

filtered_data = data[(data["Ticker"] == selected_stock)]
selected_row = filtered_data[filtered_data["Date"] == pd.to_datetime(selected_date)]

if not selected_row.empty:
    row = selected_row.iloc[0]
    st.subheader(f"Market Summary for {selected_stock} on {selected_date}")
    st.metric("Predicted % Change", f"{row['predicted_change']:.2%}")
    st.metric("Sentiment Score", f"{row['avg_sentiment']}")
    st.metric("Interest Rate", f"{row['interest_rate']:.2f}%")

    # === Real-time News Headlines ===
    if newsapi_key:
        headlines = fetch_news_headlines(selected_stock, newsapi_key)
    else:
        headlines = ["Apple launches new AI chip for edge computing", "Fed expected to maintain interest rates"]

    # Generate AI Insight
    st.markdown("### 🧠 AI Insight")
    with st.spinner("Generating GenAI insight from Ollama..."):
        prompt = build_prompt(selected_date, row['avg_sentiment'], row['predicted_change'], headlines)
        insight = generate_insight_ollama(prompt)
        st.success(insight)

    st.markdown("### 📰 News Headlines")
    for headline in headlines:
        st.write(f"- {headline}")
else:
    st.warning("No data available for selected date and stock.")

# Line chart
st.markdown("### 📊 Stock Price & Prediction Trend")
st.line_chart(filtered_data.set_index("Date")["Adj Close"])

# === Chat-like Q&A Interface ===
st.markdown("### 💬 Ask a Question About the Market")
user_question = st.text_input("Ask something like 'Why is AAPL predicted to go up?' or 'What does the sentiment look like this week?'")

if user_question:
    with st.spinner("Generating response from Ollama..."):
        chat_prompt = f"""
        Based on recent stock data and sentiment for {selected_stock}, answer this question:
        {user_question}
        """
        chat_response = generate_insight_ollama(chat_prompt)
        st.markdown(f"**🧠 GenAI:** {chat_response}")

# === 7-Day Sentiment Trend Chart ===
st.markdown("### 📈 7-Day Sentiment Trend")
recent_data = filtered_data[filtered_data["Date"] >= pd.to_datetime(selected_date) - timedelta(days=7)]
if not recent_data.empty:
    sentiment_chart = recent_data.set_index("Date")["avg_sentiment"]
    st.line_chart(sentiment_chart)
else:
    st.info("Not enough recent data to show sentiment trend.")
