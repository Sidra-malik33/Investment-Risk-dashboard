import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from io import StringIO
from dotenv import load_dotenv

load_dotenv()
import os
# ==============================================
# 🔐 LOGIN + SESSION AUTHENTICATION
# ==============================================
st.set_page_config(
    page_title="💼 Investment Risk Dashboard",
    page_icon="💹",
    layout="wide"
)


# Fake user database
USER_CREDENTIALS = {"Sidnad": "12345"}

# Check login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_page():
    st.markdown("### 🔐 Secure Investment Risk Dashboard")
    st.subheader("Login to Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.success("✅ Login successful! Redirecting...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

if not st.session_state.authenticated:
    login_page()
    st.stop()

# ==============================================
# 📊 SIDEBAR NAVIGATION
# ==============================================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to section:",
    [
        "📈 Portfolio Optimizer",
        "📉 Risk Analyzer",
        "💳 Credit Scoring",
        "🛡️ Fraud Detection"
    ]
)

# ==============================================
# ==============================================
# 1️⃣ PORTFOLIO OPTIMIZER (ALPHA VANTAGE + FALLBACK)
# ==============================================
def portfolio_optimizer():
    st.header("📈 Portfolio Optimizer (Modern Portfolio Theory)")
    st.write("Optimize your investment portfolio using historical data and risk-return tradeoff.")

    # --- Choose tickers ---
    tickers = st.multiselect(
        "Choose stock tickers (from Alpha Vantage):",
        ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "DIS", "NFLX"],
        default=["AAPL", "MSFT", "AMZN"]
    )

    investment_amount = st.number_input("💰 Total Investment Amount ($)", min_value=1000, value=10000, step=500)

    # ALPHA_VANTAGE_API_KEY = "LUPI4ELRENE8U24B"  # ⚠️ Replace with your own valid key
    ALPHA_VANTAGE_API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
    # --- Function to fetch stock data safely ---
    def get_alpha_vantage_data(symbol):
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            r = requests.get(url, params=params, timeout=10)
            data = r.json()

            if "Time Series (Daily)" not in data:
                st.warning(f"⚠️ No data for {symbol}. Retrying in 10s...")
                time.sleep(10)
                return None

            df = pd.DataFrame(data["Time Series (Daily)"]).T
            df.index = pd.to_datetime(df.index)
            df["Adj Close"] = df["4. close"].astype(float)
            return df["Adj Close"].sort_index()

        except Exception as e:
            st.error(f"Error fetching {symbol}: {e}")
            return None

    # --- Main Optimization Process ---
    if len(tickers) < 2:
        st.warning("⚠️ Please select at least two stocks.")
        return

    st.info("⏳ Fetching data from Alpha Vantage... (Free tier allows 5 requests/min)")
    prices = pd.DataFrame()

    for t in tickers:
        series = get_alpha_vantage_data(t)
        if series is not None and not series.empty:
            prices[t] = series
        else:
            st.warning(f"⚠️ Skipping {t} (no valid data).")
        time.sleep(12)  # Respect rate limit

    if prices.empty:
        st.error("❌ No valid stock data fetched. Try again later.")
        return

    # --- Portfolio calculations ---
    returns = prices.pct_change().dropna()
    if returns.empty:
        st.error("⚠️ Not enough data to compute returns.")
        return

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_portfolios = 3000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return = np.sum(weights * mean_returns) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = results[0, i] / results[1, i]

    results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe"])
    max_sharpe_idx = results_df["Sharpe"].idxmax()
    opt_weights = weights_record[max_sharpe_idx]

    st.success("✅ Optimization complete!")

    st.write("### 💡 Optimal Portfolio")
    st.write(f"**Return:** {results_df.loc[max_sharpe_idx, 'Return']*100:.2f}%")
    st.write(f"**Risk:** {results_df.loc[max_sharpe_idx, 'Volatility']*100:.2f}%")
    st.write(f"**Sharpe Ratio:** {results_df.loc[max_sharpe_idx, 'Sharpe']:.2f}")

    allocation = pd.DataFrame({
        "Stock": tickers,
        "Weight (%)": np.round(opt_weights * 100, 2),
        "Investment ($)": np.round(opt_weights * investment_amount, 2)
    })
    st.dataframe(allocation, use_container_width=True)

    # --- Efficient Frontier Visualization ---
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(results_df.Volatility, results_df.Return, c=results_df.Sharpe, cmap="viridis")
    ax.scatter(results_df.loc[max_sharpe_idx, "Volatility"],
               results_df.loc[max_sharpe_idx, "Return"],
               color="red", marker="*", s=150)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Efficient Frontier")
    st.pyplot(fig)

# ==============================================
# 2️⃣ RISK ANALYZER
# ==============================================
def risk_analyzer():
    st.header("📉 Investment Risk Analyzer")
    st.write("Analyze portfolio risk using beta, VaR, and CVaR metrics.")

    returns = np.random.normal(0.001, 0.02, 1000)
    VaR = np.percentile(returns, 5)
    CVaR = returns[returns <= VaR].mean()
    Sharpe = np.mean(returns) / np.std(returns)

    st.metric("📊 Value at Risk (5%)", f"{VaR:.3f}")
    st.metric("📉 Conditional VaR", f"{CVaR:.3f}")
    st.metric("⚖️ Sharpe Ratio", f"{Sharpe:.2f}")

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(returns, bins=30, kde=True, ax=ax)
    ax.axvline(VaR, color="r", linestyle="--", label="VaR 5%")
    ax.legend()
    st.pyplot(fig)

# ==============================================
# 3️⃣ CREDIT SCORING
# ==============================================
def credit_scoring():
    st.header("💳 Credit Scoring & Recommendation")
    st.write("Input borrower info to get credit grade & approval suggestion.")

    income = st.number_input("Monthly Income ($)", value=5000)
    debt = st.number_input("Monthly Debt ($)", value=1500)
    age = st.slider("Age", 18, 70, 30)
    history = st.slider("Credit History (years)", 0, 20, 5)

    if st.button("Calculate Credit Score"):
        score = (income / (debt + 1)) + (history * 10) - (70 - age)
        grade = "A+" if score > 300 else "B" if score > 150 else "C"
        decision = "✅ Approved" if grade in ["A+", "B"] else "❌ Rejected (High Risk)"

        st.metric("Credit Score", round(score, 1))
        st.metric("Grade", grade)
        st.success(decision) if "Approved" in decision else st.error(decision)

# ==============================================
# 4️⃣ FRAUD DETECTION (USING YOUR MODELS)
# ==============================================
def fraud_detection():
    st.header("🛡️ Fraud Detection Dashboard")
    st.write("Upload transactions and detect potential frauds.")

    iso_forest = joblib.load("models/iso_forest.pkl")
    autoencoder = keras.models.load_model("models/autoencoder.h5", compile=False)
    scaler = joblib.load("models/scaler.pkl")

    uploaded_file = st.file_uploader("📂 Upload Credit Card Transactions CSV", type=["csv"])
    threshold = st.slider("Set Autoencoder MSE Threshold", 0.001, 0.05, 0.01, step=0.001)

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        X_scaled = scaler.transform(data.drop(["Time", "Class"], axis=1))

        # Predictions
        iso_pred = iso_forest.predict(X_scaled)
        ae_recon = autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - ae_recon), axis=1)
        ae_pred = np.where(mse > threshold, -1, 1)

        combined_pred = np.where((iso_pred == -1) | (ae_pred == -1), 1, 0)
        data["Fraud_Pred"] = combined_pred

        frauds = data[data["Fraud_Pred"] == 1]
        st.metric("Total Transactions", len(data))
        st.metric("Frauds Detected", len(frauds))

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x="Fraud_Pred", data=data, palette=["green", "red"], ax=ax)
        ax.set_xticklabels(["Normal", "Fraud"])
        st.pyplot(fig)

        st.write("🚨 Detected Fraudulent Transactions")
        st.dataframe(frauds, use_container_width=True)

# ==============================================
# 🔄 PAGE ROUTING
# ==============================================
if page == "📈 Portfolio Optimizer":
    portfolio_optimizer()
elif page == "📉 Risk Analyzer":
    risk_analyzer()
elif page == "💳 Credit Scoring":
    credit_scoring()
elif page == "🛡️ Fraud Detection":
    fraud_detection()
st.markdown("🚀 **Investment Risk Dashboard** — by Sidra Malik (Fintech AI Developer)", unsafe_allow_html=True)
