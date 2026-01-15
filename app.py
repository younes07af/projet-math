import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta

# --- SETTINGS & STYLING ---
st.set_page_config(page_title="Analyse Financière Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- MATH FUNCTIONS ---
def get_stats(returns):
    """Calculates descriptive stats safely."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return None
    return {
        "Mean": np.mean(clean_returns),
        "Median": np.median(clean_returns),
        "Std Dev": np.std(clean_returns),
        "Skewness": stats.skew(clean_returns),
        "Kurtosis": stats.kurtosis(clean_returns),
        "Min": np.min(clean_returns),
        "Max": np.max(clean_returns)
    }

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_data(symbol, start, end, interval):
    try:
        data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        # Handle Multi-index headers in newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Paramètres")
ticker = st.sidebar.text_input("Symbole (ex: BTC-USD, AAPL)", value="BTC-USD").upper()
start_date = st.sidebar.date_input("Début", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("Fin", datetime.now())
interval = st.sidebar.selectbox("Intervalle", ["1d", "1h", "15m"])
calc_method = st.sidebar.radio("Calcul des Rendements", ["Arithmétique", "Logarithmique"])

# --- LOAD DATA ---
df = fetch_data(ticker, start_date, end_date, interval)

if df.empty or 'Close' not in df.columns:
    st.error(f"❌ Erreur : Impossible de récupérer les données pour '{ticker}'.")
    st.info("Vérifiez votre connexion internet ou le symbole (ex: utilisez BTC-USD au lieu de BTC).")
    st.stop()

# --- CALCULATIONS ---
if calc_method == "Arithmétique":
    df['Returns'] = df['Close'].pct_change()
else:
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# --- UI LAYOUT ---
st.title(f"📊 Dashboard Financier : {ticker}")

tab1, tab2, tab3 = st.tabs(["📈 Graphique", "🧮 Statistiques", "🧪 Backtesting"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Prix"))
    
    # Simple SMA 20 Overlay
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1), name="SMA 20"))
    
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    s = get_stats(df['Returns'])
    if s:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Moyenne", f"{s['Mean']:.5f}")
        c2.metric("Volatilité (Journalière)", f"{s['Std Dev']:.4f}")
        c3.metric("Skewness", f"{s['Skewness']:.3f}")
        c4.metric("Kurtosis", f"{s['Kurtosis']:.3f}")
        
        fig_hist = px.histogram(df['Returns'].dropna(), nbins=50, title="Distribution des Rendements", template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Normality Test
        res = stats.jarque_bera(df['Returns'].dropna())
        st.write(f"**Test de Jarque-Bera (p-value):** `{res[1]:.4f}`")
        if res[1] < 0.05:
            st.warning("La distribution n'est pas normale (p < 0.05).")
        else:
            st.success("La distribution semble normale.")

with tab3:
    st.subheader("Stratégie de Croisement SMA (20/50)")
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    df['Signal'] = 0
    df.loc[df['SMA20'] > df['SMA50'], 'Signal'] = 1
    df['Strat_Ret'] = df['Signal'].shift(1) * df['Returns']
    df['Equity'] = 1000 * (1 + df['Strat_Ret'].fillna(0)).cumprod()
    
    col_bt1, col_bt2 = st.columns(2)
    final_val = df['Equity'].iloc[-1]
    col_bt1.metric("Capital Final", f"${final_val:.2f}")
    col_bt2.metric("Performance", f"{((final_val-1000)/10):.2f}%")
    
    fig_equity = px.line(df, x=df.index, y='Equity', title="Évolution du Capital (Base $1000)", template="plotly_dark")
    st.plotly_chart(fig_equity, use_container_width=True)

st.caption("Projet Mathématiques Appliquées - Python 3.14 Compatible")