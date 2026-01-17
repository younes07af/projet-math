import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta

# ============================================
# CONFIGURATION ET THÈME (Section 7 du PDF)
# ============================================
st.set_page_config(page_title="Projet Finance - Plateforme Quantitative", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e222d; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    div[data-testid="stExpander"] { background-color: #1e222d; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1e222d; border-radius: 4px 4px 0px 0px; }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# SECTION 6.1: ACQUISITION DES DONNÉES
# ============================================
st.sidebar.header("⚙️ Configuration du Marché")

with st.sidebar:
    ticker = st.text_input("Symbole de l'actif (ex: BTC-USD, AAPL)", value="BTC-USD").upper()
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Date début", datetime(2023, 1, 1))
    end_date = col_d2.date_input("Date fin", datetime.now())
    interval = st.selectbox("Fréquence temporelle", ["1d", "1h", "15m", "5m"])
    
    st.markdown("---")
    st.subheader("Paramètres Stratégie")
    sma_f = st.number_input("SMA Courte", value=20, min_value=2)
    sma_s = st.number_input("SMA Longue", value=50, min_value=5)
    capital_init = st.number_input("Capital Initial ($)", value=1000, min_value=100)
    frais_tx = st.slider("Frais de transaction (%)", 0.0, 1.0, 0.1) / 100

@st.cache_data(ttl=3600)
def fetch_data(symbol, start, end, timeframe):
    """Importation sécurisée des données OHLC via Yahoo Finance"""
    try:
        # Correction pour les données intraday (limite de 60 jours pour 5m/15m)
        if timeframe in ["5m", "15m", "1h"]:
            data = yf.download(symbol, period="max", interval=timeframe, progress=False)
            data = data[(data.index.date >= start) & (data.index.date <= end)]
        else:
            data = yf.download(symbol, start=start, end=end, interval=timeframe, progress=False)
        
        if data.empty:
            return None
            
        # Nettoyage robuste des colonnes MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            # Si yfinance renvoie (Attribute, Ticker), on garde l'Attribute
            data.columns = data.columns.get_level_values(0)
            
        return data.reset_index()
    except Exception as e:
        st.error(f"Erreur API yfinance: {e}")
        return None

# ============================================
# SECTION 6.2 & 6.3: TRAITEMENT MATHÉMATIQUE & STATS
# ============================================
def calculate_metrics(df):
    """Calcul vectorisé des rendements et statistiques de risque"""
    df['R_Arith'] = df['Close'].pct_change()
    df['R_Log'] = np.log(df['Close'] / df['Close'].shift(1))
    
    ret = df['R_Arith'].dropna()
    if len(ret) < 2:
        return df, None
        
    stats_dict = {
        "Moyenne (Quotidienne)": ret.mean(),
        "Médiane": ret.median(),
        "Écart-type (σ)": ret.std(),
        "Volatilité Annualisée (σ * √252)": ret.std() * np.sqrt(252),
        "Skewness (Asymétrie)": stats.skew(ret),
        "Kurtosis (Aplatissement)": stats.kurtosis(ret),
        "Percentile 5% (VaR)": ret.quantile(0.05),
        "Percentile 25%": ret.quantile(0.25),
        "Percentile 75%": ret.quantile(0.75),
        "Percentile 95%": ret.quantile(0.95),
        "Max": ret.max(),
        "Min": ret.min()
    }
    return df, stats_dict

# ============================================
# SECTION 6.4: INDICATEURS TECHNIQUES
# ============================================
def apply_indicators(df, n_f, n_s):
    df['SMA_Fast'] = df['Close'].rolling(window=n_f).mean()
    df['SMA_Slow'] = df['Close'].rolling(window=n_s).mean()
    
    bb_mid = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = bb_mid + (2 * bb_std)
    df['BB_Lower'] = bb_mid - (2 * bb_std)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan) # Eviter division par zéro
    df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))
    
    return df

# ============================================
# SECTION 6.5: BACKTESTING
# ============================================
def run_backtest(df, cap_init, fees):
    df['Signal'] = np.where(df['SMA_Fast'] > df['SMA_Slow'], 1, 0)
    df['Position'] = df['Signal'].shift(1)
    df['Strat_Ret_Raw'] = df['Position'] * df['R_Arith'].fillna(0)
    
    df['Trade'] = df['Position'].diff().abs().fillna(0)
    df['Net_Strat_Ret'] = df['Strat_Ret_Raw'] - (df['Trade'] * fees)
    
    df['Equity'] = cap_init * (1 + df['Net_Strat_Ret']).cumprod()
    df['Market'] = cap_init * (1 + df['R_Arith'].fillna(0)).cumprod()
    
    r = df['Net_Strat_Ret'].dropna()
    if len(r) > 0:
        total_perf = (df['Equity'].iloc[-1] / cap_init) - 1
        sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
        peak = df['Equity'].cummax()
        mdd = ((df['Equity'] - peak) / peak).min()
    else:
        total_perf, sharpe, mdd = 0, 0, 0
    
    res = {
        "Val. Finale": df['Equity'].iloc[-1] if len(df) > 0 else cap_init,
        "Rendement Total (%)": total_perf * 100,
        "Ratio de Sharpe": sharpe,
        "Max Drawdown (%)": mdd * 100,
        "Nombre de Trades": int(df['Trade'].sum())
    }
    return df, res

# ============================================
# DASHBOARD
# ============================================
st.title(f"📊 Dashboard d'Analyse Financière : {ticker}")

raw_df = fetch_data(ticker, start_date, end_date, interval)

if raw_df is not None and len(raw_df) > sma_s:
    df_p, stats_m = calculate_metrics(raw_df)
    df_p = apply_indicators(df_p, sma_f, sma_s)
    df_p, results = run_backtest(df_p, capital_init, frais_tx)

    tab1, tab2, tab3 = st.tabs(["📉 Graphique", "🧮 Statistiques", "🎯 Backtesting"])

    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=df_p.iloc[:,0], y=df_p['Close'], name="Prix", line=dict(color='#2962ff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.iloc[:,0], y=df_p['SMA_Fast'], name="SMA Fast", line=dict(color='#ff9800', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.iloc[:,0], y=df_p['SMA_Slow'], name="SMA Slow", line=dict(color='#4caf50', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.iloc[:,0], y=df_p['BB_Upper'], line=dict(color='rgba(255,255,255,0.1)'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.iloc[:,0], y=df_p['BB_Lower'], line=dict(color='rgba(255,255,255,0.1)'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p.iloc[:,0], y=df_p['RSI'], name="RSI", line=dict(color='#9c27b0')), row=2, col=1)
        fig.update_layout(height=600, template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if stats_m:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(pd.DataFrame.from_dict(stats_m, orient='index', columns=['Valeur']).style.format("{:.4f}"))
                ret_clean = df_p['R_Arith'].dropna()
                if len(ret_clean) > 3:
                    _, p_val = stats.shapiro(ret_clean.iloc[:5000])
                    st.metric("Test Shapiro (p-value)", f"{p_val:.5f}")
            with col2:
                fig_h = go.Figure(go.Histogram(x=ret_clean, nbinsx=50, marker_color='#2962ff'))
                fig_h.update_layout(template="plotly_dark", title="Distribution")
                st.plotly_chart(fig_h, use_container_width=True)

    with tab3:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capital Final", f"${results['Val. Finale']:,.2f}")
        m2.metric("Rendement", f"{results['Rendement Total (%)']:.2f}%")
        m3.metric("Sharpe", f"{results['Ratio de Sharpe']:.2f}")
        m4.metric("Drawdown", f"{results['Max Drawdown (%)']:.2f}%")
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=df_p.iloc[:,0], y=df_p['Equity'], name="Stratégie", fill='tozeroy', line=dict(color='#00e676')))
        fig_e.add_trace(go.Scatter(x=df_p.iloc[:,0], y=df_p['Market'], name="Marché", line=dict(color='#757575', dash='dot')))
        fig_e.update_layout(template="plotly_dark", title="Évolution", height=400)
        st.plotly_chart(fig_e, use_container_width=True)
elif raw_df is not None:
    st.warning("⚠️ Pas assez de données pour calculer les indicateurs (Besoin d'au moins 50 points).")
else:
    st.error("❌ Erreur : Données introuvables. Vérifiez le symbole ou la période.")
