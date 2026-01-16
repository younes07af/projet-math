import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta

# ============================================
# SETTINGS & THEME (Section 7: Dashboard Architecture)
# ============================================
st.set_page_config(page_title="Finance Analytics - Applied Math Project", layout="wide")

# Custom Professional Styling (Dark Theme inspired by Bloomberg/TradingView)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e222d; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    div[data-testid="stExpander"] { background-color: #1e222d; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1e222d; 
        border-radius: 4px 4px 0px 0px; 
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# SECTION 6.1: ACQUISITION DES DONNÉES
# ============================================
st.sidebar.header("⚙️ Configuration du Marché")

with st.sidebar:
    st.subheader("Paramètres de l'Actif")
    ticker = st.text_input("Symbole Boursier (ex: BTC-USD, AAPL)", value="BTC-USD").upper()
    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Date de début", datetime.now() - timedelta(days=365))
    end_date = col_d2.date_input("Date de fin", datetime.now())
    
    freq = st.selectbox("Fréquence temporelle", ["1d", "1h", "15m", "5m"])
    
    st.markdown("---")
    st.subheader("Paramètres Stratégie")
    sma_f = st.number_input("SMA Courte (Période)", value=20, min_value=2)
    sma_s = st.number_input("SMA Longue (Période)", value=50, min_value=5)
    cap_initial = st.number_input("Capital de départ ($)", value=1000, min_value=100)
    frais_tx = st.slider("Frais de transaction (%)", 0.0, 1.0, 0.1) / 100

@st.cache_data(ttl=3600)
def fetch_market_data(symbol, start, end, interval):
    """Imports financial data via Yahoo Finance API (Section 6.1)"""
    try:
        data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if data.empty:
            return None
        # Robust handling for yfinance MultiIndex headers
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.reset_index()
    except Exception as e:
        st.error(f"Erreur API yfinance: {e}")
        return None

# ============================================
# SECTION 6.2 & 6.3: TRAITEMENT MATHÉMATIQUE & STATS
# ============================================
def process_data_math(df):
    """Calculates returns and descriptive statistics (Section 6.2 & 6.3)"""
    # Rendements Arithmétiques (Rt)
    df['R_Arith'] = df['Close'].pct_change()
    
    # Rendements Logarithmiques (rt)
    df['R_Log'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Statistiques Descriptives
    ret = df['R_Arith'].dropna()
    
    metrics = {
        "Moyenne (μ)": ret.mean(),
        "Médiane": ret.median(),
        "Écart-type (σ)": ret.std(),
        "Volatilité Annualisée (σ * √252)": ret.std() * np.sqrt(252),
        "Skewness (Asymétrie)": stats.skew(ret),
        "Kurtosis (Aplatissement)": stats.kurtosis(ret),
        "Percentile 5% (VaR)": ret.quantile(0.05),
        "Percentile 25%": ret.quantile(0.25),
        "Percentile 75%": ret.quantile(0.75),
        "Percentile 95%": ret.quantile(0.95),
        "Maximum": ret.max(),
        "Minimum": ret.min()
    }
    return df, metrics

# ============================================
# SECTION 6.4: INDICATEURS TECHNIQUES
# ============================================
def compute_indicators(df, n_fast, n_slow):
    """Adds technical indicators to the dataframe (Section 6.4)"""
    # SMA Fast & Slow
    df['SMA_Fast'] = df['Close'].rolling(window=n_fast).mean()
    df['SMA_Slow'] = df['Close'].rolling(window=n_slow).mean()
    
    # EMA (Exponential Moving Average)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Bandes de Bollinger (20, 2)
    bb_mid = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = bb_mid + (2 * bb_std)
    df['BB_Lower'] = bb_mid - (2 * bb_std)
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# ============================================
# SECTION 6.5: BACKTESTING
# ============================================
def execute_backtest(df, initial_capital, fees):
    """Simulates trading strategy based on SMA crossover (Section 6.5)"""
    # Signal generation (6.5.1)
    df['Signal'] = np.where(df['SMA_Fast'] > df['SMA_Slow'], 1, 0)
    
    # Shift position by 1 period to prevent look-ahead bias (Crucial Math Step)
    df['Position'] = df['Signal'].shift(1)
    
    # Strategy Returns calculation (6.5.2)
    # R_strat = Position_{t-1} * R_actif_t
    df['Strat_Ret_Raw'] = df['Position'] * df['R_Arith']
    
    # Fee deduction on every trade (change in position)
    df['Trade_Executed'] = df['Position'].diff().abs().fillna(0)
    df['Strat_Ret_Net'] = df['Strat_Ret_Raw'] - (df['Trade_Executed'] * fees)
    
    # Capital Evolution (Equity Curve)
    df['Equity'] = initial_capital * (1 + df['Strat_Ret_Net'].fillna(0)).cumprod()
    df['Benchmark'] = initial_capital * (1 + df['R_Arith'].fillna(0)).cumprod()
    
    # Performance Metrics
    r_strat = df['Strat_Ret_Net'].dropna()
    total_ret = (df['Equity'].iloc[-1] / initial_capital) - 1
    
    # Sharpe Ratio (Risk-free rate assumed 0)
    sharpe = (r_strat.mean() / r_strat.std()) * np.sqrt(252) if r_strat.std() != 0 else 0
    
    # Maximum Drawdown (MDD)
    equity_peak = df['Equity'].cummax()
    drawdown = (df['Equity'] - equity_peak) / equity_peak
    max_dd = drawdown.min()
    
    perf = {
        "Val. Finale": df['Equity'].iloc[-1],
        "Rendement Total": total_ret,
        "Ratio de Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Total Trades": int(df['Trade_Executed'].sum())
    }
    return df, perf

# ============================================
# MAIN DASHBOARD LOGIC
# ============================================
st.title(f"📈 Dashboard de Trading Quantitatif : {ticker}")

data_source = fetch_market_data(ticker, start_date, end_date, freq)

if data_source is not None:
    # 1. Processing Pipeline
    df_proc, math_stats = process_data_math(data_source)
    df_proc = compute_indicators(df_proc, sma_f, sma_s)
    df_proc, bt_results = execute_backtest(df_proc, cap_initial, frais_tx)

    # 2. Main Layout Tabs (Section 7)
    tab1, tab2, tab3 = st.tabs(["📊 Analyse Graphique", "🧮 Statistiques & Risque", "🧪 Rapport Backtesting"])

    with tab1:
        # Technical chart with Subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Row 1: Price and Overlays
        fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['Close'], name="Prix Clôture", line=dict(color='#2962ff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['SMA_Fast'], name=f"SMA {sma_f}", line=dict(color='#ff9800', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['SMA_Slow'], name=f"SMA {sma_s}", line=dict(color='#4caf50', width=1.5)), row=1, col=1)
        
        # Bollinger Bands Area
        fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['BB_Upper'], line=dict(color='rgba(255,255,255,0.1)'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['BB_Lower'], line=dict(color='rgba(255,255,255,0.1)'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)', showlegend=False), row=1, col=1)

        # Row 2: RSI
        fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['RSI'], name="RSI (14)", line=dict(color='#9c27b0')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#66bb6a", row=2, col=1)

        fig.update_layout(height=650, template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Analyse Statistique des Rendements")
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            st.subheader("Indicateurs Mathématiques")
            stats_display = pd.DataFrame.from_dict(math_stats, orient='index', columns=['Valeur'])
            st.dataframe(stats_display.style.format("{:.4f}"), use_container_width=True)
            
            # Normality Test (Section 6.3)
            # Shapiro-Wilk test (limited to first 5000 samples)
            stat_w, p_value = stats.shapiro(df_proc['R_Arith'].dropna().iloc[:5000])
            st.metric("Test Shapiro-Wilk (p-value)", f"{p_value:.5f}")
            if p_value < 0.05:
                st.error("La distribution des rendements n'est pas normale (H0 rejetée).")
            else:
                st.success("La distribution semble normale (H0 acceptée).")

        with col_s2:
            st.subheader("Distribution Fréquentielle")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df_proc['R_Arith'].dropna(), nbinsx=60, marker_color='#2962ff', opacity=0.75))
            fig_hist.update_layout(template="plotly_dark", xaxis_title="Rendement Journalier", yaxis_title="Fréquence")
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.header("Analyse de la Performance du Pipeline")
        
        # High level metrics
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        met_col1.metric("Capital Final", f"${bt_results['Val. Finale']:,.2f}")
        met_col2.metric("Rendement Total", f"{bt_results['Rendement Total']:.2%}")
        met_col3.metric("Ratio de Sharpe", f"{bt_results['Ratio de Sharpe']:.2f}")
        met_col4.metric("Max Drawdown", f"{bt_results['Max Drawdown']:.2%}")

        # Strategy vs Market chart
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['Equity'], name="Stratégie (Net de frais)", fill='tozeroy', line=dict(color='#00e676', width=2)))
        fig_equity.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['Benchmark'], name="Benchmark (Market)", line=dict(color='#757575', dash='dot')))
        fig_equity.update_layout(title="Évolution de la Valeur du Portefeuille vs Marché", template="plotly_dark", height=450)
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Section 6.5.3: Critique
        st.subheader("💡 Analyse Critique de la Stratégie")
        mkt_ret = (df_proc['Close'].iloc[-1] / df_proc['Close'].iloc[0]) - 1
        st.info(f"""
        **Observations Clés :**
        - La stratégie a réalisé un rendement net de **{bt_results['Rendement Total']:.2%}**. 
        - Comparativement, le marché (Buy & Hold) a réalisé **{mkt_ret:.2%}**.
        - Le Ratio de Sharpe de **{bt_results['Ratio de Sharpe']:.2f}** permet de juger si le risque pris a été correctement rémunéré.
        - Le nombre total de transactions effectuées s'élève à **{bt_results['Total Trades']}**, chacune impactée par des frais de **{frais_tx*100:.2f}%**.
        """)

    # Bottom Raw Data View
    with st.expander("📂 Consulter les données de calcul matricielles"):
        st.dataframe(df_proc.tail(20), use_container_width=True)

else:
    st.error("⚠️ Symbole invalide ou erreur réseau. Impossible de charger les données OHLC.")

st.markdown("---")
st.caption("Projet de Mathématiques Appliquées à la Finance | Encadrant: M. Hamza Saber | Promotion 2026")
