# ============================================================
# Portfolio Management Pro — ESSCA Project
# Prof. Olga Tatarnikova — Portfolio Management
# ============================================================
# Covers: Asset Allocation, MVO (Markowitz), Elton-Gruber,
# Merton Two-Fund, Black-Litterman, CAPM, Fama-French 3F,
# Active/Passive/Smart-Beta Strategies, Full Performance Eval
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from datetime import date, timedelta
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table as RLTable,
    TableStyle, Image as RLImage, PageBreak, HRFlowable
)
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ============================================================
# CONFIG & CONSTANTS
# ============================================================
st.set_page_config(
    page_title="Portfolio Management Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

TRADING_DAYS = 252

# ── Brand colours ──────────────────────────────────────────
PRIMARY   = "#C8102E"   # ESSCA red
DARK      = "#1C1C2E"   # deep navy
CARD_BG   = "#F7F9FC"
TEXT_DARK = "#1C1C2E"
ACCENT    = "#2E86AB"   # cool blue
SUCCESS   = "#2ECC71"
WARNING   = "#F39C12"

# ── Custom CSS ─────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background: {DARK};
  }}
  section[data-testid="stSidebar"] * {{
      color: #FFFFFF !important;
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stNumberInput label {{
      color: #CBD5E1 !important;
      font-size: 0.82rem;
  }}

  /* Header banner */
  .pm-header {{
      background: linear-gradient(135deg, {DARK} 0%, #2D2D44 100%);
      padding: 1.4rem 2rem;
      border-radius: 12px;
      margin-bottom: 1.5rem;
      border-left: 5px solid {PRIMARY};
  }}
  .pm-header h1 {{
      color: #FFFFFF;
      font-size: 1.75rem;
      font-weight: 700;
      margin: 0;
  }}
  .pm-header p {{
      color: #94A3B8;
      margin: 0.25rem 0 0 0;
      font-size: 0.9rem;
  }}

  /* Section titles */
  .section-title {{
      font-size: 1.15rem;
      font-weight: 600;
      color: {DARK};
      border-left: 4px solid {PRIMARY};
      padding-left: 0.75rem;
      margin: 1.5rem 0 0.75rem 0;
  }}

  /* KPI cards */
  .kpi-card {{
      background: {CARD_BG};
      border-radius: 10px;
      padding: 1rem 1.2rem;
      border: 1px solid #E2E8F0;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .kpi-label {{
      font-size: 0.75rem;
      color: #64748B;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }}
  .kpi-value {{
      font-size: 1.55rem;
      font-weight: 700;
      color: {DARK};
      margin-top: 0.15rem;
  }}
  .kpi-value.positive {{ color: {SUCCESS}; }}
  .kpi-value.negative {{ color: {PRIMARY}; }}

  /* Method card */
  .method-card {{
      background: white;
      border: 1px solid #E2E8F0;
      border-radius: 10px;
      padding: 1rem 1.2rem;
      margin-bottom: 0.75rem;
  }}

  /* Formula box */
  .formula-box {{
      background: #F1F5F9;
      border-left: 3px solid {ACCENT};
      border-radius: 6px;
      padding: 0.75rem 1rem;
      font-family: 'Courier New', monospace;
      font-size: 0.85rem;
      color: {DARK};
      margin: 0.5rem 0;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background: {CARD_BG};
      border-radius: 8px;
      padding: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 6px;
      font-weight: 500;
  }}
  .stTabs [aria-selected="true"] {{
      background: {PRIMARY} !important;
      color: white !important;
  }}

  /* Download button */
  .stDownloadButton button {{
      background: {PRIMARY};
      color: white;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      padding: 0.6rem 1.5rem;
      transition: opacity 0.2s;
  }}
  .stDownloadButton button:hover {{ opacity: 0.88; }}

  /* Divider */
  hr {{ border: none; border-top: 1px solid #E2E8F0; margin: 1rem 0; }}

  /* Expander */
  .streamlit-expanderHeader {{
      background: #F8FAFC;
      border-radius: 6px;
      font-weight: 500;
  }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# GEOGRAPHIC UNIVERSE
# ============================================================
UNIVERSE = {
    "Americas": {
        "🇺🇸 US Large Cap": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","JPM","JNJ","V",
                              "UNH","XOM","PG","MA","HD","CVX","ABBV","LLY","MRK","PEP"],
        "🇺🇸 US Mid Cap": ["DECK","NVR","TRGP","PODD","EG","MANH","ELS","TXRH","SAIA","FICO"],
        "🇺🇸 US Small Cap": ["BOOT","CELH","LBRT","SPSC","CRVL","QLYS","SWI","PRGS","WTS","BJ"],
        "🇧🇷 Brazil": ["VALE","PBR","ITUB","BBDC4.SA","PETR4.SA","ABEV3.SA","B3SA3.SA","WEGE3.SA"],
        "🇨🇦 Canada": ["RY","TD","ENB","CNR","BNS","BMO","TRP","CP","MFC","SU"],
        "🇲🇽 Mexico": ["AMXL.MX","FEMSA","WALMEX.MX","GFINBURO.MX","BIMBOA.MX","GMEXICOB.MX"],
    },
    "France": {
        "🇫🇷 CAC 40": ["MC.PA","TTE.PA","SAN.PA","AIR.PA","BNP.PA","OR.PA","RI.PA","SU.PA",
                        "DG.PA","AI.PA","ACA.PA","ENGI.PA","SGO.PA","ORA.PA","VIE.PA",
                        "LR.PA","CAP.PA","BN.PA","KER.PA","PUB.PA"],
        "🇫🇷 Mid Cap": ["ALSTOM.PA","BIOCAD.PA","COFA.PA","DBV.PA","ABCA.PA","FP.PA"],
    },
    "Germany": {
        "🇩🇪 DAX": ["SAP","SIE.DE","ALV.DE","MUV2.DE","DTE.DE","BMW.DE","MBG.DE","BAYN.DE",
                    "BAS.DE","VOW3.DE","RWE.DE","DB1.DE","HEI.DE","IFX.DE","HEN3.DE",
                    "DHER.DE","EOAN.DE","PAH3.DE","QIA.DE","ZAL.DE"],
    },
    "United Kingdom": {
        "🇬🇧 FTSE 100": ["AZN.L","SHEL.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","BATS.L",
                          "LLOY.L","BARC.L","VOD.L","DGE.L","NG.L","LSEG.L","IMB.L",
                          "CPG.L","RKT.L","AAL.L","PRU.L","WPP.L"],
    },
    "Spain": {
        "🇪🇸 IBEX 35": ["ITX.MC","SAN.MC","BBVA.MC","IBE.MC","REP.MC","TEF.MC","CLNX.MC",
                         "ACS.MC","ELE.MC","GRF.MC","FER.MC","MAP.MC","MTS.MC","AENA.MC","CABK.MC"],
    },
    "Italy": {
        "🇮🇹 FTSE MIB": ["ENI.MI","ENEL.MI","ISP.MI","UCG.MI","TIT.MI","ATL.MI","STM.MI",
                           "MB.MI","G.MI","LDO.MI","BAMI.MI","PIRC.MI","AMP.MI","CPR.MI","TRN.MI"],
    },
    "Europe (Full)": {
        "🌍 Eurozone": ["ASML","LVMH","SAP","NOVO-B.CO","SIE.DE","TotalEnergies","AZN.L",
                        "ROG.SW","NESN.SW","NOVN.SW","ABB.SW","UBS","CS"],
        "🇸🇪 Sweden": ["VOLV-B.ST","ERIC-B.ST","SHB-A.ST","SWED-A.ST","SKA-B.ST","INVE-B.ST"],
        "🇳🇱 Netherlands": ["ASML","AD.AS","PHIA.AS","HEIA.AS","NN.AS","RDSA.AS"],
        "🇨🇭 Switzerland": ["ROG.SW","NESN.SW","NOVN.SW","ABB.SW","LONN.SW","CFR.SW"],
    },
    "Asia & Middle East": {
        "🇯🇵 Japan": ["7203.T","6758.T","8306.T","9432.T","4502.T","6861.T","8058.T","7267.T","6902.T","4063.T"],
        "🇨🇳 China": ["BABA","JD","PDD","BIDU","NIO","XPEV","LI","TCEHY","NTES","BILI"],
        "🇮🇳 India": ["INFY","WIT","HDB","IBN","SIFY","ICICIB","AXISB","TATASTEEL.NS"],
        "🇰🇷 South Korea": ["005930.KS","000660.KS","035420.KS","051910.KS","005380.KS"],
        "🇸🇬 Singapore": ["D05.SI","O39.SI","U11.SI","Z74.SI","C6L.SI"],
        "🇸🇦 Saudi Arabia / Gulf": ["2222.SR","1120.SR","2010.SR","1150.SR","SABIC"],
        "🇷🇺 Russia (ADR/intl)": ["SBER","GAZP","LUKOY","ROSN","NVTK"],
    },
}

ALL_REGIONS = list(UNIVERSE.keys())

# Flatten universe to get all tickers
def get_tickers_for_regions(regions: list) -> list:
    tickers = []
    for region in regions:
        for subgroup, tkrs in UNIVERSE.get(region, {}).items():
            tickers.extend(tkrs)
    return list(dict.fromkeys(tickers))  # deduplicate preserving order


# ============================================================
# DATA LAYER
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    try:
        raw = yf.download(list(tickers), start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw
        prices = prices.dropna(axis=1, thresh=int(0.7 * len(prices)))
        return prices
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ff3_factors(start: str, end: str) -> pd.DataFrame:
    """
    Fetch Ken French 3-factor data via pandas_datareader (Fama-French library).
    Returns daily Mkt-RF, SMB, HML, RF columns.
    """
    try:
        import pandas_datareader.data as web
        ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                            start=start, end=end)[0]
        ff = ff / 100.0
        ff.index = pd.to_datetime(ff.index)
        return ff
    except Exception:
        return pd.DataFrame()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def annualize_returns(rets: pd.DataFrame) -> pd.Series:
    return (1 + rets).prod() ** (TRADING_DAYS / len(rets)) - 1

def annualize_vol(rets: pd.DataFrame) -> pd.Series:
    return rets.std() * np.sqrt(TRADING_DAYS)

def cov_matrix(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.cov() * TRADING_DAYS


# ============================================================
# OPTIMIZATION ENGINE
# ============================================================

def portfolio_stats(weights, mu, sigma):
    """Return (ret, vol, sharpe) for given weights."""
    w = np.array(weights)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ sigma @ w))
    sr = ret / vol if vol > 1e-10 else 0.0
    return ret, vol, sr


def gmv_portfolio(sigma: np.ndarray) -> np.ndarray:
    """Global Minimum Variance portfolio — closed-form via Lagrange."""
    n = sigma.shape[0]
    try:
        inv_sigma = np.linalg.inv(sigma)
        ones = np.ones(n)
        w = inv_sigma @ ones
        return w / w.sum()
    except np.linalg.LinAlgError:
        return np.ones(n) / n


def tangency_portfolio(mu: np.ndarray, sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Max Sharpe (Tangency) portfolio — Elton-Gruber / analytic."""
    n = sigma.shape[0]
    excess = mu - rf
    try:
        inv_sigma = np.linalg.inv(sigma)
        w = inv_sigma @ excess
        if w.sum() <= 0:
            w = np.abs(w)
        return w / w.sum()
    except np.linalg.LinAlgError:
        return np.ones(n) / n


def elton_gruber_tangency(mu: np.ndarray, sigma: np.ndarray, rf: float) -> dict:
    """
    Elton & Gruber (1977) simplified tangency portfolio.
    Returns weights + intermediate calculation steps.
    """
    n = sigma.shape[0]
    excess = mu - rf
    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        inv_sigma = np.linalg.pinv(sigma)

    z = inv_sigma @ excess
    z_sum = z.sum()
    if z_sum <= 0:
        z = np.abs(z)
        z_sum = z.sum()
    w = z / z_sum if z_sum > 0 else np.ones(n) / n

    steps = {
        "Excess returns (μ - Rf)": excess,
        "Σ⁻¹ · (μ - Rf)  [z-vector]": z,
        "Unnormalized sum": z_sum,
        "Final weights": w,
    }
    return w, steps


def merton_two_fund(mu: np.ndarray, sigma: np.ndarray, rf: float,
                    target_ret: float) -> dict:
    """
    Merton two-fund separation:
    Any efficient portfolio = combination of GMV + Tangency.
    Returns weights for given target return + derivation steps.
    """
    w_gmv = gmv_portfolio(sigma)
    w_tan = tangency_portfolio(mu, sigma, rf)

    ret_gmv, _, _ = portfolio_stats(w_gmv, mu, sigma)
    ret_tan, _, _ = portfolio_stats(w_tan, mu, sigma)

    # Interpolate: w = alpha * tan + (1 - alpha) * gmv
    denom = ret_tan - ret_gmv
    if abs(denom) < 1e-8:
        alpha = 0.5
    else:
        alpha = (target_ret - ret_gmv) / denom
    alpha = float(np.clip(alpha, -0.5, 2.0))  # allow mild leverage

    w = alpha * w_tan + (1 - alpha) * w_gmv
    w = np.clip(w, 0, None)
    w = w / w.sum()

    steps = {
        "GMV weights": w_gmv,
        "Tangency weights": w_tan,
        "Return(GMV)": ret_gmv,
        "Return(Tangency)": ret_tan,
        "Alpha (mixing coefficient)": alpha,
        "Combined weights": w,
    }
    return w, steps


def efficient_frontier_points(mu: np.ndarray, sigma: np.ndarray,
                               rf: float = 0.0, n_points: int = 60) -> pd.DataFrame:
    """
    Compute efficient frontier by solving min-variance for a range of target returns.
    Uses scipy.optimize with equality constraints (no short selling).
    """
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints_base = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    ret_min = float(mu.min())
    ret_max = float(mu.max())
    targets = np.linspace(ret_min * 1.01, ret_max * 0.99, n_points)

    frontier = []
    for target in targets:
        cons = constraints_base + [
            {"type": "eq", "fun": lambda w, t=target: (w @ mu) - t}
        ]
        w0 = np.ones(n) / n
        res = minimize(
            lambda w: w @ sigma @ w,
            w0, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"ftol": 1e-9, "maxiter": 500}
        )
        if res.success:
            w = res.x
            ret, vol, sr = portfolio_stats(w, mu, sigma)
            frontier.append({"Return": ret, "Volatility": vol, "Sharpe": sr,
                              "Weights": w})

    return pd.DataFrame(frontier)


def black_litterman(mu_eq: np.ndarray, sigma: np.ndarray,
                    P: np.ndarray, Q: np.ndarray,
                    omega_diag: np.ndarray, tau: float = 0.025) -> dict:
    """
    Black-Litterman model.
    mu_eq  : equilibrium (CAPM-implied) returns  [n]
    sigma  : annualised covariance matrix         [n x n]
    P      : picking matrix (K x n)
    Q      : view returns vector (K,)
    omega_diag : variance of views (K,) → diagonal of Ω
    tau    : scalar uncertainty on prior (typically 0.01-0.05)

    Returns BL posterior mean, posterior cov, and intermediate steps.
    """
    n = len(mu_eq)
    K = len(Q)

    tau_sigma = tau * sigma
    Omega = np.diag(omega_diag)

    # BL master formula
    A = np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P
    b = np.linalg.inv(tau_sigma) @ mu_eq + P.T @ np.linalg.inv(Omega) @ Q

    mu_bl = np.linalg.solve(A, b)
    sigma_bl = np.linalg.inv(A) + sigma  # posterior uncertainty + inherent variance

    # Optimal weights from BL posterior (max Sharpe)
    w_bl = tangency_portfolio(mu_bl, sigma_bl)

    steps = {
        "Equilibrium returns (Π)": mu_eq,
        "τ·Σ (prior covariance scaling)": tau_sigma,
        "Ω (view uncertainty matrix)": Omega,
        "P (picking matrix)": P,
        "Q (view returns)": Q,
        "BL Posterior mean (μ_BL)": mu_bl,
        "BL Posterior covariance": sigma_bl,
        "Optimal BL weights": w_bl,
    }
    return w_bl, mu_bl, sigma_bl, steps


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def portfolio_returns(rets: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(rets.columns).fillna(0.0)
    w = w / w.sum()
    return (rets * w).sum(axis=1)

def max_drawdown(cum_series: pd.Series) -> float:
    roll_max = cum_series.cummax()
    drawdown = cum_series / roll_max - 1
    return float(drawdown.min())

def sharpe(port_ret: pd.Series, rf_annual: float = 0.0) -> float:
    rf_d = rf_annual / TRADING_DAYS
    excess = port_ret - rf_d
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS)) if excess.std() > 1e-10 else 0.0

def sortino(port_ret: pd.Series, rf_annual: float = 0.0) -> float:
    rf_d = rf_annual / TRADING_DAYS
    excess = port_ret - rf_d
    downside = excess[excess < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    return float(excess.mean() * TRADING_DAYS / downside) if downside > 1e-10 else 0.0

def treynor(port_ret: pd.Series, bench_ret: pd.Series, rf_annual: float = 0.0) -> float:
    rf_d = rf_annual / TRADING_DAYS
    beta = _beta(port_ret, bench_ret)
    ann_excess = (port_ret - rf_d).mean() * TRADING_DAYS
    return float(ann_excess / beta) if abs(beta) > 1e-6 else 0.0

def information_ratio(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    active = port_ret - bench_ret
    te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return float(active.mean() * TRADING_DAYS / te) if te > 1e-10 else 0.0

def _beta(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 10:
        return 1.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / (cov[1, 1] + 1e-12))

def jensen_alpha(port_ret: pd.Series, bench_ret: pd.Series, rf_annual: float) -> float:
    rf_d = rf_annual / TRADING_DAYS
    b = _beta(port_ret, bench_ret)
    ann_port = (1 + port_ret).prod() ** (TRADING_DAYS / len(port_ret)) - 1
    ann_bench = (1 + bench_ret).prod() ** (TRADING_DAYS / len(bench_ret)) - 1
    return float(ann_port - (rf_annual + b * (ann_bench - rf_annual)))

def calmar(port_ret: pd.Series) -> float:
    cum = (1 + port_ret).cumprod()
    ann_ret = (cum.iloc[-1]) ** (TRADING_DAYS / len(cum)) - 1
    mdd = abs(max_drawdown(cum))
    return float(ann_ret / mdd) if mdd > 1e-10 else 0.0

def tracking_error(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    active = (port_ret - bench_ret).dropna()
    return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))

def compute_all_metrics(port_ret: pd.Series, bench_ret: pd.Series,
                        rf_annual: float) -> pd.DataFrame:
    cum = (1 + port_ret).cumprod()
    ann_ret = float((cum.iloc[-1]) ** (TRADING_DAYS / len(cum)) - 1)
    ann_vol = float(port_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    b = _beta(port_ret, bench_ret)
    te = tracking_error(port_ret, bench_ret)

    metrics = {
        "Annualised Return": f"{ann_ret:.2%}",
        "Annualised Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe(port_ret, rf_annual):.3f}",
        "Sortino Ratio": f"{sortino(port_ret, rf_annual):.3f}",
        "Treynor Ratio": f"{treynor(port_ret, bench_ret, rf_annual):.3f}",
        "Information Ratio": f"{information_ratio(port_ret, bench_ret):.3f}",
        "Jensen's Alpha (ann.)": f"{jensen_alpha(port_ret, bench_ret, rf_annual):.2%}",
        "Max Drawdown": f"{abs(max_drawdown(cum)):.2%}",
        "Calmar Ratio": f"{calmar(port_ret):.3f}",
        "Beta (vs Benchmark)": f"{b:.3f}",
        "Tracking Error (ann.)": f"{te:.2%}",
    }
    return pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])


# ============================================================
# STRATEGY BUILDERS
# ============================================================

def equal_weight(tickers: list) -> pd.Series:
    n = len(tickers)
    return pd.Series(1.0 / n, index=tickers)

def market_cap_weight(tickers: list) -> pd.Series:
    """Proxy market-cap weight using last available market cap from yfinance."""
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            caps[t] = getattr(info, "market_cap", None) or 1e9
        except Exception:
            caps[t] = 1e9
    s = pd.Series(caps)
    return s / s.sum()

def momentum_weight(rets: pd.DataFrame, lookback: int = 126) -> pd.Series:
    """12-1 month momentum: rank by past 6-month return, overweight top quartile."""
    if len(rets) < lookback:
        return equal_weight(rets.columns.tolist())
    recent = rets.iloc[-lookback:]
    mom = (1 + recent).prod() - 1
    ranked = mom.rank(ascending=True)
    # Top 25% get double weight, bottom 25% get half
    w = ranked.copy().astype(float)
    q75 = ranked.quantile(0.75)
    q25 = ranked.quantile(0.25)
    w[ranked >= q75] *= 2.0
    w[ranked <= q25] *= 0.5
    return w / w.sum()

def low_volatility_weight(rets: pd.DataFrame) -> pd.Series:
    """Smart-beta low-vol: weight inversely proportional to volatility."""
    vol = rets.std()
    inv_vol = 1.0 / (vol + 1e-10)
    return inv_vol / inv_vol.sum()

def value_weight(tickers: list) -> pd.Series:
    """
    Smart-beta value: proxy via book-to-market (P/B inverse).
    Fallback to equal if unavailable.
    """
    pb = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            pb_ratio = info.get("priceToBook", None)
            if pb_ratio and pb_ratio > 0:
                pb[t] = pb_ratio
        except Exception:
            pass
    if len(pb) < 2:
        return equal_weight(tickers)
    s = pd.Series(pb)
    inv_pb = 1.0 / s  # higher book-to-price = more "value"
    return inv_pb / inv_pb.sum()


# ============================================================
# FAMA-FRENCH REGRESSION
# ============================================================

def fama_french_regression(port_ret: pd.Series, ff_data: pd.DataFrame) -> dict:
    """
    Regress portfolio excess returns on FF3 factors: Mkt-RF, SMB, HML.
    Returns alpha, betas, R², t-stats.
    """
    from scipy.stats import t as t_dist

    merged = pd.concat([port_ret, ff_data], axis=1).dropna()
    if len(merged) < 30:
        return {}

    y = merged.iloc[:, 0].values - merged["RF"].values  # excess return
    X_raw = merged[["Mkt-RF", "SMB", "HML"]].values
    X = np.column_stack([np.ones(len(y)), X_raw])

    # OLS
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ y
    y_hat = X @ betas
    residuals = y - y_hat
    n, k = X.shape
    sigma2 = (residuals @ residuals) / (n - k)
    var_b = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_b))
    t_stats = betas / (se + 1e-12)
    p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n - k))

    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (residuals ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    alpha_ann = betas[0] * TRADING_DAYS

    return {
        "Alpha (daily)": betas[0],
        "Alpha (annualised)": alpha_ann,
        "β_MktRF": betas[1],
        "β_SMB": betas[2],
        "β_HML": betas[3],
        "t_alpha": t_stats[0],
        "t_MktRF": t_stats[1],
        "t_SMB": t_stats[2],
        "t_HML": t_stats[3],
        "p_alpha": p_vals[0],
        "R²": r2,
        "Adj. R²": 1 - (1 - r2) * (n - 1) / (n - k),
        "N obs": n,
    }


# ============================================================
# PLOTLY CHART HELPERS
# ============================================================

PLOTLY_TEMPLATE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", color=TEXT_DARK),
    colorway=[PRIMARY, ACCENT, SUCCESS, WARNING, "#9B59B6", "#1ABC9C", "#E67E22"],
    xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0",
                borderwidth=1, orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=20),
)

def make_equity_curve(cum_port: pd.Series, cum_bench: pd.Series,
                      port_label: str = "Portfolio",
                      bench_label: str = "Benchmark") -> go.Figure:
    base = 100.0
    p = cum_port / cum_port.iloc[0] * base
    b = cum_bench / cum_bench.iloc[0] * base

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p.values, mode="lines",
                             name=port_label, line=dict(color=PRIMARY, width=2.5)))
    fig.add_trace(go.Scatter(x=b.index, y=b.values, mode="lines",
                             name=bench_label, line=dict(color=ACCENT, width=2, dash="dot")))
    fig.update_layout(
        title="📈 Portfolio vs Benchmark — Equity Curve (base 100)",
        height=420, xaxis_title="Date", yaxis_title="Index (base 100)",
        **PLOTLY_TEMPLATE
    )
    return fig

def make_frontier_chart(frontier: pd.DataFrame, gmv_point: tuple,
                        tan_point: tuple, selected_point: tuple = None,
                        cml_rf: float = None) -> go.Figure:
    fig = go.Figure()

    # Frontier curve
    fig.add_trace(go.Scatter(
        x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
        mode="lines", name="Efficient Frontier",
        line=dict(color=PRIMARY, width=3)
    ))

    # GMV point
    if gmv_point:
        fig.add_trace(go.Scatter(
            x=[gmv_point[0] * 100], y=[gmv_point[1] * 100],
            mode="markers+text", name="Global Min Variance",
            text=["GMV"], textposition="top right",
            marker=dict(size=12, color=ACCENT, symbol="diamond",
                        line=dict(color="white", width=2))
        ))

    # Tangency
    if tan_point:
        fig.add_trace(go.Scatter(
            x=[tan_point[0] * 100], y=[tan_point[1] * 100],
            mode="markers+text", name="Tangency (Max Sharpe)",
            text=["Tangency"], textposition="top right",
            marker=dict(size=14, color=SUCCESS, symbol="star",
                        line=dict(color="white", width=2))
        ))

    # CML
    if cml_rf is not None and tan_point:
        vol_range = np.linspace(0, tan_point[0] * 1.5, 50)
        cml_rets = cml_rf + (tan_point[1] - cml_rf) / tan_point[0] * vol_range
        fig.add_trace(go.Scatter(
            x=vol_range * 100, y=cml_rets * 100,
            mode="lines", name="Capital Market Line (CML)",
            line=dict(color=WARNING, width=1.8, dash="dash")
        ))

    # Selected portfolio
    if selected_point:
        fig.add_trace(go.Scatter(
            x=[selected_point[0] * 100], y=[selected_point[1] * 100],
            mode="markers+text", name="Selected Portfolio",
            text=["▶ Your Portfolio"], textposition="top left",
            marker=dict(size=14, color=PRIMARY, symbol="circle",
                        line=dict(color="white", width=2))
        ))

    fig.update_layout(
        title="📉 Mean-Variance Efficient Frontier",
        xaxis_title="Volatility (σ) %", yaxis_title="Expected Return (μ) %",
        height=480, **PLOTLY_TEMPLATE
    )
    return fig

def make_weights_chart(weights: pd.Series, title: str = "Portfolio Weights") -> go.Figure:
    w = weights[weights > 0.001].sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=w.values * 100, y=w.index,
        orientation="h",
        marker=dict(
            color=w.values,
            colorscale=[[0, "#FEE2E2"], [0.5, PRIMARY], [1, DARK]],
            line=dict(color="white", width=1)
        ),
        text=[f"{v:.1f}%" for v in w.values * 100],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, height=max(350, len(w) * 30 + 100),
        xaxis_title="Weight (%)", **PLOTLY_TEMPLATE
    )
    return fig

def make_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    labels = corr.columns.tolist()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        title="🔗 Correlation Matrix",
        height=max(400, len(labels) * 35 + 120),
        **PLOTLY_TEMPLATE
    )
    return fig

def make_drawdown_chart(cum_series: pd.Series) -> go.Figure:
    roll_max = cum_series.cummax()
    dd = (cum_series / roll_max - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", mode="lines",
        name="Drawdown",
        line=dict(color=PRIMARY, width=1.5),
        fillcolor=f"rgba(200,16,46,0.15)"
    ))
    fig.update_layout(
        title="📉 Drawdown History",
        height=280, xaxis_title="Date", yaxis_title="Drawdown (%)",
        **PLOTLY_TEMPLATE
    )
    return fig

def make_rolling_sharpe(port_ret: pd.Series, rf_annual: float,
                        window: int = 126) -> go.Figure:
    rf_d = rf_annual / TRADING_DAYS
    excess = port_ret - rf_d
    roll_sr = excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(TRADING_DAYS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_sr.index, y=roll_sr.values,
        mode="lines", name=f"Rolling Sharpe ({window}d)",
        line=dict(color=ACCENT, width=2)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
    fig.add_hline(y=1, line_dash="dot", line_color=SUCCESS,
                  annotation_text="Sharpe = 1")
    fig.update_layout(
        title=f"📊 Rolling Sharpe Ratio ({window} trading days)",
        height=280, xaxis_title="Date", yaxis_title="Sharpe Ratio",
        **PLOTLY_TEMPLATE
    )
    return fig


# ============================================================
# PDF REPORT BUILDER
# ============================================================

def _brand_color():
    return colors.HexColor(PRIMARY)

def _build_pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontSize=26, fontName="Helvetica-Bold",
        textColor=colors.HexColor(DARK),
        alignment=TA_CENTER, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontSize=12, fontName="Helvetica",
        textColor=colors.HexColor("#64748B"),
        alignment=TA_CENTER, spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=14, fontName="Helvetica-Bold",
        textColor=colors.HexColor(DARK),
        spaceBefore=16, spaceAfter=8,
        borderPad=6,
    ))
    styles.add(ParagraphStyle(
        name="SubHeader",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor(PRIMARY),
        spaceBefore=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#374151"),
        spaceAfter=3, leading=14,
    ))
    styles.add(ParagraphStyle(
        name="FormulaText",
        fontSize=9, fontName="Courier",
        textColor=colors.HexColor(DARK),
        backColor=colors.HexColor("#F1F5F9"),
        borderPad=6, spaceAfter=6,
    ))
    return styles

def _rl_table_style(has_header: bool = True) -> TableStyle:
    style = [
        ("BACKGROUND", (0, 0), (-1, 0 if has_header else -1),
         colors.HexColor(DARK) if has_header else colors.white),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if has_header else colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]
    return TableStyle(style)

def _fig_to_image_bytes(fig) -> BytesIO:
    """Convert a plotly figure to PNG bytes for PDF embedding."""
    try:
        img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
        return BytesIO(img_bytes)
    except Exception:
        return None

def _mpl_fig_to_bytes(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf

def build_pdf_report(
    report_params: dict,
    weights: pd.Series,
    metrics_df: pd.DataFrame,
    method_name: str,
    equity_fig=None,
    frontier_fig=None,
    weights_fig=None,
    corr_fig=None,
    dd_fig=None,
    rolling_sr_fig=None,
    ff_results: dict = None,
    show_calcs: bool = False,
    calc_steps: dict = None,
) -> bytes:
    """Generate a professional PDF report."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = _build_pdf_styles()
    story = []

    # ── Cover page ─────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("PORTFOLIO MANAGEMENT", styles["ReportTitle"]))
    story.append(Paragraph("Advanced Portfolio Analysis Report", styles["ReportSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=_brand_color()))
    story.append(Spacer(1, 0.4 * cm))

    info_data = [
        ["Field", "Value"],
        ["Optimization Method", method_name],
        ["Period", f"{report_params.get('start', '')} → {report_params.get('end', '')}"],
        ["Benchmark", report_params.get("benchmark", "SPY")],
        ["Risk-Free Rate", f"{report_params.get('rf', 0):.2%}"],
        ["Number of Assets", str(report_params.get("n_assets", ""))],
        ["Date Generated", str(date.today())],
        ["ESSCA — Prof. Olga Tatarnikova", "Portfolio Management Course"],
    ]
    t = RLTable(info_data, colWidths=[7 * cm, 10 * cm])
    t.setStyle(_rl_table_style())
    story.append(t)
    story.append(PageBreak())

    # ── Step 1: Planning ────────────────────────────────────
    story.append(Paragraph("STEP 1 — PLANNING", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_brand_color()))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Investment Objectives & Capital Market Expectations",
                            styles["SubHeader"]))
    story.append(Paragraph(
        "This portfolio is constructed following a rigorous asset allocation framework aligned "
        "with Modern Portfolio Theory (Markowitz, 1952). The investment universe spans global "
        "equities selected across multiple geographic regions. The optimization methodology "
        f"applied is <b>{method_name}</b>, targeting an efficient risk-return trade-off relative "
        "to the benchmark.",
        styles["BodyText2"]
    ))
    story.append(Spacer(1, 0.3 * cm))

    params_data = [["Parameter", "Value"]] + [
        [k, str(v)] for k, v in report_params.items()
    ]
    t2 = RLTable(params_data, colWidths=[8 * cm, 9 * cm])
    t2.setStyle(_rl_table_style())
    story.append(t2)
    story.append(PageBreak())

    # ── Step 2: Execution ───────────────────────────────────
    story.append(Paragraph("STEP 2 — EXECUTION: PORTFOLIO CONSTRUCTION",
                            styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_brand_color()))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(f"Optimization Method: {method_name}", styles["SubHeader"]))

    method_desc = {
        "Markowitz MVO (Max Sharpe)": "Mean-Variance Optimization maximizes the Sharpe ratio by finding the tangency portfolio on the efficient frontier. Formula: max w'μ/√(w'Σw) subject to Σwᵢ=1, wᵢ≥0.",
        "Markowitz MVO (Min Variance)": "Global Minimum Variance portfolio minimizes total portfolio variance. Formula: min w'Σw subject to Σwᵢ=1, wᵢ≥0.",
        "Elton-Gruber (Tangency)": "Elton & Gruber (1977) simplified the Markowitz framework by computing z = Σ⁻¹(μ - Rf), then normalizing: w = z / Σzᵢ. This yields the tangency portfolio directly without quadratic programming.",
        "Merton Two-Fund": "Merton's separation theorem: any efficient portfolio is a linear combination of the GMV and Tangency portfolios. w = α·w_tan + (1-α)·w_gmv, where α is calibrated to the target return.",
        "Black-Litterman": "Black-Litterman blends CAPM equilibrium returns (Π) with investor views (Q, P) via Bayesian updating: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q].",
        "Equal Weight": "Naive diversification: wᵢ = 1/N for all assets. Benchmark strategy.",
        "Momentum": "Smart-beta: overweights past winners (top quartile by 6-month return), underweights past losers.",
        "Low Volatility": "Smart-beta: weights inversely proportional to historical volatility — wᵢ ∝ 1/σᵢ.",
    }
    desc = method_desc.get(method_name, "See application for methodology details.")
    story.append(Paragraph(desc, styles["BodyText2"]))
    story.append(Spacer(1, 0.3 * cm))

    # Weights table
    story.append(Paragraph("Final Portfolio Weights", styles["SubHeader"]))
    w_show = weights[weights > 0.001].sort_values(ascending=False)
    w_data = [["Ticker", "Weight (%)", "Rank"]]
    for i, (t, v) in enumerate(w_show.items(), 1):
        w_data.append([str(t), f"{v:.2%}", str(i)])
    wt = RLTable(w_data, colWidths=[5 * cm, 5 * cm, 5 * cm])
    wt.setStyle(_rl_table_style())
    story.append(wt)

    # Embed charts
    def _add_fig(fig, caption: str, width_cm=16):
        if fig is None:
            return
        try:
            img_b = _fig_to_image_bytes(fig)
            if img_b:
                story.append(Spacer(1, 0.3 * cm))
                story.append(RLImage(img_b, width=width_cm * cm,
                                     height=width_cm * 0.5 * cm))
                story.append(Paragraph(f"<i>{caption}</i>",
                                        ParagraphStyle("cap", fontSize=8,
                                                        textColor=colors.grey,
                                                        alignment=TA_CENTER)))
                story.append(Spacer(1, 0.2 * cm))
        except Exception:
            pass

    story.append(Spacer(1, 0.4 * cm))
    _add_fig(weights_fig, "Figure 1: Portfolio Weights Allocation")
    story.append(PageBreak())

    _add_fig(frontier_fig, "Figure 2: Mean-Variance Efficient Frontier with GMV, Tangency & CML")
    _add_fig(corr_fig, "Figure 3: Asset Correlation Matrix")
    story.append(PageBreak())

    # Show intermediate calculations if requested
    if show_calcs and calc_steps:
        story.append(Paragraph("Intermediate Calculations (Professor Review)",
                                styles["SubHeader"]))
        story.append(Paragraph(
            "The following shows the step-by-step calculations used in the optimization process.",
            styles["BodyText2"]
        ))
        for step_name, step_val in calc_steps.items():
            story.append(Paragraph(f"• {step_name}:", styles["BodyText2"]))
            if isinstance(step_val, np.ndarray):
                if step_val.ndim == 1:
                    vals = ", ".join([f"{x:.4f}" for x in step_val[:10]])
                    story.append(Paragraph(f"  [{vals}{'...' if len(step_val)>10 else ''}]",
                                           styles["FormulaText"]))
                else:
                    story.append(Paragraph(
                        f"  Matrix {step_val.shape} — see application for full display",
                        styles["FormulaText"]
                    ))
            elif isinstance(step_val, float):
                story.append(Paragraph(f"  {step_val:.6f}", styles["FormulaText"]))
            else:
                story.append(Paragraph(f"  {str(step_val)[:200]}", styles["FormulaText"]))
        story.append(PageBreak())

    # ── Step 3: Feedback ─────────────────────────────────────
    story.append(Paragraph("STEP 3 — FEEDBACK: PERFORMANCE EVALUATION",
                            styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_brand_color()))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Key Performance & Risk Metrics", styles["SubHeader"]))
    story.append(Paragraph(
        "Performance is evaluated using the full set of course metrics (S9 — Performance Evaluation). "
        "All ratios are annualised unless stated otherwise.",
        styles["BodyText2"]
    ))
    story.append(Spacer(1, 0.2 * cm))

    if not metrics_df.empty:
        m_data = [["Metric", "Value"]] + metrics_df.values.tolist()
        mt = RLTable(m_data, colWidths=[10 * cm, 7 * cm])
        mt.setStyle(_rl_table_style())
        story.append(mt)

    story.append(Spacer(1, 0.4 * cm))
    _add_fig(equity_fig, "Figure 4: Portfolio vs Benchmark Equity Curve (base 100)")
    _add_fig(dd_fig, "Figure 5: Historical Drawdown")
    _add_fig(rolling_sr_fig, "Figure 6: Rolling Sharpe Ratio (126-day window)")

    # Fama-French results
    if ff_results:
        story.append(PageBreak())
        story.append(Paragraph("Fama-French 3-Factor Regression", styles["SubHeader"]))
        story.append(Paragraph(
            "The portfolio's returns are regressed on the three Fama-French factors: "
            "MktRF (market excess return), SMB (Small minus Big), HML (High minus Low B/M). "
            "This decomposes alpha from systematic factor exposures.",
            styles["BodyText2"]
        ))
        ff_data = [["Parameter", "Value"]] + [
            [k, f"{v:.4f}" if isinstance(v, float) else str(v)]
            for k, v in ff_results.items()
        ]
        fft = RLTable(ff_data, colWidths=[8 * cm, 9 * cm])
        fft.setStyle(_rl_table_style())
        story.append(fft)

    # Footer
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")))
    story.append(Paragraph(
        f"<i>Generated by Portfolio Management Pro · ESSCA · Prof. Olga Tatarnikova · {date.today()}</i>",
        ParagraphStyle("footer", fontSize=7, textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()


# ============================================================
# KPI CARD RENDERER
# ============================================================
def kpi_card(label, value, color_class=""):
    st.markdown(
        f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {color_class}">{value}</div>
           </div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f"""<div style="text-align:center;padding:1rem 0 0.5rem;">
              <span style="font-size:2rem;">📊</span>
              <div style="font-size:1.1rem;font-weight:700;color:white;margin-top:0.3rem;">
                Portfolio Management Pro</div>
              <div style="font-size:0.75rem;color:#94A3B8;">ESSCA · Prof. Tatarnikova</div>
           </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Geographic Universe ─────────────────────────────────
    st.markdown("### 🌍 Asset Universe")
    region_choices = st.multiselect(
        "Select Regions",
        options=ALL_REGIONS,
        default=["Americas", "France", "Germany"],
        help="Select one or more regions. Selecting all gives global equity coverage."
    )

    # Show available tickers count
    candidate_tickers = get_tickers_for_regions(region_choices)
    st.caption(f"📦 {len(candidate_tickers)} tickers available in selected regions")

    # Let user further refine
    max_assets = st.slider("Max assets to include", 5, 30, 15,
                           help="App selects the most liquid from available tickers")

    st.markdown("---")
    # ── Period ──────────────────────────────────────────────
    st.markdown("### 📅 Analysis Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=date(2019, 1, 1),
                                   max_value=date.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("To", value=date.today(),
                                 min_value=start_date + timedelta(days=365))

    st.markdown("---")
    # ── Benchmark ───────────────────────────────────────────
    st.markdown("### 🎯 Benchmark")
    BENCHMARKS = {
        "S&P 500 (SPY)": "SPY",
        "MSCI World (URTH)": "URTH",
        "EURO STOXX 50 (FEZ)": "FEZ",
        "CAC 40 (EWQ)": "EWQ",
        "DAX (EWG)": "EWG",
        "FTSE 100 (EWU)": "EWU",
        "MSCI EM (EEM)": "EEM",
    }
    bench_label = st.selectbox("Benchmark", list(BENCHMARKS.keys()))
    bench_ticker = BENCHMARKS[bench_label]

    st.markdown("---")
    # ── Market Assumptions ──────────────────────────────────
    st.markdown("### 📐 Market Assumptions")
    rf_rate = st.slider("Risk-Free Rate (annual %)", 0.0, 8.0, 4.0, 0.25) / 100.0
    erp = st.slider("Equity Risk Premium (annual %)", 2.0, 10.0, 5.0, 0.25) / 100.0

    st.markdown("---")
    # ── Optimization Method ─────────────────────────────────
    st.markdown("### ⚙️ Optimization Method")
    OPT_METHODS = [
        "Markowitz MVO (Max Sharpe)",
        "Markowitz MVO (Min Variance)",
        "Elton-Gruber (Tangency)",
        "Merton Two-Fund",
        "Black-Litterman",
        "Equal Weight",
        "Momentum",
        "Low Volatility",
    ]
    method = st.selectbox("Method", OPT_METHODS)

    # Black-Litterman views
    bl_views = []
    if method == "Black-Litterman":
        st.markdown("**Investor Views (Black-Litterman)**")
        n_views = st.number_input("Number of views", 0, 5, 1, key="n_views")
        bl_views = []
        for i in range(int(n_views)):
            with st.expander(f"View {i+1}", expanded=True):
                view_type = st.selectbox(f"Type", ["Absolute", "Relative"],
                                         key=f"vt_{i}")
                ticker1 = st.text_input(f"Asset (or Asset A)", key=f"t1_{i}", value="AAPL")
                if view_type == "Relative":
                    ticker2 = st.text_input("Asset B", key=f"t2_{i}", value="MSFT")
                else:
                    ticker2 = None
                view_ret = st.number_input(f"Expected return (%)", -20.0, 50.0, 5.0,
                                           key=f"vr_{i}") / 100.0
                view_conf = st.slider(f"Confidence (%)", 10, 90, 50,
                                      key=f"vc_{i}") / 100.0
                bl_views.append((view_type, ticker1, ticker2, view_ret, view_conf))

    # Merton target return
    merton_target = 0.10
    if method == "Merton Two-Fund":
        merton_target = st.slider("Target Annual Return (%)", 1.0, 30.0, 10.0, 0.5) / 100.0

    st.markdown("---")
    # ── Report options ──────────────────────────────────────
    st.markdown("### 📄 Report Options")
    show_calcs = st.toggle("Show intermediate calculations", value=True,
                           help="Enable for professor review of calculation steps")

    st.markdown("---")
    run = st.button("🚀 Run Analysis", use_container_width=True,
                    type="primary")


# ============================================================
# MAIN CONTENT
# ============================================================

st.markdown(
    """<div class="pm-header">
         <h1>📊 Portfolio Management Pro</h1>
         <p>Advanced Portfolio Construction & Performance Analysis · ESSCA · Prof. Olga Tatarnikova</p>
       </div>""",
    unsafe_allow_html=True,
)

# ── Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Planning",
    "⚙️ Optimization",
    "📈 Performance",
    "🔬 Factor Models",
    "📄 Report",
])

# ── State ───────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None

# ============================================================
# RUN ANALYSIS
# ============================================================
if run:
    if not region_choices:
        st.error("Please select at least one region.")
        st.stop()

    with st.spinner("⏳ Fetching market data…"):
        all_tickers = get_tickers_for_regions(region_choices)
        # Limit to manageable set
        tickers_to_try = all_tickers[:min(len(all_tickers), max_assets * 3)]
        prices = fetch_prices(
            tuple(tickers_to_try),
            str(start_date), str(end_date)
        )
        if prices.empty:
            st.error("No data returned. Try different tickers or date range.")
            st.stop()

        # Keep most liquid (fewest NaN)
        prices = prices.dropna(axis=1, thresh=int(0.85 * len(prices)))
        prices = prices.iloc[:, :max_assets]  # cap at max_assets
        tickers = prices.columns.tolist()

        # Benchmark
        bench_prices = fetch_prices(
            (bench_ticker,), str(start_date), str(end_date)
        )

    with st.spinner("⏳ Computing returns & statistics…"):
        rets = compute_returns(prices)
        mu_ann = annualize_returns(rets).values
        sigma_ann = cov_matrix(rets).values
        sigma_daily = rets.cov().values
        n = len(tickers)

        if bench_prices.empty:
            bench_ret = rets.iloc[:, 0]
        else:
            bench_ret = compute_returns(bench_prices).iloc[:, 0]
            bench_ret = bench_ret.reindex(rets.index).fillna(0)

    with st.spinner(f"⏳ Running {method} optimization…"):
        calc_steps = {}
        weights = None

        if method == "Markowitz MVO (Max Sharpe)":
            w_tan = tangency_portfolio(mu_ann, sigma_ann, rf_rate)
            weights = pd.Series(w_tan, index=tickers)
            calc_steps = {
                "Expected Annual Returns (μ)": mu_ann,
                "Risk-Free Rate (Rf)": rf_rate,
                "Excess Returns (μ - Rf)": mu_ann - rf_rate,
                "Tangency weights (unnormalized)": np.linalg.pinv(sigma_ann) @ (mu_ann - rf_rate),
                "Final Tangency weights": w_tan,
            }

        elif method == "Markowitz MVO (Min Variance)":
            w_gmv = gmv_portfolio(sigma_ann)
            weights = pd.Series(w_gmv, index=tickers)
            calc_steps = {
                "Covariance Matrix Σ": sigma_ann,
                "Σ⁻¹ · 1 (unnormalized)": np.linalg.pinv(sigma_ann) @ np.ones(n),
                "GMV weights": w_gmv,
            }

        elif method == "Elton-Gruber (Tangency)":
            w_eg, eg_steps = elton_gruber_tangency(mu_ann, sigma_ann, rf_rate)
            weights = pd.Series(w_eg, index=tickers)
            calc_steps = {k: v for k, v in eg_steps.items()}

        elif method == "Merton Two-Fund":
            w_m, m_steps = merton_two_fund(mu_ann, sigma_ann, rf_rate, merton_target)
            weights = pd.Series(w_m, index=tickers)
            calc_steps = {k: v for k, v in m_steps.items()}

        elif method == "Black-Litterman":
            # Build equilibrium returns from CAPM
            mkt_var = float(bench_ret.var() * TRADING_DAYS)
            delta = (erp) / mkt_var  # risk aversion
            mu_eq = delta * sigma_ann @ np.ones(n)
            mu_eq = mu_eq / mu_eq.sum() * (rf_rate + erp)

            # Parse views
            P_rows, Q_vals, omega_vals = [], [], []
            for vtype, t1, t2, vret, vconf in bl_views:
                if t1 not in tickers:
                    continue
                row = np.zeros(n)
                idx1 = tickers.index(t1)
                row[idx1] = 1.0
                if vtype == "Relative" and t2 and t2 in tickers:
                    idx2 = tickers.index(t2)
                    row[idx2] = -1.0
                P_rows.append(row)
                Q_vals.append(vret)
                variance = ((1 - vconf) / vconf) * float(
                    np.sqrt(row @ sigma_ann @ row))
                omega_vals.append(max(variance ** 2, 1e-6))

            if len(P_rows) == 0:
                P_rows = [np.eye(n)[0]]
                Q_vals = [mu_eq[0]]
                omega_vals = [1e-4]

            P = np.array(P_rows)
            Q = np.array(Q_vals)
            omega_diag = np.array(omega_vals)

            w_bl, mu_bl, sigma_bl, bl_steps = black_litterman(
                mu_eq, sigma_ann, P, Q, omega_diag, tau=0.025
            )
            weights = pd.Series(w_bl, index=tickers)
            calc_steps = {"Equilibrium returns (Π)": mu_eq,
                          "BL Posterior mean (μ_BL)": mu_bl,
                          "Optimal BL weights": w_bl}

        elif method == "Equal Weight":
            weights = equal_weight(tickers)

        elif method == "Momentum":
            weights = momentum_weight(rets)

        elif method == "Low Volatility":
            weights = low_volatility_weight(rets)

        if weights is None:
            weights = equal_weight(tickers)

        # Normalize
        weights = weights.clip(lower=0)
        weights = weights / weights.sum()

    with st.spinner("⏳ Computing efficient frontier…"):
        frontier_df = efficient_frontier_points(mu_ann, sigma_ann, rf_rate, n_points=60)

    with st.spinner("⏳ Computing performance metrics…"):
        port_ret = portfolio_returns(rets, weights)
        cum_port = (1 + port_ret).cumprod()
        cum_bench = (1 + bench_ret.reindex(port_ret.index).fillna(0)).cumprod()
        metrics_df = compute_all_metrics(port_ret, bench_ret.reindex(port_ret.index).fillna(0), rf_rate)

        # Frontier key points
        w_gmv_arr = gmv_portfolio(sigma_ann)
        w_tan_arr = tangency_portfolio(mu_ann, sigma_ann, rf_rate)
        gmv_ret, gmv_vol, _ = portfolio_stats(w_gmv_arr, mu_ann, sigma_ann)
        tan_ret, tan_vol, _ = portfolio_stats(w_tan_arr, mu_ann, sigma_ann)
        sel_ret, sel_vol, _ = portfolio_stats(weights.values, mu_ann, sigma_ann)

    with st.spinner("⏳ Fetching Fama-French factors…"):
        ff_data = fetch_ff3_factors(str(start_date), str(end_date))
        ff_results = {}
        if not ff_data.empty:
            ff_results = fama_french_regression(port_ret, ff_data)

    # Build charts
    equity_fig = make_equity_curve(cum_port, cum_bench, "Portfolio", bench_label)
    frontier_fig = make_frontier_chart(
        frontier_df, (gmv_vol, gmv_ret), (tan_vol, tan_ret),
        (sel_vol, sel_ret), rf_rate
    )
    weights_fig = make_weights_chart(weights, f"Weights — {method}")
    corr_fig = make_correlation_heatmap(rets.corr())
    dd_fig = make_drawdown_chart(cum_port)
    rolling_sr_fig = make_rolling_sharpe(port_ret, rf_rate)

    st.session_state["results"] = {
        "tickers": tickers,
        "prices": prices,
        "rets": rets,
        "weights": weights,
        "port_ret": port_ret,
        "bench_ret": bench_ret.reindex(port_ret.index).fillna(0),
        "cum_port": cum_port,
        "cum_bench": cum_bench,
        "metrics_df": metrics_df,
        "frontier_df": frontier_df,
        "gmv_point": (gmv_vol, gmv_ret),
        "tan_point": (tan_vol, tan_ret),
        "sel_point": (sel_vol, sel_ret),
        "ff_results": ff_results,
        "calc_steps": calc_steps,
        "equity_fig": equity_fig,
        "frontier_fig": frontier_fig,
        "weights_fig": weights_fig,
        "corr_fig": corr_fig,
        "dd_fig": dd_fig,
        "rolling_sr_fig": rolling_sr_fig,
        "method": method,
        "mu_ann": mu_ann,
        "sigma_ann": sigma_ann,
        "rf_rate": rf_rate,
    }

    st.success("✅ Analysis complete! Browse the tabs below.")


# ============================================================
# DISPLAY RESULTS
# ============================================================
R = st.session_state.get("results")

# ── TAB 1: Planning ─────────────────────────────────────────
with tab1:
    if R is None:
        st.info("👈 Configure your parameters in the sidebar and click **Run Analysis**.")
        st.markdown("""
        ### About this Application
        This application implements the **three-step asset allocation framework** from the course:
        
        **Step 1 — Planning:** Define investment objectives, select asset universe, set benchmark and capital market expectations.
        
        **Step 2 — Execution:** Apply one of several portfolio optimization methodologies (Markowitz, Elton-Gruber, Merton, Black-Litterman, Smart-Beta strategies).
        
        **Step 3 — Feedback:** Evaluate portfolio performance using all course metrics (Sharpe, Sortino, Treynor, Jensen's Alpha, Fama-French regression, etc.)
        
        ---
        **Supported optimization methods:**
        - 📐 Markowitz MVO (Max Sharpe & Min Variance)
        - 📐 Elton-Gruber Tangency Portfolio
        - 📐 Merton Two-Fund Separation
        - 🧠 Black-Litterman (with manual investor views)
        - 📊 Equal Weight / Momentum / Low-Volatility (Smart-Beta)
        """)
    else:
        st.markdown('<div class="section-title">Asset Universe & Capital Market Expectations</div>',
                    unsafe_allow_html=True)

        tickers = R["tickers"]
        rets = R["rets"]
        mu_ann = R["mu_ann"]
        sigma_ann = R["sigma_ann"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Assets Selected", str(len(tickers)))
        with c2:
            kpi_card("Start Date", str(start_date))
        with c3:
            kpi_card("End Date", str(end_date))
        with c4:
            kpi_card("Benchmark", bench_label.split("(")[0].strip())

        st.markdown('<div class="section-title">Expected Returns & Risk (Annualised)</div>',
                    unsafe_allow_html=True)
        cme_df = pd.DataFrame({
            "Ticker": tickers,
            "Exp. Return (ann.)": [f"{r:.2%}" for r in mu_ann],
            "Volatility (ann.)": [f"{v:.2%}" for v in np.sqrt(np.diag(sigma_ann))],
            "Sharpe (individual)": [
                f"{(mu_ann[i] - rf_rate) / (np.sqrt(sigma_ann[i, i]) + 1e-10):.3f}"
                for i in range(len(tickers))
            ],
        })
        st.dataframe(cme_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Correlation Matrix</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(R["corr_fig"], use_container_width=True)

        if show_calcs:
            with st.expander("📐 Covariance Matrix (Σ) — Annual", expanded=False):
                cov_df = pd.DataFrame(sigma_ann, index=tickers, columns=tickers)
                st.dataframe(cov_df.style.background_gradient(cmap="YlOrRd"), use_container_width=True)


# ── TAB 2: Optimization ─────────────────────────────────────
with tab2:
    if R is None:
        st.info("👈 Run the analysis first.")
    else:
        st.markdown(f'<div class="section-title">{R["method"]}</div>',
                    unsafe_allow_html=True)

        # Method explanation
        method_explanations = {
            "Markowitz MVO (Max Sharpe)": {
                "desc": "Maximizes the Sharpe Ratio by finding the tangency portfolio on the efficient frontier. "
                        "This is the portfolio that offers the best risk-adjusted return.",
                "formula": "max  (w'μ - Rf) / √(w'Σw)   s.t.  Σwᵢ = 1,  wᵢ ≥ 0",
                "ref": "Markowitz (1952) — Portfolio Selection, Journal of Finance"
            },
            "Markowitz MVO (Min Variance)": {
                "desc": "Finds the Global Minimum Variance (GMV) portfolio — the leftmost point of the efficient frontier. "
                        "Ideal for risk-averse investors prioritizing stability over return.",
                "formula": "min  w'Σw   s.t.  Σwᵢ = 1,  wᵢ ≥ 0",
                "ref": "Markowitz (1952) — Portfolio Selection, Journal of Finance"
            },
            "Elton-Gruber (Tangency)": {
                "desc": "Elton & Gruber (1977) simplified Markowitz by providing a closed-form solution for the "
                        "tangency portfolio using the inverse covariance matrix.",
                "formula": "z = Σ⁻¹·(μ - Rf)   →   w* = z / Σzᵢ",
                "ref": "Elton & Gruber (1977) — Modern Portfolio Theory and Investment Analysis"
            },
            "Merton Two-Fund": {
                "desc": "Merton's Separation Theorem states that any efficient portfolio can be expressed as "
                        "a linear combination of the GMV portfolio and the Tangency portfolio. "
                        "The mixing parameter α is calibrated to your target return.",
                "formula": "w* = α·w_tan + (1-α)·w_gmv   where   α = (μ_target - μ_gmv) / (μ_tan - μ_gmv)",
                "ref": "Merton (1969) — Lifetime Portfolio Selection, Review of Economics and Statistics"
            },
            "Black-Litterman": {
                "desc": "Black-Litterman blends CAPM equilibrium returns with analyst views via Bayesian updating. "
                        "Views can be absolute (stock X will return Y%) or relative (X will outperform Y by Z%).",
                "formula": "μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹Π + P'Ω⁻¹Q]",
                "ref": "Black & Litterman (1992) — Global Portfolio Optimization, Financial Analysts Journal"
            },
            "Equal Weight": {
                "desc": "Naive 1/N diversification. Despite its simplicity, this strategy often performs "
                        "competitively as a benchmark for more sophisticated methods.",
                "formula": "wᵢ = 1/N   for all i",
                "ref": "DeMiguel et al. (2009) — Optimal Versus Naive Diversification"
            },
            "Momentum": {
                "desc": "Smart-beta strategy that overweights recent winners and underweights recent losers, "
                        "exploiting the momentum anomaly documented by Carhart (1997).",
                "formula": "w ∝ rank(return₋₁₂₆ days)²,  top quartile doubled, bottom quartile halved",
                "ref": "Carhart (1997) — On Persistence in Mutual Fund Performance"
            },
            "Low Volatility": {
                "desc": "Smart-beta strategy that inverts the CAPM prediction: lower-volatility stocks "
                        "tend to earn higher risk-adjusted returns (low-vol anomaly).",
                "formula": "wᵢ ∝ 1/σᵢ   (inverse volatility weighting)",
                "ref": "Baker, Bradley & Wurgler (2011) — Benchmarks as Limits to Arbitrage"
            },
        }

        expl = method_explanations.get(R["method"], {})
        if expl:
            with st.container():
                st.markdown(f"""<div class="method-card">
                    <strong>{R["method"]}</strong><br/>
                    <span style="color:#64748B;font-size:0.9rem;">{expl.get('desc','')}</span>
                    <div class="formula-box">{expl.get('formula','')}</div>
                    <span style="font-size:0.8rem;color:#94A3B8;">📚 {expl.get('ref','')}</span>
                </div>""", unsafe_allow_html=True)

        # Efficient Frontier
        st.markdown('<div class="section-title">Efficient Frontier</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(R["frontier_fig"], use_container_width=True)

        # Portfolio Weights
        st.markdown('<div class="section-title">Optimal Portfolio Weights</div>',
                    unsafe_allow_html=True)
        col_w1, col_w2 = st.columns([1.3, 1])
        with col_w1:
            st.plotly_chart(R["weights_fig"], use_container_width=True)
        with col_w2:
            w_table = R["weights"].sort_values(ascending=False)
            w_table_df = pd.DataFrame({
                "Ticker": w_table.index,
                "Weight": [f"{v:.2%}" for v in w_table.values],
            })
            st.dataframe(w_table_df, use_container_width=True, hide_index=True)

            # Key portfolio stats
            sel_ret, sel_vol, sel_sr = portfolio_stats(
                R["weights"].values, R["mu_ann"], R["sigma_ann"]
            )
            st.markdown("**Portfolio Statistics**")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Expected Return | {sel_ret:.2%} |
            | Volatility | {sel_vol:.2%} |
            | Sharpe Ratio | {sel_sr:.3f} |
            """)

        # Intermediate calculations
        if show_calcs and R["calc_steps"]:
            st.markdown('<div class="section-title">🔬 Intermediate Calculations (Professor Review)</div>',
                        unsafe_allow_html=True)
            for step_name, step_val in R["calc_steps"].items():
                with st.expander(f"📐 {step_name}", expanded=False):
                    if isinstance(step_val, np.ndarray):
                        if step_val.ndim == 1:
                            df_step = pd.DataFrame(
                                {"Asset": R["tickers"][:len(step_val)], step_name: step_val}
                            )
                            st.dataframe(df_step, use_container_width=True, hide_index=True)
                        elif step_val.ndim == 2:
                            df_step = pd.DataFrame(step_val,
                                                    index=R["tickers"][:step_val.shape[0]],
                                                    columns=R["tickers"][:step_val.shape[1]])
                            st.dataframe(df_step.style.background_gradient(cmap="coolwarm"),
                                         use_container_width=True)
                    elif isinstance(step_val, float):
                        st.metric(step_name, f"{step_val:.6f}")
                    else:
                        st.write(step_val)


# ── TAB 3: Performance ──────────────────────────────────────
with tab3:
    if R is None:
        st.info("👈 Run the analysis first.")
    else:
        metrics_df = R["metrics_df"]

        # KPI row
        st.markdown('<div class="section-title">Key Performance Indicators</div>',
                    unsafe_allow_html=True)

        m_dict = dict(zip(metrics_df["Metric"], metrics_df["Value"]))

        cols = st.columns(4)
        metrics_to_show = [
            ("Annualised Return", "Annualised Return"),
            ("Sharpe Ratio", "Sharpe Ratio"),
            ("Max Drawdown", "Max Drawdown"),
            ("Jensen's Alpha (ann.)", "Jensen's Alpha"),
        ]
        for col, (k, label) in zip(cols, metrics_to_show):
            with col:
                val = m_dict.get(k, "—")
                color = ""
                if "%" in str(val):
                    try:
                        num = float(str(val).replace("%", ""))
                        color = "positive" if num > 0 else "negative"
                    except Exception:
                        pass
                kpi_card(label, val, color)

        st.markdown("")
        cols2 = st.columns(4)
        m2 = [
            ("Sortino Ratio", "Sortino Ratio"),
            ("Treynor Ratio", "Treynor Ratio"),
            ("Information Ratio", "Information Ratio"),
            ("Calmar Ratio", "Calmar Ratio"),
        ]
        for col, (k, label) in zip(cols2, m2):
            with col:
                kpi_card(label, m_dict.get(k, "—"))

        st.markdown('<div class="section-title">Equity Curve vs Benchmark</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(R["equity_fig"], use_container_width=True)

        col_dd, col_rs = st.columns(2)
        with col_dd:
            st.plotly_chart(R["dd_fig"], use_container_width=True)
        with col_rs:
            st.plotly_chart(R["rolling_sr_fig"], use_container_width=True)

        st.markdown('<div class="section-title">Full Metrics Table (S9 — Performance Evaluation)</div>',
                    unsafe_allow_html=True)

        if show_calcs:
            st.markdown("""
            <div class="formula-box">
            Sharpe = (Rp - Rf) / σp  ·  Sortino = (Rp - Rf) / σ_downside  ·  
            Treynor = (Rp - Rf) / βp  ·  IR = (Rp - Rb) / TE  ·  
            Jensen's α = Rp - [Rf + βp·(Rm - Rf)]  ·  Calmar = CAGR / |Max DD|
            </div>
            """, unsafe_allow_html=True)

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Multi-strategy comparison
        st.markdown('<div class="section-title">Strategy Comparison</div>',
                    unsafe_allow_html=True)
        with st.expander("📊 Compare all strategies side-by-side", expanded=False):
            with st.spinner("Computing all strategies…"):
                comp_rows = []
                strats = {
                    "Max Sharpe": pd.Series(tangency_portfolio(R["mu_ann"], R["sigma_ann"], R["rf_rate"]), index=R["tickers"]),
                    "Min Variance": pd.Series(gmv_portfolio(R["sigma_ann"]), index=R["tickers"]),
                    "Equal Weight": equal_weight(R["tickers"]),
                    "Momentum": momentum_weight(R["rets"]),
                    "Low Volatility": low_volatility_weight(R["rets"]),
                    "Selected": R["weights"],
                }
                for sname, sw in strats.items():
                    sw = sw.clip(lower=0)
                    sw = sw / sw.sum()
                    pr = portfolio_returns(R["rets"], sw)
                    cum = (1 + pr).cumprod()
                    ann_r = (cum.iloc[-1]) ** (TRADING_DAYS / len(cum)) - 1
                    ann_v = pr.std() * np.sqrt(TRADING_DAYS)
                    sh = sharpe(pr, R["rf_rate"])
                    mdd = abs(max_drawdown(cum))
                    comp_rows.append({
                        "Strategy": sname,
                        "Ann. Return": f"{ann_r:.2%}",
                        "Volatility": f"{ann_v:.2%}",
                        "Sharpe": f"{sh:.3f}",
                        "Max Drawdown": f"{mdd:.2%}",
                        "Jensen's Alpha": f"{jensen_alpha(pr, R['bench_ret'], R['rf_rate']):.2%}",
                    })

                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Radar chart
                categories = ["Ann. Return", "Sharpe", "Sortino", "Calmar"]
                fig_radar = go.Figure()
                for row in comp_rows[:5]:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[float(str(row["Ann. Return"]).replace("%", "").replace("−", "-")),
                           float(row["Sharpe"]),
                           0, 0],  # placeholder
                        theta=categories, fill="toself", name=row["Strategy"]
                    ))
                fig_radar.update_layout(title="Strategy Comparison",
                                        polar=dict(radialaxis=dict(visible=True)),
                                        height=400, **{k: v for k, v in PLOTLY_TEMPLATE.items()
                                                       if k not in ["xaxis", "yaxis"]})
                st.plotly_chart(fig_radar, use_container_width=True)


# ── TAB 4: Factor Models ─────────────────────────────────────
with tab4:
    if R is None:
        st.info("👈 Run the analysis first.")
    else:
        # CAPM
        st.markdown('<div class="section-title">CAPM Analysis</div>',
                    unsafe_allow_html=True)
        if show_calcs:
            st.markdown("""
            <div class="formula-box">
            CAPM:  E[Rᵢ] - Rf = βᵢ · (E[Rm] - Rf)
            Jensen's Alpha:  αᵢ = Rᵢ_realized - E[Rᵢ]_CAPM
            R² measures how much return variation is explained by the market factor
            </div>
            """, unsafe_allow_html=True)

        tickers = R["tickers"]
        rets = R["rets"]
        bench_ret = R["bench_ret"]
        rf_d = rf_rate / TRADING_DAYS
        aligned = pd.concat([rets, bench_ret.rename("BENCH")], axis=1).dropna()

        capm_rows = []
        for t in tickers:
            if t not in aligned.columns:
                continue
            y = aligned[t].values
            X = aligned["BENCH"].values
            x_var = np.var(X, ddof=1) + 1e-12
            beta = float(np.cov(y, X, ddof=1)[0, 1] / x_var)
            a = float(y.mean() - beta * X.mean())
            y_hat = a + beta * X
            r2 = float(1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            ann_ret = float((1 + pd.Series(y)).prod() ** (TRADING_DAYS / len(y)) - 1)
            capm_exp = rf_rate + beta * erp
            alpha_ann = ann_ret - capm_exp
            capm_rows.append({
                "Ticker": t,
                "Beta (β)": f"{beta:.3f}",
                "CAPM Exp. Return": f"{capm_exp:.2%}",
                "Realized Return": f"{ann_ret:.2%}",
                "Jensen's Alpha": f"{alpha_ann:.2%}",
                "R²": f"{r2:.3f}",
            })

        capm_df = pd.DataFrame(capm_rows)
        st.dataframe(capm_df, use_container_width=True, hide_index=True)

        # Security Market Line
        betas = [float(r["Beta (β)"]) for r in capm_rows]
        alphas = [float(r["Jensen's Alpha"].replace("%", "")) / 100 for r in capm_rows]
        exp_rets = [float(r["CAPM Exp. Return"].replace("%", "")) / 100 for r in capm_rows]
        real_rets = [float(r["Realized Return"].replace("%", "")) / 100 for r in capm_rows]

        beta_range = np.linspace(min(min(betas), 0), max(max(betas), 2), 50)
        sml_rets = rf_rate + beta_range * erp

        fig_sml = go.Figure()
        fig_sml.add_trace(go.Scatter(
            x=beta_range, y=sml_rets * 100,
            mode="lines", name="Security Market Line (SML)",
            line=dict(color=ACCENT, width=2, dash="dash")
        ))
        fig_sml.add_trace(go.Scatter(
            x=betas, y=[r * 100 for r in real_rets],
            mode="markers+text", name="Assets (Realized)",
            text=tickers, textposition="top center",
            marker=dict(
                size=10, color=[a * 1000 for a in alphas],
                colorscale="RdYlGn", showscale=True,
                colorbar=dict(title="Alpha (bps)"),
                line=dict(color=DARK, width=1)
            )
        ))
        fig_sml.update_layout(
            title="📊 Security Market Line (CAPM) — Assets vs SML",
            xaxis_title="Beta (β)", yaxis_title="Return (%)",
            height=480, **PLOTLY_TEMPLATE
        )
        st.plotly_chart(fig_sml, use_container_width=True)

        # Fama-French
        st.markdown('<div class="section-title">Fama-French 3-Factor Model</div>',
                    unsafe_allow_html=True)
        if show_calcs:
            st.markdown("""
            <div class="formula-box">
            FF3:  Rₚ - Rf = α + β_MKT·(Rm - Rf) + β_SMB·SMB + β_HML·HML + ε
            
            MktRF = Market excess return  |  SMB = Small minus Big (size factor)
            HML   = High minus Low B/M   (value factor)  |  α = abnormal return
            </div>
            """, unsafe_allow_html=True)

        ff_results = R.get("ff_results", {})
        if ff_results:
            col_ff1, col_ff2 = st.columns(2)
            with col_ff1:
                st.markdown("**Factor Loadings & Alpha**")
                ff_display = {
                    "Alpha (annualised)": f"{ff_results.get('Alpha (annualised)', 0):.2%}",
                    "β Market (MktRF)": f"{ff_results.get('β_MktRF', 0):.3f}",
                    "β Size (SMB)": f"{ff_results.get('β_SMB', 0):.3f}",
                    "β Value (HML)": f"{ff_results.get('β_HML', 0):.3f}",
                    "R²": f"{ff_results.get('R²', 0):.3f}",
                    "Adj. R²": f"{ff_results.get('Adj. R²', 0):.3f}",
                    "N observations": str(ff_results.get("N obs", "")),
                }
                ff_df = pd.DataFrame(list(ff_display.items()), columns=["Parameter", "Value"])
                st.dataframe(ff_df, use_container_width=True, hide_index=True)

            with col_ff2:
                st.markdown("**T-Statistics (|t| > 1.96 → significant at 5%)**")
                t_display = {
                    "t-stat Alpha": f"{ff_results.get('t_alpha', 0):.3f}",
                    "t-stat MktRF": f"{ff_results.get('t_MktRF', 0):.3f}",
                    "t-stat SMB": f"{ff_results.get('t_SMB', 0):.3f}",
                    "t-stat HML": f"{ff_results.get('t_HML', 0):.3f}",
                    "p-value Alpha": f"{ff_results.get('p_alpha', 1):.4f}",
                }
                t_df = pd.DataFrame(list(t_display.items()), columns=["Statistic", "Value"])
                st.dataframe(t_df, use_container_width=True, hide_index=True)

            # Factor bar chart
            fig_ff = go.Figure(go.Bar(
                x=["Market (β_MKT)", "Size (β_SMB)", "Value (β_HML)"],
                y=[ff_results.get("β_MktRF", 0),
                   ff_results.get("β_SMB", 0),
                   ff_results.get("β_HML", 0)],
                marker_color=[ACCENT, SUCCESS, WARNING],
                text=[f"{v:.3f}" for v in [ff_results.get("β_MktRF", 0),
                                            ff_results.get("β_SMB", 0),
                                            ff_results.get("β_HML", 0)]],
                textposition="outside",
            ))
            fig_ff.add_hline(y=0, line_color=DARK)
            fig_ff.update_layout(
                title=f"📊 Fama-French Factor Exposures  |  Ann. Alpha = {ff_results.get('Alpha (annualised)', 0):.2%}  |  R² = {ff_results.get('R²', 0):.3f}",
                yaxis_title="Factor Loading (β)",
                height=380, **PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_ff, use_container_width=True)

            if show_calcs:
                with st.expander("📐 Regression details (OLS formula & interpretation)"):
                    st.markdown("""
                    **OLS Estimation:**
                    
                    β = (X'X)⁻¹ X'y   where X = [1, MktRF, SMB, HML]
                    
                    **Interpretation:**
                    - **β_MKT > 1** → portfolio amplifies market moves (aggressive)
                    - **β_SMB > 0** → tilted toward small-cap stocks
                    - **β_HML > 0** → tilted toward value stocks (high book-to-market)
                    - **α > 0** → portfolio generates returns above what FF3 factors predict
                    - **R²** → fraction of portfolio return variation explained by the three factors
                    """)
        else:
            st.warning("Fama-French factor data not available for this period. Check your internet connection or try a different date range.")


# ── TAB 5: Report ────────────────────────────────────────────
with tab5:
    if R is None:
        st.info("👈 Run the analysis first to generate the report.")
    else:
        st.markdown('<div class="section-title">📄 Export PDF Report</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        The PDF report follows the three-step asset allocation structure required by the course:
        
        - **Step 1 — Planning**: Investment objectives, benchmark, capital market expectations
        - **Step 2 — Execution**: Optimization methodology, intermediate calculations, portfolio weights, efficient frontier
        - **Step 3 — Feedback**: Full performance evaluation with all course metrics, Fama-French analysis
        """)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            include_calcs = st.toggle("Include intermediate calculations in PDF",
                                       value=show_calcs,
                                       help="Show calculation steps for professor review")
        with col_opt2:
            report_title_input = st.text_input("Portfolio Name", value="Global Equity Portfolio")

        if st.button("📄 Generate Report", type="primary"):
            with st.spinner("Building PDF report…"):
                report_params_pdf = {
                    "Portfolio Name": report_title_input,
                    "Method": R["method"],
                    "start": str(start_date),
                    "end": str(end_date),
                    "benchmark": bench_label,
                    "rf": rf_rate,
                    "ERP": erp,
                    "n_assets": len(R["tickers"]),
                    "Regions": ", ".join(region_choices),
                }

                pdf_bytes = build_pdf_report(
                    report_params=report_params_pdf,
                    weights=R["weights"],
                    metrics_df=R["metrics_df"],
                    method_name=R["method"],
                    equity_fig=R["equity_fig"],
                    frontier_fig=R["frontier_fig"],
                    weights_fig=R["weights_fig"],
                    corr_fig=R["corr_fig"],
                    dd_fig=R["dd_fig"],
                    rolling_sr_fig=R["rolling_sr_fig"],
                    ff_results=R.get("ff_results"),
                    show_calcs=include_calcs,
                    calc_steps=R.get("calc_steps"),
                )

                st.download_button(
                    label="⬇️ Download Full Portfolio Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"portfolio_report_{date.today()}.pdf",
                    mime="application/pdf",
                )
                st.success("✅ Report ready for download!")

        # Quick stats preview
        st.markdown('<div class="section-title">Report Preview</div>',
                    unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Portfolio Weights (Top 10)**")
            w_preview = R["weights"].sort_values(ascending=False).head(10)
            fig_pie = go.Figure(go.Pie(
                labels=w_preview.index,
                values=w_preview.values,
                hole=0.4,
                marker=dict(colors=px.colors.sequential.RdBu[:len(w_preview)])
            ))
            fig_pie.update_layout(
                height=380, showlegend=True,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="white"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_p2:
            st.markdown("**Performance Summary**")
            st.dataframe(R["metrics_df"], use_container_width=True, hide_index=True)
