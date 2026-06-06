# ============================================================
# Portfolio Management Pro — ESSCA Project
# Prof. Benoit Seguret — Portfolio Management
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
# GEOGRAPHIC UNIVERSE  (imported from universe.py)
# ============================================================
from universe import UNIVERSE, ALL_REGIONS, get_tickers_for_regions


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

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_ff3_factors(start: str, end: str) -> pd.DataFrame:
    """
    Build proxy Fama-French 3-factor series from freely available ETF data via yfinance.
    No external data provider required — works on Streamlit Cloud.

    Proxies:
      Mkt-RF  = SPY daily return − RF (daily)
      SMB     = IWM (small cap) − IWB (large cap)   [size factor]
      HML     = IVE (value)     − IVW (growth)      [value factor]
      RF      = SHY daily return (short-term T-bills proxy)
    """
    try:
        etfs = ["SPY", "IWM", "IWB", "IVE", "IVW", "SHY"]
        raw = yf.download(etfs, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw

        prices = prices.dropna(how="all")
        rets = prices.pct_change().dropna()

        available = rets.columns.tolist()

        # RF proxy
        if "SHY" in available:
            rf_daily = rets["SHY"]
        else:
            rf_daily = pd.Series(0.04 / 252, index=rets.index)

        # Market excess return
        if "SPY" in available:
            mkt_rf = rets["SPY"] - rf_daily
        else:
            return pd.DataFrame()

        # SMB: small minus large
        if "IWM" in available and "IWB" in available:
            smb = rets["IWM"] - rets["IWB"]
        else:
            smb = pd.Series(0.0, index=rets.index)

        # HML: value minus growth
        if "IVE" in available and "IVW" in available:
            hml = rets["IVE"] - rets["IVW"]
        else:
            hml = pd.Series(0.0, index=rets.index)

        ff = pd.DataFrame({
            "Mkt-RF": mkt_rf,
            "SMB": smb,
            "HML": hml,
            "RF": rf_daily,
        }).dropna()
        return ff

    except Exception as e:
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
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, sans-serif", color="#1C1C2E", size=12),
    colorway=[PRIMARY, ACCENT, SUCCESS, WARNING, "#9B59B6", "#1ABC9C", "#E67E22"],
    xaxis=dict(
        showgrid=True, gridcolor="#E8ECF0", linecolor="#CBD5E1",
        tickfont=dict(color="#374151", size=11),
        titlefont=dict(color="#1C1C2E", size=12),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#E8ECF0", linecolor="#CBD5E1",
        tickfont=dict(color="#374151", size=11),
        titlefont=dict(color="#1C1C2E", size=12),
    ),
    title=dict(font=dict(color="#1C1C2E", size=14, family="Inter, sans-serif")),
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#CBD5E1",
        borderwidth=1, orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(color="#374151", size=11),
    ),
    margin=dict(l=20, r=20, t=50, b=20),
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
        fontSize=24, fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_CENTER, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontSize=11, fontName="Helvetica",
        textColor=colors.HexColor("#CBD5E1"),
        alignment=TA_CENTER, spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.white,
        spaceBefore=0, spaceAfter=0,
        leftIndent=8,
    ))
    styles.add(ParagraphStyle(
        name="SubHeader",
        fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.HexColor(PRIMARY),
        spaceBefore=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        fontSize=8.5, fontName="Helvetica",
        textColor=colors.HexColor("#374151"),
        spaceAfter=4, leading=13,
    ))
    styles.add(ParagraphStyle(
        name="FormulaText",
        fontSize=8, fontName="Courier",
        textColor=colors.HexColor("#1E293B"),
        backColor=colors.HexColor("#F1F5F9"),
        borderPad=5, spaceAfter=5, leading=12,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        fontSize=7.5, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#6B7280"),
        alignment=TA_CENTER, spaceAfter=4,
    ))
    return styles

def _rl_table_style(has_header: bool = True) -> TableStyle:
    style = [
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor(DARK) if has_header else colors.HexColor("#F8FAFC")),
        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.white if has_header else colors.HexColor(DARK)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    return TableStyle(style)

def _section_banner(title: str, subtitle: str = "") -> RLTable:
    """Red banner for section headers."""
    cell_content = [Paragraph(title, ParagraphStyle(
        "BannerTitle", fontSize=12, fontName="Helvetica-Bold",
        textColor=colors.white, spaceAfter=0,
    ))]
    if subtitle:
        cell_content.append(Paragraph(subtitle, ParagraphStyle(
            "BannerSub", fontSize=8, fontName="Helvetica",
            textColor=colors.HexColor("#FECACA"), spaceAfter=0,
        )))
    t = RLTable([[cell_content]], colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PRIMARY)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return t

# ── Matplotlib chart builders for PDF ──────────────────────

MPL_COLORS = {
    "primary": "#C8102E",
    "accent": "#2E86AB",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "dark": "#1C1C2E",
    "light_bg": "#F7F9FC",
    "grid": "#E8ECF0",
}

def _mpl_base_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(MPL_COLORS["light_bg"])
    ax.grid(True, color=MPL_COLORS["grid"], linewidth=0.6, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MPL_COLORS["grid"])
    ax.spines["bottom"].set_color(MPL_COLORS["grid"])
    ax.tick_params(colors=MPL_COLORS["dark"], labelsize=7)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold",
                     color=MPL_COLORS["dark"], pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5, color=MPL_COLORS["dark"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5, color=MPL_COLORS["dark"])

def _mpl_fig_to_bytes(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf

def _pdf_chart_equity(cum_port: pd.Series, cum_bench: pd.Series,
                      bench_label: str) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    base = 100.0
    p = cum_port / cum_port.iloc[0] * base
    b = cum_bench / cum_bench.iloc[0] * base
    ax.plot(p.index, p.values, color=MPL_COLORS["primary"], lw=1.8,
            label="Portfolio", zorder=3)
    ax.plot(b.index, b.values, color=MPL_COLORS["accent"], lw=1.4,
            linestyle="--", label=bench_label, zorder=2)
    ax.fill_between(p.index, p.values, b.values,
                    where=p.values >= b.values,
                    alpha=0.12, color=MPL_COLORS["success"], label="Outperformance")
    ax.fill_between(p.index, p.values, b.values,
                    where=p.values < b.values,
                    alpha=0.12, color=MPL_COLORS["primary"])
    _mpl_base_style(ax, "Portfolio vs Benchmark — Equity Curve (base 100)",
                    "Date", "Index (base 100)")
    ax.legend(fontsize=7, loc="upper left",
              facecolor="white", edgecolor=MPL_COLORS["grid"])
    fig.tight_layout()
    return _mpl_fig_to_bytes(fig)

def _pdf_chart_frontier(frontier_df: pd.DataFrame,
                        gmv_point: tuple, tan_point: tuple,
                        sel_point: tuple, rf: float) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    if not frontier_df.empty:
        ax.plot(frontier_df["Volatility"] * 100, frontier_df["Return"] * 100,
                color=MPL_COLORS["primary"], lw=2.2, label="Efficient Frontier", zorder=3)
    if gmv_point:
        ax.scatter([gmv_point[0] * 100], [gmv_point[1] * 100],
                   marker="D", s=80, color=MPL_COLORS["accent"],
                   zorder=5, label="GMV Portfolio")
        ax.annotate("GMV", (gmv_point[0] * 100, gmv_point[1] * 100),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=MPL_COLORS["accent"], fontweight="bold")
    if tan_point:
        ax.scatter([tan_point[0] * 100], [tan_point[1] * 100],
                   marker="*", s=160, color=MPL_COLORS["success"],
                   zorder=5, label="Tangency (Max Sharpe)")
        ax.annotate("Tangency", (tan_point[0] * 100, tan_point[1] * 100),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=MPL_COLORS["success"], fontweight="bold")
        # CML
        vol_range = np.linspace(0, tan_point[0] * 1.6, 50)
        cml = rf * 100 + (tan_point[1] - rf) / tan_point[0] * vol_range
        ax.plot(vol_range * 100, cml, color=MPL_COLORS["warning"],
                lw=1.4, linestyle="--", label="CML", zorder=2)
    if sel_point:
        ax.scatter([sel_point[0] * 100], [sel_point[1] * 100],
                   marker="o", s=100, color=MPL_COLORS["primary"],
                   edgecolors="white", linewidths=1.5, zorder=6,
                   label="Selected Portfolio")
        ax.annotate("Portfolio", (sel_point[0] * 100, sel_point[1] * 100),
                    textcoords="offset points", xytext=(-50, 6),
                    fontsize=7, color=MPL_COLORS["primary"], fontweight="bold")
    _mpl_base_style(ax, "Mean-Variance Efficient Frontier",
                    "Volatility (σ) %", "Expected Return (μ) %")
    ax.legend(fontsize=7, facecolor="white", edgecolor=MPL_COLORS["grid"])
    fig.tight_layout()
    return _mpl_fig_to_bytes(fig)

def _pdf_chart_weights(weights: pd.Series) -> BytesIO:
    w = weights[weights > 0.001].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, len(w) * 0.32)))
    bar_colors = [MPL_COLORS["primary"] if v > w.median() else MPL_COLORS["accent"]
                  for v in w.values]
    bars = ax.barh(range(len(w)), w.values * 100, color=bar_colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(w)))
    ax.set_yticklabels(w.index, fontsize=7.5)
    for bar, val in zip(bars, w.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7,
                color=MPL_COLORS["dark"], fontweight="bold")
    _mpl_base_style(ax, "Portfolio Weights Allocation", "Weight (%)", "")
    ax.set_xlim(0, w.values.max() * 100 * 1.18)
    fig.tight_layout()
    return _mpl_fig_to_bytes(fig)

def _pdf_chart_corr(corr: pd.DataFrame) -> BytesIO:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, n * 0.5)))
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(corr.index, fontsize=6.5)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            text_col = "white" if abs(val) > 0.6 else MPL_COLORS["dark"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=text_col, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title("Asset Correlation Matrix", fontsize=9,
                 fontweight="bold", color=MPL_COLORS["dark"], pad=6)
    fig.tight_layout()
    return _mpl_fig_to_bytes(fig)

def _pdf_chart_drawdown(cum_port: pd.Series) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    roll_max = cum_port.cummax()
    dd = (cum_port / roll_max - 1) * 100
    ax.fill_between(dd.index, dd.values, 0,
                    color=MPL_COLORS["primary"], alpha=0.35)
    ax.plot(dd.index, dd.values, color=MPL_COLORS["primary"], lw=1)
    _mpl_base_style(ax, "Historical Drawdown", "Date", "Drawdown (%)")
    ax.axhline(0, color=MPL_COLORS["dark"], lw=0.8)
    mdd_val = dd.min()
    mdd_date = dd.idxmin()
    ax.annotate(f"Max DD: {mdd_val:.1f}%",
                xy=(mdd_date, mdd_val),
                xytext=(20, 10), textcoords="offset points",
                fontsize=7, color=MPL_COLORS["primary"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MPL_COLORS["primary"],
                                lw=0.8))
    fig.tight_layout()
    return _mpl_fig_to_bytes(fig)

def _pdf_chart_rolling_sharpe(port_ret: pd.Series, rf_annual: float) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    rf_d = rf_annual / TRADING_DAYS
    excess = port_ret - rf_d
    roll_sr = (excess.rolling(126).mean() /
               excess.rolling(126).std() * np.sqrt(TRADING_DAYS))
    ax.plot(roll_sr.index, roll_sr.values,
            color=MPL_COLORS["accent"], lw=1.4, label="Rolling Sharpe (126d)")
    ax.axhline(0, color="#94A3B8", lw=0.8, linestyle="--")
    ax.axhline(1, color=MPL_COLORS["success"], lw=0.8, linestyle=":",
               label="Sharpe = 1.0")
    ax.fill_between(roll_sr.index, roll_sr.values, 0,
                    where=roll_sr.values >= 0,
                    alpha=0.12, color=MPL_COLORS["success"])
    ax.fill_between(roll_sr.index, roll_sr.values, 0,
                    where=roll_sr.values < 0,
                    alpha=0.12, color=MPL_COLORS["primary"])
    _mpl_base_style(ax, "Rolling Sharpe Ratio (126-day window)",
                    "Date", "Sharpe Ratio")
    ax.legend(fontsize=7, facecolor="white", edgecolor=MPL_COLORS["grid"])
    fig.tight_layout()
    return _mpl_fig_to_bytes(fig)

def _pdf_chart_ff_betas(ff_results: dict) -> BytesIO:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))

    # Left: Factor betas bar
    ax = axes[0]
    factors = ["β_MktRF", "β_SMB", "β_HML"]
    labels = ["Market\n(MktRF)", "Size\n(SMB)", "Value\n(HML)"]
    vals = [ff_results.get(f, 0) for f in factors]
    bar_cols = [MPL_COLORS["accent"] if v >= 0 else MPL_COLORS["primary"]
                for v in vals]
    bars = ax.bar(labels, vals, color=bar_cols, edgecolor="white", width=0.5)
    ax.axhline(0, color=MPL_COLORS["dark"], lw=0.8)
    for bar, v in zip(bars, vals):
        ypos = v + 0.02 if v >= 0 else v - 0.06
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:.3f}", ha="center", fontsize=8, fontweight="bold",
                color=MPL_COLORS["dark"])
    _mpl_base_style(ax, "Factor Loadings (β)", "", "Beta Coefficient")

    # Right: R² and alpha summary
    ax2 = axes[1]
    ax2.axis("off")
    alpha_ann = ff_results.get("Alpha (annualised)", 0)
    r2 = ff_results.get("R²", 0)
    adj_r2 = ff_results.get("Adj. R²", 0)
    t_alpha = ff_results.get("t_alpha", 0)
    p_alpha = ff_results.get("p_alpha", 1)
    summary_text = (
        f"Fama-French 3-Factor Results\n\n"
        f"Alpha (ann.):   {alpha_ann:.2%}\n"
        f"t-stat alpha:   {t_alpha:.3f}\n"
        f"p-value alpha:  {p_alpha:.4f}\n"
        f"R²:             {r2:.3f}\n"
        f"Adj. R²:        {adj_r2:.3f}\n"
        f"N obs:          {ff_results.get('N obs', '')}\n\n"
        f"{'✓ Alpha significant (5%)' if p_alpha < 0.05 else '✗ Alpha not significant'}"
    )
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=8, va="top", fontfamily="monospace",
             color=MPL_COLORS["dark"],
             bbox=dict(boxstyle="round,pad=0.5", facecolor=MPL_COLORS["light_bg"],
                       edgecolor=MPL_COLORS["grid"]))

    fig.tight_layout()
    return _mpl_fig_to_bytes(fig)

def _add_image_to_story(story: list, img_bytes: BytesIO,
                        caption: str, width_cm: float = 17):
    if img_bytes is None:
        return
    try:
        img_bytes.seek(0)
        story.append(RLImage(img_bytes, width=width_cm * cm,
                              height=width_cm * 0.42 * cm))
        story.append(Paragraph(caption, ParagraphStyle(
            "cap", fontSize=7.5, fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#6B7280"),
            alignment=TA_CENTER, spaceAfter=6,
        )))
        story.append(Spacer(1, 0.15 * cm))
    except Exception:
        pass


def build_pdf_report(
    report_params: dict,
    weights: pd.Series,
    metrics_df: pd.DataFrame,
    method_name: str,
    # raw data for matplotlib charts (preferred over plotly for PDF)
    cum_port: pd.Series = None,
    cum_bench: pd.Series = None,
    port_ret: pd.Series = None,
    rets: pd.DataFrame = None,
    frontier_df: pd.DataFrame = None,
    gmv_point: tuple = None,
    tan_point: tuple = None,
    sel_point: tuple = None,
    rf: float = 0.0,
    bench_label: str = "Benchmark",
    ff_results: dict = None,
    show_calcs: bool = False,
    calc_steps: dict = None,
) -> bytes:
    """Generate a professional, chart-rich PDF report."""
    buf = BytesIO()
    PAGE_W = 17 * cm  # usable width on A4 with 2cm margins each side

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )
    styles = _build_pdf_styles()
    story = []

    # ── COVER PAGE ──────────────────────────────────────────────
    # Dark header block
    cover_title = RLTable(
        [[Paragraph("PORTFOLIO MANAGEMENT", ParagraphStyle(
              "CT", fontSize=22, fontName="Helvetica-Bold",
              textColor=colors.white, alignment=TA_CENTER, spaceAfter=2)),
          ],
         [Paragraph("Advanced Portfolio Analysis Report", ParagraphStyle(
              "CS", fontSize=11, fontName="Helvetica",
              textColor=colors.HexColor("#CBD5E1"), alignment=TA_CENTER, spaceAfter=0)),
          ],
         [Paragraph("ESSCA · Prof. Benoit Seguret · Portfolio Management Course",
                     ParagraphStyle("CI", fontSize=9, fontName="Helvetica-Oblique",
                                    textColor=colors.HexColor("#94A3B8"),
                                    alignment=TA_CENTER, spaceAfter=0)),
          ]
         ],
        colWidths=[PAGE_W]
    )
    cover_title.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(DARK)),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))
    story.append(cover_title)
    story.append(Spacer(1, 0.5 * cm))

    # Cover info table
    info_data = [
        ["Field", "Value"],
        ["Optimization Method", method_name],
        ["Analysis Period", f"{report_params.get('start', '')}  →  {report_params.get('end', '')}"],
        ["Benchmark", report_params.get("benchmark", "SPY")],
        ["Risk-Free Rate (Rf)", f"{report_params.get('rf', 0):.2%}"],
        ["Equity Risk Premium", f"{report_params.get('ERP', 0.05):.2%}"],
        ["Number of Assets", str(report_params.get("n_assets", ""))],
        ["Geographic Regions", str(report_params.get("Regions", ""))],
        ["Date Generated", str(date.today())],
    ]
    t_cover = RLTable(info_data, colWidths=[6.5 * cm, 10.5 * cm])
    t_cover.setStyle(_rl_table_style())
    story.append(t_cover)
    story.append(PageBreak())

    # ── STEP 1: PLANNING ────────────────────────────────────────
    story.append(_section_banner(
        "STEP 1 — PLANNING",
        "Investment Objectives · Benchmark · Capital Market Expectations"
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Investment Framework", styles["SubHeader"]))
    story.append(Paragraph(
        "This portfolio follows the <b>three-step asset allocation framework</b> (S2 — Asset Allocation). "
        "The execution applies <b>Modern Portfolio Theory</b> (Markowitz, 1952): constructing a portfolio "
        f"on the efficient frontier using <b>{method_name}</b>. "
        "The objective is to maximise risk-adjusted return relative to the selected benchmark, "
        "respecting the investor's risk tolerance and return objectives.",
        styles["BodyText2"]
    ))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Capital Market Expectations (CME)", styles["SubHeader"]))
    story.append(Paragraph(
        "Expected returns are estimated from historical geometric returns annualised over the analysis period. "
        "The covariance matrix is computed from daily returns and annualised (×252 trading days). "
        f"Risk-free rate: {report_params.get('rf', 0):.2%} · "
        f"Equity Risk Premium: {report_params.get('ERP', 0.05):.2%}.",
        styles["BodyText2"]
    ))
    story.append(Spacer(1, 0.25 * cm))

    params_data = [["Parameter", "Value"]] + [
        [k, str(v)] for k, v in report_params.items()
        if k not in ["start", "end"]
    ]
    t_params = RLTable(params_data, colWidths=[7 * cm, 10 * cm])
    t_params.setStyle(_rl_table_style())
    story.append(t_params)
    story.append(PageBreak())

    # ── STEP 2: EXECUTION ───────────────────────────────────────
    story.append(_section_banner(
        "STEP 2 — EXECUTION: PORTFOLIO CONSTRUCTION",
        "Optimization Methodology · Portfolio Weights · Efficient Frontier · Correlation"
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(f"Optimization Method: {method_name}", styles["SubHeader"]))

    method_details = {
        "Markowitz MVO (Max Sharpe)": (
            "Mean-Variance Optimization (Markowitz, 1952) identifies the tangency portfolio — "
            "the point on the efficient frontier with the highest Sharpe ratio.",
            "Objective:  max  (w'μ - Rf) / √(w'Σw)\n"
            "Subject to: Σwᵢ = 1,  wᵢ ≥ 0  (long-only)\n"
            "Solution:   w* = Σ⁻¹(μ - Rf) / [1'·Σ⁻¹(μ - Rf)]  (closed-form)",
        ),
        "Markowitz MVO (Min Variance)": (
            "The Global Minimum Variance (GMV) portfolio minimises total variance — "
            "the leftmost point of the efficient frontier, optimal for maximally risk-averse investors.",
            "Objective:  min  w'Σw\n"
            "Subject to: Σwᵢ = 1,  wᵢ ≥ 0\n"
            "Solution:   w_GMV = Σ⁻¹·1 / (1'·Σ⁻¹·1)  (closed-form)",
        ),
        "Elton-Gruber (Tangency)": (
            "Elton & Gruber (1977) provided a closed-form simplification of Markowitz, "
            "computing the tangency portfolio directly from the inverse covariance matrix.",
            "Step 1:  Compute excess returns  e = μ - Rf\n"
            "Step 2:  z = Σ⁻¹ · e  (z-vector)\n"
            "Step 3:  w* = z / Σzᵢ  (normalise to sum to 1)",
        ),
        "Merton Two-Fund": (
            "Merton's Separation Theorem (1969): any efficient portfolio is a linear combination "
            "of two funds — the GMV and the Tangency portfolio.",
            "w*(target) = α · w_tan + (1 - α) · w_gmv\n"
            "where  α = (μ_target - μ_gmv) / (μ_tan - μ_gmv)\n"
            "Interpretation: α > 1 → leveraged, α < 0 → short market",
        ),
        "Black-Litterman": (
            "Black & Litterman (1992) blend CAPM equilibrium returns (Π) with investor views "
            "via Bayesian updating, producing a more stable and diversified allocation.",
            "Prior:   Π = δ·Σ·w_mkt  (CAPM-implied equilibrium returns)\n"
            "Posterior: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹Π + P'Ω⁻¹Q]\n"
            "Views: absolute (asset i returns r%) or relative (i outperforms j by r%)",
        ),
        "Equal Weight": (
            "Naïve 1/N diversification assigns equal weight to all assets. "
            "Despite its simplicity, it is competitive with MVO out-of-sample (DeMiguel et al., 2009).",
            "wᵢ = 1/N  for all i = 1, ..., N",
        ),
        "Momentum": (
            "Smart-beta momentum strategy (Carhart, 1997): overweight past winners, "
            "underweight past losers. Exploits the momentum factor (UMD).",
            "wᵢ ∝ rank(Ret₋₁₂₆ days)  · tilt multiplier\n"
            "Top 25%: weight × 2  |  Bottom 25%: weight × 0.5",
        ),
        "Low Volatility": (
            "Smart-beta low-volatility strategy: weights inversely proportional to historical σᵢ. "
            "Exploits the low-vol anomaly (Baker, Bradley & Wurgler, 2011).",
            "wᵢ ∝ 1/σᵢ  (inverse volatility weighting)\n"
            "σᵢ = historical daily std × √252  (annualised)",
        ),
    }

    desc, formula = method_details.get(method_name, ("See application.", ""))
    story.append(Paragraph(desc, styles["BodyText2"]))
    if formula:
        story.append(Paragraph(formula, styles["FormulaText"]))
    story.append(Spacer(1, 0.2 * cm))

    # Weights table
    story.append(Paragraph("Final Portfolio Weights", styles["SubHeader"]))
    w_show = weights[weights > 0.001].sort_values(ascending=False)
    w_data = [["Rank", "Ticker", "Weight (%)", "Contribution"]]
    for i, (ticker, v) in enumerate(w_show.items(), 1):
        bar = "█" * int(v * 40)
        w_data.append([str(i), str(ticker), f"{v:.2%}", bar])
    wt = RLTable(w_data, colWidths=[1.5 * cm, 4 * cm, 4 * cm, 7.5 * cm])
    wt.setStyle(_rl_table_style())
    story.append(wt)
    story.append(Spacer(1, 0.3 * cm))

    # Weights chart
    if cum_port is not None:
        w_img = _pdf_chart_weights(weights)
        _add_image_to_story(story, w_img, "Figure 1: Portfolio Weights Allocation")

    story.append(PageBreak())

    # Efficient Frontier chart
    if frontier_df is not None and not frontier_df.empty:
        story.append(Paragraph("Efficient Frontier & Capital Market Line", styles["SubHeader"]))
        story.append(Paragraph(
            "The efficient frontier (Markowitz, 1952) plots all mean-variance optimal portfolios. "
            "The GMV portfolio (◆) is the minimum-risk portfolio. "
            "The Tangency portfolio (★) maximises the Sharpe ratio and lies at the tangent point of the CML. "
            "The CML (dashed) connects the risk-free rate to the tangency portfolio — "
            "all investors should combine the tangency portfolio with the risk-free asset.",
            styles["BodyText2"]
        ))
        frontier_img = _pdf_chart_frontier(frontier_df, gmv_point, tan_point, sel_point, rf)
        _add_image_to_story(story, frontier_img,
                             "Figure 2: Mean-Variance Efficient Frontier with GMV, Tangency Portfolio & CML")

    # Correlation matrix
    if rets is not None:
        story.append(Paragraph("Asset Correlation Matrix", styles["SubHeader"]))
        story.append(Paragraph(
            "Correlation measures co-movement between assets. Lower correlations → higher diversification benefit. "
            "Values close to +1 indicate assets move together; close to −1 indicates inverse movement.",
            styles["BodyText2"]
        ))
        corr_img = _pdf_chart_corr(rets.corr())
        _add_image_to_story(story, corr_img, "Figure 3: Asset Correlation Matrix")

    story.append(PageBreak())

    # Intermediate calculations
    if show_calcs and calc_steps:
        story.append(Paragraph("Intermediate Calculations (Professor Review)",
                                styles["SubHeader"]))
        story.append(Paragraph(
            f"Detailed step-by-step derivation for {method_name}:",
            styles["BodyText2"]
        ))
        for step_name, step_val in calc_steps.items():
            story.append(Paragraph(f"<b>{step_name}:</b>", styles["BodyText2"]))
            if isinstance(step_val, np.ndarray) and step_val.ndim == 1:
                tickers_short = [str(x) for x in range(len(step_val))]
                rows = [["Asset"] + [f"w{i}" for i in range(min(len(step_val), 10))]]
                rows.append(["Value"] + [f"{x:.4f}" for x in step_val[:10]])
                if len(step_val) > 10:
                    rows[0].append("...")
                    rows[1].append("...")
                small_t = RLTable(rows)
                small_t.setStyle(_rl_table_style())
                story.append(small_t)
            elif isinstance(step_val, float):
                story.append(Paragraph(f"  = {step_val:.6f}", styles["FormulaText"]))
            elif isinstance(step_val, np.ndarray) and step_val.ndim == 2:
                story.append(Paragraph(
                    f"  [{step_val.shape[0]} × {step_val.shape[1]} matrix — see app for full display]",
                    styles["FormulaText"]
                ))
            else:
                story.append(Paragraph(f"  {str(step_val)[:300]}", styles["FormulaText"]))
            story.append(Spacer(1, 0.1 * cm))
        story.append(PageBreak())

    # ── STEP 3: FEEDBACK ────────────────────────────────────────
    story.append(_section_banner(
        "STEP 3 — FEEDBACK: PERFORMANCE EVALUATION",
        "Risk-Adjusted Metrics · Benchmark Comparison · Factor Analysis (S9)"
    ))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Performance & Risk Metrics (S9 — Performance Evaluation)",
                            styles["SubHeader"]))
    story.append(Paragraph(
        "All ratios are annualised. Benchmark: " + report_params.get("benchmark", "SPY") + ".",
        styles["BodyText2"]
    ))
    story.append(Spacer(1, 0.15 * cm))

    # Split metrics into two columns
    if not metrics_df.empty:
        half = len(metrics_df) // 2 + len(metrics_df) % 2
        left_m = metrics_df.iloc[:half]
        right_m = metrics_df.iloc[half:]
        m_left = [["Metric", "Value"]] + left_m.values.tolist()
        m_right = [["Metric", "Value"]] + right_m.values.tolist()
        # Pad if uneven
        while len(m_right) < len(m_left):
            m_right.append(["", ""])

        combined = [[RLTable(m_left, colWidths=[5 * cm, 3 * cm]),
                     Spacer(0.4 * cm, 0),
                     RLTable(m_right, colWidths=[5 * cm, 3 * cm])]]
        t_metrics = RLTable(combined, colWidths=[8.2 * cm, 0.6 * cm, 8.2 * cm])
        t_metrics_left = RLTable(m_left, colWidths=[5.5 * cm, 3 * cm])
        t_metrics_left.setStyle(_rl_table_style())
        t_metrics_right = RLTable(m_right, colWidths=[5.5 * cm, 3 * cm])
        t_metrics_right.setStyle(_rl_table_style())
        row = [[t_metrics_left, t_metrics_right]]
        t_side_by_side = RLTable(row, colWidths=[8.4 * cm, 8.6 * cm],
                                  hAlign="LEFT", spaceAfter=0.2 * cm)
        t_side_by_side.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(t_side_by_side)

    # Metric formulas reference
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "Sharpe = (Rp-Rf)/σp  ·  Sortino = (Rp-Rf)/σ_down  ·  Treynor = (Rp-Rf)/β  ·  "
        "IR = (Rp-Rb)/TE  ·  Jensen's α = Rp - [Rf + β(Rm-Rf)]  ·  Calmar = CAGR/|MaxDD|",
        styles["FormulaText"]
    ))
    story.append(Spacer(1, 0.25 * cm))

    # Equity curve
    if cum_port is not None and cum_bench is not None:
        eq_img = _pdf_chart_equity(cum_port, cum_bench, bench_label)
        _add_image_to_story(story, eq_img,
                             "Figure 4: Portfolio vs Benchmark Equity Curve (base 100)")

    # Drawdown + Rolling Sharpe side by side
    if port_ret is not None and cum_port is not None:
        dd_img = _pdf_chart_drawdown(cum_port)
        rs_img = _pdf_chart_rolling_sharpe(port_ret, rf)
        story.append(Spacer(1, 0.1 * cm))
        # Place side by side
        dd_img.seek(0)
        rs_img.seek(0)
        try:
            row2 = [[RLImage(dd_img, width=8.2 * cm, height=3.4 * cm),
                     RLImage(rs_img, width=8.2 * cm, height=3.4 * cm)]]
            t_two = RLTable(row2, colWidths=[8.4 * cm, 8.6 * cm])
            t_two.setStyle(TableStyle([
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(t_two)
            story.append(Paragraph(
                "Figure 5: Drawdown History (left)  ·  Figure 6: Rolling Sharpe Ratio (right)",
                styles["Caption"]
            ))
        except Exception:
            pass

    story.append(PageBreak())

    # CAPM & Fama-French
    story.append(Paragraph("CAPM Analysis", styles["SubHeader"]))
    story.append(Paragraph(
        "The Capital Asset Pricing Model (Sharpe, 1964) describes the relationship between "
        "systematic risk (β) and expected return: <b>E[Ri] − Rf = βᵢ · (E[Rm] − Rf)</b>. "
        "Jensen's Alpha (α) measures the abnormal return above/below the CAPM-implied return. "
        "A positive α indicates portfolio outperformance after adjusting for market risk.",
        styles["BodyText2"]
    ))
    story.append(Spacer(1, 0.15 * cm))

    # Extract CAPM metrics from metrics_df
    m_dict = {}
    if not metrics_df.empty:
        m_dict = dict(zip(metrics_df["Metric"], metrics_df["Value"]))
    capm_data = [
        ["CAPM Metric", "Value", "Interpretation"],
        ["Beta (β)", m_dict.get("Beta (vs Benchmark)", "—"),
         "< 1: Defensive  |  > 1: Aggressive"],
        ["Jensen's Alpha (ann.)", m_dict.get("Jensen's Alpha (ann.)", "—"),
         "> 0: Outperforms CAPM prediction"],
        ["Tracking Error", m_dict.get("Tracking Error (ann.)", "—"),
         "Volatility of active returns vs benchmark"],
        ["Information Ratio", m_dict.get("Information Ratio", "—"),
         "Alpha per unit of active risk (> 0.5 = good)"],
    ]
    t_capm = RLTable(capm_data, colWidths=[5 * cm, 3 * cm, 9 * cm])
    t_capm.setStyle(_rl_table_style())
    story.append(t_capm)
    story.append(Spacer(1, 0.3 * cm))

    # Fama-French
    story.append(Paragraph("Fama-French 3-Factor Regression", styles["SubHeader"]))
    story.append(Paragraph(
        "The Fama-French model (1993) extends CAPM with two additional factors: "
        "<b>SMB</b> (Small Minus Big — size premium) and <b>HML</b> (High Minus Low — value premium). "
        "The regression decomposes portfolio returns into factor exposures and a pure alpha:",
        styles["BodyText2"]
    ))
    story.append(Paragraph(
        "Rp - Rf  =  α  +  β_MKT·(Rm-Rf)  +  β_SMB·SMB  +  β_HML·HML  +  ε",
        styles["FormulaText"]
    ))

    if ff_results:
        ff_img = _pdf_chart_ff_betas(ff_results)
        _add_image_to_story(story, ff_img,
                             "Figure 7: Fama-French Factor Loadings & Regression Summary")

        ff_table_data = [
            ["Parameter", "Value", "Statistical Significance"],
            ["Alpha (daily)", f"{ff_results.get('Alpha (daily)', 0):.6f}",
             f"t = {ff_results.get('t_alpha', 0):.3f}  |  p = {ff_results.get('p_alpha', 1):.4f}"],
            ["Alpha (annualised)", f"{ff_results.get('Alpha (annualised)', 0):.2%}",
             "★ Significant at 5%" if ff_results.get("p_alpha", 1) < 0.05 else "Not significant at 5%"],
            ["β Market (MktRF)", f"{ff_results.get('β_MktRF', 0):.4f}",
             f"t = {ff_results.get('t_MktRF', 0):.3f}"],
            ["β Size (SMB)", f"{ff_results.get('β_SMB', 0):.4f}",
             f"t = {ff_results.get('t_SMB', 0):.3f}"],
            ["β Value (HML)", f"{ff_results.get('β_HML', 0):.4f}",
             f"t = {ff_results.get('t_HML', 0):.3f}"],
            ["R²", f"{ff_results.get('R²', 0):.4f}", "Fraction of return variance explained"],
            ["Adj. R²", f"{ff_results.get('Adj. R²', 0):.4f}", "R² adjusted for 3 regressors"],
            ["N observations", str(ff_results.get("N obs", "")), "Trading days in sample"],
        ]
        t_ff = RLTable(ff_table_data, colWidths=[4.5 * cm, 3.5 * cm, 9 * cm])
        t_ff.setStyle(_rl_table_style())
        story.append(t_ff)
    else:
        story.append(Paragraph(
            "Fama-French data unavailable — please check internet connection or date range.",
            styles["BodyText2"]
        ))

    # Footer
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        f"Generated by Portfolio Management Pro  ·  ESSCA  ·  Prof. Benoit Seguret  ·  {date.today()}",
        ParagraphStyle("footer", fontSize=7, fontName="Helvetica-Oblique",
                       textColor=colors.HexColor("#9CA3AF"), alignment=TA_CENTER)
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
    st.caption(f"📦 {len(candidate_tickers)} tickers in universe · app tests all, keeps most liquid")

    # Let user further refine
    max_assets = st.slider("Assets in portfolio", 5, 50, 15,
                           help="Fetches ALL region tickers, drops illiquid ones, keeps top N by data completeness")

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
         <p>Advanced Portfolio Construction & Performance Analysis · ESSCA · Prof. Benoit Seguret</p>
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

    with st.spinner("⏳ Fetching market data for all selected regions…"):
        all_tickers = get_tickers_for_regions(region_choices)

        # Fetch ALL tickers in batches of 60 to avoid yfinance timeouts
        BATCH = 60
        price_chunks = []
        for i in range(0, len(all_tickers), BATCH):
            batch = all_tickers[i : i + BATCH]
            chunk = fetch_prices(tuple(batch), str(start_date), str(end_date))
            if not chunk.empty:
                price_chunks.append(chunk)

        if not price_chunks:
            st.error("No data returned. Try different tickers or date range.")
            st.stop()

        # Merge all chunks on the date index
        prices_all = pd.concat(price_chunks, axis=1)
        prices_all = prices_all.loc[:, ~prices_all.columns.duplicated()]

        # Rank by data completeness — keep >= 70% coverage
        completeness = prices_all.notna().sum().sort_values(ascending=False)
        min_rows = int(0.70 * len(prices_all))
        good_tickers = completeness[completeness >= min_rows].index.tolist()
        prices_all = prices_all[good_tickers].ffill().dropna()

        if prices_all.empty or len(prices_all.columns) == 0:
            st.error("No tickers had sufficient data. Broaden regions or extend date range.")
            st.stop()

        # Keep the top max_assets by completeness
        top_tickers = completeness[good_tickers].head(max_assets).index.tolist()
        prices = prices_all[top_tickers]
        tickers = prices.columns.tolist()

        st.caption(
            f"✅ {len(good_tickers)} tickers had sufficient data · "
            f"keeping top {len(tickers)} by data completeness"
        )

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

            Factor proxies (ETF-based, fully self-contained):
            Mkt-RF = SPY − SHY  |  SMB = IWM − IWB (small − large cap)
            HML    = IVE − IVW  (value − growth)    |  RF = SHY daily return
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
                    cum_port=R["cum_port"],
                    cum_bench=R["cum_bench"],
                    port_ret=R["port_ret"],
                    rets=R["rets"],
                    frontier_df=R["frontier_df"],
                    gmv_point=R["gmv_point"],
                    tan_point=R["tan_point"],
                    sel_point=R["sel_point"],
                    rf=rf_rate,
                    bench_label=bench_label,
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
