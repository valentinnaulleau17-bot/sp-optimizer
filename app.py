# app_portfolio_sp500.py
# ------------------------------------------------------------
# Streamlit — S&P 500/400/600 Optimizer + Custom Portfolio
# - Weight adjustment via +/- buttons (delta equally redistributed)
# - Interactive Plotly equity curve + forward projection to horizon
# - Correlation matrix (heatmap) WITH coefficients inside each cell
# - Performance & risk metrics (Sharpe, Sortino, VaR, CVaR, Beta, Alpha, etc.)
# - Export CSV + Export PDF (summary / full report incl. equity + corr + full table split)
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from io import StringIO, BytesIO
import requests
import math

# --- Optional: Matplotlib for PDF images (recommended)
# If your IDE shows "could not be resolved", it's usually a VSCode interpreter issue.
# The app can still run; PDF will gracefully degrade if matplotlib is missing.
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

# --- PDF (ReportLab)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table as RLTable, TableStyle, Image as RLImage,
    PageBreak
)
from reportlab.lib.units import cm


# ============================================================
# App Config
# ============================================================
st.set_page_config(page_title="Portfolio Optimizer Pro", layout="wide")
TRADING_DAYS = 252


# ============================================================
# Helpers — weights
# ============================================================
def _normalize_weights(w: pd.Series) -> pd.Series:
    w = w.copy()
    w = w.clip(lower=0.0)
    s = float(w.sum())
    return (w / s) if s > 0 else w

def capm_portfolio_metrics(
    asset_rets: pd.DataFrame,
    weights: pd.Series,
    mkt_rets: pd.Series,
    rf_annual: float,
    erp_annual: float,
) -> dict:
    df = asset_rets.join(mkt_rets.rename("MKT"), how="inner").dropna()
    w = weights.reindex(asset_rets.columns).fillna(0.0)

    port = (df[asset_rets.columns] @ w).rename("PORT")
    X = df["MKT"].values
    y = port.values
    x_var = float(np.var(X, ddof=0) + 1e-12)

    beta_p = float(np.cov(y, X, ddof=0)[0, 1] / x_var)

    a = float(y.mean() - beta_p * X.mean())
    y_hat = a + beta_p * X
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum() + 1e-12)
    r2_p = 1.0 - ss_res / ss_tot

    realized_ann = float((1.0 + port).prod() ** (TRADING_DAYS / len(port)) - 1.0)
    capm_exp_ann = float(rf_annual + beta_p * erp_annual)
    alpha_ann = float(realized_ann - capm_exp_ann)

    return {
        "CAPM Beta (Portfolio)": beta_p,
        "CAPM Exp Return (ann.) (Portfolio)": capm_exp_ann,
        "CAPM Alpha (ann.) (Portfolio)": alpha_ann,
        "CAPM R² (Portfolio)": r2_p,
    }

def capm_by_asset(
    asset_rets: pd.DataFrame,
    mkt_rets: pd.Series,
    rf_annual: float,
    erp_annual: float,
) -> pd.DataFrame:
    """
    Compute CAPM metrics per asset (annualized where relevant).

    Outputs (per asset):
    - CAPM Beta
    - CAPM Expected Return (ann.) = rf + beta * ERP
    - CAPM Alpha (ann.) = realized_ann - capm_exp_ann
    - CAPM R² (regression fit vs market)
    """

    # Align dates
    df = asset_rets.join(mkt_rets.rename("MKT"), how="inner").dropna()
    X = df["MKT"].values
    x_var = float(np.var(X, ddof=0) + 1e-12)

    rf_annual = float(rf_annual)
    erp_annual = float(erp_annual)

    out = {}
    for col in asset_rets.columns:
        if col not in df.columns:
            continue

        y = df[col].values

        # CAPM beta
        beta = float(np.cov(y, X, ddof=0)[0, 1] / x_var)

        # R² of regression y ~ a + beta * X
        a = float(y.mean() - beta * X.mean())
        y_hat = a + beta * X
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum() + 1e-12)
        r2 = 1.0 - ss_res / ss_tot

        # Realized annual return (geometric)
        realized_ann = float(
            (1.0 + df[col]).prod() ** (TRADING_DAYS / len(df)) - 1.0
        )

        # CAPM expected return & alpha
        capm_exp_ann = float(rf_annual + beta * erp_annual)
        capm_alpha_ann = float(realized_ann - capm_exp_ann)

        out[col] = {
            "CAPM Beta": beta,
            "CAPM Exp Return (ann.)": capm_exp_ann,
            "CAPM Alpha (ann.)": capm_alpha_ann,
            "CAPM R²": r2,
        }

    return pd.DataFrame.from_dict(out, orient="index")

def rebalance_equal_others(w: pd.Series, changed: str, new_value: float) -> pd.Series:
    """
    Set w[changed] = new_value, redistribute delta equally across all other tickers.
    Enforces w>=0 and sum(w)=1.
    """
    w = w.copy()
    tickers = w.index.tolist()
    n = len(tickers)
    if n <= 1:
        w.iloc[0] = 1.0
        return w

    old_value = float(w[changed])
    new_value = float(np.clip(new_value, 0.0, 1.0))
    delta = new_value - old_value
    if abs(delta) < 1e-12:
        return _normalize_weights(w)

    w[changed] = new_value
    others = [t for t in tickers if t != changed]

    remaining = -delta  # must be distributed among others
    active = others.copy()

    # Robust redistribution loop (handles hitting 0 constraints)
    for _ in range(50):
        if len(active) == 0 or abs(remaining) < 1e-12:
            break

        share = remaining / len(active)

        if share >= 0:
            for t in active:
                w[t] = float(w[t]) + share
            remaining = 0.0
            break

        # share < 0 => decreasing others; some may hit 0
        new_vals = {}
        hit_zero = []
        for t in active:
            nv = float(w[t]) + share
            if nv < 0:
                nv = 0.0
                hit_zero.append(t)
            new_vals[t] = nv

        before = float(w[active].sum())
        after = float(sum(new_vals.values()))
        actually_applied = after - before  # <= 0
        remaining -= actually_applied  # remaining still to distribute

        for t, nv in new_vals.items():
            w[t] = nv

        active = [t for t in active if t not in hit_zero]

    return _normalize_weights(w)


# ============================================================
# Universe: S&P tickers
# ============================================================
@st.cache_data(show_spinner=False)
def get_sp_tickers_csv(url: str):
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return tickers, df


@st.cache_data(show_spinner=False)
def get_sp_tickers_wiki(url: str, table_idx: int = 0, symbol_col: str = "Symbol"):
    html = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
    tables = pd.read_html(StringIO(html))
    df = tables[table_idx]
    tickers = df[symbol_col].astype(str).str.replace(".", "-", regex=False).tolist()
    return tickers, df


def get_sp500_tickers():
    return get_sp_tickers_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")


def get_sp400_tickers():
    return get_sp_tickers_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", table_idx=0, symbol_col="Symbol")


def get_sp600_tickers():
    return get_sp_tickers_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", table_idx=0, symbol_col="Symbol")


# ============================================================
# Data fetching
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices(tickers, years=5):
    end = date.today()
    start = end - timedelta(days=int(years * 365.25) + 10)

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned an empty DataFrame.")

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0).unique().tolist()
        field = "Close" if "Close" in level0 else ("Adj Close" if "Adj Close" in level0 else None)
        if field is None:
            raise RuntimeError(f"Price field not found. Available fields: {level0}")
        px = data[field].copy()
    else:
        cols = data.columns.tolist()
        if "Close" in cols:
            px = data[["Close"]].copy()
        elif "Adj Close" in cols:
            px = data[["Adj Close"]].copy()
        else:
            raise RuntimeError(f"Close column not found. Available columns: {cols}")
        if len(tickers) == 1:
            px.columns = [tickers[0]]
        else:
            px.columns = ["PRICE"]

    px = px.dropna(how="all").ffill()
    px.columns = [str(c).replace(".", "-") for c in px.columns]
    px = px.dropna(axis=1, how="all").dropna(how="all")

    if px.shape[1] == 0:
        raise RuntimeError("All tickers failed to download (empty columns).")

    return px


@st.cache_data(show_spinner=False)
def fetch_fundamentals_yf(tickers):
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker": t,
                "Name": info.get("shortName"),
                "Sector": info.get("sector"),
                "MarketCap": info.get("marketCap"),
                "Beta": info.get("beta"),
                "PE_TTM": info.get("trailingPE"),
                "PE_FWD": info.get("forwardPE"),
                "PB": info.get("priceToBook"),
                "ROE": info.get("returnOnEquity"),
                "ProfitMargin": info.get("profitMargins"),
                "DebtToEquity": info.get("debtToEquity"),
                "DividendYield": info.get("dividendYield"),
            })
        except Exception:
            rows.append({"Ticker": t})
    return pd.DataFrame(rows).set_index("Ticker")


def compute_returns(prices: pd.DataFrame):
    rets = prices.pct_change().dropna()
    mu_daily = rets.mean()
    cov_daily = rets.cov()
    return rets, mu_daily, cov_daily


def zscore(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


# ============================================================
# Scenario-conditioned Monte Carlo (correlated GBM)
# ============================================================
def adjust_corr(corr: np.ndarray, shrink: float):
    n = corr.shape[0]
    ones = np.ones((n, n))
    a = float(np.clip(shrink, 0.0, 0.95))
    out = (1 - a) * corr + a * ones
    np.fill_diagonal(out, 1.0)
    return np.clip(out, -0.99, 0.99)

def mc_terminal_gbm_corr(S0: pd.Series, mu: pd.Series, sigma: np.ndarray, corr: np.ndarray,
                        years: float, n_sims: int, seed: int):
    rng = np.random.default_rng(seed)
    tickers = list(S0.index)
    n = len(tickers)

    L = _safe_cholesky(corr)
    T = float(years)

    Z = rng.standard_normal((n_sims, n)) @ L.T

    mu_vec = np.asarray(mu, dtype=float)
    sig_vec = np.asarray(sigma, dtype=float)

    drift = (mu_vec - 0.5 * sig_vec**2) * T
    diffusion = sig_vec * np.sqrt(T) * Z

    ST = S0.values[None, :] * np.exp(drift[None, :] + diffusion)
    return ST, tickers


def compute_sigma_and_corr(returns_df: pd.DataFrame, vol_mult: float, corr_shrink: float):
    cov_annual = returns_df.cov() * TRADING_DAYS
    sigma = np.sqrt(np.diag(cov_annual)) * float(vol_mult)

    corr = returns_df.corr().values
    corr_adj = adjust_corr(corr, float(corr_shrink))
    return sigma, corr_adj


def _safe_cholesky(corr: np.ndarray, max_tries: int = 10):
    eps = 1e-10
    A = corr.copy()
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            eps *= 10
            A = A.copy()
            np.fill_diagonal(A, 1.0 + eps)

    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, 1e-8, None)
    A = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(A))
    A = A / (d[:, None] * d[None, :])
    np.fill_diagonal(A, 1.0)
    return np.linalg.cholesky(A)


def mc_paths_gbm_corr(S0: pd.Series, mu: pd.Series, sigma: np.ndarray, corr: np.ndarray,
                      years: float, steps_per_year: int, n_sims: int, seed: int):
    rng = np.random.default_rng(seed)
    tickers = list(S0.index)
    n = len(tickers)
    dt = 1.0 / steps_per_year
    steps = int(years * steps_per_year)

    L = _safe_cholesky(corr)

    S = np.zeros((n_sims, steps + 1, n), dtype=float)
    S[:, 0, :] = S0.values

    mu_vec = mu.reindex(tickers).values.astype(float)
    sig_vec = np.asarray(sigma, dtype=float)

    for t in range(steps):
        Z = rng.standard_normal((n_sims, n)) @ L.T
        drift = (mu_vec - 0.5 * sig_vec**2) * dt
        diffusion = (sig_vec * np.sqrt(dt)) * Z
        S[:, t + 1, :] = S[:, t, :] * np.exp(drift + diffusion)

    return S, tickers


def compute_smart_mu(
    fund: pd.DataFrame,
    score: pd.Series,
    rf: float,
    erp: float,
    k_alpha: float,
    lambda_val: float
):
    # --- Normalisation
    if isinstance(fund, pd.Series):
        fund = fund.to_frame().T

    df = fund.copy()
    df["score"] = score.reindex(df.index)

    # --- Beta (toujours Series)
    if "Beta" not in df.columns:
        df["Beta"] = np.nan

    beta = pd.to_numeric(df["Beta"], errors="coerce")

    if np.isscalar(beta):
        beta = pd.Series(beta, index=df.index)

    beta = beta.fillna(1.0).clip(0.0, 3.0)

    # --- Score normalisé
    sc = pd.to_numeric(df["score"], errors="coerce")
    sc_z = (sc - sc.mean()) / (sc.std(ddof=0) + 1e-12)
    alpha = float(k_alpha) * sc_z.fillna(0.0)

    # --- Espérance CAPM + alpha + mean reversion
    mu_capm = rf + beta * erp
    mu_smart = mu_capm + alpha

    if lambda_val > 0:
        mu_smart = (1.0 - lambda_val) * mu_smart + lambda_val * mu_capm.mean()

    # --- GARANTIE: on retourne toujours une Series
    mu_smart = pd.to_numeric(mu_smart, errors="coerce").fillna(rf + erp)
    return mu_smart


def portfolio_projection_from_asset_paths(S_paths: np.ndarray, tickers: list[str], S0: pd.Series,
                                         weights: pd.Series, V0: float):
    w = weights.reindex(tickers).fillna(0.0).values
    w = w / (w.sum() + 1e-12)
    S0_vec = S0.reindex(tickers).values
    rel = S_paths / S0_vec[None, None, :]
    V = V0 * (rel * w[None, None, :]).sum(axis=2)
    return V


# ============================================================
# Horizon-driven selection & weight optimization
# ============================================================
def compute_cvar(series_2d: pd.DataFrame, alpha: float = 0.05):
    q = series_2d.quantile(alpha, axis=0)
    mask = series_2d.le(q, axis=1)
    return series_2d.where(mask).mean(axis=0)


def optimize_weights_terminal(term_prices: pd.DataFrame, S0: pd.Series,
                              iters: int, seed: int,
                              objective: str, risk_lambda: float,
                              rf: float = 0.0, years: float = 1.0):

    rng = np.random.default_rng(seed)
    cols = list(term_prices.columns)
    n = len(cols)

    rel = term_prices.values / S0.reindex(cols).values
    best_w, best_val = None, -1e18

    for _ in range(int(iters)):
        w = rng.random(n)
        w = w / w.sum()
        VT = (rel * w[None, :]).sum(axis=1)  # V0 = 1

        # 1) Max expected terminal value
        if objective == "Maximum Expected Return (Horizon-Based)":
            val = float(VT.mean())

        # 2) Max compounded growth (log-utility)
        elif objective == "Compounded Growth Optimization":
            val = float(np.log(VT + 1e-12).mean())

        # 3) Max Sharpe ratio (terminal, cross-scenario)
        elif objective == "Maximum Sharpe Ratio (Aggressive)":
            rT = VT - 1.0
            rf_h = (1.0 + float(rf))**float(years) - 1.0
            ex = float(rT.mean() - rf_h)
            sd = float(rT.std(ddof=0) + 1e-12)
            val = ex / sd

        # 4) Risk-adjusted return with downside protection (CVaR)
        elif objective == "Risk-Adjusted Return":
            rT = VT - 1.0
            q = np.quantile(rT, 0.05)
            cvar = float(rT[rT <= q].mean())
            tail_loss = -cvar
            val = float(rT.mean() - float(risk_lambda) * tail_loss)

        else:
            raise ValueError(f"Unknown objective: {objective}")

        if val > best_val:
            best_val = val
            best_w = w.copy()

    return pd.Series(best_w, index=cols), best_val


def portfolio_series(returns: pd.DataFrame, weights: pd.Series):
    weights = weights.reindex(returns.columns).fillna(0.0)
    port_ret = returns @ weights
    cum = (1 + port_ret).cumprod()
    return port_ret, cum


# ============================================================
# Metrics
# ============================================================
def max_drawdown(cum_curve: pd.Series) -> float:
    peak = cum_curve.cummax()
    dd = (cum_curve / peak) - 1.0
    return float(dd.min())


def sharpe_ratio(daily_ret: pd.Series, rf_annual: float) -> float:
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    ex = daily_ret - rf_daily
    return float(ex.mean() / (ex.std(ddof=0) + 1e-12) * np.sqrt(TRADING_DAYS))


def sortino_ratio(daily_ret: pd.Series, rf_annual: float) -> float:
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    ex = daily_ret - rf_daily
    downside = ex.copy()
    downside[downside > 0] = 0.0
    dd = np.sqrt((downside**2).mean())
    return float(ex.mean() / (dd + 1e-12) * np.sqrt(TRADING_DAYS))


def beta_alpha_vs_benchmark(port_daily: pd.Series, bench_daily: pd.Series, rf_annual: float):
    # beta = cov(port, bench)/var(bench), alpha annualized from CAPM line
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    y = (port_daily - rf_daily).values
    x = (bench_daily - rf_daily).values
    cov = np.cov(x, y, ddof=0)[0, 1]
    var = np.var(x, ddof=0)
    beta = cov / (var + 1e-12)
    alpha_daily = np.mean(y) - beta * np.mean(x)
    alpha_annual = (1 + alpha_daily) ** TRADING_DAYS - 1
    return float(beta), float(alpha_annual)


def var_parametric(daily_ret: pd.Series, alpha: float = 0.05):
    # Normal VaR (loss as positive)
    mu = float(daily_ret.mean())
    sigma = float(daily_ret.std(ddof=0))
    z = float(pd.Series(np.random.normal(size=200000)).quantile(alpha))  # stable approx
    # VaR on return distribution: quantile of returns
    q = mu + z * sigma
    return float(-q)  # loss positive


def var_historical(daily_ret: pd.Series, alpha: float = 0.05):
    q = float(daily_ret.quantile(alpha))
    return float(-q)


def cvar_historical(daily_ret: pd.Series, alpha: float = 0.05):
    q = float(daily_ret.quantile(alpha))
    tail = daily_ret[daily_ret <= q]
    return float(-tail.mean()) if len(tail) else float("nan")


# ============================================================
# Plotly — correlation heatmap with coefficients inside cells
# ============================================================
def plot_corr_heatmap_with_text(corr: pd.DataFrame, title: str):
    z = corr.values
    txt = np.vectorize(lambda x: f"{x:.2f}")(z)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1, zmax=1,
            colorscale="Blues",
            colorbar=dict(title="Corr", thickness=12),
            hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>"
        )
    )
    # text layer
    fig.add_trace(
        go.Scatter(
            x=np.repeat(corr.columns.values, len(corr.index)),
            y=np.tile(corr.index.values, len(corr.columns)),
            mode="text",
            text=txt.flatten(),
            textfont=dict(size=12, color="black"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=40, t=50, b=40),
        height=520,
    )
    fig.update_xaxes(tickangle=-45)
    return fig


# ============================================================
# PDF helpers
# ============================================================
def _df_to_rl_table(df: pd.DataFrame, title: str, max_cols_per_block: int = 8):
    """
    Build one or multiple ReportLab tables to avoid width overflow.
    Strategy:
      - If too many columns, split columns into blocks and stack them vertically.
      - Use small font + repeat header.
    Returns list of Flowables.
    """
    styles = getSampleStyleSheet()
    flow = [Paragraph(f"<b>{title}</b>", styles["Heading3"]), Spacer(1, 0.2 * cm)]

    df2 = df.copy()
    df2 = df2.reset_index()

    # Split columns into blocks if needed
    cols = df2.columns.tolist()
    blocks = [cols[i:i + max_cols_per_block] for i in range(0, len(cols), max_cols_per_block)]

    for bi, bcols in enumerate(blocks, start=1):
        block_df = df2[bcols].copy()

        # Convert to strings (prevents reportlab trying to interpret floats weirdly)
        table_data = [bcols] + block_df.astype(object).fillna("").values.tolist()

        t = RLTable(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        if len(blocks) > 1:
            flow.append(Paragraph(f"<i>Columns block {bi}/{len(blocks)}</i>", styles["Normal"]))
            flow.append(Spacer(1, 0.15 * cm))

        flow.append(t)
        flow.append(Spacer(1, 0.4 * cm))

    return flow


def _matplotlib_save_equity_png(hist_idx, cum_hist, future_idx, V_med, V_p05, V_p95):
    if not MATPLOTLIB_OK:
        return None

    fig = plt.figure(figsize=(10, 4))
    plt.plot(hist_idx, cum_hist.values, label="Historical (cumulative)")
    if future_idx is not None and V_med is not None:
        plt.plot(future_idx, V_med[1:], label="Median projection")
        plt.fill_between(future_idx, V_p05[1:], V_p95[1:], alpha=0.2, label="5%-95% band")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Value (base 1)")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


def _matplotlib_save_corr_png(corr: pd.DataFrame, title: str):
    if not MATPLOTLIB_OK:
        return None

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)

    # Annotate coefficients in each cell
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, fraction=0.02, pad=0.02)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf_summary(title: str, params: dict, weights_opt: pd.Series, weights_final: pd.Series,
                      metrics_perf: pd.DataFrame, metrics_risk: pd.DataFrame) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("<b>Parameters</b>", styles["Heading2"]))
    p_lines = "<br/>".join([f"{k}: {v}" for k, v in params.items()])
    story.append(Paragraph(p_lines, styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    # Weights (opt + final)
    w_df = pd.DataFrame({
        "Weight_OPT": weights_opt.reindex(weights_final.index).fillna(0.0),
        "Weight_FINAL": weights_final,
    }).reset_index().rename(columns={"index": "Ticker"})
    story += _df_to_rl_table(w_df, "Portfolio weights (optimal vs final)", max_cols_per_block=6)

    story.append(Spacer(1, 0.2 * cm))
    story += _df_to_rl_table(metrics_perf, "Performance metrics", max_cols_per_block=6)
    story += _df_to_rl_table(metrics_risk, "Risk metrics", max_cols_per_block=6)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


def build_pdf_full_report(title: str, params: dict,
                          weights_opt: pd.Series, weights_final: pd.Series,
                          equity_png: BytesIO | None,
                          corr_png: BytesIO | None,
                          metrics_perf: pd.DataFrame, metrics_risk: pd.DataFrame,
                          details_table: pd.DataFrame) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("<b>Parameters</b>", styles["Heading2"]))
    p_lines = "<br/>".join([f"{k}: {v}" for k, v in params.items()])
    story.append(Paragraph(p_lines, styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    # Equity curve image
    story.append(Paragraph("<b>Equity curve (historical + projection)</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    if equity_png is not None:
        img = RLImage(equity_png, width=24 * cm, height=8.5 * cm)
        story.append(img)
    else:
        story.append(Paragraph("Equity curve image unavailable (matplotlib not installed).", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    # Correlation image
    story.append(Paragraph("<b>Correlation matrix (historical)</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    if corr_png is not None:
        img = RLImage(corr_png, width=24 * cm, height=9.0 * cm)
        story.append(img)
    else:
        story.append(Paragraph("Correlation matrix image unavailable (matplotlib not installed).", styles["Normal"]))
    story.append(PageBreak())

    # Metrics
    story.append(Paragraph("<b>Performance & Risk metrics</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    story += _df_to_rl_table(metrics_perf, "Performance metrics", max_cols_per_block=7)
    story += _df_to_rl_table(metrics_risk, "Risk metrics", max_cols_per_block=7)

    story.append(PageBreak())

    # Portfolio details — IMPORTANT: avoid truncation by splitting columns into blocks and stacking
    story.append(Paragraph("<b>Portfolio details (table)</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    story += _df_to_rl_table(details_table, "Portfolio details", max_cols_per_block=7)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# ============================================================
# UI — Sidebar
# ============================================================
st.title("Portfolio Optimizer Pro (S&P 500/400/600 + Custom)")

with st.sidebar:
    st.header("Parameters")

    years_hist = st.slider("History (years)", 2, 10, 5)
    target_n = st.slider("Number of assets (Top N)", 5, 20, 10)
    iters = st.slider("Optimization iterations", 5000, 60000, 25000, step=5000)

    st.divider()
    st.subheader("Universe")
    universe_choice = st.multiselect(
        "Select indexes",
        ["S&P 500 (Large)", "S&P 400 (Mid)", "S&P 600 (Small)"],
        default=["S&P 500 (Large)"]
    )

    st.subheader("Portfolio mode")
    mode = st.radio("Choose mode", ["Optimization (S&P)", "Custom portfolio"], index=0)

    custom_tickers_input = ""
    if mode == "Custom portfolio":
        custom_tickers_input = st.text_area(
            "Custom tickers (comma-separated, e.g., AAPL, MSFT, AMZN)",
            value="AAPL, MSFT, GOOGL, AMZN, MCD, UNH, JNJ, JPM, XOM, KO"
        )

    st.divider()
    st.subheader("Horizon & Objective")
    years_fwd = st.slider("Projection horizon (years)", 1, 5, 2)
    candidates_n = st.slider("Candidate pre-selection (scalable)", 30, 200, 80, step=10)

    objective = st.selectbox(
        "Objective",
        ["Risk-Adjusted Return", "Maximum Expected Return (Horizon-Based)", "Compounded Growth Optimization","Maximum Sharpe Ratio (Aggressive)"]
    )
    risk_lambda = st.slider("λ (risk aversion / CVaR)", 0.0, 5.0, 1.0, 0.1)

    st.divider()
    st.subheader("Scenario (macro)")
    scenario = st.selectbox("Macro scenario", ["Base", "Stress (rates+inflation)", "Risk-on (rates down)"])

    SCENARIOS = {
        "Base": {"rf": 0.018, "erp": 0.048, "vol_mult": 0.68, "corr_shrink": 0.20},
        "Stress (rates+inflation)": {"rf": 0.045, "erp": 0.065, "vol_mult": 1.35, "corr_shrink": 0.35},
        "Risk-on (rates down)": {"rf": 0.02, "erp": 0.040, "vol_mult": 0.85, "corr_shrink": 0.10},
    }
    scen = SCENARIOS[scenario]

    rf_scen = st.slider("Scenario risk-free (rf)", 0.00, 0.10, float(scen["rf"]), 0.005)
    erp_scen = st.slider("Scenario equity risk premium (ERP)", 0.00, 0.10, float(scen["erp"]), 0.005)
    vol_mult = st.slider("Volatility multiplier", 0.50, 2.00, float(scen["vol_mult"]), 0.05)
    corr_shrink = st.slider("Correlation stress (shrink → 1)", 0.00, 0.70, float(scen["corr_shrink"]), 0.05)

    k_alpha = st.slider("Factor alpha scale (k)", 0.00, 0.08, 0.03, 0.005)
    lambda_val = st.slider("Valuation mean reversion (lambda)", 0.00, 0.50, 0.20, 0.05)

    st.divider()
    st.subheader("Monte Carlo")
    n_sims = st.slider("Simulations", 1000, 20000, 5000, step=1000)

    st.divider()
    st.subheader("Weight adjustment")
    w_step = st.number_input("Step per click (+/-)", min_value=0.001, max_value=0.20, value=0.03, step=0.01, format="%.3f")

    # --- run control (prevents full recompute on every +/- click)
    def _trigger_run():
        st.session_state["do_run"] = True

    st.button("🚀 Run calculation", type="primary", on_click=_trigger_run)
    st.caption("First run can be slow (prices + fundamentals).")


st.info("Pick a horizon + macro scenario. The model selects assets and weights for that horizon. You can then adjust weights (+/-) with equal redistribution.")


# ============================================================
# Compute ONLY when explicitly requested
# ============================================================
do_run = bool(st.session_state.get("do_run", False))

# Stored results (to keep UI reactive without recompute)
RESULT_KEYS = [
    "prices", "returns10", "top", "w_opt", "w_final",
    "fund10", "table_details",
    "cum_hist", "port_ret_hist",
    "V_med", "V_p05", "V_p95", "future_idx",
    "corr_hist",
    "bench_cum", "bench_ret",
    "metrics_perf", "metrics_risk",
    "pdf_summary_bytes", "pdf_full_bytes"
]

if do_run:
    status = st.status("Starting…", expanded=True)

    try:
        # 1/7 — Build universe
        status.write("1/7 — Building universe / portfolio list")
        universe = []

        if mode == "Optimization (S&P)":
            if "S&P 500 (Large)" in universe_choice:
                t, _ = get_sp500_tickers()
                universe += t
            if "S&P 400 (Mid)" in universe_choice:
                try:
                    t, _ = get_sp400_tickers()
                    universe += t
                except Exception as e:
                    st.warning(f"S&P 400 unavailable: {e}")
            if "S&P 600 (Small)" in universe_choice:
                try:
                    t, _ = get_sp600_tickers()
                    universe += t
                except Exception as e:
                    st.warning(f"S&P 600 unavailable: {e}")

            universe = sorted(set(universe))
            if len(universe) < 10:
                raise RuntimeError("Universe too small; select at least one index with enough constituents.")

        else:
            # Custom portfolio tickers
            raw = (custom_tickers_input or "").strip()
            if len(raw) == 0:
                raise RuntimeError("Please provide custom tickers.")
            universe = [x.strip().upper().replace(".", "-") for x in raw.split(",") if x.strip()]
            universe = sorted(set(universe))
            if len(universe) < 2:
                raise RuntimeError("Please provide at least 2 tickers for a portfolio.")

        status.write(f"✔ Universe size: {len(universe)}")

        # 2/7 — Prices
        status.write("2/7 — Downloading prices (EOD)")
        prices_all = fetch_prices(universe, years=years_hist)

        # Benchmark: use SPY as S&P500 proxy (more reliable than ^SPX on yfinance)
        bench_px = fetch_prices(["SPY"], years=years_hist)
        bench_px.columns = ["SPY"]

        status.write(f"✔ Prices loaded: {prices_all.shape[1]} tickers, {prices_all.shape[0]} dates")

        # 3/7 — Filter minimal history
        status.write("3/7 — Cleaning (minimum history)")
        min_obs = int(0.80 * len(prices_all))
        valid_cols = [c for c in prices_all.columns if prices_all[c].count() >= min_obs]
        prices = prices_all[valid_cols].copy()

        # Keep benchmark aligned
        common_idx = prices.index.intersection(bench_px.index)
        prices = prices.loc[common_idx]
        bench_px = bench_px.loc[common_idx]

        bench_rets = bench_px["SPY"].pct_change().dropna()

        status.write(f"✔ After filter: {prices.shape[1]} tickers")
        if prices.shape[1] < 2:
            raise RuntimeError("After filtering, fewer than 2 tickers remain with usable data.")

        # 4/7 — Pre-selection score (only in Optimization mode)
        status.write("4/7 — Fast scoring (candidate pre-selection)")

        returns_all, mu_daily_all, _ = compute_returns(prices)

        mu_annual_all = (1 + mu_daily_all) ** TRADING_DAYS - 1
        vol_annual_all = returns_all.std() * np.sqrt(TRADING_DAYS)
        sharpe_like = (mu_annual_all - rf_scen) / (vol_annual_all + 1e-12)

        lookback = min(TRADING_DAYS, len(prices) - 1)
        mom_1y = prices.iloc[-1] / prices.iloc[-lookback] - 1

        score_fast = (0.50 * zscore(sharpe_like) + 0.50 * zscore(mom_1y)).replace([np.inf, -np.inf], np.nan)
        score_fast = score_fast.fillna(score_fast.median())

        if mode == "Custom portfolio":
            candidates = prices.columns.tolist()
        else:
            ranked = score_fast.sort_values(ascending=False)
            candidates = ranked.head(int(min(candidates_n, len(ranked)))).index.tolist()

        status.write(f"✔ Candidates: {len(candidates)}")

        # 5/7 — Scenario + Monte Carlo on candidates
        status.write("5/7 — Scenario estimation + Monte Carlo (candidates)")
        pricesC = prices[candidates]
        returnsC, _, _ = compute_returns(pricesC)
        fundC = fetch_fundamentals_yf(candidates)

        mu_smart_C = compute_smart_mu(
            fund=fundC,
            score=score_fast.reindex(candidates),
            rf=rf_scen,
            erp=erp_scen,
            k_alpha=k_alpha,
            lambda_val=lambda_val
        )

        sigmaC, corrC = compute_sigma_and_corr(returnsC, vol_mult=vol_mult, corr_shrink=corr_shrink)
        S0C = pricesC.iloc[-1].astype(float)

        ST, tickersC = mc_terminal_gbm_corr(
            S0=S0C.reindex(candidates),
            mu=mu_smart_C.reindex(candidates),
            sigma=sigmaC,
            corr=corrC,
            years=float(years_fwd),
            n_sims=int(n_sims),
            seed=42
        )

        terminal_prices = pd.DataFrame(ST, columns=tickersC)
        terminal_returns = terminal_prices.div(S0C.reindex(tickersC), axis=1) - 1.0

        E = terminal_returns.mean(axis=0)
        logE = np.log1p(terminal_returns).mean(axis=0)
        cvar05 = compute_cvar(terminal_returns, alpha=0.05)

        if objective == "Max E[VT]":
            asset_score = E
        elif objective == "Max E[log(VT)]":
            asset_score = logE
        else:
            asset_score = E - float(risk_lambda) * (-cvar05)

        # Select top N
        top = asset_score.sort_values(ascending=False).head(int(min(target_n, len(asset_score)))).index.tolist()
        if len(top) < 2:
            raise RuntimeError("Top selection produced fewer than 2 assets.")

        status.write(f"✔ Top {len(top)} selected for horizon {years_fwd}Y (objective: {objective})")

        # 6/7 — Optimize weights (Optimization mode) or optimized weights (Custom mode)
        status.write("6/7 — Weight optimization / initialization")

        term_top = terminal_prices[top]
        S0_top = S0C.reindex(top)

        w_opt, best_obj = optimize_weights_terminal(
            term_prices=term_top,
            S0=S0_top,
            iters=int(iters),
            seed=123,
            objective=objective,
            risk_lambda=float(risk_lambda),
            rf=float(rf_scen),
            years=float(years_fwd),
        )

        w_opt = _normalize_weights(w_opt.reindex(top).fillna(0.0))

        # Initialize live weights in session (final weights start = optimal)
        st.session_state["w_live"] = w_opt.copy()
        st.session_state["w_live_tickers"] = top.copy()

        # 7/7 — Build full analytics with final weights (initially = w_opt)
        status.write("7/7 — Building tables, equity curve, projection, metrics, PDFs")

        # Historical series
        prices10 = prices[top].copy()
        returns10, mu10_daily, cov10_daily = compute_returns(prices10)

        # Benchmark returns aligned
        bench_ret = bench_px["SPY"].pct_change().dropna()
        returns10_aligned = returns10.join(bench_ret.rename("SPY"), how="inner")
        bench_ret = returns10_aligned["SPY"]
        returns10_aligned = returns10_aligned.drop(columns=["SPY"])


        # Scenario mu for top (from candidates)
        mu_top_scen = mu_smart_C.reindex(top)

        # Re-run MC on top for stable trajectory
        sigma10, corr10 = compute_sigma_and_corr(returns10, vol_mult=vol_mult, corr_shrink=corr_shrink)
        S0_10 = prices10.iloc[-1].astype(float)

        S_paths_10, tickers10 = mc_paths_gbm_corr(
            S0=S0_10.reindex(top),
            mu=mu_top_scen.reindex(top),
            sigma=sigma10,
            corr=corr10,
            years=float(years_fwd),
            steps_per_year=TRADING_DAYS,
            n_sims=int(n_sims),
            seed=7
        )

        # initial final weights = optimal at run time
        w_final = st.session_state["w_live"].copy()

        port_ret_hist, cum_hist = portfolio_series(returns10_aligned, w_final)
        bench_cum = (1 + bench_ret).cumprod()

        V0 = float(cum_hist.iloc[-1])

        V_paths = portfolio_projection_from_asset_paths(
            S_paths=S_paths_10,
            tickers=tickers10,
            S0=S0_10.reindex(tickers10),
            weights=w_final,
            V0=V0
        )

        V_med = np.median(V_paths, axis=0)
        V_p05 = np.percentile(V_paths, 5, axis=0)
        V_p95 = np.percentile(V_paths, 95, axis=0)

        hist_idx = cum_hist.index
        future_idx = pd.bdate_range(hist_idx[-1] + pd.Timedelta(days=1), periods=len(V_med) - 1)

        # Correlation matrix (historical) incl benchmark
        corr_assets = returns10_aligned.copy()
        corr_assets["SPY"] = bench_ret
        corr_hist = corr_assets.corr()

        # Fundamentals + scenario stats
        fund10 = fetch_fundamentals_yf(top)

        mu_annual_hist = (1 + returns10_aligned.mean()) ** TRADING_DAYS - 1
        sig_annual_hist = returns10_aligned.std() * np.sqrt(TRADING_DAYS)

        # Terminal return stats
        terminal_prices_top = pd.DataFrame(S_paths_10[:, -1, :], columns=tickers10)
        terminal_returns_top = terminal_prices_top.div(S0_10.reindex(tickers10), axis=1) - 1.0
        E_top = terminal_returns_top.mean(axis=0)
        Vol_top = terminal_returns_top.std(axis=0)
        CVaR_top = compute_cvar(terminal_returns_top, alpha=0.05)

        # Table details
        rows = []
        for t in top:
            j = tickers10.index(t)
            term_p = S_paths_10[:, -1, j]
            rows.append({
                "Ticker": t,
                "Weight_FINAL": float(w_final[t]),
                "Weight_OPT": float(w_opt[t]),
                "LastPrice": float(S0_10[t]),
                "Mu_Ann_Scenario": float(mu_top_scen[t]),
                f"E[Ret]_{years_fwd}Y": float(E_top[t]),
                f"Vol[Ret]_{years_fwd}Y": float(Vol_top[t]),
                f"CVaR5%[Ret]_{years_fwd}Y": float(CVaR_top[t]),
                "ExpReturn_Ann_Hist": float(mu_annual_hist[t]),
                "Vol_Ann_Hist": float(sig_annual_hist[t]),
                f"ForecastPrice_{years_fwd}Y_P05": float(np.percentile(term_p, 5)),
                f"ForecastPrice_{years_fwd}Y_Med": float(np.percentile(term_p, 50)),
                f"ForecastPrice_{years_fwd}Y_P95": float(np.percentile(term_p, 95)),
            })

        forecast_df = pd.DataFrame(rows).set_index("Ticker")
        capm_df = capm_by_asset(asset_rets=returns10[top], mkt_rets=bench_ret, rf_annual=float(rf_scen), erp_annual=float(erp_scen))
        table_details = fund10.join(forecast_df, how="right")
        table_details = table_details.join(capm_df, how="left")

        # Metrics
        cagr = float(cum_hist.iloc[-1] ** (TRADING_DAYS / len(cum_hist)) - 1)
        vol_ann = float(port_ret_hist.std(ddof=0) * np.sqrt(TRADING_DAYS))
        sharpe = sharpe_ratio(port_ret_hist, rf_scen)
        sortino = sortino_ratio(port_ret_hist, rf_scen)
        mdd = max_drawdown(cum_hist)
        beta, alpha_ann = beta_alpha_vs_benchmark(port_ret_hist, bench_ret, rf_scen)


        var95_p = var_parametric(port_ret_hist, 0.05)
        var99_p = var_parametric(port_ret_hist, 0.01)
        var95_h = var_historical(port_ret_hist, 0.05)
        cvar95 = cvar_historical(port_ret_hist, 0.05)
        cvar99 = cvar_historical(port_ret_hist, 0.01)

        metrics_perf = pd.DataFrame({
            "Metric": [
                "CAGR", "Volatility (ann.)", "Sharpe Ratio", "Sortino Ratio",
                "Beta", "Alpha (ann.)",
                "Max Drawdown"
            ],
            "Value": [
                f"{cagr:.2%}",
                f"{vol_ann:.2%}",
                f"{sharpe:.3f}",
                f"{sortino:.3f}",
                f"{beta:.3f}",
                f"{alpha_ann:.2%}",
                f"{abs(mdd):.2%}",
            ]
        })

        metrics_risk = pd.DataFrame({
            "Metric": ["VaR 95% (Parametric)", "VaR 99% (Parametric)", "VaR 95% (Historical)", "CVaR 95%", "CVaR 99%"],
            "Value": [f"{var95_p:.2%}", f"{var99_p:.2%}", f"{var95_h:.2%}", f"{cvar95:.2%}", f"{cvar99:.2%}"]
        })

        # PDFs (images via matplotlib)
        eq_png = _matplotlib_save_equity_png(hist_idx, cum_hist, future_idx, V_med, V_p05, V_p95)
        corr_png = _matplotlib_save_corr_png(corr_hist, "Correlation matrix (historical)")

        params = {
            "Mode": mode,
            "Indexes": ", ".join(universe_choice) if mode == "Optimization (S&P)" else "Custom",
            "Objective": objective,
            "Horizon": f"{years_fwd} years",
            "Scenario": scenario,
            "rf": f"{rf_scen:.2%}",
            "ERP": f"{erp_scen:.2%}",
            "Vol multiplier": f"{vol_mult:.2f}",
            "Corr shrink": f"{corr_shrink:.2f}",
            "Simulations": f"{n_sims}",
        }

        pdf_summary = build_pdf_summary(
            title="Portfolio Summary",
            params=params,
            weights_opt=w_opt,
            weights_final=w_final,
            metrics_perf=metrics_perf,
            metrics_risk=metrics_risk
        )

        pdf_full = build_pdf_full_report(
            title="Portfolio Full Report",
            params=params,
            weights_opt=w_opt,
            weights_final=w_final,
            equity_png=eq_png,
            corr_png=corr_png,
            metrics_perf=metrics_perf,
            metrics_risk=metrics_risk,
            details_table=table_details
        )

        # Persist results
        st.session_state["prices"] = prices
        st.session_state["returns10"] = returns10_aligned
        st.session_state["top"] = top
        st.session_state["w_opt"] = w_opt
        st.session_state["w_final"] = w_final
        st.session_state["fund10"] = fund10
        st.session_state["table_details"] = table_details
        st.session_state["cum_hist"] = cum_hist
        st.session_state["port_ret_hist"] = port_ret_hist
        st.session_state["V_med"] = V_med
        st.session_state["V_p05"] = V_p05
        st.session_state["V_p95"] = V_p95
        st.session_state["future_idx"] = future_idx
        st.session_state["corr_hist"] = corr_hist
        st.session_state["bench_cum"] = bench_cum
        st.session_state["bench_ret"] = bench_ret
        st.session_state["metrics_perf"] = metrics_perf
        st.session_state["metrics_risk"] = metrics_risk
        st.session_state["pdf_summary_bytes"] = pdf_summary
        st.session_state["pdf_full_bytes"] = pdf_full

        # IMPORTANT: stop re-running automatically
        st.session_state["do_run"] = False

        status.update(label="✅ Done", state="complete", expanded=False)

    except Exception as e:
        st.session_state["do_run"] = False
        status.update(label="❌ Error during execution", state="error", expanded=True)
        st.exception(e)


# ============================================================
# UI — Display (uses stored results; no recompute on +/- clicks)
# ============================================================
if st.session_state.get("top") is not None and st.session_state.get("prices") is not None:
    top = st.session_state["top"]
    w_opt = st.session_state["w_opt"].copy()

    # Ensure w_live exists
    if "w_live" not in st.session_state or st.session_state.get("w_live_tickers") != top:
        st.session_state["w_live"] = _normalize_weights(w_opt.copy())
        st.session_state["w_live_tickers"] = top.copy()

    # --- 1) Portfolio weights + +/- controls
    st.subheader("1) Portfolio (assets + weights)")
    c1, c2 = st.columns([1, 2])

    with c1:
        st.write(f"**Mode:** {mode}")
        st.write(f"**Objective:** {objective}")
        st.write(f"**Horizon:** {years_fwd} year(s)")
        st.write("**Optimized weights (baseline)**")
        st.dataframe(w_opt.sort_values(ascending=False).to_frame("Weight_OPT"), use_container_width=True)

    def _apply_click(ticker: str, direction: int):
        w = st.session_state["w_live"].reindex(top).fillna(0.0).copy()
        step = float(w_step)
        new_val = float(np.clip(float(w[ticker]) + direction * step, 0.0, 1.0))
        st.session_state["w_live"] = rebalance_equal_others(w, changed=ticker, new_value=new_val)

    with c2:
        st.write("Adjust weights with +/- (delta equally redistributed across other assets):")
        w_live = _normalize_weights(st.session_state["w_live"].reindex(top).fillna(0.0))

        # Controls table-like layout
        header = st.columns([2.2, 1.0, 0.8, 0.8])
        header[0].markdown("**Ticker**")
        header[1].markdown("**Current**")
        header[2].markdown("**-**")
        header[3].markdown("**+**")

        for t in top:
            row = st.columns([2.2, 1.0, 0.8, 0.8])
            row[0].write(t)
            row[1].write(f"{w_live[t]:.4f}")

            row[2].button("−", key=f"minus_{t}", on_click=_apply_click, args=(t, -1), use_container_width=True)
            row[3].button("+", key=f"plus_{t}", on_click=_apply_click, args=(t, +1), use_container_width=True)

        w_final = _normalize_weights(st.session_state["w_live"].reindex(top).fillna(0.0))
        st.session_state["w_final"] = w_final

        st.write("**Final weights**")
        st.dataframe(w_final.sort_values(ascending=False).to_frame("Weight_FINAL"), use_container_width=True)

    # --- Recompute downstream views based on w_final ONLY (no yfinance calls)
    prices = st.session_state["prices"]
    returns10 = st.session_state["returns10"]
    bench_ret = st.session_state["bench_ret"]

    port_ret_hist, cum_hist = portfolio_series(returns10, st.session_state["w_final"])
    bench_cum = (1 + bench_ret.loc[returns10.index]).cumprod()

    # --- 2) Interactive equity curve + projection horizon
    st.subheader("2) Equity Curve (interactive)")
    cagr = float(cum_hist.iloc[-1] ** (TRADING_DAYS / len(cum_hist)) - 1)
    vol_ann = float(port_ret_hist.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = sharpe_ratio(port_ret_hist, rf_scen)
    mdd = max_drawdown(cum_hist)

    # Projection arrays from previous MC run (kept stable)
    V_med = st.session_state.get("V_med")
    V_p05 = st.session_state.get("V_p05")
    V_p95 = st.session_state.get("V_p95")
    future_idx = st.session_state.get("future_idx")

    # Build Plotly figure (historical + projection band + benchmark)
    fig = go.Figure()

    # Historical normalized to 100
    hist_base = 100.0
    port_hist_plot = (cum_hist / cum_hist.iloc[0]) * hist_base
    bench_hist_plot = (bench_cum / bench_cum.iloc[0]) * hist_base

    fig.add_trace(go.Scatter(
        x=port_hist_plot.index, y=port_hist_plot.values,
        mode="lines", name="Portfolio (hist.)"
    ))
    fig.add_trace(go.Scatter(
        x=bench_hist_plot.index, y=bench_hist_plot.values,
        mode="lines", name="Benchmark (SPY)"
    ))

    # Forward projection to horizon (base = last hist point)
    if future_idx is not None and V_med is not None:
        last = float(port_hist_plot.iloc[-1])
        proj_med = last * (V_med[1:] / V_med[0])
        proj_p05 = last * (V_p05[1:] / V_p05[0])
        proj_p95 = last * (V_p95[1:] / V_p95[0])

        fig.add_trace(go.Scatter(
            x=future_idx, y=proj_med,
            mode="lines", name=f"Projection median ({years_fwd}Y)", line=dict(dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([future_idx.values, future_idx.values[::-1]]),
            y=np.concatenate([proj_p95, proj_p05[::-1]]),
            fill="toself",
            name="Projection band (5%-95%)",
            opacity=0.25,
            line=dict(width=0),
            showlegend=True
        ))

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date",
        yaxis_title="Index (base 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # KPI row (as in your screenshot)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CAGR", f"{cagr:.2%}")
    k2.metric("Volatility", f"{vol_ann:.2%}")
    k3.metric("Sharpe Ratio", f"{sharpe:.3f}")
    k4.metric("Max Drawdown", f"{abs(mdd):.2%}")

    # --- 2bis) Correlation matrix with coefficients INSIDE cells
    st.subheader("2bis) Correlation matrix (historical)")
    corr_hist = st.session_state["corr_hist"].copy()

    # Keep SPY at end if present
    if "SPY" in corr_hist.columns:
        cols = [c for c in corr_hist.columns if c != "SPY"] + ["SPY"]
        corr_hist = corr_hist.loc[cols, cols]

    corr_fig = plot_corr_heatmap_with_text(corr_hist, "Correlation matrix (historical)")
    st.plotly_chart(corr_fig, use_container_width=True)
    st.caption("Note: SPY is used as S&P 500 benchmark proxy.")

    # --- 3) Performance & Risk metrics (recomputed with new weights)
    st.subheader("3) Performance & Risk Metrics")
    beta, alpha_ann = beta_alpha_vs_benchmark(port_ret_hist, bench_ret.loc[returns10.index], rf_scen)

    metrics_perf = pd.DataFrame({
        "Metric": ["Sortino Ratio", "Beta", "Alpha (annual)", "Treynor Ratio", "Information Ratio"],
        "Value": [
            f"{sortino_ratio(port_ret_hist, rf_scen):.3f}",
            f"{beta:.3f}",
            f"{alpha_ann:.2%}",
            f"{( (port_ret_hist.mean() * TRADING_DAYS) / (beta + 1e-12) ):.3f}",
            f"{((port_ret_hist - bench_ret.loc[returns10.index]).mean() / ((port_ret_hist - bench_ret.loc[returns10.index]).std(ddof=0)+1e-12) * np.sqrt(TRADING_DAYS)):.3f}",
        ]
    })

    metrics_risk = pd.DataFrame({
        "Metric": ["VaR 95% (Parametric)", "VaR 99% (Parametric)", "VaR 95% (Historical)", "CVaR 95%", "CVaR 99%"],
        "Value": [
            f"{var_parametric(port_ret_hist, 0.05):.2%}",
            f"{var_parametric(port_ret_hist, 0.01):.2%}",
            f"{var_historical(port_ret_hist, 0.05):.2%}",
            f"{cvar_historical(port_ret_hist, 0.05):.2%}",
            f"{cvar_historical(port_ret_hist, 0.01):.2%}",
        ]
    })

    a, b = st.columns(2)
    with a:
        st.markdown("**Performance Metrics**")
        st.dataframe(metrics_perf, use_container_width=True)
    with b:
        st.markdown("**Risk Metrics**")
        st.dataframe(metrics_risk, use_container_width=True)

    # --- 4) Portfolio details table (weights updated)
    st.subheader("4) Portfolio details (table)")
    table_details = st.session_state["table_details"].copy()
    table_details["Weight_FINAL"] = st.session_state["w_final"].reindex(table_details.index).fillna(0.0)
    table_details["Weight_OPT"] = st.session_state["w_opt"].reindex(table_details.index).fillna(0.0)
    # Bring weights columns first
    cols = ["Weight_FINAL", "Weight_OPT"] + [c for c in table_details.columns if c not in ["Weight_FINAL", "Weight_OPT"]]
    table_details = table_details[cols]
    st.dataframe(table_details, use_container_width=True)

    # --- Export section (PDFs rebuilt with updated weights + updated equity/corr images)
    st.subheader("5) Export")
    e1, e2, e3 = st.columns([1, 1, 1])

    # CSV weights
    weights_csv = pd.DataFrame({
        "Ticker": top,
        "Weight_OPT": st.session_state["w_opt"].reindex(top).values,
        "Weight_FINAL": st.session_state["w_final"].reindex(top).values
    })
    csv_bytes = weights_csv.to_csv(index=False).encode("utf-8")

    with e1:
        st.download_button(
            "Download portfolio weights (CSV)",
            data=csv_bytes,
            file_name="portfolio_weights.csv",
            mime="text/csv"
        )

    # Rebuild PDFs with updated weights + updated metrics
    params = {
        "Mode": mode,
        "Indexes": ", ".join(universe_choice) if mode == "Optimization (S&P)" else "Custom",
        "Objective": objective,
        "Horizon": f"{years_fwd} years",
        "Scenario": scenario,
        "rf": f"{rf_scen:.2%}",
        "ERP": f"{erp_scen:.2%}",
        "Vol multiplier": f"{vol_mult:.2f}",
        "Corr shrink": f"{corr_shrink:.2f}",
        "Simulations": f"{n_sims}",
    }

    # PDF images updated (correlation WITH coefficients + equity curve with projection)
    hist_idx = cum_hist.index
    V0_last = float(cum_hist.iloc[-1])

    # For PDF, we use matplotlib images (if available)
    # Projection arrays: V_med etc from initial MC, scaled in plot; image uses those arrays directly
    eq_png = _matplotlib_save_equity_png(hist_idx, cum_hist, st.session_state.get("future_idx"),
                                         st.session_state.get("V_med"), st.session_state.get("V_p05"), st.session_state.get("V_p95"))

    # Correlation image WITH coefficients inside
    corr_png = _matplotlib_save_corr_png(corr_hist, "Correlation matrix (historical)")

    pdf_summary = build_pdf_summary(
        title="Portfolio Summary",
        params=params,
        weights_opt=st.session_state["w_opt"],
        weights_final=st.session_state["w_final"],
        metrics_perf=metrics_perf,
        metrics_risk=metrics_risk
    )

    pdf_full = build_pdf_full_report(
        title="Portfolio Full Report",
        params=params,
        weights_opt=st.session_state["w_opt"],
        weights_final=st.session_state["w_final"],
        equity_png=eq_png,
        corr_png=corr_png,
        metrics_perf=metrics_perf,
        metrics_risk=metrics_risk,
        details_table=table_details
    )

    with e2:
        st.download_button(
            "Download summary report (PDF)",
            data=pdf_summary,
            file_name="portfolio_summary_report.pdf",
            mime="application/pdf"
        )

    with e3:
        st.download_button(
            "Download full report (PDF)",
            data=pdf_full,
            file_name="portfolio_full_report.pdf",
            mime="application/pdf"
        )

else:
    st.warning("Run the calculation from the sidebar to generate a portfolio.")
