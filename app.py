# ============================================================
#  BAGHDAD CLIMATE ANOMALY PREDICTOR
#  A single-file Streamlit application for university
#  Probability coursework — Poisson & Exponential distributions
#
#  Author  : Expert Python Developer / Probability Tutor
#  Stack   : streamlit · requests · scipy.stats · plotly
#  Run     : streamlit run app.py
# ============================================================

# ── Standard library ────────────────────────────────────────
import math
import datetime

# ── Third-party ─────────────────────────────────────────────
import numpy as np
import requests
import streamlit as st
from scipy.stats import poisson, expon   # Poisson PMF & Exponential CDF
import plotly.graph_objects as go


# ════════════════════════════════════════════════════════════
#  0.  GLOBAL DESIGN TOKENS
#      All colour / font constants live here so every chart
#      and CSS injection stays in sync.
# ════════════════════════════════════════════════════════════
DARK_BG       = "#0A0A0F"      # near-black page background
CARD_BG       = "#12121A"      # slightly lighter card surface
GRID_COLOR    = "#1E1E2E"      # chart grid lines
AMBER         = "#F59E0B"      # primary accent  (Poisson)
CORAL         = "#F97316"      # secondary accent (Exponential)
TEXT_PRIMARY  = "#F1F5F9"      # headings
TEXT_MUTED    = "#94A3B8"      # body / muted labels
FONT_MONO     = "'JetBrains Mono', 'Fira Mono', monospace"


# ════════════════════════════════════════════════════════════
#  1.  STREAMLIT PAGE CONFIGURATION
#      Must be called before any other st.* command.
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title   = "Baghdad Climate Anomaly Predictor",
    page_icon    = "🌪️",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)


# ════════════════════════════════════════════════════════════
#  2.  GLOBAL CSS INJECTION
#      Override Streamlit's default chrome with a strict dark-
#      mode palette.  All colours come from design tokens above.
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  /* ── Google Font import ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Root overrides ── */
  html, body, [class*="css"] {{
      font-family: 'Space Grotesk', sans-serif;
      background-color: {DARK_BG} !important;
      color: {TEXT_PRIMARY};
  }}

  /* ── Remove default Streamlit padding ── */
  .main .block-container {{
      padding: 1.5rem 2.5rem 2rem 2.5rem;
      max-width: 1400px;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background-color: {CARD_BG} !important;
      border-right: 1px solid {GRID_COLOR};
  }}
  [data-testid="stSidebar"] .stMarkdown p {{
      color: {TEXT_MUTED};
      font-size: 0.82rem;
  }}

  /* ── Metric cards ── */
  [data-testid="metric-container"] {{
      background: {CARD_BG};
      border: 1px solid {GRID_COLOR};
      border-radius: 12px;
      padding: 1rem 1.25rem;
  }}
  [data-testid="metric-container"] label {{
      color: {TEXT_MUTED} !important;
      font-size: 0.75rem !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
      color: {AMBER} !important;
      font-size: 1.9rem !important;
      font-family: {FONT_MONO};
  }}

  /* ── Dividers ── */
  hr {{ border-color: {GRID_COLOR}; }}

  /* ── Code / formula blocks ── */
  code, .formula-box {{
      background: {GRID_COLOR};
      color: {AMBER};
      border-radius: 6px;
      font-family: {FONT_MONO};
      font-size: 0.88rem;
      padding: 0.15em 0.4em;
  }}

  /* ── Custom card wrapper ── */
  .stat-card {{
      background: {CARD_BG};
      border: 1px solid {GRID_COLOR};
      border-radius: 14px;
      padding: 1.4rem 1.6rem;
      height: 100%;
  }}

  /* ── Section headings ── */
  .section-label {{
      font-size: 0.70rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: {TEXT_MUTED};
      margin-bottom: 0.25rem;
  }}

  /* ── Alert / info boxes ── */
  [data-testid="stAlert"] {{
      background: {CARD_BG} !important;
      border-color: {AMBER} !important;
      border-radius: 10px !important;
  }}

  /* ── Spinner ── */
  [data-testid="stSpinner"] > div {{
      border-top-color: {AMBER} !important;
  }}

  /* ── Selectbox / slider labels ── */
  .stSelectbox label, .stSlider label, .stCheckbox label {{
      color: {TEXT_MUTED} !important;
      font-size: 0.82rem !important;
  }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  3.  CONSTANTS & OFFLINE DEMO DATA
#      Fallback values used when the API is unreachable.
# ════════════════════════════════════════════════════════════
BAGHDAD_LAT  = 33.31
BAGHDAD_LON  = 44.36
API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Historical years to fetch from Open-Meteo
HISTORY_START = "2015-01-01"
HISTORY_END   = datetime.date.today().strftime("%Y-%m-%d")

# Pre-calculated offline demo λ values (events / month)
DEMO_LAMBDA = {
    "Extreme Wind  (max wind speed > 40 km/h)": 4.2,
    "Extreme Heat  (max temp > 48 °C)"         : 1.8,
}


# ════════════════════════════════════════════════════════════
#  4.  API & DATA LAYER
#      fetch_open_meteo()  → raw daily data dict
#      compute_lambda()    → historical λ (events / month)
# ════════════════════════════════════════════════════════════

def fetch_open_meteo(variable: str, threshold: float) -> dict:
    """
    Fetches daily weather data from the Open-Meteo Historical API
    for Baghdad and returns:
        {
          "lambda"       : float,   # average extreme events per month
          "total_events" : int,
          "total_months" : float,
          "years_covered": int,
        }

    Parameters
    ----------
    variable  : Open-Meteo daily variable name, e.g. 'windspeed_10m_max'
    threshold : The numeric threshold that qualifies as an "extreme event"
    """
    params = {
        "latitude"        : BAGHDAD_LAT,
        "longitude"       : BAGHDAD_LON,
        "start_date"      : HISTORY_START,
        "end_date"        : HISTORY_END,
        "daily"           : variable,
        "timezone"        : "Asia/Baghdad",
    }
    response = requests.get(API_BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    values = data["daily"][variable]          # list of daily readings
    dates  = data["daily"]["time"]            # list of "YYYY-MM-DD" strings

    # Count days that exceed the threshold → each qualifies as an extreme event
    extreme_days = [v for v in values if v is not None and v > threshold]
    total_events = len(extreme_days)

    # Determine how many months the data spans
    start_dt = datetime.date.fromisoformat(dates[0])
    end_dt   = datetime.date.fromisoformat(dates[-1])
    total_months = (
        (end_dt.year - start_dt.year) * 12
        + (end_dt.month - start_dt.month)
        + 1   # inclusive
    )

    # ── KEY FORMULA ────────────────────────────────────────
    #  λ (lambda) = total extreme events / total months
    #  This is the Maximum Likelihood Estimate of the Poisson
    #  rate parameter.  For the Exponential distribution,
    #  the same λ is used as the rate (1/mean waiting time).
    # ───────────────────────────────────────────────────────
    lam = total_events / total_months if total_months > 0 else 1.0

    return {
        "lambda"        : round(lam, 4),
        "total_events"  : total_events,
        "total_months"  : total_months,
        "years_covered" : (end_dt.year - start_dt.year + 1),
    }


@st.cache_data(show_spinner=False)
def get_data(anomaly_type: str, offline: bool) -> dict:
    """
    Cached wrapper: returns data dict with key 'lambda'.
    Uses offline demo values when `offline=True` or API fails.
    """
    if offline:
        return {
            "lambda"       : DEMO_LAMBDA[anomaly_type],
            "total_events" : "—  (demo)",
            "total_months" : "—  (demo)",
            "years_covered": "—  (demo)",
            "source"       : "offline",
        }

    # Map UI label → (Open-Meteo variable, threshold)
    api_config = {
        "Extreme Wind  (max wind speed > 40 km/h)": ("windspeed_10m_max", 40.0),
        "Extreme Heat  (max temp > 48 °C)"         : ("temperature_2m_max", 48.0),
    }
    var, threshold = api_config[anomaly_type]

    try:
        result = fetch_open_meteo(var, threshold)
        result["source"] = "live"
        return result
    except Exception as err:
        st.warning(f"⚠️  API unavailable ({err}). Falling back to demo data.")
        return {
            "lambda"       : DEMO_LAMBDA[anomaly_type],
            "total_events" : "—  (fallback)",
            "total_months" : "—  (fallback)",
            "years_covered": "—  (fallback)",
            "source"       : "fallback",
        }


# ════════════════════════════════════════════════════════════
#  5.  STATISTICAL MATH ENGINE
#      Both functions are thin wrappers around scipy.stats
#      that also return the intermediate values needed for
#      the formula display panel.
# ════════════════════════════════════════════════════════════

def compute_poisson_pmf(lam: float, k_max: int = 14) -> dict:
    """
    Computes the Poisson Probability Mass Function (PMF) for
    k = 0, 1, 2, …, k_max.

    POISSON PMF FORMULA:
        P(X = k) = (λ^k · e^{−λ}) / k!

    scipy maps to this as:
        poisson.pmf(k, mu=λ)
        where mu is the rate parameter (our λ).

    Returns
    -------
    dict with:
        k_values  : list[int]   — 0..k_max
        pmf_values: list[float] — P(X=k) for each k
        mode_k    : int         — most-probable k  (floor(λ))
        mode_prob : float       — P(X = mode_k)
    """
    k_values   = list(range(k_max + 1))
    # scipy.stats.poisson.pmf(k, mu) implements (λ^k · e^{-λ}) / k!
    pmf_values = [float(poisson.pmf(k, mu=lam)) for k in k_values]
    mode_k     = int(math.floor(lam))
    mode_prob  = float(poisson.pmf(mode_k, mu=lam))

    return {
        "k_values"  : k_values,
        "pmf_values": pmf_values,
        "mode_k"    : mode_k,
        "mode_prob" : mode_prob,
    }


def compute_exponential_cdf(lam: float, t_max: float = None) -> dict:
    """
    Computes the Exponential CDF — the probability that the
    WAITING TIME until the next extreme event is ≤ t months.

    EXPONENTIAL CDF FORMULA:
        F(t) = 1 − e^{−λt}       for t ≥ 0

    scipy's expon uses the *scale* parameter = 1/λ (the mean),
    so:
        expon.cdf(t, scale=1/λ) == 1 − e^{−λt}

    Returns
    -------
    dict with:
        t_values  : np.ndarray  — time axis in months
        cdf_values: np.ndarray  — F(t) for each t
        mean_wait : float       — E[T] = 1/λ  (mean waiting time)
        p_one_month: float      — P(T ≤ 1) = F(1) = 1 − e^{−λ}
    """
    if t_max is None:
        t_max = max(4.0, 3.0 / lam)   # show at least 3 mean waiting times

    t_values   = np.linspace(0, t_max, 400)
    scale      = 1.0 / lam            # scale = mean waiting time
    # expon.cdf(t, scale=1/λ) ≡ 1 − e^{−λt}
    cdf_values = expon.cdf(t_values, scale=scale)
    mean_wait  = scale
    p_one_month = float(expon.cdf(1.0, scale=scale))

    return {
        "t_values"    : t_values,
        "cdf_values"  : cdf_values,
        "mean_wait"   : round(mean_wait, 4),
        "p_one_month" : round(p_one_month, 4),
        "t_max"       : t_max,
    }


# ════════════════════════════════════════════════════════════
#  6.  CHART BUILDERS  (Plotly)
#      All charts share the same dark-mode layout template.
# ════════════════════════════════════════════════════════════

def _base_layout(title: str) -> dict:
    """Returns a Plotly layout dict with the global dark theme."""
    return dict(
        title       = dict(text=title, font=dict(size=14, color=TEXT_MUTED,
                           family="Space Grotesk"), x=0.03),
        paper_bgcolor = DARK_BG,
        plot_bgcolor  = CARD_BG,
        font          = dict(family="Space Grotesk", color=TEXT_MUTED, size=11),
        margin        = dict(l=40, r=20, t=50, b=40),
        xaxis = dict(
            gridcolor  = GRID_COLOR,
            zerolinecolor = GRID_COLOR,
            tickfont   = dict(size=10),
        ),
        yaxis = dict(
            gridcolor  = GRID_COLOR,
            zerolinecolor = GRID_COLOR,
            tickfont   = dict(size=10),
        ),
        hoverlabel = dict(
            bgcolor    = GRID_COLOR,
            font_size  = 11,
            font_family= "JetBrains Mono",
        ),
    )


def build_poisson_chart(pmf_data: dict, lam: float, k_query: int) -> go.Figure:
    """
    Bar chart of the Poisson PMF.
    The queried k bar is highlighted in AMBER; others in muted coral.
    """
    k_vals  = pmf_data["k_values"]
    pmf_vals= pmf_data["pmf_values"]

    colors = [AMBER if k == k_query else "#374151" for k in k_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x           = k_vals,
        y           = pmf_vals,
        marker_color= colors,
        marker_line_color = DARK_BG,
        marker_line_width = 1.5,
        hovertemplate = "<b>k = %{x}</b><br>P(X=%{x}) = %{y:.4f}<extra></extra>",
        name        = "P(X = k)",
    ))

    # Annotation for highlighted bar
    if 0 <= k_query <= max(k_vals):
        fig.add_annotation(
            x=k_query, y=pmf_vals[k_query],
            text=f"<b>{pmf_vals[k_query]:.4f}</b>",
            showarrow=True, arrowhead=2,
            arrowcolor=AMBER, font=dict(color=AMBER, size=12),
            ay=-32,
        )

    layout = _base_layout(f"Poisson PMF  (λ = {lam})")
    layout["xaxis"]["title"] = dict(text="k  (number of events)", font=dict(size=11))
    layout["yaxis"]["title"] = dict(text="Probability  P(X = k)", font=dict(size=11))
    fig.update_layout(**layout)
    return fig


def build_exponential_chart(exp_data: dict, lam: float, t_query: float) -> go.Figure:
    """
    Line chart of the Exponential CDF with shaded area under the curve
    up to t_query.
    """
    t  = exp_data["t_values"]
    F  = exp_data["cdf_values"]

    # Clip fill to t_query
    mask  = t <= t_query
    t_fill = np.append(t[mask], t_query)
    F_fill = np.append(F[mask], float(expon.cdf(t_query, scale=1/lam)))

    fig = go.Figure()

    # Filled area up to t_query
    fig.add_trace(go.Scatter(
        x=t_fill, y=F_fill,
        fill="tozeroy",
        fillcolor=f"rgba(249,115,22,0.15)",  # CORAL transparent
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Main CDF line
    fig.add_trace(go.Scatter(
        x=t, y=F,
        mode="lines",
        line=dict(color=CORAL, width=2.5),
        hovertemplate="t = %{x:.2f} months<br>F(t) = %{y:.4f}<extra></extra>",
        name="F(t) = 1 − e^{−λt}",
    ))

    # Vertical reference line at t_query
    F_at_tq = float(expon.cdf(t_query, scale=1/lam))
    fig.add_shape(type="line",
        x0=t_query, x1=t_query, y0=0, y1=F_at_tq,
        line=dict(color=AMBER, width=1.5, dash="dot"),
    )
    fig.add_shape(type="line",
        x0=0, x1=t_query, y0=F_at_tq, y1=F_at_tq,
        line=dict(color=AMBER, width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=t_query, y=F_at_tq,
        text=f"<b>F({t_query:.1f}) = {F_at_tq:.4f}</b>",
        showarrow=True, arrowhead=2,
        arrowcolor=AMBER, font=dict(color=AMBER, size=12),
        ay=-30, ax=30,
    )

    layout = _base_layout(f"Exponential CDF  (λ = {lam})")
    layout["xaxis"]["title"] = dict(text="t  (months)", font=dict(size=11))
    layout["yaxis"]["title"] = dict(text="F(t) = P(T ≤ t)", font=dict(size=11))
    fig.update_layout(**layout)
    return fig


# ════════════════════════════════════════════════════════════
#  7.  SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <p style='font-size:1.15rem; font-weight:600; color:#F1F5F9; margin-bottom:0.1rem'>
    🌪️ Baghdad CAP
    </p>
    <p style='font-size:0.72rem; color:#64748B; margin-top:0; margin-bottom:1.5rem'>
    Climate Anomaly Predictor
    </p>
    """, unsafe_allow_html=True)

    offline_mode = st.toggle(
        "Offline Demo Mode",
        value=False,
        help="Disables live API calls — use this during presentations with no internet.",
    )
    if offline_mode:
        st.info("🟡 Using pre-calculated demo λ values.")

    st.markdown("---")

    anomaly_options = list(DEMO_LAMBDA.keys())
    anomaly_type = st.selectbox(
        "Anomaly Type",
        options=anomaly_options,
        index=0,
        help="Select which type of extreme weather event to analyse.",
    )

    st.markdown("---")

    st.markdown(
        f"<p class='section-label'>Poisson — Query</p>",
        unsafe_allow_html=True,
    )
    k_query = st.slider(
        "k  (exact events in next month)",
        min_value=0, max_value=14, value=4,
        help="P(X = k) will be highlighted on the PMF chart.",
    )

    st.markdown("---")

    st.markdown(
        f"<p class='section-label'>Exponential — Query</p>",
        unsafe_allow_html=True,
    )
    t_query = st.slider(
        "t  (waiting time in months)",
        min_value=0.1, max_value=6.0, value=1.0, step=0.1,
        help="F(t) = P(next event occurs within t months) will be shown.",
    )

    st.markdown("---")
    st.markdown(f"""
    <p style='font-size:0.70rem; color:#334155; line-height:1.6'>
    Data: Open-Meteo Historical API<br>
    Location: Baghdad (33.31 N, 44.36 E)<br>
    Period: {HISTORY_START} → today<br>
    Distributions: scipy.stats
    </p>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  8.  DATA FETCH
# ════════════════════════════════════════════════════════════

with st.spinner("Fetching historical climate data from Open-Meteo…"):
    data = get_data(anomaly_type, offline_mode)

lam = data["lambda"]

# Pre-compute stats
pmf_data = compute_poisson_pmf(lam, k_max=14)
exp_data = compute_exponential_cdf(lam)
p_k      = float(poisson.pmf(k_query, mu=lam))
F_t      = exp_data["p_one_month"] if t_query == 1.0 else float(
               expon.cdf(t_query, scale=1/lam))


# ════════════════════════════════════════════════════════════
#  9.  PAGE HEADER
# ════════════════════════════════════════════════════════════

st.markdown(f"""
<div style='margin-bottom:0.5rem'>
  <span style='font-size:0.70rem; letter-spacing:0.12em; text-transform:uppercase;
               color:{TEXT_MUTED}'>Baghdad, Iraq · Probability Analysis</span>
  <h1 style='font-size:1.75rem; font-weight:700; color:{TEXT_PRIMARY};
             margin:0.1rem 0 0 0; letter-spacing:-0.02em;'>
      Baghdad Climate Anomaly Predictor
  </h1>
  <p style='color:{TEXT_MUTED}; font-size:0.88rem; margin-top:0.3rem;
            max-width:680px; line-height:1.5'>
      Applying <b style='color:{AMBER}'>Poisson</b> and
      <b style='color:{CORAL}'>Exponential</b> distributions to model
      extreme weather frequencies using {data.get("years_covered","—")} years
      of historical data.
  </p>
</div>
""", unsafe_allow_html=True)

source_badge = (
    f"<span style='color:#22C55E; font-size:0.72rem'>● Live API</span>"
    if data.get("source") == "live"
    else f"<span style='color:{AMBER}; font-size:0.72rem'>● Demo / Fallback</span>"
)
st.markdown(source_badge, unsafe_allow_html=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════
#  10.  METRICS ROW
# ════════════════════════════════════════════════════════════

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="λ  (events / month)",
        value=f"{lam:.3f}",
        help="Maximum Likelihood Estimate of the Poisson rate parameter.",
    )
with m2:
    st.metric(
        label="Mean waiting time  1/λ",
        value=f"{exp_data['mean_wait']:.2f} mo",
        help="Expected months until the next extreme event.",
    )
with m3:
    st.metric(
        label=f"P(X = {k_query})  — Poisson",
        value=f"{p_k:.4f}",
        help=f"Probability of exactly {k_query} extreme events next month.",
    )
with m4:
    st.metric(
        label=f"F({t_query:.1f} mo)  — Exponential",
        value=f"{F_t:.4f}",
        help=f"Probability next event occurs within {t_query:.1f} month(s).",
    )

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  11.  POISSON SECTION
#        Left  → formula workings
#        Right → PMF bar chart
# ════════════════════════════════════════════════════════════

st.markdown(
    f"<p class='section-label' style='font-size:0.78rem; font-weight:600;"
    f"color:{AMBER}; letter-spacing:0.1em'>① POISSON DISTRIBUTION — DISCRETE</p>",
    unsafe_allow_html=True,
)

p_col_left, p_col_right = st.columns([1, 1.6], gap="large")

with p_col_left:
    st.markdown(f"""
    <div class='stat-card'>
      <p style='color:{TEXT_MUTED}; font-size:0.80rem; margin-bottom:1rem; line-height:1.7'>
        Models the <b style='color:{TEXT_PRIMARY}'>number of discrete events</b>
        in a fixed interval, assuming each event is independent and the average
        rate λ is constant.
      </p>

      <p class='section-label'>General Formula</p>
      <p style='font-family:{FONT_MONO}; color:{AMBER}; font-size:1.05rem;
                background:{GRID_COLOR}; padding:0.7rem 1rem; border-radius:8px;
                letter-spacing:0.02em; margin:0.4rem 0 1rem 0'>
        P(X = k) = (λᵏ · e<sup>−λ</sup>) / k!
      </p>

      <p class='section-label'>Your Values</p>
      <table style='width:100%; font-size:0.82rem; border-collapse:collapse'>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>λ  =</td>
          <td style='font-family:{FONT_MONO}; color:{AMBER}'>{lam}</td>
        </tr>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>k  =</td>
          <td style='font-family:{FONT_MONO}; color:{AMBER}'>{k_query}</td>
        </tr>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>λᵏ =</td>
          <td style='font-family:{FONT_MONO}; color:{TEXT_PRIMARY}'>{lam**k_query:.4f}</td>
        </tr>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>e⁻λ =</td>
          <td style='font-family:{FONT_MONO}; color:{TEXT_PRIMARY}'>{math.exp(-lam):.4f}</td>
        </tr>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>k! =</td>
          <td style='font-family:{FONT_MONO}; color:{TEXT_PRIMARY}'>{math.factorial(k_query)}</td>
        </tr>
      </table>

      <div style='margin-top:1rem; padding:0.8rem 1rem; background:{GRID_COLOR};
                  border-radius:8px; border-left: 3px solid {AMBER}'>
        <p style='color:{TEXT_MUTED}; font-size:0.75rem; margin:0 0 0.2rem 0'>
          Result
        </p>
        <p style='font-family:{FONT_MONO}; color:{AMBER}; font-size:1.3rem; margin:0'>
          P(X = {k_query}) = {p_k:.6f}
        </p>
        <p style='color:{TEXT_MUTED}; font-size:0.75rem; margin:0.4rem 0 0 0'>
          ≈ {p_k*100:.2f}% chance of exactly {k_query} event(s)
        </p>
      </div>

      <p style='color:{TEXT_MUTED}; font-size:0.72rem; margin-top:1rem'>
        Modal k (most likely) = <b style='color:{TEXT_PRIMARY}'>{pmf_data["mode_k"]}</b>
        with probability <b style='color:{TEXT_PRIMARY}'>{pmf_data["mode_prob"]:.4f}</b>
      </p>
    </div>
    """, unsafe_allow_html=True)

with p_col_right:
    poisson_fig = build_poisson_chart(pmf_data, lam, k_query)
    st.plotly_chart(poisson_fig, use_container_width=True, config={"displayModeBar": False})


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════
#  12.  EXPONENTIAL SECTION
#        Left  → formula workings
#        Right → CDF line chart
# ════════════════════════════════════════════════════════════

st.markdown(
    f"<p class='section-label' style='font-size:0.78rem; font-weight:600;"
    f"color:{CORAL}; letter-spacing:0.1em'>② EXPONENTIAL DISTRIBUTION — CONTINUOUS</p>",
    unsafe_allow_html=True,
)

e_col_left, e_col_right = st.columns([1, 1.6], gap="large")

F_t_query = float(expon.cdf(t_query, scale=1/lam))

with e_col_left:
    st.markdown(f"""
    <div class='stat-card'>
      <p style='color:{TEXT_MUTED}; font-size:0.80rem; margin-bottom:1rem; line-height:1.7'>
        Models the <b style='color:{TEXT_PRIMARY}'>continuous waiting time</b>
        T until the next extreme event.  Naturally paired with the Poisson
        process — if events arrive at rate λ, waiting times are Exponential(λ).
      </p>

      <p class='section-label'>CDF Formula</p>
      <p style='font-family:{FONT_MONO}; color:{CORAL}; font-size:1.05rem;
                background:{GRID_COLOR}; padding:0.7rem 1rem; border-radius:8px;
                letter-spacing:0.02em; margin:0.4rem 0 1rem 0'>
        F(t) = 1 − e<sup>−λt</sup>
      </p>

      <p class='section-label'>Your Values</p>
      <table style='width:100%; font-size:0.82rem; border-collapse:collapse'>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>λ   =</td>
          <td style='font-family:{FONT_MONO}; color:{CORAL}'>{lam}</td>
        </tr>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>t   =</td>
          <td style='font-family:{FONT_MONO}; color:{CORAL}'>{t_query:.1f} months</td>
        </tr>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>λt  =</td>
          <td style='font-family:{FONT_MONO}; color:{TEXT_PRIMARY}'>{lam*t_query:.4f}</td>
        </tr>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>e⁻λᵗ =</td>
          <td style='font-family:{FONT_MONO}; color:{TEXT_PRIMARY}'>{math.exp(-lam*t_query):.4f}</td>
        </tr>
        <tr>
          <td style='color:{TEXT_MUTED}; padding:0.3rem 0'>1/λ =</td>
          <td style='font-family:{FONT_MONO}; color:{TEXT_PRIMARY}'>{exp_data["mean_wait"]:.4f} mo</td>
        </tr>
      </table>

      <div style='margin-top:1rem; padding:0.8rem 1rem; background:{GRID_COLOR};
                  border-radius:8px; border-left: 3px solid {CORAL}'>
        <p style='color:{TEXT_MUTED}; font-size:0.75rem; margin:0 0 0.2rem 0'>
          Result
        </p>
        <p style='font-family:{FONT_MONO}; color:{CORAL}; font-size:1.3rem; margin:0'>
          F({t_query:.1f}) = {F_t_query:.6f}
        </p>
        <p style='color:{TEXT_MUTED}; font-size:0.75rem; margin:0.4rem 0 0 0'>
          ≈ {F_t_query*100:.2f}% chance next event within {t_query:.1f} month(s)
        </p>
      </div>

      <p style='color:{TEXT_MUTED}; font-size:0.72rem; margin-top:1rem'>
        Memoryless property: P(T > s+t | T > s) = P(T > t).<br>
        The distribution has no "memory" of past waiting time.
      </p>
    </div>
    """, unsafe_allow_html=True)

with e_col_right:
    exp_fig = build_exponential_chart(exp_data, lam, t_query)
    st.plotly_chart(exp_fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════
#  13.  FOOTER
# ════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:center;
            flex-wrap:wrap; gap:0.5rem'>
  <p style='color:{TEXT_MUTED}; font-size:0.70rem; margin:0'>
    Baghdad Climate Anomaly Predictor · University Probability Project ·
    Data © Open-Meteo (CC BY 4.0)
  </p>
  <p style='color:#334155; font-size:0.70rem; margin:0;
            font-family:{FONT_MONO}'>
    scipy.stats &nbsp;·&nbsp; plotly &nbsp;·&nbsp; streamlit
  </p>
</div>
""", unsafe_allow_html=True)
