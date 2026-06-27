"""
app.py
------
Streamlit application for the Uber & Lyft Dynamic Pricing Prediction project.
Imports predict_price() from predict.py; never touches model artifacts directly.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import sys

# ── Path setup — allows importing predict.py from the same directory ───────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import (
    predict_price,
    get_time_context,
    VALID_CAB_TYPES,
    VALID_RIDE_NAMES,
    VALID_LOCATIONS,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uber & Lyft Dynamic Pricing Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font import ──────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Remove top padding Streamlit adds */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── Header banner ───────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 60%, #16213e 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.07);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 14px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0 0 1.4rem 0;
    font-weight: 400;
    line-height: 1.6;
    max-width: 600px;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    color: #cbd5e1;
    font-size: 0.76rem;
    font-weight: 500;
    padding: 4px 12px;
}

/* ── Section labels ──────────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.6rem;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1.2rem;
}

/* ── Cards ───────────────────────────────────────────── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card-dark {
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Prediction result card ──────────────────────────── */
.prediction-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
}
.prediction-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 0.5rem;
}
.prediction-price {
    font-size: 3.4rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.prediction-note {
    font-size: 0.8rem;
    color: #818cf8;
    margin: 0;
}

/* ── Metric chips ────────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.metric-chip {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
    flex: 1;
    min-width: 80px;
    text-align: center;
}
.metric-chip-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    font-family: 'DM Mono', monospace;
}
.metric-chip-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-top: 2px;
}

/* ── Context badges ──────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    margin: 3px 3px 3px 0;
}
.badge-rush {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}
.badge-weekend {
    background: #dbeafe;
    color: #1e40af;
    border: 1px solid #bfdbfe;
}
.badge-neutral {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}

/* ── Dataset stats ───────────────────────────────────── */
.stat-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.stat-block {
    flex: 1;
    min-width: 100px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e293b;
    font-family: 'DM Mono', monospace;
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-top: 3px;
}

/* ── Model table ─────────────────────────────────────── */
.model-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.model-table th {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    padding: 0 12px 10px 0;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
}
.model-table td {
    padding: 9px 12px 9px 0;
    color: #334155;
    border-bottom: 1px solid #f1f5f9;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
}
.model-table tr:last-child td {
    border-bottom: none;
}
.model-table td:first-child {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: #1e293b;
}
.best-row td {
    color: #4f46e5 !important;
    font-weight: 600;
}
.best-row td:first-child {
    color: #4338ca !important;
}

/* ── Features list ───────────────────────────────────── */
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
.feature-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.8rem;
    color: #334155;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 7px;
}
.feature-dot {
    width: 6px; height: 6px;
    background: #6366f1;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .sidebar-header {
    color: #f1f5f9;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.5rem 0 0.2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
section[data-testid="stSidebar"] .sidebar-section {
    color: #6366f1;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.5rem 0;
}

/* ── Footer ──────────────────────────────────────────── */
.footer {
    border-top: 1px solid #e2e8f0;
    padding-top: 1.5rem;
    margin-top: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
.footer-left {
    font-size: 0.78rem;
    color: #94a3b8;
}
.footer-links {
    display: flex;
    gap: 16px;
}
.footer-link {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6366f1;
    text-decoration: none;
}

/* ── Streamlit overrides ─────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 10px;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.55rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.88;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _model_exists() -> bool:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'best_model.pkl')
    return os.path.exists(model_path)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input Panel
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown('<div class="sidebar-header">🚕 Ride Parameters</div>', unsafe_allow_html=True)
    st.markdown("Configure your ride details below to estimate the fare.")

    # ── Ride type ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🚖 Ride Type</div>', unsafe_allow_html=True)

    cab_type = st.selectbox(
        "Platform",
        options=VALID_CAB_TYPES,
        help="Uber or Lyft"
    )

    ride_options = VALID_RIDE_NAMES[cab_type]
    ride_name = st.selectbox(
        "Service Tier",
        options=ride_options,
        help="The specific ride product"
    )

    # ── Trip details ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">📍 Trip Details</div>', unsafe_allow_html=True)

    source = st.selectbox(
        "Pickup Location",
        options=VALID_LOCATIONS,
        index=0,
        help="Your starting point in Boston"
    )

    destination = st.selectbox(
        "Drop-off Location",
        options=VALID_LOCATIONS,
        index=5,
        help="Your destination in Boston"
    )

    distance = st.slider(
        "Distance (miles)",
        min_value=0.1,
        max_value=10.0,
        value=2.5,
        step=0.1,
        help="Trip distance in miles"
    )

    surge_multiplier = st.select_slider(
        "Surge Multiplier",
        options=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
        value=1.0,
        help="Demand-based price multiplier (1.0 = no surge)"
    )

    # ── Time details ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🕐 Time of Ride</div>', unsafe_allow_html=True)

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_name_selected = st.selectbox(
        "Day of Week",
        options=day_names,
        index=0,
    )
    day_of_week = day_names.index(day_name_selected)

    hour = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=9,
        step=1,
        format="%d:00",
        help="0 = midnight, 12 = noon, 23 = 11 PM"
    )

    st.markdown("---")
    predict_btn = st.button("⚡ Predict Fare", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-tag">End-to-End Machine Learning Portfolio Project</div>
    <h1 class="hero-title">Uber &amp; Lyft Dynamic Pricing Predictor</h1>
    <p class="hero-subtitle">
        A machine learning system that predicts Uber &amp; Lyft ride fares across Boston
        using distance, surge pricing, ride type, and temporal patterns — trained on 693,000 real trips.
    </p>
    <div class="hero-pills">
        <span class="pill">🗂 693K rides</span>
        <span class="pill">📍 Boston, MA</span>
        <span class="pill">🤖 XGBoost (Tuned)</span>
        <span class="pill">📊 R² = 0.97</span>
        <span class="pill">💵 MAE = $1.02</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Two-column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.05, 1], gap="large")


# ════════════════════════════════════════════════════════
# LEFT COLUMN — Prediction result + trip context
# ════════════════════════════════════════════════════════
with col_left:

    # ── Model not trained yet ──────────────────────────
    if not _model_exists():
        st.warning(
            "**Model artifacts not found.**  \n"
            "Train the model first by running `python train.py`, "
            "then reload this page.",
            icon="⚠️"
        )

    # ── Prediction triggered ───────────────────────────
    if predict_btn:
        if source == destination:
            st.error(
                "Pickup and drop-off locations must be different. "
                "Please select two distinct locations.",
                icon="🚫"
            )
        elif not _model_exists():
            st.error(
                "The trained model could not be found. "
                "Please run `python train.py` before launching the application.",
                icon="🚫"
            )
        else:
            with st.spinner("Running inference…"):
                try:
                    price = predict_price(
                        cab_type=cab_type,
                        ride_name=ride_name,
                        distance=distance,
                        surge_multiplier=surge_multiplier,
                        source=source,
                        destination=destination,
                        hour=hour,
                        day_of_week=day_of_week,
                    )

                    ctx = get_time_context(hour, day_of_week)

                    # Store in session state so the result persists after
                    # the user changes sidebar inputs without re-predicting
                    st.session_state['result'] = {
                        'price':            price,
                        'ctx':              ctx,
                        'cab_type':         cab_type,
                        'ride_name':        ride_name,
                        'source':           source,
                        'destination':      destination,
                        'distance':         distance,
                        'surge_multiplier': surge_multiplier,
                    }

                except ValueError as e:
                    st.error(f"Invalid input: {e}", icon="🚫")
                except FileNotFoundError as e:
                    st.error(str(e), icon="🚫")
                except Exception as e:
                    st.error(
                        f"An unexpected error occurred during prediction: {e}",
                        icon="🚫"
                    )

    # ── Show stored result ─────────────────────────────
    if 'result' in st.session_state:
        r = st.session_state['result']

        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated Fare</div>
            <div class="prediction-price">${r['price']:.2f}</div>
            <div class="prediction-note">
                {r['cab_type']} · {r['ride_name']} · {r['distance']:.1f} mi
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Time context badges
        ctx = r['ctx']
        badge_html = f'<span class="badge badge-neutral">🗓 {ctx["day_name"]}</span>'
        badge_html += f'<span class="badge badge-neutral">🕐 {ctx["time_str"]}</span>'
        if ctx['is_rush_hour']:
            badge_html += '<span class="badge badge-rush">⚡ Rush Hour</span>'
        else:
            badge_html += '<span class="badge badge-neutral">Off-Peak</span>'
        if ctx['is_weekend']:
            badge_html += '<span class="badge badge-weekend">Weekend</span>'
        else:
            badge_html += '<span class="badge badge-neutral">Weekday</span>'

        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Trip insights
        st.markdown('<div class="section-label">Ride Insights</div>', unsafe_allow_html=True)
        surge_status = "Active" if r['surge_multiplier'] > 1.0 else "None"
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-value">{r['distance']:.1f} mi</div>
                <div class="metric-chip-label">Distance</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-value">{r['surge_multiplier']:.2f}×</div>
                <div class="metric-chip-label">Surge — {surge_status}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-value">${r['price'] / r['distance']:.2f}</div>
                <div class="metric-chip-label">Cost per Mile</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Route
        st.markdown(f"""
        <div class="card" style="padding: 1rem 1.4rem;">
            <div class="section-label">Route</div>
            <div style="display:flex; align-items:center; gap:10px; margin-top:4px;">
                <div style="font-size:0.85rem; font-weight:600; color:#1e293b;">{r['source']}</div>
                <div style="color:#94a3b8; font-size:0.85rem;">→</div>
                <div style="font-size:0.85rem; font-weight:600; color:#1e293b;">{r['destination']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty state — shown before the first prediction
        st.markdown("""
        <div class="card" style="text-align:center; padding:2.8rem 1.8rem; border-style:dashed;">
            <div style="font-size:2rem; margin-bottom:0.8rem;">🚕</div>
            <div style="font-size:0.93rem; font-weight:600; color:#1e293b; margin-bottom:0.5rem;">
                Ready to predict
            </div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6; max-width:280px; margin:0 auto;">
                Configure the ride details in the sidebar and click
                <strong style="color:#6366f1;">Predict Fare</strong> to estimate the
                cost using the trained XGBoost model.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# RIGHT COLUMN — Dataset info + Model performance
# ════════════════════════════════════════════════════════
with col_right:

    # ── Dataset summary ────────────────────────────────
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Uber & Lyft · Boston, MA</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-block">
            <div class="stat-value">693K</div>
            <div class="stat-label">Rides</div>
        </div>
        <div class="stat-block">
            <div class="stat-value">57</div>
            <div class="stat-label">Raw Cols</div>
        </div>
        <div class="stat-block">
            <div class="stat-value">10</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-block">
            <div class="stat-value">12</div>
            <div class="stat-label">Locations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="padding:1.1rem 1.3rem; margin-top:-4px;">
        <div class="feature-grid">
            <div class="feature-item"><span class="feature-dot"></span>cab_type</div>
            <div class="feature-item"><span class="feature-dot"></span>ride name</div>
            <div class="feature-item"><span class="feature-dot"></span>distance</div>
            <div class="feature-item"><span class="feature-dot"></span>surge ×</div>
            <div class="feature-item"><span class="feature-dot"></span>source</div>
            <div class="feature-item"><span class="feature-dot"></span>destination</div>
            <div class="feature-item"><span class="feature-dot"></span>hour</div>
            <div class="feature-item"><span class="feature-dot"></span>day of week</div>
            <div class="feature-item"><span class="feature-dot"></span>rush hour flag</div>
            <div class="feature-item"><span class="feature-dot"></span>weekend flag</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model performance ──────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Benchmark Comparison</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="padding:1.2rem 1.4rem;">
        <table class="model-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R²</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Linear Regression</td>
                    <td>5.08</td>
                    <td>6.31</td>
                    <td>0.52</td>
                </tr>
                <tr>
                    <td>Decision Tree</td>
                    <td>1.10</td>
                    <td>1.65</td>
                    <td>0.97</td>
                </tr>
                <tr>
                    <td>Random Forest</td>
                    <td>1.12</td>
                    <td>1.75</td>
                    <td>0.96</td>
                </tr>
                <tr>
                    <td>XGBoost</td>
                    <td>1.08</td>
                    <td>1.61</td>
                    <td>0.97</td>
                </tr>
                <tr class="best-row">
                    <td>✦ XGBoost (Tuned)</td>
                    <td>1.02</td>
                    <td>1.55</td>
                    <td>0.97</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT SECTION — Full width, below the two columns
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">About This Project</div>', unsafe_allow_html=True)

about_col1, about_col2, about_col3 = st.columns(3, gap="medium")

with about_col1:
    st.markdown("""
    <div class="card" style="height:100%;">
        <div style="font-size:1.5rem; margin-bottom:0.6rem;">🔄</div>
        <div style="font-size:0.9rem; font-weight:600; color:#1e293b; margin-bottom:0.5rem;">End-to-End Pipeline</div>
        <div style="font-size:0.8rem; color:#64748b; line-height:1.6;">
            The project covers the full ML workflow: data preprocessing,
            feature engineering, training four regression models, hyperparameter
            tuning with RandomizedSearchCV, and deployment via Streamlit.
            Each stage is separated into its own module for clarity.
        </div>
    </div>
    """, unsafe_allow_html=True)

with about_col2:
    st.markdown("""
    <div class="card" style="height:100%;">
        <div style="font-size:1.5rem; margin-bottom:0.6rem;">⚙️</div>
        <div style="font-size:0.9rem; font-weight:600; color:#1e293b; margin-bottom:0.5rem;">Key Design Decisions</div>
        <div style="font-size:0.8rem; color:#64748b; line-height:1.6;">
            Outliers are capped using the IQR method to stabilise the linear baseline.
            Label encoders are saved alongside the model so inference uses the exact
            same encoding as training. Rush-hour and weekend flags are derived from
            the Unix timestamp to capture demand patterns without leaking raw time data.
        </div>
    </div>
    """, unsafe_allow_html=True)

with about_col3:
    st.markdown("""
    <div class="card" style="height:100%;">
        <div style="font-size:1.5rem; margin-bottom:0.6rem;">🔭</div>
        <div style="font-size:0.9rem; font-weight:600; color:#1e293b; margin-bottom:0.5rem;">Potential Improvements</div>
        <div style="font-size:0.8rem; color:#64748b; line-height:1.6;">
            Weather and traffic data as additional features, one-hot encoding
            comparison, route-aware distance calculation, periodic model retraining
            on fresh data, and cloud deployment via Streamlit Community Cloud
            or a containerised API.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">
        Built with Python &bull; XGBoost &bull; Streamlit &nbsp;&nbsp;|&nbsp;&nbsp;
        Dataset: Uber &amp; Lyft Boston (Kaggle) &nbsp;&nbsp;|&nbsp;&nbsp;
        Created by <strong>Gaurav Verma</strong>
    </div>
    <div class="footer-links">
        <a class="footer-link" href="https://github.com/Gaurav4421/Dynamic-Pricing-Prediction" target="_blank">
            GitHub
        </a>
        <a class="footer-link" href="https://www.linkedin.com/in/gaurav-verma-00a524342/" target="_blank">
            LinkedIn
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
