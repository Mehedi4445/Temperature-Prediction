import streamlit as st
import numpy as np
import joblib
from datetime import date

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi Climate Predictor",
    page_icon="🌡️",
    layout="centered",
)

# ── Google Fonts + Global CSS ─────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #f78166;
    --accent2:   #ffa657;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --success:   #3fb950;
    --card:      #1c2128;
}

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'Sora', sans-serif;
    color: var(--text);
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Block container ── */
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 680px !important;
    padding: 2.5rem 1.5rem 4rem;
    margin: 0 auto;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2rem 0 2.5rem;
}
.hero .badge {
    display: inline-block;
    background: rgba(247,129,102,0.15);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 0.5rem;
    background: linear-gradient(90deg, #e6edf3 30%, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0;
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin: 1.8rem 0 0.6rem;
    padding-left: 2px;
}

/* ── Slider overrides ── */
div[data-testid="stSlider"] > label {
    color: var(--text) !important;
    font-weight: 600;
    font-size: 0.88rem;
}
div[data-testid="stSlider"] .stSlider > div { padding: 0; }

/* ── Number inputs ── */
div[data-testid="stNumberInput"] > label {
    color: var(--text) !important;
    font-weight: 600;
    font-size: 0.88rem;
}
div[data-testid="stNumberInput"] input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(247,129,102,0.2) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 0.85rem 1.5rem !important;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #0d1117 !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s !important;
    margin-top: 0.5rem;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
}

/* ── Result card ── */
.result-card {
    margin-top: 2rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    animation: slideUp 0.4s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card .temp-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    color: var(--accent2);
    line-height: 1;
    white-space: nowrap;
}
.result-card .temp-unit {
    font-size: 1.4rem;
    color: var(--muted);
}
.result-card .result-meta {
    flex: 1;
}
.result-card .result-meta .label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}
.result-card .result-meta .desc {
    font-size: 1rem;
    color: var(--text);
    font-weight: 600;
}

/* ── Feature summary chips ── */
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1rem;
}
.chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
}
.chip span { color: var(--text); font-weight: 600; }

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    color: var(--muted);
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">🌡️ Linear Regression · Delhi Climate</div>
    <h1>Temperature Predictor</h1>
    <p>Enter weather conditions to forecast the mean temperature for New Delhi.</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📅 Date</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year)
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=date.today().month)
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=date.today().day)

st.markdown('<div class="section-label">🌦️ Atmospheric Conditions</div>', unsafe_allow_html=True)

humidity = st.slider(
    "💧 Humidity (%)",
    min_value=0.0, max_value=100.0, value=60.0, step=0.5,
    help="Relative humidity as a percentage (0–100%)"
)

wind_speed = st.slider(
    "💨 Wind Speed (km/h)",
    min_value=0.0, max_value=200.0, value=8.0, step=0.5,
    help="Wind speed in kilometres per hour"
)

meanpressure = st.slider(
    "🔵 Mean Pressure (hPa)",
    min_value=900.0, max_value=1100.0, value=1010.0, step=0.5,
    help="Atmospheric pressure at sea level in hectopascals"
)

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
predict = st.button("🧠  Predict Temperature")

# ── Result ────────────────────────────────────────────────────────────────────
if predict:
    features = np.array([[humidity, wind_speed, meanpressure, year, month, day]])
    prediction = model.predict(features)
    temp = round(float(prediction[0]), 2)

    # Describe the temperature
    if temp < 10:
        desc = "❄️ Cold — bundle up!"
    elif temp < 20:
        desc = "🌤️ Cool & pleasant"
    elif temp < 30:
        desc = "☀️ Warm & comfortable"
    elif temp < 38:
        desc = "🥵 Hot — stay hydrated"
    else:
        desc = "🔥 Extreme heat advisory"

    st.markdown(f"""
    <div class="result-card">
        <div>
            <div class="temp-value">{temp}<span class="temp-unit"> °C</span></div>
        </div>
        <div class="result-meta">
            <div class="label">Predicted Mean Temperature</div>
            <div class="desc">{desc}</div>
        </div>
    </div>
    <div class="chips">
        <div class="chip">Humidity <span>{humidity}%</span></div>
        <div class="chip">Wind <span>{wind_speed} km/h</span></div>
        <div class="chip">Pressure <span>{meanpressure} hPa</span></div>
        <div class="chip">Date <span>{int(year)}-{int(month):02d}-{int(day):02d}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · Linear Regression on DailyDelhiClimate dataset
</div>
""", unsafe_allow_html=True)
