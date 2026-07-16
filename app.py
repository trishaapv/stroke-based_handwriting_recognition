from pathlib import Path
import pickle
import streamlit as st
import pandas as pd
import numpy as np

from scripts.data_manager import DataManager
from scripts.predictor import Predictor
from scripts.draw_utils import (
    draw_character,
    draw_single_stroke,
    draw_order
)
from scripts.feature_info import (
    feature_names,
    feature_description
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Explainable Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)

# ==========================================================
# GLOBAL THEME / CSS
# ==========================================================
# One shared visual language for the whole app: a soft gradient
# background, rounded "card" panels, a colourful gradient title
# banner, nicer metrics/tabs/expanders, and a few reusable helper
# classes (ez-card, ez-pipe, ez-arrow, ez-big-number, badges).
# Everything here is pure CSS, so it never changes app logic.

# ==========================================================
# GLOBAL THEME / CSS  —  "Ink & Paper" notebook theme
# ==========================================================
# A warm, paper-like theme (nods to handwriting) instead of a
# generic purple SaaS gradient. Sidebar stays light on purpose —
# forcing light text everywhere breaks contrast against the
# white boxes Streamlit renders for inputs/expanders, so instead
# we style specific elements individually.

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #2b2420; }
    h1, h2, h3, h4 { font-family: 'Fraunces', serif; color: #2b2420; }

    /* Warm paper background with faint ruled lines */
    .stApp {
        background-color: #fbf6ec;
        background-image: repeating-linear-gradient(
            180deg, rgba(44,62,112,0.035) 0px, rgba(44,62,112,0.035) 1px,
            transparent 1px, transparent 34px
        );
    }

    /* Sidebar: light paper tone, ink text, no wildcard overrides */
    section[data-testid="stSidebar"] {
        background-color: #f3ead9;
        border-right: 1px solid #e6d9c2;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #2b2420;
    }
    section[data-testid="stSidebar"] hr { border-color: #ddcfb2; }

    /* Sidebar nav pills */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #ffffff;
        border: 1px solid #e6d9c2;
        border-radius: 10px;
        padding: 6px 10px;
        margin-bottom: 4px;
        display: block;
    }

    /* Hero banner: notebook-page header with a torn-edge shadow */
    .ez-hero {
        background: linear-gradient(100deg, #e8734a 0%, #d9622b 100%);
        padding: 26px 30px;
        border-radius: 6px 18px 18px 6px;
        border-left: 8px solid #2c3e70;
        color: #fff8f0 !important;
        margin-bottom: 10px;
        box-shadow: 0 10px 24px rgba(217, 98, 43, 0.25);
    }
    .ez-hero h1 { color: #fff8f0 !important; margin: 0 0 6px 0; font-size: 30px; }
    .ez-hero p { color: rgba(255,248,240,0.92) !important; margin: 0; font-size: 15px; }

    /* Equal-height card rows. Cards must be rendered as ONE HTML block
       inside .ez-grid-3 (not as separate st.markdown calls inside
       st.columns) — CSS Grid auto-sizes each row to its tallest cell and
       stretches every cell to match by default, which is reliable across
       Streamlit versions unlike depending on Streamlit's internal column
       markup. */
    .ez-grid-3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 16px;
    }
    @media (max-width: 900px) {
        .ez-grid-3 { grid-template-columns: 1fr; }
    }

    /* Reusable "feature" cards — 3-colour rotation for variety. */
    .ez-card {
        border: 1px solid #ecdfc9;
        border-radius: 12px;
        padding: 18px 20px;
        min-height: 220px;
        height: 100%;
        box-sizing: border-box;
        background: #fffdf8;
        color: #2b2420;
        box-shadow: 0 2px 10px rgba(44, 62, 112, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        border-top: 6px solid #2c3e70;
        display: flex;
        flex-direction: column;
    }
    .ez-card:hover { transform: translateY(-3px); box-shadow: 0 12px 22px rgba(44, 62, 112, 0.15); }
    .ez-card h3, .ez-card h4 {
        margin-top: 0;
        line-height: 1.25;
    }
    .ez-card p { color: #4a4238; line-height: 1.55; margin-bottom: 0; }

    .ez-c1 { border-top-color: #d9622b; } .ez-c1 h3 { color: #b74f1f; }
    .ez-c2 { border-top-color: #1f7a6c; } .ez-c2 h3 { color: #175f54; }
    .ez-c3 { border-top-color: #2c3e70; } .ez-c3 h3 { color: #24335e; }

    .ez-card-top    { border-top-color: #1f7a6c; } .ez-card-top h3    { color: #175f54; }
    .ez-card-mid    { border-top-color: #d9a02b; } .ez-card-mid h3    { color: #a3791c; }
    .ez-card-low    { border-top-color: #9a9184; } .ez-card-low h3    { color: #6b6355; }
    .ez-big-number  { font-family: 'Fraunces', serif; font-size: 32px; font-weight: 700; margin: 4px 0 6px 0; color: #2b2420; }

    /* Pipeline step cards (Overview page) — individual Streamlit columns */
    .ez-pipe {
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        min-height: 150px;
        background: #fffdf8;
        color: #2b2420;
        border: 1px solid #ecdfc9;
        border-bottom: 4px solid #2c3e70;
        box-shadow: 0 2px 8px rgba(44, 62, 112, 0.06);
    }
    .ez-arrow {
        text-align: center;
        margin-top: 55px;
        font-size: 22px;
        color: #d9622b;
        font-weight: 700;
    }

    /* Recognition Pipeline (Transformer Prediction page) — single flex row,
       so cards size to their content instead of being forced into equal
       narrow Streamlit columns (which broke long words like "Handwriting"
       into single-character lines). */
    .ez-pipe-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        overflow-x: auto;
        padding-bottom: 8px;
    }
    .ez-pipe-flex {
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        min-height: 150px;
        width: 150px;
        flex: 0 0 150px;
        background: #fffdf8;
        color: #2b2420;
        border: 1px solid #ecdfc9;
        border-bottom: 4px solid #2c3e70;
        box-shadow: 0 2px 8px rgba(44, 62, 112, 0.06);
        word-break: normal;
        overflow-wrap: break-word;
    }
    .ez-pipe-title {
        font-weight: 700;
        margin-top: 8px;
        line-height: 1.25;
    }
    .ez-pipe-subtitle {
        margin-top: 10px;
        color: #475569;
        font-size: 13px;
        line-height: 1.3;
    }
    .ez-arrow-flex {
        flex: 0 0 auto;
        text-align: center;
        font-size: 22px;
        color: #d9622b;
        font-weight: 700;
    }

    /* Confidence badges — soft semantic colours, paper-pill style */
    .ez-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 14px;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .ez-badge-high { background: #e4f2ec; color: #1f7a6c; }
    .ez-badge-mid  { background: #fbf0da; color: #a3791c; }
    .ez-badge-low  { background: #fbe7e0; color: #b7481f; }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #fffdf8;
        border: 1px solid #ecdfc9;
        border-radius: 12px;
        padding: 14px 12px;
        box-shadow: 0 2px 8px rgba(44, 62, 112, 0.05);
    }
    div[data-testid="stMetricLabel"] { color: #7a7163; }
    div[data-testid="stMetricValue"] { color: #2c3e70; font-family: 'Fraunces', serif; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #fffdf8;
        border-radius: 10px 10px 0 0;
        padding: 8px 16px;
        border: 1px solid #ecdfc9;
    }
    .stTabs [aria-selected="true"] {
        background: #2c3e70 !important;
        color: #fff8f0 !important;
    }

    /* Expanders (main content area only — sidebar handled above) */
    div[data-testid="stExpander"] details {
        background: #fffdf8;
        border: 1px solid #ecdfc9;
        border-radius: 10px;
        padding: 2px 8px;
    }

    /* Alerts: rounder corners, keep Streamlit's own text colours */
    div[data-testid="stAlert"] { border-radius: 10px; }

    /* Buttons */
    .stButton button, .stDownloadButton button {
        background: #2c3e70;
        color: #fff8f0;
        border-radius: 8px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def confidence_badge(conf_value: float) -> str:
    """Return a small HTML pill that colour-codes model confidence
    so a non-technical reader can tell 'good vs risky' at a glance."""
    pct = conf_value * 100
    if pct >= 75:
        css_class, label = "ez-badge-high", "High confidence"
    elif pct >= 45:
        css_class, label = "ez-badge-mid", "Medium confidence"
    else:
        css_class, label = "ez-badge-low", "Low confidence"
    return f'<span class="ez-badge {css_class}">{label} · {pct:.1f}%</span>'


# ==========================================================
# LOAD DATA
# ==========================================================

ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
PROCESSED_DIR = ROOT / "processed"

dm = DataManager()
predictor = Predictor()

profiles = pd.read_csv(FILES_DIR / "character_profiles.csv")
taxonomy = pd.read_csv(FILES_DIR / "character_taxonomy.csv")
profiles["character"] = profiles["character"].astype(str).str.strip()
taxonomy["character"] = taxonomy["character"].astype(str).str.strip()

# Optional files
removal_path = FILES_DIR / "stroke_removal_results.csv"
removal = pd.read_csv(removal_path) if removal_path.exists() else pd.DataFrame()

cooperation_path = FILES_DIR / "character_cooperation.csv"
cooperation = pd.read_csv(cooperation_path) if cooperation_path.exists() else pd.DataFrame()

synergy_path = FILES_DIR / "character_synergy.csv"
synergy = pd.read_csv(synergy_path) if synergy_path.exists() else pd.DataFrame()

# Per-sample Shapley values (optional). If this file is present, the
# Stroke Importance and Research Interpretation pages will show the
# TRUE values for the exact selected sample instead of a character-wide
# average. Column expected: sample_id, true_label, stroke_1, stroke_2, stroke_3.
sample_shapley_path = FILES_DIR / "shapley_results_clean.csv"
sample_shapley = pd.read_csv(sample_shapley_path) if sample_shapley_path.exists() else pd.DataFrame()


def get_stroke_contribution(sample_id, predicted_char):
    """
    Returns (stroke_1, stroke_2, stroke_3, is_sample_specific).

    Prefers the real per-sample Shapley values for the exact sample
    currently selected. Falls back to the character-level average
    (from character_profiles.csv) if per-sample data isn't available,
    and tells the caller which one it used so the UI can be honest
    about it instead of silently showing an average as if it were
    sample-specific.
    """
    if not sample_shapley.empty and "sample_id" in sample_shapley.columns:
        match = sample_shapley[sample_shapley["sample_id"] == sample_id]
        if len(match) > 0:
            r = match.iloc[0]
            return float(r["stroke_1"]), float(r["stroke_2"]), float(r["stroke_3"]), True

    row = profiles[profiles["character"] == predicted_char]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return float(r["stroke_1"]), float(r["stroke_2"]), float(r["stroke_3"]), False


# Parsed raw handwriting samples for drawing
PARSED_PATH = FILES_DIR / "parsed_samples.pkl"
with open(PARSED_PATH, "rb") as f:
    parsed_samples = pickle.load(f)

def get_parsed_sample(idx):
    return parsed_samples[idx]

# ==========================================================
# HELPERS
# ==========================================================

def direction_text(theta):
    if -np.pi/8 <= theta < np.pi/8:
        return "Mostly horizontal → right"
    elif np.pi/8 <= theta < 3*np.pi/8:
        return "Diagonal ↗"
    elif 3*np.pi/8 <= theta < 5*np.pi/8:
        return "Mostly vertical ↑"
    elif 5*np.pi/8 <= theta < 7*np.pi/8:
        return "Diagonal ↖"
    elif theta >= 7*np.pi/8 or theta < -7*np.pi/8:
        return "Mostly horizontal ←"
    elif -7*np.pi/8 <= theta < -5*np.pi/8:
        return "Diagonal ↙"
    elif -5*np.pi/8 <= theta < -3*np.pi/8:
        return "Mostly vertical ↓"
    else:
        return "Diagonal ↘"

def size_text(v):
    if v < 0.15:
        return "Very small"
    elif v < 0.35:
        return "Small"
    elif v < 0.75:
        return "Medium"
    else:
        return "Large"

def curvature_text(v):
    if v < 0.15:
        return "Low curvature"
    elif v < 0.45:
        return "Moderate curvature"
    else:
        return "High curvature"

def count_real_strokes(feature_matrix):
    count = 0
    for row in feature_matrix:
        if np.any(np.abs(row) > 1e-8):
            count += 1
    return count

# Ground-truth stroke-presence mask, if you've generated it via the corrected
# preprocess_dataset.py / normalise.py pipeline. When present, this is used
# instead of the `!= 0` proxy check above, which is fragile: it silently
# misreads normalized zero-padding as a "real" stroke (this was the exact
# bug that affected the Shapley/cooperation/synergy analysis before the fix).
_mask_path_candidates = [
    PROCESSED_DIR / "mask_test.npy",
    PROCESSED_DIR / "mask.npy",
]
real_stroke_mask = None
for _p in _mask_path_candidates:
    if _p.exists():
        try:
            real_stroke_mask = np.load(_p)
        except Exception:
            real_stroke_mask = None
        break


def get_real_stroke_indices(sample_id, feature_matrix):
    """
    Returns the list of stroke slot indices (0,1,2) that are genuinely
    present for this sample. Uses the ground-truth mask when available
    and shape-compatible; otherwise falls back to the `!= 0` proxy check.
    """
    if real_stroke_mask is not None and 0 <= sample_id < len(real_stroke_mask):
        row_mask = real_stroke_mask[sample_id]
        if len(row_mask) == feature_matrix.shape[0]:
            return [i for i, present in enumerate(row_mask) if present]

    return [i for i in range(feature_matrix.shape[0]) if np.any(np.abs(feature_matrix[i]) > 1e-8)]

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("## 🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Project Overview",
        "✍️ Character Explorer",
        "🔍 Feature View",
        "🤖 Point Transformer Prediction",
        "🧠 Stroke Importance",
        "✂️ Stroke Removal Lab",
        "📘 Research Interpretation",
        "🚀 Future Scope"
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

total_samples = dm.total_samples()

if "sample_idx" not in st.session_state:
    st.session_state["sample_idx"] = 0

if page != "🏠 Project Overview":
    st.sidebar.markdown("### 🎯 Sample Selection")
    sample = st.sidebar.number_input(
        "Choose sample index",
        min_value=0,
        max_value=total_samples - 1,
        step=1,
        key="sample_idx"
    )
else:
    sample = st.session_state["sample_idx"]

true_label = dm.get_label(sample)
features = dm.get_sample(sample)
parsed_sample = get_parsed_sample(sample)
pred, conf, top5 = predictor.predict(parsed_sample)

if page != "🏠 Project Overview":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Current Sample")
    st.sidebar.write(f"**Sample ID:** {sample}")
    st.sidebar.write(f"**Ground Truth:** {true_label}")
    st.sidebar.write(f"**Prediction:** {pred}")
    st.sidebar.markdown(confidence_badge(conf), unsafe_allow_html=True)
    st.sidebar.progress(min(max(conf, 0.0), 1.0))


# ==========================================================
# APP TITLE
# ==========================================================

st.markdown(
    """
    <div class="ez-hero">
        <h1>✍️ Explainable Handwriting Recognition</h1>
        <p>An AI model reads handwritten characters stroke by stroke — and this
        dashboard shows not just <b>what</b> it thinks you wrote, but <b>why</b>,
        in plain language.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# PAGE 1 — PROJECT OVERVIEW
# ==========================================================

if page == "🏠 Project Overview":

    st.markdown(
        """
        ## Understanding Handwritten Characters, One Stroke at a Time

        This project builds a **point-sequence Transformer-based handwritten character recognition system**
        using the **UJI Pen Characters dataset**.  
        Instead of only predicting the final character, the system also explains **why**
        a prediction was made by measuring how much each stroke contributed to the decision.

        The current model is the final **point-sequence Transformer**. It reads a 96-point resampled pen trajectory, preserving local curve and movement information that stroke-summary features can lose.
        """
    )

    st.markdown("---")

    st.subheader("What this dashboard does")

    st.markdown(
        """
        <div class="ez-grid-3">
            <div class="ez-card ez-c1">
                <h3>✍️ Character Exploration</h3>
                <p>
                View handwritten characters from the dataset and inspect
                how they are formed using individual pen strokes.
                </p>
            </div>
            <div class="ez-card ez-c2">
                <h3>🤖 Point Transformer Prediction</h3>
                <p>
                See how the point-sequence Transformer predicts the handwritten character from the resampled pen trajectory.
                </p>
            </div>
            <div class="ez-card ez-c3">
                <h3>🧠 Stroke-Level Explainability</h3>
                <p>
                Understand which stroke contributed most to the final prediction
                using game-theoretic explanation techniques.
                </p>
            </div>
        </div>
        <div class="ez-grid-3">
            <div class="ez-card ez-c1">
                <h3>✂️ Stroke Removal Lab</h3>
                <p>
                Interactively remove strokes and observe how the prediction
                and confidence change when parts of the character disappear.
                </p>
            </div>
            <div class="ez-card ez-c2">
                <h3>📊 Research Interpretation</h3>
                <p>
                Explore cooperation, dominance, and interaction between strokes
                in a way that connects the model output back to research insights.
                </p>
            </div>
            <div class="ez-card ez-c3">
                <h3>🚀 Future Scope</h3>
                <p>
                Extend the idea from isolated handwritten characters
                toward full handwritten word recognition.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Pipeline at a glance")

    # NOTE: this used to be 5 cards + 4 arrows squeezed into 9 equal
    # Streamlit columns. That forced each card into a very narrow column,
    # which made long words like "Handwritten" wrap one character per
    # line. Switched to the same single flex-row HTML block used by the
    # "Recognition Pipeline" section further down, so cards size to their
    # own fixed width instead of an arbitrary 1/9th of the page.
    steps = [
        ("✍️", "Handwritten Character"),
        ("📊", "Stroke Features"),
        ("🤖", "Transformer Prediction"),
        ("🧠", "Game-Theoretic Explanation"),
        ("✂️", "Stroke Removal Validation"),
    ]

    glance_html = '<div class="ez-pipe-row">'
    for i, (icon, text) in enumerate(steps):
        glance_html += f"""
            <div class="ez-pipe-flex">
                <div style="font-size:34px;">{icon}</div>
                <div class="ez-pipe-title">{text}</div>
            </div>
        """
        if i < len(steps) - 1:
            glance_html += '<div class="ez-arrow-flex">&#8594;</div>'
    glance_html += '</div>'

    st.markdown(glance_html, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Why this project matters")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
            """
            ### Key Idea

            Traditional handwriting recognition systems usually return **only a final prediction**.
            In contrast, this project tries to answer:

            - **Which stroke mattered most?**
            - **How much did each stroke contribute to the final decision?**
            - **What happens if an important stroke is removed?**
            - **Can we make handwriting recognition more transparent and interpretable?**

            This makes the model easier to **analyze, explain, and trust**.
            """
        )

    with right:
        st.info(
            """
**Core Contributions**

- Point-sequence Transformer handwritten character recognition  
- Stroke-wise feature representation with padding masks  
- Game-theoretic explanation of stroke contribution  
- Stroke removal validation for interpretability  
- Dashboard for visual and interactive analysis
"""
        )

    st.markdown("---")

    st.subheader("📖 New here? Quick glossary")

    with st.expander("Show glossary", expanded=True):
        st.markdown(
            """
            **Stroke** — one continuous pen movement, from pen-down to pen-up.
            A letter is usually made of 1–3 strokes.

            **Feature** — a number that describes a stroke (its length, shape,
            direction, speed, etc.) so a computer can work with it.

            **Point-sequence Transformer** — the AI model used here. Instead of
            summarizing each stroke into a handful of features, it reads the
            handwriting as a resampled 96-point pen trajectory and learns
            directly from that sequence of points to guess which character
            was written.

            **Confidence** — how sure the model is about its guess, shown
            as a percentage. Higher = more sure.

            **Shapley value / stroke contribution** — a fair way (borrowed
            from game theory) of asking *"if we removed this stroke, how
            much worse would the guess get?"* — that tells us how important
            the stroke was.

            **Cooperation category** — each character is labelled *Balanced
            Cooperation*, *Asymmetric Cooperation*, *Dominant Stroke*, *Interference*,
            or *Mixed*, based on how its strokes' Shapley contributions combine —
            see the Research Interpretation page for what each one means.
            """
        )

# ==========================================================
# PAGE 2 — CHARACTER EXPLORER
# ==========================================================

elif page == "✍️ Character Explorer":

    st.header("✍️ Character Explorer")
    st.caption("Visualize the handwritten character and inspect how it is built stroke by stroke.")

    # Use the same mask-aware source of truth as the Stroke Removal Lab
    # (get_real_stroke_indices) instead of the fragile `!= 0` proxy. The two
    # previously disagreed on some samples, which is why this page could
    # report a different stroke count than the Removal Lab for the same
    # sample.
    real_stroke_idx_explorer = get_real_stroke_indices(sample, features)
    stroke_count = len(real_stroke_idx_explorer)

    st.markdown(
        f"""
This sample corresponds to the handwritten character **{true_label}**.
The model currently predicts it as **{pred}** with **{conf*100:.2f}% confidence**.

This character contains **{stroke_count} stroke(s)** in the processed representation.
Below, you can inspect the full character, the stroke order, and each stroke individually.
"""
    )

    st.markdown("---")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Complete Character")
        img = draw_character(parsed_sample)
        st.image(img, use_container_width=True)

    with right:
        st.subheader("Sample Summary")

        st.metric("Sample ID", sample)
        st.metric("Ground Truth", true_label)
        st.metric("Predicted Character", pred)
        st.metric("Confidence", f"{conf*100:.2f}%")
        st.markdown(confidence_badge(conf), unsafe_allow_html=True)
        st.metric("Detected Strokes", stroke_count)

        st.info(
            """
**How to read this page**

- The complete character shows all available strokes together.
- The stroke order view shows the sequence in which the character was written.
- The individual stroke panels isolate each stroke so you can inspect its role separately.
"""
        )

    st.markdown("---")

    st.subheader("Stroke Order View")

    try:
        img_order = draw_order(parsed_sample)
        st.image(img_order, use_container_width=True)
    except Exception:
        st.warning("Stroke order visualization is not available with the current draw utility.")

    st.markdown("---")

    st.subheader("Individual Stroke Inspection")

    stroke_cols = st.columns(3)
    stroke_names = ["Stroke 1", "Stroke 2", "Stroke 3"]

    for i in range(3):
        with stroke_cols[i]:
            st.markdown(f"### {stroke_names[i]}")

            if i in real_stroke_idx_explorer:
                try:
                    img_s = draw_single_stroke(parsed_sample, i)
                    st.image(img_s, use_container_width=True)
                except Exception:
                    st.warning(f"{stroke_names[i]} could not be displayed.")
            else:
                st.info("No stroke present in this slot.")

    st.markdown("---")

    st.subheader("What you are seeing")

    st.success(
        """
A handwritten character is not treated as a single image here.  
Instead, it is broken into **individual pen strokes**.  
Each stroke is studied separately, and the Transformer later combines all of them to decide which character was written.
"""
    )

    with st.expander("Advanced details"):
        st.write(f"Feature matrix shape: **{features.shape}**")
        st.write(
            """
The current implementation stores each character as a **3 × 15 feature matrix**:
- **3 rows** → up to 3 strokes
- **15 columns** → numeric features extracted for each stroke
"""
        )

# ==========================================================
# PAGE 3 — FEATURE VIEW
# ==========================================================

elif page == "🔍 Feature View":

    st.header("🔍 Feature View")
    st.caption("See how the handwritten character is converted into numerical stroke features before entering the Transformer.")

    st.markdown(
        """
The point-sequence Transformer does not directly receive a picture of the character.  
Instead, each real stroke is converted into a **15-dimensional feature vector** describing its geometry and shape. Empty stroke slots are kept as padding and ignored by the mask.

Together, the character is represented as a **3 × 15 feature matrix**:
- **3 rows** → one row per stroke
- **15 columns** → one numerical feature per stroke property
"""
    )

    st.markdown("---")

    sample_features = features

    st.subheader("Stroke-wise feature summary")

    stroke_tabs = st.tabs(["Stroke 1", "Stroke 2", "Stroke 3"])
    real_stroke_idx_fv = get_real_stroke_indices(sample, sample_features)

    for i, tab in enumerate(stroke_tabs):
        with tab:

            f = sample_features[i]

            if i not in real_stroke_idx_fv:
                st.info("This stroke slot is empty for the selected character.")
                continue

            length = float(f[0])
            width = float(f[1])
            height = float(f[2])
            orientation = float(f[3])
            point_count = int(round(float(f[4])))
            mean_speed = float(f[5])
            curvature = float(f[6])

            x_start = float(f[7])
            y_start = float(f[8])
            x_end = float(f[9])
            y_end = float(f[10])

            x_mean = float(f[11])
            y_mean = float(f[12])
            x_std = float(f[13])
            y_std = float(f[14])

            st.markdown(
                f"""
                <div class="ez-grid-3">
                    <div class="ez-card ez-c1">
                        <h4>📏 Shape</h4>
                        <p><b>Length:</b> {length:.3f}</p>
                        <p><b>Width:</b> {width:.3f} ({size_text(width)})</p>
                        <p><b>Height:</b> {height:.3f} ({size_text(height)})</p>
                        <p><b>Points:</b> {point_count}</p>
                    </div>
                    <div class="ez-card ez-c2">
                        <h4>🧭 Motion</h4>
                        <p><b>Orientation:</b> {orientation:.3f} rad</p>
                        <p><b>Direction:</b> {direction_text(orientation)}</p>
                        <p><b>Mean Speed:</b> {mean_speed:.3f}</p>
                        <p><b>Curvature:</b> {curvature:.3f} ({curvature_text(curvature)})</p>
                    </div>
                    <div class="ez-card ez-c3">
                        <h4>📍 Position</h4>
                        <p><b>Start:</b> ({x_start:.3f}, {y_start:.3f})</p>
                        <p><b>End:</b> ({x_end:.3f}, {y_end:.3f})</p>
                        <p><b>Mean Position:</b> ({x_mean:.3f}, {y_mean:.3f})</p>
                        <p><b>Spread:</b> ({x_std:.3f}, {y_std:.3f})</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("---")

            stroke_col1, stroke_col2 = st.columns([1, 1.2])

            with stroke_col1:
                st.markdown(f"#### Visual of Stroke {i+1}")
                try:
                    img_s = draw_single_stroke(parsed_sample, i)
                    st.image(img_s, use_container_width=True)
                except Exception:
                    st.warning("Stroke visualization not available.")

            with stroke_col2:
                st.markdown(f"#### Plain-English interpretation of Stroke {i+1}")

                explanation = []

                if length > 0.75:
                    explanation.append("This stroke is relatively long and likely forms a major part of the character.")
                elif length > 0.3:
                    explanation.append("This stroke has a moderate length and may provide supporting structure.")
                else:
                    explanation.append("This is a relatively short stroke, likely acting as a detail or accent.")

                if height > width:
                    explanation.append("It is taller than it is wide, so it behaves more like a vertical component.")
                elif width > height:
                    explanation.append("It is wider than it is tall, so it behaves more like a horizontal or sweeping component.")
                else:
                    explanation.append("Its width and height are fairly balanced.")

                if curvature > 0.45:
                    explanation.append("The stroke is strongly curved, which may help distinguish rounded characters.")
                elif curvature > 0.15:
                    explanation.append("The stroke has some curvature and is not completely straight.")
                else:
                    explanation.append("The stroke is fairly straight and geometrically stable.")

                st.write(" ".join(explanation))

    st.markdown("---")

    st.subheader("Transformer Input Matrix")

    feature_df = pd.DataFrame(
        sample_features,
        columns=feature_names,
        index=["Stroke 1", "Stroke 2", "Stroke 3"]
    )

    st.dataframe(
        feature_df.round(4),
        use_container_width=True
    )

    st.info(
        f"""
**Current sample tensor shape:** `{sample_features.shape}`

This means the Transformer receives:
- **{sample_features.shape[0]} stroke slots**
- **{sample_features.shape[1]} features per stroke**
- **{stroke_count} actual stroke(s) present in this character**
- **{sample_features.size} total scalar values**
"""
    )

    with st.expander("Feature descriptions"):
        info_df = pd.DataFrame({
            "Feature": feature_names,
            "Meaning": feature_description
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Why feature extraction matters")

    st.success(
        """
The Transformer cannot directly reason about pen motion unless the handwriting is represented numerically.  
These stroke features capture **shape, direction, size, curvature, and position**, allowing the model to compare one handwritten character against many learned examples.

So before the Transformer predicts the final letter, it first sees the handwriting as a structured **3 × 15 matrix of stroke information**.
"""
    )

# ==========================================================
# PLACEHOLDERS FOR REMAINING PAGES
# ==========================================================

elif page == "🤖 Point Transformer Prediction":

    st.header("🤖 Point Transformer Prediction")
    st.caption("See how the point-sequence Transformer uses real strokes, ignores padding, and predicts the handwritten character.")

    st.markdown(
        f"""
The selected handwriting sample is represented as a **3 × 15 stroke-feature matrix**.
The raw strokes are resampled into a 96-point sequence and passed through a point-sequence Transformer encoder. The model uses local movement features to learn the written shape before predicting the character.

Architecture used in this demo: **128-dimensional point embeddings**, **4 Transformer encoder layers**, **8 attention heads**, and a learned classification token over a 96-point resampled handwriting sequence.

For the current sample:

- **Ground Truth:** `{true_label}`
- **Predicted Character:** `{pred}`
- **Confidence:** `{conf*100:.2f}%`
"""
    )

    st.markdown("---")

    # ======================================================
    # 1. MODEL PIPELINE
    # ======================================================

    st.subheader("Transformer Recognition Pipeline")

    pipeline = [
        ("✍️", "Handwriting Input", "Character strokes"),
        ("📊", "Point Sequence", "96 resampled points"),
        ("🔵", "Point Embedding", "2 → 128"),
        ("🧠", "Point Transformer Encoder", "4 layers • 8 heads"),
        ("📦", "Classification Token", "Learned CLS pooling"),
        ("⚙️", "Classifier", "128 → 62"),
        ("🎯", "Prediction", pred),
    ]

    pipeline_html = '<div class="ez-pipe-row">'
    for i, (icon, title, subtitle) in enumerate(pipeline):
        pipeline_html += f"""
            <div class="ez-pipe-flex">
                <div style="font-size:34px;">{icon}</div>
                <div class="ez-pipe-title">{title}</div>
                <div class="ez-pipe-subtitle">{subtitle}</div>
            </div>
        """
        if i < len(pipeline) - 1:
            pipeline_html += '<div class="ez-arrow-flex">&#8594;</div>'
    pipeline_html += '</div>'

    st.markdown(pipeline_html, unsafe_allow_html=True)

    st.markdown("---")

    # ======================================================
    # 2. PREDICTION SUMMARY
    # ======================================================

    st.subheader("Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Ground Truth", true_label)

    with col2:
        st.metric("Prediction", pred)

    with col3:
        st.metric("Confidence", f"{conf*100:.2f}%")
        st.markdown(confidence_badge(conf), unsafe_allow_html=True)

    with col4:
        verdict = "Correct" if pred == true_label else "Incorrect"
        st.metric("Prediction Status", verdict)

    st.markdown("---")

    # ======================================================
    # 3. TOP-5 PREDICTIONS
    # ======================================================

    st.subheader("Top-5 Predicted Characters")

    top5_df = pd.DataFrame(top5, columns=["Character", "Probability"])
    top5_df["Probability %"] = top5_df["Probability"] * 100

    left, right = st.columns([1.2, 1])

    with left:
        chart_df = top5_df.iloc[::-1]

        st.bar_chart(
            chart_df.set_index("Character")["Probability %"],
            use_container_width=True
        )

    with right:
        st.dataframe(
            top5_df[["Character", "Probability %"]].style.format({
                "Probability %": "{:.2f}"
            }),
            hide_index=True,
            use_container_width=True
        )

    st.info(
        """
The **Top-5 predictions** show the model’s most likely character classes for this handwriting sample.  
A higher probability means the Transformer found stronger evidence that the handwriting belongs to that class.
"""
    )

    st.markdown("---")

    # ======================================================
    # 4. HOW TO READ THE PREDICTION
    # ======================================================

    st.subheader("How to interpret this prediction")

    if pred == true_label:
        st.success(
            f"""
The model correctly recognized this handwriting sample as **{pred}**.

This means the Transformer successfully used the extracted stroke features
to match the writing pattern to the correct character class.
"""
        )
    else:
        st.warning(
            f"""
The model predicted **{pred}**, but the true character is **{true_label}**.

This usually means one of the following:
- the handwriting shape is visually similar to another character,
- some strokes are ambiguous or weak,
- or the model learned a competing pattern from the training data.
"""
        )

    st.markdown("---")

    # ======================================================
    # 5. SAMPLE VISUAL + EXPLANATION
    # ======================================================

    left, right = st.columns([1, 1.15])

    with left:
        st.subheader("Current Character Sample")
        try:
            img = draw_character(parsed_sample)
            st.image(img, use_container_width=True)
        except Exception:
            st.warning("Character image could not be displayed.")

    with right:
        st.subheader("Plain-English Explanation")

        explanation_lines = [
            f"1. The handwriting sample is first broken into **individual strokes**.",
            f"2. Each stroke is converted into a **15-feature numeric vector**.",
            f"3. These vectors form a **3 × 15 input matrix** for the Transformer.",
            f"4. The Transformer studies how the strokes relate to each other and builds an internal representation of the character.",
            f"5. A classifier then converts that representation into probabilities over all possible characters.",
            f"6. For this sample, the most likely prediction is **{pred}** with **{conf*100:.2f}% confidence**."
        ]

        for line in explanation_lines:
            st.write(line)

        if pred == true_label:
            st.success(
                f"The model’s prediction matches the actual label, so this is a successful recognition example."
            )
        else:
            st.warning(
                f"The model confused this sample with **{pred}**, which suggests that the handwritten structure overlaps with another learned pattern."
            )

    st.markdown("---")

    # ======================================================
    # 6. TECHNICAL INPUT VIEW
    # ======================================================

    with st.expander("See the exact Transformer input matrix"):
        feature_df = pd.DataFrame(
            features,
            columns=feature_names,
            index=["Stroke 1", "Stroke 2", "Stroke 3"]
        )
        st.dataframe(feature_df.round(4), use_container_width=True)

elif page == "🧠 Stroke Importance":

    st.header("🧠 Stroke Importance")
    st.caption("Understand which strokes mattered most for the model’s prediction using game-theoretic stroke contributions.")

    st.markdown(
        """
Not all strokes contribute equally to handwritten character recognition.  
This section shows **how important each stroke was** in the final prediction.

The contribution values are based on the game-theoretic analysis you computed for each character.
They help answer questions like:

- **Which stroke mattered most?**
- **Did all strokes cooperate equally?**
- **Was one stroke dominant?**
- **Did any stroke interfere with recognition?**
"""
    )

    st.markdown("---")

    # ======================================================
    # LOAD CHARACTER PROFILE ROW
    # ======================================================

    row = profiles[profiles["character"] == str(pred).strip()]

    if len(row) == 0:
        st.warning("No stroke-importance profile is available for this predicted character.")
    else:
        row = row.iloc[0]

        stroke_1, stroke_2, stroke_3, is_sample_specific = get_stroke_contribution(sample, pred)

        if is_sample_specific:
            st.caption("✅ Showing the exact stroke contributions computed for **this specific sample**.")
        else:
            st.caption(
                "ℹ️ Per-sample Shapley values aren't available for this sample, so the numbers below "
                f"are the **average across all test-set '{pred}' samples**, not this exact handwriting instance."
            )

        contrib = np.array([stroke_1, stroke_2, stroke_3], dtype=float)

        # keep non-negative for display
        contrib = np.maximum(contrib, 0)

        if contrib.sum() == 0:
            contrib[:] = 1.0

        contrib = contrib / contrib.sum()

        colors = ["#ef4444", "#3b82f6", "#22c55e"]

        # ==================================================
        # 1. SUMMARY METRICS
        # ==================================================

        st.subheader("Stroke Contribution Summary")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Predicted Character", pred)

        with c2:
            st.metric("Cooperation Index", f"{row['cooperation_index']:.3f}")

        with c3:
            st.metric("Synergy", f"{row['synergy']:.3f}")

        with c4:
            st.metric("Dominance", f"{row['dominance_score']:.3f}")

        st.markdown("---")

        # ==================================================
        # 2. CONTRIBUTION VISUALS
        # ==================================================

        st.subheader("How much each stroke contributed")

        left, right = st.columns([1.25, 1])

        with left:
            contrib_df = pd.DataFrame({
                "Stroke": ["Stroke 1", "Stroke 2", "Stroke 3"],
                "Contribution": contrib * 100
            })

            st.bar_chart(
                contrib_df.set_index("Stroke")["Contribution"],
                use_container_width=True
            )

        with right:
            st.dataframe(
                contrib_df.style.format({"Contribution": "{:.2f}"}),
                hide_index=True,
                use_container_width=True
            )

        st.info(
            """
A higher contribution means that stroke had a stronger role in the model’s final recognition of the character.  
If one stroke dominates, the prediction depends heavily on that stroke.  
If the values are balanced, the model relies on the full combination of strokes.
"""
        )

        st.markdown("---")

        # ==================================================
        # 3. DONUT + CHARACTER VIEW
        # ==================================================

        vis1, vis2 = st.columns([1, 1])

        with vis1:
            st.subheader("Contribution Distribution")

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.pie(
                contrib,
                labels=["S1", "S2", "S3"],
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.45)
            )
            ax.set_title(f"Stroke Contribution for '{pred}'")
            st.pyplot(fig, use_container_width=False)

        with vis2:
            st.subheader("Character Sample")

            try:
                img = draw_character(parsed_sample)
                st.image(img, use_container_width=True)
            except Exception:
                st.warning("Character image could not be displayed.")

            st.markdown("#### Contribution ranking")

            ranked = sorted(
                [
                    ("Stroke 1", contrib[0]),
                    ("Stroke 2", contrib[1]),
                    ("Stroke 3", contrib[2]),
                ],
                key=lambda x: x[1],
                reverse=True
            )

            for i, (name, value) in enumerate(ranked, start=1):
                st.write(f"**{i}. {name}** — {value*100:.2f}%")

        st.markdown("---")

        # ==================================================
        # 4. CHARACTER TAXONOMY
        # ==================================================

        st.subheader("Character Behaviour Category")

        tax_row = taxonomy[taxonomy["character"] == str(pred).strip()]

        if len(tax_row) == 0:
            st.warning("No taxonomy entry available for this character.")
            category = "Unknown"
        else:
            category = tax_row.iloc[0]["category"]

            if category == "Balanced Cooperation":
                st.success(f"**{category}**")
            elif category == "Asymmetric Cooperation":
                st.info(f"**{category}**")
            elif category == "Dominant Stroke":
                st.warning(f"**{category}**")
            elif category == "Interference":
                st.error(f"**{category}**")
            else:
                st.write(category)

        st.markdown("---")

        # ==================================================
        # 5. PLAIN-ENGLISH INTERPRETATION
        # ==================================================

        st.subheader("Plain-English Interpretation")

        top_idx = int(np.argmax(contrib)) + 1
        top_value = contrib.max() * 100

        if category == "Balanced Cooperation":
            st.success(
                f"""
This character shows **balanced cooperation** between strokes.

That means the Transformer does **not rely on only one stroke**.
Instead, it combines information from all strokes to recognize the character.

For **{pred}**, the strokes work together fairly evenly, so the final prediction is based on the overall shape rather than a single dominant part.
"""
            )

        elif category == "Asymmetric Cooperation":
            st.info(
                f"""
This character shows **asymmetric cooperation**.

That means all strokes help, but **Stroke {top_idx}** contributes the most
at **{top_value:.1f}%**. The remaining strokes still provide useful supporting information,
but the model leans more heavily on one particular structural component.
"""
            )

        elif category == "Dominant Stroke":
            st.warning(
                f"""
This character is dominated by **Stroke {top_idx}**.

That stroke contributes **{top_value:.1f}%** of the total importance,
which means the Transformer relies strongly on it when recognizing the character.

The other strokes are still present, but they play a smaller role in the final decision.
"""
            )

        elif category == "Interference":
            st.error(
                f"""
This character falls into the **interference** category.

That suggests some strokes may be confusing the model rather than helping it.
In other words, the full character may contain a stroke pattern that competes with the model’s learned representation.

This is exactly the kind of case where **stroke removal validation** becomes useful.
"""
            )

        else:
            st.info(
                f"""
This character shows a **mixed stroke interaction pattern**.

The Transformer uses multiple strokes together, but the importance is not perfectly balanced.
For this sample, **Stroke {top_idx}** contributes the most, while the remaining strokes still provide supporting evidence.
"""
            )

        st.markdown("---")

        # ==================================================
        # 6. STROKE-BY-STROKE CONTRIBUTION CARDS
        # ==================================================

        st.subheader("Stroke-by-Stroke Importance")

        contrib_cards_html = '<div class="ez-grid-3">'
        for i in range(3):
            label = f"Stroke {i+1}"
            value = contrib[i] * 100

            if i == np.argmax(contrib):
                tag = "🏆 Most important stroke"
                card_class = "ez-card ez-card-top"
            elif value > 25:
                tag = "💪 Strong supporting stroke"
                card_class = "ez-card ez-card-mid"
            elif value > 10:
                tag = "🙂 Moderate supporting stroke"
                card_class = "ez-card ez-card-mid"
            else:
                tag = "🌫️ Low influence stroke"
                card_class = "ez-card ez-card-low"

            contrib_cards_html += f"""
                <div class="{card_class}">
                    <h3>{label}</h3>
                    <p class="ez-big-number">{value:.2f}%</p>
                    <p>{tag}</p>
                </div>
            """
        contrib_cards_html += '</div>'

        st.markdown(contrib_cards_html, unsafe_allow_html=True)

elif page == "✂️ Stroke Removal Lab":

    st.header("✂️ Stroke Removal Lab")
    st.caption("Interactively remove strokes and observe how the Transformer’s prediction changes.")

    st.markdown(
        """
This section validates the explanation results by answering a very practical question:

> **If a stroke is important, what happens when we remove it?**

You can switch individual strokes on or off, rebuild the character representation, and check how the model’s prediction changes.
This helps verify whether the strokes identified as important are actually important in practice.
"""
    )

    st.markdown("---")

    # ======================================================
    # BASIC SETUP
    # ======================================================

    sample_features = dm.get_sample(sample).copy()
    original_pred, original_conf, original_top5 = predictor.predict(parsed_sample)

    real_strokes = get_real_stroke_indices(sample, sample_features)

    stroke_count = len(real_strokes)

    st.subheader("Original Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Ground Truth", true_label)
    with c2:
        st.metric("Original Prediction", original_pred)
    with c3:
        st.metric("Original Confidence", f"{original_conf*100:.2f}%")

    st.markdown("---")

    # ======================================================
    # STROKE TOGGLES
    # ======================================================

    st.subheader("Stroke Controls")

    st.write(
        "Turn available strokes ON or OFF to see how the model behaves when parts of the character are removed."
    )

    # real_strokes contains indices of strokes that actually exist in this sample
    # e.g. [0] for 1-stroke character, [0,1] for 2-stroke, [0,1,2] for 3-stroke

    keep_flags = [False, False, False]

    if stroke_count == 1:
        toggle_cols = st.columns(1)
    elif stroke_count == 2:
        toggle_cols = st.columns(2)
    else:
        toggle_cols = st.columns(3)

    for col_idx, stroke_idx in enumerate(real_strokes):
        with toggle_cols[col_idx]:
            keep_flags[stroke_idx] = st.toggle(
                f"Keep Stroke {stroke_idx + 1}",
                value=True,
                key=f"stroke_toggle_{sample}_{stroke_idx}"
            )

    # show unavailable strokes separately
    missing_strokes = [i for i in range(3) if i not in real_strokes]

    if missing_strokes:
        st.caption(
            "Not present in this sample: "
            + ", ".join([f"Stroke {i+1}" for i in missing_strokes])
        )

    # ======================================================
    # BUILD MODIFIED FEATURE MATRIX
    # ======================================================

    modified_features = sample_features.copy()
    removed_strokes = []

    for i in range(3):

        # if the stroke doesn't exist in the original sample,
        # keep it zero and never let it participate
        if i not in real_strokes:
            modified_features[i] = np.zeros_like(modified_features[i])
            continue

        # if the stroke exists but user turned it OFF -> remove it
        if not keep_flags[i]:
            modified_features[i] = np.zeros_like(modified_features[i])
            removed_strokes.append(i + 1)

    active_strokes = [
        i + 1
        for i in range(3)
        if np.any(np.abs(modified_features[i]) > 1e-8)
    ]

    # If all strokes removed, avoid model crash / nonsense interpretation
    no_strokes_left = len(active_strokes) == 0

    if no_strokes_left:
        modified_pred = "—"
        modified_conf = 0.0
        modified_top5 = []
    else:
        modified_pred, modified_conf, modified_top5 = predictor.predict_without_strokes(parsed_sample, [s - 1 for s in removed_strokes])

    st.markdown("---")

    # ======================================================
    # COMPARISON
    # ======================================================

    st.subheader("Original vs Modified Prediction")

    left, right = st.columns(2)

    with left:
        st.markdown("### Original Character")
        try:
            img = draw_character(parsed_sample)
            st.image(img, use_container_width=True)
        except Exception:
            st.warning("Character image could not be displayed.")

        st.metric("Prediction", original_pred)
        st.metric("Confidence", f"{original_conf*100:.2f}%")
        st.markdown(confidence_badge(original_conf), unsafe_allow_html=True)

    with right:
        st.markdown("### After Stroke Removal")

        # simple text summary instead of redrawing altered stroke image
        if no_strokes_left:
            st.info("All strokes were removed. No character remains to classify.")
        else:
            st.success(f"Active strokes: {', '.join([f'S{i}' for i in active_strokes])}")

        if removed_strokes:
            st.warning(f"Removed strokes: {', '.join([f'S{i}' for i in removed_strokes])}")
        else:
            st.info("No strokes removed.")

        st.metric("Prediction", modified_pred)
        st.metric("Confidence", f"{modified_conf*100:.2f}%")
        st.markdown(confidence_badge(modified_conf), unsafe_allow_html=True)

    st.markdown("---")

    # ======================================================
    # EFFECT OF REMOVAL
    # ======================================================

    st.subheader("Impact of Stroke Removal")

    if no_strokes_left:
        st.error(
            """
All strokes were removed, so the model no longer has any handwriting information to classify.
This acts like a complete ablation of the sample.
"""
        )
    else:
        conf_drop = (original_conf - modified_conf) * 100

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Original Prediction", original_pred)

        with m2:
            st.metric("Modified Prediction", modified_pred)

        with m3:
            st.metric(
                "Confidence Change",
                f"{modified_conf*100 - original_conf*100:+.2f}%"
            )

        if modified_pred != original_pred:
            st.error(
                f"""
Removing the selected stroke(s) changed the model’s predicted class from **{original_pred}** to **{modified_pred}**.

This is strong evidence that the removed stroke(s) were important to the Transformer’s decision.
"""
            )
        else:
            if modified_conf < original_conf:
                st.warning(
                    f"""
The prediction stayed the same (**{modified_pred}**), but confidence dropped by **{conf_drop:.2f}%**.

This suggests the removed stroke(s) still contributed useful information,
even though they were not strong enough to completely change the final class.
"""
                )
            elif modified_conf > original_conf:
                st.success(
                    f"""
The prediction stayed the same and confidence actually increased.

This suggests that one of the removed strokes may have been weak, noisy, or slightly confusing for the model.
"""
                )
            else:
                st.info(
                    """
The prediction and confidence stayed almost unchanged,
which suggests the removed stroke(s) had limited influence on the final decision.
"""
                )

    st.markdown("---")

    # ======================================================
    # TOP-5 AFTER REMOVAL
    # ======================================================

    st.subheader("Top-5 Predictions After Removal")

    if no_strokes_left:
        st.info("No top-5 predictions available because all strokes were removed.")
    else:
        mod_df = pd.DataFrame(modified_top5, columns=["Character", "Probability"])
        mod_df["Probability %"] = mod_df["Probability"] * 100

        colA, colB = st.columns([1.2, 1])

        with colA:
            chart_df = mod_df.iloc[::-1]
            st.bar_chart(
                chart_df.set_index("Character")["Probability %"],
                use_container_width=True
            )

        with colB:
            st.dataframe(
                mod_df[["Character", "Probability %"]].style.format({
                    "Probability %": "{:.2f}"
                }),
                hide_index=True,
                use_container_width=True
            )

    st.markdown("---")

    # ======================================================
    # RESEARCH VALIDATION MESSAGE
    # ======================================================

    st.subheader("What this tells us")

    if no_strokes_left:
        st.info(
            """
Removing every stroke eliminates all evidence needed for recognition.
This is the extreme case of stroke ablation.
"""
        )
    else:
        if modified_pred != original_pred:
            st.success(
                """
This is a strong **stroke removal validation result**:
the removed stroke(s) were clearly important because the model changed its final decision.
"""
            )
        elif modified_conf < original_conf:
            st.info(
                """
This is a **soft validation result**:
the prediction stayed the same, but confidence decreased,
showing that the removed stroke(s) still helped the model.
"""
            )
        elif modified_conf > original_conf:
            st.warning(
                """
This is an interesting case:
removing a stroke improved confidence, which may indicate that the removed stroke introduced ambiguity.
"""
            )
        else:
            st.info(
                """
The removed stroke(s) had little effect on this particular sample,
so the model likely relied more heavily on the remaining strokes.
"""
            )

    with st.expander("See modified feature matrix"):
        mod_df = pd.DataFrame(
            modified_features,
            columns=feature_names,
            index=["Stroke 1", "Stroke 2", "Stroke 3"]
        )
        st.dataframe(mod_df.round(4), use_container_width=True)

elif page == "📘 Research Interpretation":

    st.header("📘 Research Interpretation")
    st.caption("Interpret the model’s behaviour in both a layman-friendly and research-oriented way.")

    st.markdown(
        """
This page combines the prediction result, stroke contributions, and stroke-interaction statistics
to explain **how the Transformer recognized the current character**.

It is meant to answer two audiences at once:

- **A layman / dashboard user** → “Which stroke mattered most and why?”
- **A research audience** → “What does this say about cooperation, dominance, and interpretability?”
"""
    )

    st.markdown("---")

    # ======================================================
    # LOAD PROFILE + TAXONOMY
    # ======================================================

    row = profiles[profiles["character"] == str(pred).strip()]
    tax_row = taxonomy[taxonomy["character"] == str(pred).strip()]

    if len(row) == 0:
        st.warning("No character profile available for this predicted character.")
    else:
        row = row.iloc[0]

        stroke_1, stroke_2, stroke_3, is_sample_specific = get_stroke_contribution(sample, pred)

        if is_sample_specific:
            st.caption("✅ Stroke contributions below are computed for **this specific sample**.")
        else:
            st.caption(
                "ℹ️ Showing the **average** stroke contribution across all test-set "
                f"'{pred}' samples, since per-sample Shapley data isn't loaded for this sample."
            )

        contrib = np.array([stroke_1, stroke_2, stroke_3], dtype=float)

        contrib = np.maximum(contrib, 0)
        if contrib.sum() == 0:
            contrib[:] = 1.0
        contrib = contrib / contrib.sum()

        top_idx = int(np.argmax(contrib)) + 1
        top_val = contrib.max() * 100

        if len(tax_row) == 0:
            category = "Unknown"
        else:
            category = tax_row.iloc[0]["category"]

        # ==================================================
        # SECTION 1 — EXECUTIVE SUMMARY
        # ==================================================

        st.subheader("Executive Summary")

        summary_cols = st.columns(4)

        with summary_cols[0]:
            st.metric("Ground Truth", true_label)

        with summary_cols[1]:
            st.metric("Prediction", pred)

        with summary_cols[2]:
            st.metric("Top Stroke", f"Stroke {top_idx}")

        with summary_cols[3]:
            st.metric("Top Contribution", f"{top_val:.2f}%")

        st.markdown("---")

        # ==================================================
        # SECTION 2 — LAYMAN VIEW
        # ==================================================

        st.subheader("Layman-Friendly Interpretation")

        layman_left, layman_right = st.columns([1, 1.2])

        with layman_left:
            try:
                img = draw_character(parsed_sample)
                st.image(img, use_container_width=True)
            except Exception:
                st.warning("Character image could not be displayed.")

        with layman_right:
            st.markdown("### What the model is saying")

            if pred == true_label:
                st.success(
                    f"""
The model correctly recognized this handwritten character as **{pred}**.
"""
                )
            else:
                st.warning(
                    f"""
The model predicted **{pred}**, while the actual character is **{true_label}**.
This means the handwriting pattern looked more similar to **{pred}** according to the model.
"""
                )

            st.write(
                f"""
The most influential part of the character was **Stroke {top_idx}**,
which contributed **{top_val:.2f}%** of the total stroke importance.

So if someone asks:

> **“What part of the handwriting mattered most?”**

the answer is:

> **Stroke {top_idx} had the biggest influence on the Transformer’s decision.**
"""
            )

            if category == "Balanced Cooperation":
                st.info(
                    """
The strokes worked together in a balanced way.
The model did not rely on just one part — it used the overall structure of the character.
"""
                )

            elif category == "Asymmetric Cooperation":
                st.info(
                    """
One stroke mattered more than the others, but the remaining strokes still helped.
So the model used one strong structural clue plus supporting evidence from the rest.
"""
                )

            elif category == "Dominant Stroke":
                st.warning(
                    """
One stroke was doing most of the work.
That means the model heavily relied on a single stroke to recognize the character.
"""
                )

            elif category == "Interference":
                st.error(
                    """
Some strokes may be confusing the model rather than helping it.
This means the full handwritten character contains competing stroke information.
"""
                )

            else:
                st.info(
                    """
The character shows a mixed interaction pattern between strokes.
Some strokes help more than others, but the behaviour is not purely balanced or purely dominated by one stroke.
"""
                )

        st.markdown("---")

        # ==================================================
        # SECTION 3 — RESEARCH VIEW
        # ==================================================

        st.subheader("Research-Oriented Interpretation")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Cooperation Index", f"{row['cooperation_index']:.3f}")

        with r2:
            st.metric("Synergy", f"{row['synergy']:.3f}")

        with r3:
            st.metric("Dominance Score", f"{row['dominance_score']:.3f}")

        st.markdown("### Interpretation of the metrics")

        st.write(
            """
**1. Stroke contribution values** estimate how much each stroke helps the model recognize a character.  
These are useful for identifying whether the Transformer depends on one critical stroke or on a combination of strokes.

**2. Cooperation index** reflects how evenly the strokes contribute.  
A higher value usually suggests that multiple strokes work together rather than one stroke completely dominating the decision.

**3. Dominance score** indicates whether a single stroke carries most of the recognition burden.  
High dominance implies that one stroke acts as the main discriminative feature.

**4. Synergy** reflects interaction effects between strokes.  
If synergy is strong, the model benefits from seeing certain strokes together rather than independently.
"""
        )

        st.markdown("---")

        # ==================================================
        # SECTION 4 — CATEGORY EXPLANATION
        # ==================================================

        st.subheader("Category-Based Behaviour Analysis")

        if category == "Balanced Cooperation":
            st.success("Balanced Cooperation")

            st.write(
                """
In a **Balanced Cooperation** pattern, no single stroke is solely responsible for recognition.
The Transformer appears to encode the character as a joint combination of strokes,
which is desirable for structurally complete characters.

**Interpretation:**  
The character identity is distributed across multiple pen movements rather than concentrated in one stroke.
"""
            )

        elif category == "Asymmetric Cooperation":
            st.info("Asymmetric Cooperation")

            st.write(
                f"""
In **Asymmetric Cooperation**, all strokes contribute, but one stroke contributes more strongly than the others.
For this character, **Stroke {top_idx}** appears to be the main structural cue,
while the remaining strokes provide supporting information.

**Interpretation:**  
The character is recognized through a mix of dominance and cooperation.
"""
            )

        elif category == "Dominant Stroke":
            st.warning("Dominant Stroke")

            st.write(
                f"""
This character exhibits a **Dominant Stroke** pattern.
The model relies heavily on **Stroke {top_idx}** for recognition,
suggesting that a single stroke contains most of the class-discriminative information.

**Interpretation:**  
The learned representation may be vulnerable if that stroke is missing, distorted, or noisy.
"""
            )

        elif category == "Interference":
            st.error("Interference")

            st.write(
                """
The **Interference** category suggests that some strokes may reduce clarity rather than improve it.
This is especially interesting from an explainability perspective because it means
that more handwriting information is not always better.

**Interpretation:**  
Certain strokes may introduce ambiguity or overlap with competing character classes.
"""
            )

        else:
            st.info("Mixed Behaviour")

            st.write(
                """
The character shows a mixed interaction pattern.
Some strokes are clearly more useful than others, but the overall behaviour does not fit neatly into one category.
"""
            )

        st.markdown("---")

        # ==================================================
        # SECTION 5 — VALIDATION LINK
        # ==================================================

        st.subheader("Connection to Stroke Removal Validation")

        st.write(
            """
The purpose of stroke-removal validation is to test whether these explanation scores are meaningful.

If the explanation says a stroke is important, then **removing that stroke should usually reduce confidence or change the prediction**.
If removing a supposedly unimportant stroke has almost no effect, that also supports the explanation.

So the dashboard creates a loop:

1. **Predict the character**
2. **Measure stroke importance**
3. **Remove selected strokes**
4. **Check whether the model behaviour changes**

This is what makes the explanation more convincing than a static importance score alone.
"""
        )

        st.markdown("---")

        # ==================================================
        # SECTION 6 — TAKEAWAY BOX
        # ==================================================

        st.subheader("Takeaway")

        st.success(
            f"""
For the current sample, the point-sequence Transformer predicts **{pred}** and relies most strongly on **Stroke {top_idx}**.

From an explainability perspective, this character is best described as **{category}**,
which tells us how the model distributes importance across strokes and whether recognition is driven by cooperation, dominance, or interference.
"""
        )

elif page == "🚀 Future Scope":

    st.header("🚀 Future Scope")
    st.caption("Where this project can go next — especially from letters to words.")

    st.markdown(
        """
The current system focuses on **single handwritten characters**.
That already allows us to study stroke-level contributions, point-sequence Transformer predictions, and interpretability.

But the bigger long-term goal is more exciting:

> **Can we move from explaining single letters to explaining full handwritten words?**
"""
    )

    st.markdown("---")

    # ======================================================
    # 1. CURRENT STAGE
    # ======================================================

    st.subheader("Where the project stands right now")

    st.success(
        """
### Current achievement
- Handwritten **character-level recognition**
- Point-sequence Transformer classifier
- Stroke-wise feature representation with padding masks
- Game-theoretic stroke importance analysis using the masked model
- Stroke-removal validation dashboard
- Research interpretation of cooperation / dominance / interference
"""
    )

    st.markdown("---")

    # ======================================================
    # 2. WORD-LEVEL EXTENSION
    # ======================================================

    st.subheader("Next major direction: From letters to words")

    st.markdown(
        """
Moving from **letters to words** would be a natural and very meaningful extension of this project.

Instead of recognizing a single character like **a** or **b**, the system would try to recognize full handwritten words such as:

- `cat`
- `hello`
- `transformer`
- `BITS`

This introduces a richer problem because the model must now understand:

- multiple characters in sequence,
- spacing and segmentation,
- relationships between letters,
- and possibly writer-specific variations across the whole word.
"""
    )

    st.markdown("---")

    # ======================================================
    # 3. WHAT WOULD CHANGE
    # ======================================================

    st.subheader("What changes when we move to words?")

    st.markdown(
        """
        <div class="ez-grid-3">
            <div class="ez-card ez-c1">
                <h3>1. Larger Input Structure</h3>
                <p>
                A word contains many more strokes than a single character.
                So the model would need to process a longer stroke sequence,
                or a sequence of character-level embeddings.
                </p>
            </div>
            <div class="ez-card ez-c2">
                <h3>2. Character Segmentation</h3>
                <p>
                The system must decide where one handwritten character ends
                and the next begins — or learn to handle the whole word directly.
                </p>
            </div>
            <div class="ez-card ez-c3">
                <h3>3. Sequence Prediction</h3>
                <p>
                Instead of predicting one label, the model would predict a sequence of letters
                or directly output the final word.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ======================================================
    # 4. POSSIBLE TECHNICAL PATHS
    # ======================================================

    st.subheader("Possible technical extensions")

    st.markdown(
        """
### A. Character-level word pipeline
One practical approach would be:

1. **Segment the handwritten word into characters**
2. Run the current **character recognizer** on each segment
3. Combine the predictions into a word
4. Apply language constraints or dictionary correction

This would reuse a large part of the current project.

---

### B. Word-level Transformer
Another approach would be to build a **word-level point-sequence Transformer** that directly receives a longer sequence of real strokes for the whole word.

This model could learn:
- which strokes belong to which letter,
- how letters connect,
- and which parts of the word are most informative.

---

### C. Hierarchical explainability
A very interesting research direction would be to build **multi-level explanations**:

- **Stroke importance within a letter**
- **Letter importance within a word**
- **Word-level confidence and ambiguity**

That would make the dashboard much more powerful.
"""
    )

    st.markdown("---")

    # ======================================================
    # 5. FUTURE DASHBOARD IDEAS
    # ======================================================

    st.subheader("Future dashboard ideas")

    st.info(
        """
### If this project grows to words, the dashboard could include:

- **Word canvas** showing the full handwritten word
- **Character segmentation view** to split the word into letters
- **Letter-by-letter prediction timeline**
- **Stroke importance inside each character**
- **Word-level explanation** showing which letters caused ambiguity
- **Interactive removal of letters or strokes** to see how the final word prediction changes
"""
    )

    st.markdown("---")

    # ======================================================
    # 6. OTHER FUTURE DIRECTIONS
    # ======================================================

    st.subheader("Other future improvements")

    st.write(
        """
Beyond the letters-to-words extension, this project can also be improved in several other directions:

### 1. Better visual explainability
- richer animations for stroke formation
- dynamic highlighting of influential strokes
- hover-based explanations for feature values

### 2. Better modeling
- carefully tuned point-sequence Transformer encoders
- attention visualization
- contrastive learning for confusing characters

### 3. Better datasets
- more writers
- more writing styles
- more difficult or noisy handwriting samples

### 4. Real-time handwriting input
- allow a user to draw directly in the dashboard
- run prediction instantly
- explain the result live

### 5. Educational / accessibility use
- show children how a letter should be formed
- detect missing or malformed strokes
- provide corrective feedback during handwriting learning
"""
    )

    st.markdown("---")

    # ======================================================
    # 7. FINAL TAKEAWAY
    # ======================================================

    st.subheader("Final Vision")

    st.success(
        """
This project starts with a simple question:

> **How does a point-sequence Transformer recognize a handwritten character, and which stroke matters most?**

But it can grow into something much larger:

- an **interactive explainable handwriting system**
- a **stroke-aware word recognizer**
- a **research platform for game-theoretic interpretability**
- and eventually a tool that explains not just *what* was written, but *why the model understood it that way*.
"""
    )


