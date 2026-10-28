"""
================================================================================
Mobile Phone Customer Churn Classification Using Machine Learning
================================================================================
File: app.py
Description: Clean, professional Streamlit dashboard and real-time churn
             prediction system for two telecom churn datasets.
Theme: White background, navy-blue (#0F172A / #1E3A8A) and teal (#0D9488).
Datasets: Telco Customer Churn, Cell2Cell Telecom Churn (CSV upload only).
Tech Stack: Streamlit, Plotly, Scikit-learn, Pandas, NumPy, Joblib.
================================================================================
"""

import hashlib
import io

import pandas as pd
import streamlit as st

import dataset_views as views
from dataset_manager import (
    SUPPORTED_DATASETS,
    DatasetError,
    dataset_diagnostics,
    detect_target,
    ensure_supported_dataset,
    identify_dataset,
    prepare_dataset,
    scan_for_leakage,
    suggest_excluded_columns,
    train_models,
    validate_target,
)

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & METADATA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnPulse | Customer Churn Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 2. CLEAN THEME (WHITE · NAVY · TEAL)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy: #0F172A;
        --blue: #1E3A8A;
        --teal: #0D9488;
        --teal-dark: #0F766E;
        --muted: #64748B;
        --border: #E5E7EB;
        --surface: #FFFFFF;
        --canvas: #F8FAFC;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, Roboto,
                     'Helvetica Neue', Arial, sans-serif;
        background: #FFFFFF;
        color: var(--navy);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    /* ------------------------------------------------------------ upload box */
    [data-testid="stFileUploaderDropzone"] {
        border: 1.6px dashed #5EEAD4;
        border-radius: 12px;
        background: #F0FDFA;
        padding: 1rem;
        min-height: 88px;
        transition: border-color 0.22s ease, background 0.22s ease, box-shadow 0.22s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #0D9488;
        background: #ECFDF5;
        box-shadow: 0 0 0 4px rgba(13, 148, 136, 0.07);
    }

    /* ------------------------------------------------------- motion & welcome */
    @keyframes fade-up {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fade-in {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes shimmer {
        from { background-position: 0% 50%; }
        to   { background-position: 200% 50%; }
    }
    @keyframes float-soft {
        0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.55; }
        50%      { transform: translate(-50%, -50%) scale(1.1); opacity: 0.85; }
    }

    .welcome-hero {
        position: relative;
        text-align: center;
        padding: 0.6rem 0 0.2rem;
        margin: 0 auto 1.6rem;
        max-width: 920px;
    }
    .welcome-hero h1 {
        position: relative;
        z-index: 1;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        color: var(--navy) !important;
        margin: 0 !important;
        line-height: 1.2;
        animation: fade-up 0.6s cubic-bezier(0.22, 1, 0.36, 1) backwards;
    }
    .welcome-glow {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 420px;
        height: 260px;
        transform: translate(-50%, -50%);
        background: radial-gradient(circle, rgba(13, 148, 136, 0.16) 0%,
                    rgba(56, 189, 248, 0.10) 45%, transparent 72%);
        filter: blur(24px);
        z-index: 0;
        pointer-events: none;
        animation: float-soft 9s ease-in-out infinite;
    }
    .welcome-underline {
        position: relative;
        z-index: 1;
        width: 78px;
        height: 4px;
        margin: 1rem auto 0;
        border-radius: 999px;
        background: linear-gradient(90deg, #0D9488, #38BDF8, #0D9488);
        background-size: 200% 100%;
        animation: shimmer 5.5s linear infinite, fade-in 0.9s ease 0.15s backwards;
    }
    .st-key-welcome_upload {
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        transition: transform 0.25s ease, box-shadow 0.25s ease,
                    border-color 0.25s ease;
        animation: fade-up 0.65s cubic-bezier(0.22, 1, 0.36, 1) 0.1s backwards;
    }
    .st-key-welcome_upload:hover {
        transform: translateY(-2px);
    }
    .st-key-welcome_upload:hover,
    .st-key-welcome_upload:hover [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #99F6E4 !important;
        box-shadow: 0 18px 40px rgba(13, 148, 136, 0.10);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: inherit;
        color: var(--navy);
        font-weight: 700;
        letter-spacing: -0.2px;
    }

    p, li, label { color: #334155; }

    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

    /* ---------------------------------------------------------------- sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    [data-testid="stSidebar"] > div:first-child {
        animation: fade-in 0.45s ease-out backwards;
    }
    ::selection { background: #CCFBF1; color: #0F172A; }

    /* ------------------------------------------------------------------- hero */
    .hero-banner {
        background: linear-gradient(135deg, #F8FAFC 0%, #F0FDFA 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--teal);
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 6px 24px rgba(15, 23, 42, 0.04);
    }
    .hero-banner .hero-badge {
        display: inline-block;
        background: #ECFDF5;
        color: var(--teal-dark);
        border: 1px solid #CCFBF1;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.71rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .hero-banner .hero-title {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: var(--navy) !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
        line-height: 1.25;
    }
    .hero-banner .hero-subtitle {
        color: #475569 !important;
        font-size: 0.97rem !important;
        line-height: 1.6;
        margin: 0.45rem 0 0 0 !important;
        max-width: 75ch;
    }

    /* -------------------------------------------------------------------- KPI */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.05rem 1.15rem;
        transition: box-shadow 0.22s ease, border-color 0.22s ease,
                    transform 0.22s ease;
        animation: fade-up 0.5s cubic-bezier(0.22, 1, 0.36, 1) backwards;
    }
    .kpi-container .kpi-card:nth-child(1) { animation-delay: 0.03s; }
    .kpi-container .kpi-card:nth-child(2) { animation-delay: 0.08s; }
    .kpi-container .kpi-card:nth-child(3) { animation-delay: 0.13s; }
    .kpi-container .kpi-card:nth-child(4) { animation-delay: 0.18s; }
    .kpi-card:hover {
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
        border-color: #CCFBF1;
        transform: translateY(-3px);
    }
    .kpi-navy { border-top: 3px solid var(--blue); }
    .kpi-teal { border-top: 3px solid var(--teal); }
    .kpi-rose { border-top: 3px solid #EF4444; }
    .kpi-amber { border-top: 3px solid #D97706; }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
    }
    .kpi-title {
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-icon {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        flex: 0 0 auto;
    }
    .icon-navy { background: #EFF6FF; }
    .icon-teal { background: #F0FDFA; }
    .icon-rose { background: #FEF2F2; }
    .icon-amber { background: #FFFBEB; }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--navy);
        line-height: 1.15;
        margin-top: 0.4rem;
    }
    .kpi-desc {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.15rem;
    }
    .kpi-desc span, .kpi-desc strong { color: var(--muted) !important; }

    /* ------------------------------------------------------------------ cards */
    .dashboard-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
        transition: box-shadow 0.22s ease, border-color 0.22s ease,
                    transform 0.22s ease;
    }
    .dashboard-card:hover {
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.07);
        border-color: #E2E8F0;
        transform: translateY(-2px);
    }
    .dashboard-card h4 {
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--navy) !important;
        margin: 0 0 0.55rem 0;
    }
    .dashboard-card p, .dashboard-card li { color: #475569; }

    .insight-box {
        background: var(--canvas);
        border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #334155 !important;
        font-size: 0.88rem;
        line-height: 1.6;
        margin: 0.9rem 0 0 0;
    }
    .insight-box strong { color: var(--navy) !important; }

    /* --------------------------------------------------------------- prediction */
    .result-card-danger {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #EF4444;
        border-radius: 12px;
        padding: 1.35rem;
    }
    .result-card-safe {
        background: #F0FDFA;
        border: 1px solid #99F6E4;
        border-left: 4px solid var(--teal);
        border-radius: 12px;
        padding: 1.35rem;
    }
    .result-card-danger h2 { color: #991B1B !important; }
    .result-card-safe h2 { color: var(--teal-dark) !important; }
    .result-card-danger p, .result-card-safe p { color: #475569 !important; }
    .result-card-danger strong, .result-card-safe strong { color: var(--navy) !important; }

    .badge-churn-yes {
        display: inline-block;
        background: #FEE2E2;
        color: #B91C1C;
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.4px;
    }
    .badge-churn-no {
        display: inline-block;
        background: #CCFBF1;
        color: var(--teal-dark);
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.4px;
    }

    .chip-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.7rem;
        margin-top: 1rem;
    }
    .metric-chip {
        background: var(--canvas);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.7rem 0.85rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease,
                    border-color 0.18s ease;
    }
    .metric-chip:hover {
        transform: translateY(-2px);
        border-color: #CCFBF1;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
    }
    .metric-chip-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .metric-chip-value {
        font-size: 1.12rem;
        font-weight: 800;
        color: var(--navy);
        margin-top: 0.15rem;
    }

    /* ---------------------------------------------------------------- controls */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        border: 1px solid var(--border);
        transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease,
                    transform 0.18s ease, box-shadow 0.18s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(15, 23, 42, 0.09);
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.99);
        box-shadow: none;
    }
    .stButton > button p,
    .stButton > button div,
    .stButton > button span,
    .stDownloadButton > button p,
    .stDownloadButton > button div,
    .stDownloadButton > button span,
    [data-testid="stBaseButton-primary"] p {
        color: inherit !important;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--teal);
        border: 1px solid var(--teal);
        color: #FFFFFF;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
        color: #FFFFFF;
    }
    .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid var(--border);
        font-weight: 600;
        transition: border-color 0.18s ease, color 0.18s ease,
                    transform 0.18s ease, box-shadow 0.18s ease;
    }
    .stDownloadButton > button:hover {
        border-color: var(--teal);
        color: var(--teal-dark);
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(15, 23, 42, 0.08);
    }
    .stLinkButton a, [data-testid="stLinkButton"] a {
        transition: transform 0.18s ease, box-shadow 0.18s ease,
                    border-color 0.18s ease;
    }
    .stLinkButton a:hover, [data-testid="stLinkButton"] a:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(15, 23, 42, 0.07);
    }

    div[data-testid="stSegmentedControl"] button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    [data-testid="stButtonGroup"] button {
        transition: background 0.18s ease, color 0.18s ease,
                    border-color 0.18s ease, transform 0.18s ease;
    }
    [data-testid="stButtonGroup"] button:hover {
        transform: translateY(-1px);
    }

    [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
        transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }
    [data-baseweb="input"]:focus-within,
    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="textarea"]:focus-within {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.12);
    }
    [data-testid="stFileUploaderDropzone"] button {
        transition: background 0.18s ease, border-color 0.18s ease,
                    transform 0.18s ease, box-shadow 0.18s ease;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
    }
    button:focus-visible,
    a:focus-visible,
    input:focus-visible,
    [role="button"]:focus-visible {
        outline: 2px solid var(--teal);
        outline-offset: 2px;
    }

    [data-testid="stForm"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        background: #FFFFFF;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border) !important;
        border-radius: 12px !important;
        background: #FFFFFF;
    }

    /* --------------------------------------------------------------- responsive */
    @media (max-width: 1100px) {
        .kpi-container { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .hero-banner .hero-title { font-size: 1.55rem !important; }
    }
    @media (max-width: 640px) {
        .kpi-container { grid-template-columns: 1fr; }
        .hero-banner { padding: 1.2rem 1.25rem; }
        .hero-banner .hero-title { font-size: 1.35rem !important; }
        .welcome-hero h1 { font-size: 1.95rem !important; }
        .welcome-glow { width: 300px; height: 200px; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 3. PAGE NAVIGATION
# ------------------------------------------------------------------------------
PAGE_OPTIONS = [
    "Dashboard",
    "Prediction",
    "Visualization",
    "Performance",
    "About",
]

#: Session keys that describe the current upload; cleared on reset / new file.
UPLOAD_STATE_KEYS = (
    "upload_bytes",
    "upload_name",
    "upload_id",
    "target_col",
    "excluded_cols",
    "settings_target",
    "settings_excluded",
    "_target_seen",
    "train_request",
    "gen_last_prediction",
)


@st.cache_data(show_spinner=False)
def parse_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parses an uploaded CSV (delimiter and encoding auto-detected)."""
    last_error: Exception | None = None
    frame: pd.DataFrame | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(
                io.BytesIO(raw_bytes), sep=None, engine="python", encoding=encoding
            )
            break
        except Exception as error:  # try the next encoding
            last_error = error

    if frame is None:
        raise ValueError(f"Could not read the uploaded file as CSV: {last_error}")

    if frame.empty or frame.shape[1] < 2:
        raise ValueError(
            "The file has no usable tabular data. A churn dataset needs several columns "
            "and at least a few rows."
        )
    if frame.shape[0] < 10:
        raise ValueError(
            f"The file has only {frame.shape[0]:,} row(s). Upload a complete churn dataset "
            "with at least 10 rows so a train/test split is possible."
        )

    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


@st.cache_resource(show_spinner=False)
def train_uploaded_dataset(raw_bytes: bytes, target: str, drop_cols: tuple) -> dict:
    """
    Prepares the uploaded dataset and benchmarks all four algorithms on it.
    Cached by file content + target + excluded columns, so changing UI controls
    does not retrain the models.
    """
    frame = parse_uploaded_csv(raw_bytes)
    dataset = ensure_supported_dataset(frame)
    X, y, schema = prepare_dataset(frame, target, drop_cols)
    result = train_models(X, y, schema)
    result["dataset"] = dataset
    return result


@st.cache_data(show_spinner=False)
def preview_schema(raw_bytes: bytes, target: str, drop_cols: tuple):
    """Feature schema without fitting anything (used before training)."""
    frame = parse_uploaded_csv(raw_bytes)
    _, _, schema = prepare_dataset(frame, target, drop_cols)
    return schema


@st.cache_data(show_spinner=False)
def leakage_scan(raw_bytes: bytes, target: str):
    """Cached heuristic leakage investigation for the current upload."""
    frame = parse_uploaded_csv(raw_bytes)
    return scan_for_leakage(frame, target)


def reset_upload() -> None:
    """Clears the uploaded dataset and every derived value."""
    for key in UPLOAD_STATE_KEYS:
        st.session_state.pop(key, None)


def change_dataset() -> None:
    """
    Returns to the welcome screen with an empty file picker.

    Safe to call from any page because the uploader widget is not rendered
    there, so its session key can be cleared without a Streamlit state error.
    """
    reset_upload()
    st.session_state.pop("dataset_uploader", None)
    st.rerun()


def build_context() -> dict:
    """
    Resolves the current upload into one dictionary that every page reads from.
    Never raises: problems are reported through the `error`/`unsupported` keys.
    """
    context = {
        "df": None,
        "filename": st.session_state.get("upload_name", "dataset.csv"),
        "dataset": None,
        "target": None,
        "target_report": None,
        "excluded": [],
        "schema": None,
        "training": None,
        "payload": None,
        "results_df": None,
        "error": None,
        "unsupported": False,
        "ready": False,
        "raw_bytes": None,
        "leakage": [],
    }

    raw_bytes = st.session_state.get("upload_bytes")
    if not raw_bytes:
        return context

    context["raw_bytes"] = raw_bytes

    try:
        frame = parse_uploaded_csv(raw_bytes)
    except ValueError as error:
        context["error"] = str(error)
        return context

    context["df"] = frame
    dataset = identify_dataset(frame)
    if dataset is None:
        context["error"] = (
            "This CSV is not one of the two supported churn datasets. Upload "
            "Telco-Customer-Churn.csv or cell2celltrain.csv."
        )
        context["unsupported"] = True
        return context
    context["dataset"] = dataset

    # `settings_*` are plain session keys (not widgets), so the confirmed setup
    # survives even when the setup screen is not rendered and Streamlit cleans
    # up the widget state for this run.
    target = st.session_state.get("settings_target")
    if target not in frame.columns:
        target = st.session_state.get("target_col")
    if target not in frame.columns:
        target = detect_target(frame, dataset.target_candidates)
    context["target"] = target

    report = validate_target(frame, target)
    context["target_report"] = report
    if not report.is_valid:
        context["error"] = report.error
        return context

    try:
        context["leakage"] = leakage_scan(raw_bytes, target)
    except Exception:  # pragma: no cover - the scan must never block training
        context["leakage"] = []

    if "settings_excluded" in st.session_state:
        excluded = tuple(st.session_state["settings_excluded"])
    else:
        excluded = tuple(st.session_state.get("excluded_cols", []))
    context["excluded"] = list(excluded)

    # Train only when the user asked for it with this exact configuration.
    training = None
    request = st.session_state.get("train_request")
    if request == (st.session_state.get("upload_id"), target, excluded):
        try:
            with st.spinner("Training and benchmarking the four algorithms…"):
                training = train_uploaded_dataset(raw_bytes, target, excluded)
        except DatasetError as error:
            context["error"] = str(error)
        except Exception as error:  # pragma: no cover - defensive UI guard
            context["error"] = f"Training failed: {error}"

    if training is not None:
        schema = training["schema"]
        context["schema"] = schema
        context["training"] = training
        context["results_df"] = training["results_df"]
        context["payload"] = {
            "pipeline": training["best_pipeline"],
            "best_model_name": training["best_model_name"],
            "metrics": training["best_metrics"],
            "feature_names": schema.feature_cols,
            "num_cols": schema.numeric_cols,
            "cat_cols": schema.categorical_cols,
            "target_col": schema.target,
            "positive_label": schema.positive_label,
            "negative_label": schema.negative_label,
            "confusion_matrix": training["confusion_matrix"].tolist(),
        }
        context["ready"] = True
        return context

    # Not trained yet: build a lightweight schema so charts still work.
    if context["error"] is None:
        try:
            context["schema"] = preview_schema(raw_bytes, target, excluded)
        except DatasetError as error:
            context["error"] = str(error)

    context["ready"] = context["schema"] is not None and context["error"] is None
    return context


# ------------------------------------------------------------------------------
# 4. SIDEBAR — DATASET INFO (THE APP ALSO WORKS WITHOUT IT)
# ------------------------------------------------------------------------------
ctx = build_context()

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.6rem;">
            <div style="background:#0D9488;width:36px;height:36px;border-radius:9px;display:flex;
                        align-items:center;justify-content:center;font-size:1.05rem;color:#FFFFFF;">📈</div>
            <div>
                <div style="font-size:1.02rem;font-weight:800;color:#0F172A;letter-spacing:-0.3px;">ChurnPulse</div>
                <div style="font-size:0.66rem;font-weight:600;color:#64748B;text-transform:uppercase;
                            letter-spacing:0.7px;">Telecom churn studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if ctx.get("dataset") is not None:
        dataset = ctx["dataset"]
        report = ctx.get("target_report")
        rows = len(ctx["df"])
        cols = ctx["df"].shape[1]
        target = ctx.get("target") or "—"
        classes = (
            f"{report.negative_label} / {report.positive_label}"
            if report is not None and report.is_valid
            else "—"
        )
        st.markdown(
            f"""
            <div style="background:#F8FAFC;border:1px solid #E5E7EB;border-radius:12px;
                        padding:0.85rem 0.9rem;">
                <div style="font-size:0.64rem;font-weight:700;color:#0D9488;text-transform:uppercase;
                            letter-spacing:0.6px;">Current dataset</div>
                <div style="font-weight:800;color:#0F172A;font-size:0.92rem;margin-top:0.1rem;">
                    {dataset.name}</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:0.3rem;word-break:break-all;">
                    📄 {ctx.get('filename', 'dataset.csv')}</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:0.35rem;">
                    Rows <strong>{rows:,}</strong> · Columns <strong>{cols}</strong>
                </div>
                <div style="font-size:0.75rem;color:#475569;margin-top:0.2rem;">
                    Target <strong>{target}</strong> · Classes <strong>{classes}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ctx.get("training"):
            metrics = ctx["training"]["best_metrics"]
            st.markdown(
                f"""
                <div style="background:#F0FDFA;border:1px solid #99F6E4;border-radius:12px;
                            padding:0.8rem 0.9rem;margin-top:0.6rem;">
                    <div style="font-size:0.64rem;font-weight:700;color:#0F766E;text-transform:uppercase;
                                letter-spacing:0.6px;">Active model</div>
                    <div style="font-weight:800;color:#0F172A;font-size:0.9rem;margin-top:0.1rem;">
                        {ctx['training']['best_model_name']}</div>
                    <div style="font-size:0.76rem;color:#475569;margin-top:0.3rem;">
                        Accuracy <strong style="color:#0D9488;">{metrics['Accuracy'] * 100:.1f}%</strong>
                        · F1 <strong style="color:#0D9488;">{metrics['F1 Score'] * 100:.1f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Change dataset", use_container_width=True):
            change_dataset()
    elif ctx.get("df") is not None:
        st.warning("This CSV is not a supported dataset.")
        if st.button(
            "Change dataset",
            use_container_width=True,
            key="change_unsupported",
        ):
            change_dataset()

    if ctx.get("leakage"):
        flagged = ", ".join(c for finding in ctx["leakage"] for c in finding.columns)
        st.warning(f"Possible leakage: {flagged}")


# ------------------------------------------------------------------------------
# 5. HERO BANNERS
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def render_hero(badge: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="hero-badge">{badge}</span>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataset_link_markdown(dataset) -> str:
    """Markdown bullet list of download links for one supported dataset."""
    return "\n".join(
        f'- [{label}]({url})' for label, url in dataset.sources
    )


# ------------------------------------------------------------------------------
# 6. WELCOME / ERROR SCREENS
# ------------------------------------------------------------------------------
def render_welcome(error: str | None = None) -> None:
    st.markdown(
        "<div style='height: clamp(2rem, 10vh, 8rem);'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="welcome-hero">
            <div class="welcome-glow"></div>
            <h1>Customer Churn Prediction</h1>
            <div class="welcome-underline"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, middle, _ = st.columns([1, 1.35, 1])
    with middle:
        if error:
            st.error(error)

        with st.container(border=True, key="welcome_upload"):
            st.markdown("#### 📤 Upload Dataset (CSV)")
            st.markdown(
                "<div style='font-size:0.85rem;color:#0F766E;font-weight:600;margin:0 0 0.45rem 0;'>"
                "Drag and drop a CSV here, or click <strong>Upload</strong>."
                "</div>",
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "📤 Upload Dataset (CSV)",
                type=["csv", "txt"],
                key="dataset_uploader",
                label_visibility="collapsed",
                help="Telco Customer Churn or Cell2Cell Telecom Churn — one CSV at a time.",
            )

            if uploaded_file is not None:
                raw_bytes = uploaded_file.getvalue()
                file_id = hashlib.sha256(raw_bytes).hexdigest()
                if st.session_state.get("upload_id") != file_id:
                    reset_upload()
                    st.session_state["upload_bytes"] = raw_bytes
                    st.session_state["upload_name"] = uploaded_file.name
                    st.session_state["upload_id"] = file_id
                    st.session_state["active_page"] = PAGE_OPTIONS[0]
                    st.rerun()


def render_unsupported(ctx: dict) -> None:
    render_hero(
        "UNSUPPORTED DATASET",
        "This CSV does not match a supported dataset",
        "The app accepts only Telco Customer Churn and Cell2Cell Telecom Churn, so that "
        "validation, leakage checks and benchmarks stay meaningful.",
    )

    st.error(ctx["error"] or "The uploaded file could not be matched to a supported dataset.")

    if st.button("Use a different CSV", type="primary"):
        change_dataset()

    st.markdown("### What the app looked for")
    st.dataframe(
        pd.DataFrame(dataset_diagnostics(ctx["df"])),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Columns found in your file")
    st.code(", ".join(map(str, ctx["df"].columns)), language="text")

    st.markdown("### Download a supported dataset")
    for dataset in SUPPORTED_DATASETS:
        with st.expander(f"{dataset.name} — {dataset.expected_rows}"):
            st.markdown(f"{dataset.tagline}\n\n{dataset_link_markdown(dataset)}")
            st.caption(f"Target column: {' / '.join(dataset.target_candidates)}")


def render_setup(ctx: dict) -> None:
    render_hero(
        "STEP 2 · PREPARE & TRAIN",
        "Confirm the target and train",
        "The dataset was detected successfully. Check the summary, keep the suggested "
        "exclusions and train the four classifiers.",
    )

    df = ctx["df"]
    dataset = ctx.get("dataset")
    report = ctx.get("target_report")

    if ctx.get("error") and report is not None and report.is_valid:
        st.error(ctx["error"])

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### 🎯 Target & features")
            columns = list(df.columns)
            current = st.session_state.get("target_col")
            if current not in columns:
                current = ctx.get("target")
            index = columns.index(current) if current in columns else len(columns) - 1

            target_value = st.selectbox(
                "Target column",
                options=columns,
                index=index,
                key="target_col",
                help="The column to predict. It must contain exactly two classes, e.g. Yes/No or 1/0.",
            )

            if st.session_state.get("_target_seen") != target_value:
                st.session_state["_target_seen"] = target_value
                suggested = suggest_excluded_columns(df, target_value)
                st.session_state["excluded_cols"] = suggested
                st.session_state["settings_excluded"] = tuple(suggested)
                st.session_state.pop("train_request", None)

            if "excluded_cols" not in st.session_state:
                if "settings_excluded" in st.session_state:
                    st.session_state["excluded_cols"] = list(
                        st.session_state["settings_excluded"]
                    )
                else:
                    st.session_state["excluded_cols"] = suggest_excluded_columns(
                        df, target_value
                    )

            with st.expander("Advanced · excluded columns", expanded=False):
                st.multiselect(
                    "Do not use these columns",
                    options=[c for c in columns if c != target_value],
                    key="excluded_cols",
                    help="Identifiers, free text and > 50-level columns are pre-selected to "
                    "prevent leakage and overfitting.",
                )
                st.caption(
                    "IDs and free-text fields are removed automatically. Un-exclude a column "
                    "if you want it used as a feature."
                )

            # Persist the confirmed setup as plain keys so it is not lost when the
            # widget state is cleaned up after this screen stops rendering.
            st.session_state["settings_target"] = target_value
            st.session_state["settings_excluded"] = tuple(
                st.session_state.get("excluded_cols", [])
            )

            check = validate_target(df, target_value)
            if check.is_valid:
                st.markdown(
                    f"""
                    <div style="background:#ECFDF5;border:1px solid #A7F3D0;border-radius:10px;
                                padding:0.65rem 0.8rem;font-size:0.85rem;color:#065F46;">
                        ✅ Target <strong>{target_value}</strong> has two classes —
                        <strong>{check.negative_label}</strong> ({check.n_negative:,}) vs
                        <strong>{check.positive_label}</strong> ({check.n_positive:,}).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if check.n_missing:
                    st.caption(
                        f"⚠️ {check.n_missing:,} row(s) have an empty target and will be dropped."
                    )
            else:
                st.error(check.error)

            if st.button(
                "Train models",
                type="primary",
                use_container_width=True,
                disabled=not check.is_valid,
            ):
                st.session_state["train_request"] = (
                    st.session_state.get("upload_id"),
                    target_value,
                    tuple(st.session_state.get("excluded_cols", [])),
                )
                st.session_state.pop("gen_last_prediction", None)
                st.rerun()

            if st.button(
                "Use a different CSV",
                use_container_width=True,
                key="setup_change_dataset",
            ):
                change_dataset()

    with right:
        with st.container(border=True):
            st.markdown("#### 📄 Dataset summary")
            excluded = tuple(st.session_state.get("excluded_cols", []))
            try:
                preview = preview_schema(ctx["raw_bytes"], target_value, excluded)
                features_kept = str(preview.n_features)
            except DatasetError:
                features_kept = "—"

            summary = (
                ("Dataset", dataset.name if dataset else "—"),
                ("File", ctx.get("filename", "dataset.csv")),
                ("Rows", f"{len(df):,}"),
                ("Columns", str(df.shape[1])),
                ("Target", target_value),
                ("Features kept", features_kept),
                ("Excluded", str(len(excluded))),
            )
            body = "".join(
                "<div style='display:flex;justify-content:space-between;gap:1rem;padding:0.42rem 0;"
                "border-bottom:1px solid #F1F5F9;font-size:0.86rem;'>"
                f"<span style='color:#64748B;'>{label}</span>"
                f"<strong style='color:#0F172A;text-align:right;word-break:break-all;'>{value}</strong></div>"
                for label, value in summary
            )
            st.markdown(f"<div style='margin-top:0.2rem;'>{body}</div>", unsafe_allow_html=True)
            if check.is_valid:
                st.caption(
                    f"Positive class: **{check.positive_label}** · "
                    f"negative class: **{check.negative_label}**"
                )

    with st.expander("🔎 Preview the uploaded data (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)

    st.caption(
        "Want a different file? Click **Use a different CSV** above, or **Change dataset** in the sidebar."
    )


def render_nav() -> None:
    """Main navigation, visible even when the sidebar is collapsed."""
    nav_col, action_col = st.columns([5, 1], vertical_alignment="center")
    with nav_col:
        st.pills(
            "Pages",
            options=PAGE_OPTIONS,
            default=PAGE_OPTIONS[0],
            key="active_page",
            label_visibility="collapsed",
        )
    with action_col:
        if st.button("Change dataset", use_container_width=True):
            change_dataset()


# ------------------------------------------------------------------------------
# 7. PAGE ROUTING
# ------------------------------------------------------------------------------
if ctx["df"] is None:
    render_welcome(ctx.get("error"))
    st.stop()

if ctx["unsupported"]:
    render_unsupported(ctx)
    st.stop()

if ctx["training"] is None:
    render_setup(ctx)
    st.stop()

render_nav()

page = st.session_state.get("active_page") or PAGE_OPTIONS[0]

dataset_name = ctx["dataset"].name if ctx.get("dataset") else "your dataset"
target_name = ctx.get("target") or "the target"

if page == "Dashboard":
    render_hero(
        "OVERVIEW",
        "Churn Dashboard",
        f"Customers, churn rate and class balance for {dataset_name}.",
    )
    views.render_generic_dashboard(ctx)

elif page == "Prediction":
    render_hero(
        "REAL-TIME INFERENCE",
        f"Predict {target_name}",
        "The form is generated from the uploaded dataset — no hard-coded fields or categories.",
    )
    views.render_generic_prediction(ctx)

elif page == "Visualization":
    render_hero(
        "EXPLORATORY DATA ANALYSIS",
        "Visual Churn Insights",
        "Interactive charts for the columns that exist in the uploaded dataset.",
    )
    views.render_generic_eda(ctx)

elif page == "Performance":
    render_hero(
        "BENCHMARKING & VALIDATION",
        "Model Performance",
        "Four classifiers on the same 80/20 split, ranked by F1 score.",
    )
    views.render_generic_performance(ctx)

elif page == "About":
    render_hero(
        "ABOUT THE PROJECT",
        "How the system works",
        "Objective, datasets, preprocessing, evaluation metrics and limitations.",
    )

    st.markdown(
        """
        <div class="dashboard-card">
            <h4 style="margin-top:0;">🎯 Objective</h4>
            <p style="color:#475569;font-size:0.92rem;line-height:1.65;margin-bottom:0;">
                Predict whether a telecom customer will <strong>churn or stay</strong> from their
                account, usage and service attributes, then compare four classical classifiers on an
                identical 80/20 split (<code>random_state=42</code>) using Accuracy, Precision, Recall
                and F1 Score. The model with the highest F1 Score becomes the active model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <h4 style="margin-top:0;">🗂️ Supported datasets</h4>
                <p style="color:#475569;font-size:0.9rem;line-height:1.6;">
                    Exactly two public churn datasets are accepted, both through the same CSV upload.
                </p>
                {''.join(
                    f'<div style="margin-bottom:0.85rem;">'
                    f'<div style="font-weight:700;color:#0F172A;">{d.name}</div>'
                    f'<div style="font-size:0.84rem;color:#475569;">{d.tagline}</div>'
                    f'<div style="font-size:0.8rem;color:#64748B;">Target: {d.target_candidates[0]} · '
                    f'{d.expected_rows}</div></div>'
                    for d in SUPPORTED_DATASETS
                )}
                <div style="font-size:0.82rem;color:#475569;margin-top:0.2rem;">
                    Download links are on the welcome screen.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="dashboard-card">
                <h4 style="margin-top:0;">🧪 Algorithms</h4>
                <ul style="color:#475569;font-size:0.9rem;line-height:1.7;margin-bottom:0;">
                    <li><strong>Logistic Regression</strong> — linear baseline, easy to interpret.</li>
                    <li><strong>Decision Tree</strong> — captures non-linear rules.</li>
                    <li><strong>K-Nearest Neighbors</strong> — similarity-based predictions.</li>
                    <li><strong>Gaussian Naive Bayes</strong> — fast probabilistic baseline.</li>
                </ul>
                <div style="font-size:0.82rem;color:#64748B;margin-top:0.6rem;">
                    Fixed baseline settings — no hyper-parameter search.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        findings = ctx.get("leakage") or []
        if findings:
            leakage_html = "".join(
                f'<li><strong>{", ".join(f.columns)}</strong> — {f.detail}</li>' for f in findings
            )
            leakage_block = (
                f'<ul style="color:#B45309;font-size:0.85rem;line-height:1.6;margin:0.5rem 0 0 0;">'
                f"{leakage_html}</ul>"
            )
        else:
            leakage_block = (
                '<div style="font-size:0.85rem;color:#0F766E;margin-top:0.5rem;">'
                "✅ No leakage indicators found in this upload.</div>"
            )

        st.markdown(
            f"""
            <div class="dashboard-card">
                <h4 style="margin-top:0;">⚙️ Preprocessing &amp; leakage control</h4>
                <ul style="color:#475569;font-size:0.9rem;line-height:1.7;">
                    <li><strong>Detection:</strong> the uploaded columns are matched against the two
                        supported signatures; anything else is rejected.</li>
                    <li><strong>Target:</strong> exactly two classes, otherwise training is blocked.</li>
                    <li><strong>Identifiers:</strong> customer IDs, row numbers and free-text columns
                        are excluded before training.</li>
                    <li><strong>Pipeline:</strong> imputers, scaler and one-hot encoder are fitted
                        inside a Scikit-Learn <code>Pipeline</code> on the training split only.</li>
                    <li><strong>Missing values:</strong> median (numeric) and most-frequent
                        (categorical) imputation; empty target rows are dropped.</li>
                    <li><strong>Leakage scan:</strong> duplicate labels, post-outcome column names and
                        perfect class separators are checked before training.</li>
                </ul>
                {leakage_block}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="dashboard-card">
                <h4 style="margin-top:0;">📏 Evaluation metrics</h4>
                <ul style="color:#475569;font-size:0.9rem;line-height:1.7;margin-bottom:0;">
                    <li><strong>Accuracy</strong> — share of all predictions that are correct.</li>
                    <li><strong>Precision</strong> — of customers predicted to churn, how many did.</li>
                    <li><strong>Recall</strong> — of customers who churned, how many were caught.</li>
                    <li><strong>F1 Score</strong> — harmonic mean of precision and recall; used to
                        choose the best model.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="dashboard-card">
            <h4 style="margin-top:0;">⚠️ Limitations</h4>
            <ul style="color:#475569;font-size:0.9rem;line-height:1.7;margin-bottom:0;">
                <li>Only Telco Customer Churn and Cell2Cell Telecom Churn are accepted.</li>
                <li>Binary classification only — targets with more than two classes are blocked.</li>
                <li>The four algorithms use fixed baseline settings: no tuning, class weighting
                    or SMOTE resampling.</li>
                <li>Cell2Cell is noisy and imbalanced, so its F1 scores are lower and are reported
                    exactly as measured.</li>
                <li>The leakage scan flags suspicious patterns; it does not prove causality.
                    Cell2Cell retention-call columns are legitimate business actions, not copies
                    of the outcome.</li>
                <li>Kaggle downloads may require an account; the Telco IBM mirror is public.</li>
                <li>Uploaded data stays in the Streamlit session — nothing is persisted unless you
                    download an artifact.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
