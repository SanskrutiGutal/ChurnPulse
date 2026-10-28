"""
================================================================================
Generic Dataset Views (Streamlit)
================================================================================
File: dataset_views.py
Purpose:
    Renders the dashboard, EDA, prediction and model-performance pages for the
    uploaded dataset. Everything is generated from the inferred schema, so no
    column name, category or target label is hard-coded.

Design rule: every chart/field here is generated from the dataset schema, so no
column name is hard-coded.
================================================================================
"""

from __future__ import annotations

import io
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from dataset_manager import (
    DatasetError,
    column_profile,
    encode_binary_target,
    normalize_name,
    predict_single,
)

# Consistent navy/teal palette (matches app.py's theme).
NAVY = "#1E3A8A"
TEAL = "#0D9488"
ROSE = "#EF4444"
AMBER = "#D97706"
PLOT_FONT = dict(color="#0F172A", family="Segoe UI, Arial, sans-serif")


# ------------------------------------------------------------------------------
# 1. SMALL HELPERS
# ------------------------------------------------------------------------------
def target_series(ctx: dict):
    """Returns (binary_target_series, positive_label, negative_label) or None."""
    df = ctx.get("df")
    target = ctx.get("target")
    if df is None or not target or target not in df.columns:
        return None
    try:
        encoded, positive, negative = encode_binary_target(df[target])
    except DatasetError:
        return None
    series = pd.Series(encoded, index=df.index).dropna().astype(int)
    return series, positive, negative


def class_labels(ctx: dict) -> tuple[str, str]:
    """Human-readable (negative, positive) class names."""
    training = ctx.get("training") or {}
    schema = training.get("schema") or ctx.get("schema")
    if schema is not None:
        return schema.negative_label, schema.positive_label
    resolved = target_series(ctx)
    if resolved:
        return resolved[2], resolved[1]
    return "0", "1"


def dataset_strip(ctx: dict) -> None:
    """Compact banner describing the active dataset and uploaded file."""
    df = ctx.get("df")
    if df is None:
        return
    negative, positive = class_labels(ctx)
    resolved = target_series(ctx)
    rate = f"{resolved[0].mean() * 100:.2f}%" if resolved else "n/a"
    dataset = ctx.get("dataset")
    dataset_name = getattr(dataset, "name", None) or "Uploaded dataset"
    filename = ctx.get("filename") or "dataset.csv"
    st.markdown(
        f"""
        <div class="insight-box" style="margin-top:0; display:flex; flex-wrap:wrap; gap:1.4rem;">
            <span>🏷️ <strong>{dataset_name}</strong></span>
            <span>📄 <strong>{filename}</strong></span>
            <span>Rows: <strong>{len(df):,}</strong></span>
            <span>Columns: <strong>{df.shape[1]}</strong></span>
            <span>Target: <strong>{ctx.get('target') or 'not selected'}</strong></span>
            <span>Classes: <strong>{negative} / {positive}</strong></span>
            <span>Churn rate: <strong>{rate}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _style(fig, height: int = 320):
    fig.update_layout(
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        legend=dict(font=dict(color="#0F172A")),
    )
    fig.update_xaxes(tickfont=dict(color="#0F172A"), title_font=dict(color="#0F172A"))
    fig.update_yaxes(
        gridcolor="#E2E8F0", tickfont=dict(color="#0F172A"), title_font=dict(color="#0F172A")
    )
    return fig


@contextmanager
def _card(title: str, subtitle: str = ""):
    """Bordered card that wraps the Streamlit widgets rendered inside it."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if subtitle:
            st.caption(subtitle)
        yield


def render_training_required(ctx: dict, action: str) -> None:
    """Friendly gate shown when an uploaded dataset has not been trained yet."""
    name = getattr(ctx.get("dataset"), "name", None) or "this dataset"
    st.warning(f"Train a model on **{name}** to unlock {action}.")
    st.markdown(
        """
        <div class="dashboard-card">
            <h4 style="margin-top:0;">How to continue</h4>
            <ol style="line-height:1.9; margin-bottom:0;">
                <li>Return to the welcome screen and upload a CSV with <strong>📤 Upload Dataset (CSV)</strong>.</li>
                <li>Confirm the <strong>target column</strong> and the columns to exclude.</li>
                <li>Click <strong>Train models</strong>.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# 2. DASHBOARD
# ------------------------------------------------------------------------------
def render_generic_dashboard(ctx: dict) -> None:
    df = ctx["df"]
    dataset_strip(ctx)

    resolved = target_series(ctx)
    negative, positive = class_labels(ctx)

    total_rows = len(df)
    churned = int(resolved[0].sum()) if resolved else 0
    labelled = int(len(resolved[0])) if resolved else 0
    retained = labelled - churned
    positive_rate = (churned / labelled * 100) if labelled else 0.0
    missing_labels = total_rows - labelled

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card kpi-navy">
                <div class="kpi-header"><span class="kpi-title">Total customers</span><div class="kpi-icon icon-navy">👥</div></div>
                <div class="kpi-value">{total_rows:,}</div>
                <div class="kpi-desc"><span>Rows in the uploaded file</span></div>
            </div>
            <div class="kpi-card kpi-rose">
                <div class="kpi-header"><span class="kpi-title">Churned</span><div class="kpi-icon icon-rose">⚠️</div></div>
                <div class="kpi-value">{churned:,}</div>
                <div class="kpi-desc"><span>{positive} · {positive_rate:.1f}% of labelled rows</span></div>
            </div>
            <div class="kpi-card kpi-teal">
                <div class="kpi-header"><span class="kpi-title">Retained</span><div class="kpi-icon icon-teal">✅</div></div>
                <div class="kpi-value">{retained:,}</div>
                <div class="kpi-desc"><span>{negative} · {100 - positive_rate:.1f}% of labelled rows</span></div>
            </div>
            <div class="kpi-card kpi-amber">
                <div class="kpi-header"><span class="kpi-title">Churn rate</span><div class="kpi-icon icon-amber">📉</div></div>
                <div class="kpi-value">{positive_rate:.2f}%</div>
                <div class="kpi-desc"><span>{churned:,} of {labelled:,} labelled customers</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if missing_labels:
        st.caption(
            f"⚠️ {missing_labels:,} row(s) have an empty target and are excluded from the "
            "churn counts and from training."
        )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        with _card(f"{positive} vs {negative} distribution"):
            if resolved:
                chart_df = pd.DataFrame(
                    {
                        "Class": [negative, positive],
                        "Customers": [retained, churned],
                    }
                )
                fig = px.pie(
                    chart_df,
                    names="Class",
                    values="Customers",
                    hole=0.62,
                    color="Class",
                    color_discrete_map={negative: TEAL, positive: ROSE},
                )
                fig.update_traces(
                    textinfo="percent+label",
                    marker=dict(line=dict(color="#FFFFFF", width=3)),
                )
                _style(fig).update_layout(
                    showlegend=False,
                    annotations=[
                        dict(
                            text=f"<b>{labelled:,}</b><br><span style='font-size:11px;color:#475569;'>labelled</span>",
                            x=0.5,
                            y=0.5,
                            showarrow=False,
                            font=dict(size=16, color="#0F172A"),
                        )
                    ],
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f"""
                    <div class="insight-box">
                        💡 <strong>{positive_rate:.1f}%</strong> of labelled customers are
                        <strong>{positive}</strong>. The classes are {"imbalanced" if positive_rate < 40 or positive_rate > 60 else "reasonably balanced"},
                        so F1 Score and Recall matter more than accuracy alone.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("The selected target column could not be encoded as two classes.")

    with right:
        with _card("Dataset summary"):
            schema = ctx.get("schema")
            dataset_name = getattr(ctx.get("dataset"), "name", None) or "Uploaded dataset"
            feature_count = schema.n_features if schema is not None else "—"
            excluded_count = len(schema.dropped_cols) if schema is not None else "—"
            findings = ctx.get("leakage") or []
            leakage_text = (
                f"<span style='color:#B45309;'>{len(findings)} indicator(s) to review</span>"
                if findings
                else "<span style='color:#0F766E;'>No indicators found</span>"
            )
            rows = (
                ("Dataset", dataset_name),
                ("File", ctx.get("filename") or "dataset.csv"),
                ("Rows", f"{total_rows:,}"),
                ("Columns", f"{df.shape[1]}"),
                ("Target", str(ctx.get("target") or "not selected")),
                ("Classes", f"{negative} / {positive}"),
                ("Features used", str(feature_count)),
                ("Excluded columns", str(excluded_count)),
                ("Leakage check", leakage_text),
            )
            body = "".join(
                f"<div style='display:flex;justify-content:space-between;gap:1rem;padding:0.42rem 0;"
                f"border-bottom:1px solid #F1F5F9;font-size:0.86rem;'>"
                f"<span style='color:#64748B;'>{label}</span>"
                f"<strong style='color:#0F172A;text-align:right;word-break:break-all;'>{value}</strong></div>"
                for label, value in rows
            )
            st.markdown(f"<div style='margin-top:0.2rem;'>{body}</div>", unsafe_allow_html=True)

    with st.expander("🔎 Feature profile used by the model"):
        if schema:
            profile = column_profile(df, schema).copy()
            profile["Positive rate range"] = profile["Positive rate range"].str.replace(
                "–", "to", regex=False
            )
            st.dataframe(profile, use_container_width=True, hide_index=True)
            if schema.dropped_cols:
                st.markdown("**Excluded columns**")
                st.dataframe(
                    pd.DataFrame(
                        [{"Column": k, "Reason": v} for k, v in schema.dropped_cols.items()]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Train a model to see the exact feature set and per-feature signal.")
            st.dataframe(df.head(20), use_container_width=True)


# ------------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------------------
def _find_column(df, aliases: tuple[str, ...]) -> str | None:
    """Case/space-insensitive lookup for the first available alias."""
    lookup: dict[str, str] = {}
    for column in df.columns:
        lookup.setdefault(normalize_name(column), column)
    for alias in aliases:
        match = lookup.get(normalize_name(alias))
        if match is not None:
            return match
    return None


def _rate_by_category(df, feature: str, target, max_levels: int = 12) -> pd.DataFrame:
    """Churn rate (%) and row count for each value of a categorical column."""
    frame = pd.DataFrame(
        {
            "Category": df[feature].astype(str).str.strip(),
            "Target": target.reindex(df.index),
        }
    ).dropna()
    frame = frame[~frame["Category"].isin(["", "nan", "None", "<NA>"])]
    grouped = (
        frame.groupby("Category")["Target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Rate", "count": "Customers"})
    )
    grouped = grouped[grouped["Customers"] > 0]
    grouped["Rate"] = grouped["Rate"] * 100
    return grouped.sort_values("Rate", ascending=False).head(max_levels)


def _auto_categorical(df, target, target_name: str) -> str | None:
    """Pick the low-cardinality feature with the widest churn-rate spread."""
    # Feature selection is a heuristic; use a sample for speed on large files.
    sample = df if len(df) <= 20_000 else df.sample(20_000, random_state=42)
    sample_target = target.reindex(sample.index)
    best, best_spread = None, -1.0
    for column in sample.columns:
        if column == target_name:
            continue
        values = sample[column].astype(str).str.strip()
        levels = values[~values.isin(["", "nan", "None", "<NA>"])].nunique()
        if not 3 <= levels <= 10:
            continue
        grouped = _rate_by_category(sample, column, sample_target)
        frequent = grouped[grouped["Customers"] >= max(5, int(0.01 * len(sample)))]
        if len(frequent) < 2:
            continue
        spread = float(frequent["Rate"].max() - frequent["Rate"].min())
        if spread > best_spread:
            best, best_spread = column, spread
    return best


def _binned_by_class(values, classes, bins: int = 30) -> pd.DataFrame:
    """Histogram counts per class, pre-binned so Plotly receives ~60 points."""
    frame = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"), "class": classes}
    ).dropna()
    if frame.empty:
        return frame
    _, edges = np.histogram(frame["value"], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    rows = []
    for label, group in frame.groupby("class"):
        counts, _ = np.histogram(group["value"], bins=edges)
        rows.append(pd.DataFrame({"value": centers, "class": label, "customers": counts}))
    return pd.concat(rows, ignore_index=True)


def _churn_rate_chart(df, feature: str, target, negative: str, positive: str):
    grouped = _rate_by_category(df, feature, target)
    if grouped.empty:
        return None, None
    fig = px.bar(
        grouped,
        x="Category",
        y="Rate",
        color="Rate",
        color_continuous_scale=["#CCFBF1", TEAL, AMBER, ROSE],
        text="Rate",
        hover_data={"Customers": True, "Rate": ":.1f"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="",
        yaxis_title=f"{positive} rate (%)",
        xaxis_tickangle=-20,
    )
    return fig, grouped


def _render_category_chart(df, feature: str, target) -> None:
    """Bar chart of churn rate per category plus a one-line insight."""
    grouped = _rate_by_category(df, feature, target)
    if grouped.empty:
        st.info("This column has no usable categories.")
        return

    plot_df = grouped.sort_values("Rate", ascending=False)
    fig = px.bar(
        plot_df,
        x="Category",
        y="Rate",
        color="Rate",
        color_continuous_scale=["#CCFBF1", TEAL, AMBER, ROSE],
        text="Rate",
        hover_data={"Customers": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="",
        yaxis_title="Churn rate (%)",
        xaxis_tickangle=-20,
    )
    st.plotly_chart(_style(fig), use_container_width=True)

    high, low = plot_df.iloc[0], plot_df.iloc[-1]
    st.markdown(
        f"""
        <div class="insight-box">
            💡 <strong>{high['Category']}</strong> has the highest churn rate
            (<strong>{high['Rate']:.1f}%</strong>, {int(high['Customers']):,} customers);
            <strong>{low['Category']}</strong> has the lowest
            (<strong>{low['Rate']:.1f}%</strong>).
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_generic_eda(ctx: dict) -> None:
    df = ctx["df"]
    dataset_strip(ctx)
    resolved = target_series(ctx)
    negative, positive = class_labels(ctx)

    if resolved is None:
        st.info("Pick a binary target column on the setup screen to unlock target-based analysis.")
        st.dataframe(df.head(50), use_container_width=True)
        return

    target, positive, negative = resolved
    schema = ctx.get("schema")
    numeric_cols = (
        schema.numeric_cols if schema else list(df.select_dtypes(include=np.number).columns)
    )

    # ---- Row 1: churn distribution + contract type (or best categorical) ------
    left, right = st.columns(2, gap="large")

    with left:
        with _card(f"{positive} vs {negative} distribution"):
            counts = pd.DataFrame(
                {
                    "Class": [negative, positive],
                    "Customers": [int((target == 0).sum()), int((target == 1).sum())],
                }
            )
            fig = px.bar(
                counts,
                x="Class",
                y="Customers",
                color="Class",
                text="Customers",
                color_discrete_map={negative: TEAL, positive: ROSE},
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
            st.plotly_chart(_style(fig), use_container_width=True)
            rate = target.mean() * 100
            st.markdown(
                f"""
                <div class="insight-box">
                    💡 {int((target == 1).sum()):,} customers are labelled {positive} out of
                    {len(target):,} labelled records — a churn rate of <strong>{rate:.1f}%</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    contract_col = _find_column(
        df, ("contract", "contract type", "contracttype", "plan", "plan type", "tariff")
    )

    with right:
        if contract_col:
            with _card("Churn rate by contract type", f"Column: {contract_col}"):
                _render_category_chart(df, contract_col, target)
        else:
            cardinality = df.nunique()
            candidates = [
                column
                for column in df.columns
                if column != ctx["target"]
                and 2 <= int(cardinality.get(column, 0)) <= 12
                and column not in (ctx.get("excluded") or [])
            ]
            if not candidates:
                with _card("Churn rate by category"):
                    st.info("No suitable categorical column was found in this dataset.")
            else:
                auto = _auto_categorical(df, target, ctx["target"])
                default = candidates.index(auto) if auto in candidates else 0
                chosen = st.selectbox(
                    "Explore a categorical feature",
                    candidates,
                    index=default,
                    help="This dataset has no contract column, so you can explore any "
                    "low-cardinality feature.",
                )
                with _card(f"Churn rate by {chosen}", f"Column: {chosen}"):
                    _render_category_chart(df, chosen, target)

    # ---- Row 2: tenure + monthly charges --------------------------------------
    tenure_col = _find_column(
        df,
        (
            "tenure",
            "months in service",
            "monthsinservice",
            "months with company",
            "customer tenure",
            "account length",
        ),
    )
    charge_col = _find_column(
        df,
        (
            "monthly charges",
            "monthlycharges",
            "monthly revenue",
            "monthlyrevenue",
            "monthly charge",
            "monthly fee",
            "monthly bill",
        ),
    )

    if tenure_col or charge_col:
        left, right = st.columns(2, gap="large")

        if tenure_col:
            with left:
                with _card("Customer tenure by churn", f"Column: {tenure_col}"):
                    numeric = pd.to_numeric(df[tenure_col], errors="coerce")
                    classes = target.map({0: negative, 1: positive}).reindex(df.index)
                    plot_df = pd.DataFrame({"value": numeric, "class": classes}).dropna()
                    if plot_df.empty:
                        st.info("This column is not numeric.")
                    else:
                        binned = _binned_by_class(numeric, classes, bins=30)
                        fig = px.bar(
                            binned,
                            x="value",
                            y="customers",
                            color="class",
                            barmode="overlay",
                            opacity=0.75,
                            color_discrete_map={negative: TEAL, positive: ROSE},
                            labels={"value": tenure_col, "customers": "Customers", "class": ""},
                        )
                        fig.update_traces(marker_line_width=0)
                        fig.update_layout(bargap=0.03, legend_title="")
                        st.plotly_chart(_style(fig), use_container_width=True)
                        medians = plot_df.groupby("class")["value"].median()
                        if {negative, positive}.issubset(medians.index):
                            st.markdown(
                                f"""
                                <div class="insight-box">
                                    💡 Median tenure is <strong>{medians[positive]:.0f}</strong> for
                                    customers who churned versus <strong>{medians[negative]:.0f}</strong>
                                    for customers who stayed.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        if charge_col:
            with right:
                with _card("Monthly charges by churn", f"Column: {charge_col}"):
                    numeric = pd.to_numeric(df[charge_col], errors="coerce")
                    classes = target.map({0: negative, 1: positive}).reindex(df.index)
                    plot_df = pd.DataFrame({"value": numeric, "class": classes}).dropna()
                    if plot_df.empty:
                        st.info("This column is not numeric.")
                    else:
                        binned = _binned_by_class(numeric, classes, bins=30)
                        fig = px.bar(
                            binned,
                            x="value",
                            y="customers",
                            color="class",
                            barmode="overlay",
                            opacity=0.75,
                            color_discrete_map={negative: TEAL, positive: ROSE},
                            labels={
                                "value": charge_col,
                                "customers": "Customers",
                                "class": "",
                            },
                        )
                        fig.update_traces(marker_line_width=0)
                        fig.update_layout(bargap=0.03, legend_title="")
                        st.plotly_chart(_style(fig), use_container_width=True)
                        medians = plot_df.groupby("class")["value"].median()
                        if {negative, positive}.issubset(medians.index):
                            st.markdown(
                                f"""
                                <div class="insight-box">
                                    💡 Median monthly charge is <strong>{medians[positive]:.2f}</strong>
                                    for customers who churned versus
                                    <strong>{medians[negative]:.2f}</strong> for customers who stayed.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # ---- Row 3: internet service + payment method (only if present) -----------
    internet_col = _find_column(
        df, ("internet service", "internetservice", "internet", "broadband", "connection type")
    )
    payment_col = _find_column(
        df, ("payment method", "paymentmethod", "payment", "pay method", "payment type")
    )
    optional = [(internet_col, "Internet service vs churn"), (payment_col, "Payment method vs churn")]
    optional = [(column, title) for column, title in optional if column]

    if optional:
        columns = st.columns(len(optional), gap="large")
        for container, (column, title) in zip(columns, optional):
            with container:
                with _card(title, f"Column: {column}"):
                    fig, grouped = _churn_rate_chart(df, column, target, negative, positive)
                    if fig is None:
                        st.info("This column has no usable categories.")
                    else:
                        fig.update_layout(xaxis_tickangle=-15)
                        st.plotly_chart(_style(fig), use_container_width=True)
                        high = grouped.iloc[0]
                        st.markdown(
                            f"""
                            <div class="insight-box">
                                💡 <strong>{high['Category']}</strong> has the highest {positive.lower()}
                                rate at <strong>{high['Rate']:.1f}%</strong>
                                ({int(high['Customers']):,} customers).
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # ---- Missing values -------------------------------------------------------
    with _card("Missing value analysis"):
        counts = df.isna().sum()
        affected = int((counts > 0).sum())
        total_missing = int(counts.sum())
        if affected == 0:
            st.success("✅ No missing values detected — the pipeline has nothing to impute.")
        else:
            st.markdown(
                f"""
                <div class="insight-box" style="margin-top:0;">
                    🕳️ <strong>{total_missing:,}</strong> missing cells across
                    <strong>{affected}</strong> of {df.shape[1]} columns
                    ({total_missing / (df.shape[0] * df.shape[1]) * 100:.2f}% of all cells).
                    They are imputed inside the Scikit-Learn pipeline — median for numeric
                    columns, most frequent value for categorical columns.
                </div>
                """,
                unsafe_allow_html=True,
            )
            missing_df = pd.DataFrame(
                {
                    "Column": counts.index,
                    "Missing values": counts.values,
                    "Missing %": (counts.values / len(df) * 100).round(2),
                    "Non-null values": (len(df) - counts.values),
                }
            )
            missing_df = missing_df[missing_df["Missing values"] > 0].sort_values(
                "Missing %", ascending=False
            )
            chart = missing_df.head(15).sort_values("Missing %")
            fig = px.bar(
                chart,
                x="Missing %",
                y="Column",
                orientation="h",
                color="Missing %",
                color_continuous_scale=["#CCFBF1", TEAL, AMBER, ROSE],
                text="Missing %",
                hover_data=["Missing values"],
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Missing (%)", yaxis_title="")
            st.plotly_chart(_style(fig, 380), use_container_width=True)
            with st.expander("View the full missing-value table"):
                st.dataframe(missing_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# 4. PREDICTION
# ------------------------------------------------------------------------------
def render_generic_prediction(ctx: dict) -> None:
    training = ctx.get("training")
    if not training:
        render_training_required(ctx, "real-time prediction")
        return

    df = ctx["df"]
    schema = training["schema"]
    pipeline = training["best_pipeline"]
    negative, positive = schema.negative_label, schema.positive_label
    resolved = target_series(ctx)
    overall_rate = float(resolved[0].mean()) if resolved else float(np.mean(training["y_test"]))

    dataset_strip(ctx)
    st.markdown(
        f"##### 🧾 Customer record — predict **{positive}** vs **{negative}**"
        f" &nbsp;·&nbsp; model: `{training['best_model_name']}`"
    )

    with st.form("generic_prediction_form"):
        values: dict = {}
        input_columns = st.columns(3)
        for index, feature in enumerate(schema.numeric_cols):
            series = pd.to_numeric(df[feature], errors="coerce").dropna()
            if series.empty:
                continue
            low, high = float(series.min()), float(series.max())
            default = float(series.median())
            container = input_columns[index % 3]
            with container:
                all_integers = bool(np.allclose(series, series.round()))
                if all_integers and (high - low) <= 200:
                    values[feature] = st.slider(
                        feature,
                        min_value=int(np.floor(low)),
                        max_value=int(np.ceil(high)),
                        value=int(np.clip(round(default), low, high)),
                        key=f"gen_in_{feature}",
                    )
                else:
                    values[feature] = st.number_input(
                        feature,
                        value=float(round(default, 2)),
                        step=1.0 if all_integers else 0.1,
                        key=f"gen_in_{feature}",
                    )

        for index, feature in enumerate(schema.categorical_cols):
            series = (
                df[feature].astype(str).str.strip().replace({"nan": np.nan}).dropna()
            )
            options = sorted(series.unique().tolist())
            if not options:
                continue
            default = series.mode().iloc[0] if not series.mode().empty else options[0]
            with input_columns[index % 3]:
                values[feature] = st.selectbox(
                    feature,
                    options,
                    index=options.index(default) if default in options else 0,
                    key=f"gen_in_{feature}",
                )

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if not submitted and "gen_last_prediction" not in st.session_state:
        st.info("Set the feature values above, then press **Predict**.")
        return

    if submitted:
        try:
            pred_class, proba = predict_single(pipeline, values)
        except Exception as error:  # pragma: no cover - defensive UI guard
            st.error(f"Prediction failed: {error}")
            return
        st.session_state["gen_last_prediction"] = {
            "values": values,
            "pred": pred_class,
            "proba": proba,
            "model": training["best_model_name"],
        }

    result = st.session_state["gen_last_prediction"]
    pred_class, proba = result["pred"], result["proba"]
    retention = 1.0 - proba

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        if pred_class == 1:
            st.markdown(
                f"""
                <div class="result-card-danger">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge-churn-yes">⚠️ {positive.upper()}</span>
                        <span style="font-weight:800; color:#EF4444;">ELEVATED RISK</span>
                    </div>
                    <h2 style="margin-top:0.8rem; margin-bottom:0.2rem; font-size:1.6rem;">Predicted: {positive}</h2>
                    <p style="font-size:0.95rem; margin-bottom:0;">
                        The model estimates a <strong>{proba*100:.1f}% probability</strong> for
                        <strong>{positive}</strong> on this record.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card-safe">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="badge-churn-no">✅ {negative.upper()}</span>
                        <span style="font-weight:800; color:#0D9488;">LOW RISK</span>
                    </div>
                    <h2 style="margin-top:0.8rem; margin-bottom:0.2rem; font-size:1.6rem;">Predicted: {negative}</h2>
                    <p style="font-size:0.95rem; margin-bottom:0;">
                        The model estimates a <strong>{retention*100:.1f}% probability</strong> for
                        <strong>{negative}</strong> on this record.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="chip-container">
                <div class="metric-chip">
                    <div class="metric-chip-title">{positive} Probability</div>
                    <div class="metric-chip-value" style="color:{ROSE if proba >= 0.5 else TEAL};">{proba*100:.1f}%</div>
                </div>
                <div class="metric-chip">
                    <div class="metric-chip-title">{negative} Probability</div>
                    <div class="metric-chip-value" style="color:#2563EB;">{retention*100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 🕵️ What drives this prediction")
        drivers = _explain_row(df, resolved, values, schema, overall_rate)
        if drivers:
            for driver in drivers:
                st.markdown(f"- {driver}")
        else:
            st.markdown("- No standout single-feature signal for this record.")

    with right:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={"suffix": "%", "font": {"size": 34, "color": "#0F172A"}},
                title={"text": f"{positive} probability", "font": {"size": 14, "color": "#475569"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                    "bar": {"color": ROSE if proba >= 0.5 else TEAL, "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 1,
                    "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [0, 40], "color": "#E6FFFA"},
                        {"range": [40, 70], "color": "#FEF3C7"},
                        {"range": [70, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": "#DC2626", "width": 4}, "thickness": 0.75, "value": 70},
                },
            )
        )
        gauge.update_layout(
            height=260,
            margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=PLOT_FONT,
        )
        st.plotly_chart(gauge, use_container_width=True)

        st.caption(
            f"Model: **{result['model']}** · dataset baseline {positive} rate: "
            f"**{overall_rate*100:.1f}%**"
        )
        st.markdown(
            f"""
            <div class="insight-box">
                💡 <strong>Reading the score:</strong> compare the predicted probability with the
                dataset baseline of {overall_rate*100:.1f}%. Anything above the baseline means the
                record looks riskier than a typical customer.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _explain_row(df, resolved, values, schema, overall_rate) -> list[str]:
    """
    Lightweight, model-agnostic explanation: compares the record's values with
    historical target rates (categorical) and class means (numeric).
    """
    if resolved is None:
        return []
    target = resolved[0]
    drivers: list[str] = []

    for feature, value in values.items():
        if feature in schema.categorical_cols:
            grouped = target.groupby(df[feature].astype(str).str.strip(), dropna=True).mean()
            if str(value) in grouped.index:
                rate = float(grouped[str(value)])
                delta = rate - overall_rate
                if abs(delta) >= 0.05:
                    direction = "higher" if delta > 0 else "lower"
                    drivers.append(
                        f"**{feature} = {value}**: historical churn rate "
                        f"**{rate*100:.1f}%**, {direction} than the {overall_rate*100:.1f}% baseline."
                    )
        elif feature in schema.numeric_cols:
            numeric = pd.to_numeric(df[feature], errors="coerce")
            means = numeric.groupby(target).mean()
            if len(means) == 2:
                gap = means.get(1, np.nan) - means.get(0, np.nan)
                span = float(numeric.std()) or 1.0
                position = (float(value) - float(means.get(0, 0))) / span
                if abs(gap) > 0.25 * span and abs(position) > 0.5:
                    drivers.append(
                        f"**{feature} = {value}**: sits towards the "
                        f"churned group (mean {means.get(1, float('nan')):.2f} "
                        f"vs {means.get(0, float('nan')):.2f})."
                    )

    return drivers[:6]


# ------------------------------------------------------------------------------
# 5. MODEL PERFORMANCE
# ------------------------------------------------------------------------------
def render_generic_performance(ctx: dict) -> None:
    training = ctx.get("training")
    if not training:
        render_training_required(ctx, "model performance analytics")
        return

    schema = training["schema"]
    results = training["results_df"]
    best = training["best_model_name"]
    metrics = training["best_metrics"]
    negative, positive = schema.negative_label, schema.positive_label
    dataset_strip(ctx)

    st.markdown(
        f"##### 🏆 Best model by F1 Score: **{best}** &nbsp;·&nbsp; trained on "
        f"{training['train_size']:,} rows, tested on {training['test_size']:,} rows"
    )

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card kpi-navy">
                <div class="kpi-header"><span class="kpi-title">Accuracy</span><div class="kpi-icon icon-navy">🎯</div></div>
                <div class="kpi-value">{metrics['Accuracy']*100:.2f}%</div>
                <div class="kpi-desc"><span>Overall correctness</span></div>
            </div>
            <div class="kpi-card kpi-teal">
                <div class="kpi-header"><span class="kpi-title">Precision</span><div class="kpi-icon icon-teal">🔍</div></div>
                <div class="kpi-value">{metrics['Precision']*100:.2f}%</div>
                <div class="kpi-desc"><span>Of flagged, how many churned</span></div>
            </div>
            <div class="kpi-card kpi-rose">
                <div class="kpi-header"><span class="kpi-title">Recall</span><div class="kpi-icon icon-rose">📡</div></div>
                <div class="kpi-value">{metrics['Recall']*100:.2f}%</div>
                <div class="kpi-desc"><span>Churned cases caught</span></div>
            </div>
            <div class="kpi-card kpi-amber">
                <div class="kpi-header"><span class="kpi-title">F1 Score</span><div class="kpi-icon icon-amber">⚖️</div></div>
                <div class="kpi-value">{metrics['F1 Score']*100:.2f}%</div>
                <div class="kpi-desc"><span>Selection criterion</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85])

    with left:
        with _card("Model Comparison"):
            melted = results.melt(id_vars="Model", var_name="Metric", value_name="Score")
            melted["Score"] = melted["Score"] * 100
            fig = px.bar(
                melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text=melted["Score"].round(1),
                color_discrete_sequence=[NAVY, "#3B82F6", AMBER, TEAL],
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis_title="Score (%)", yaxis_range=[0, 112], xaxis_title="")
            st.plotly_chart(_style(fig, 380), use_container_width=True)

    with right:
        with _card("Confusion Matrix"):
            available = list(training["pipelines"].keys())
            chosen = st.selectbox(
                "Model",
                available,
                index=available.index(best) if best in available else 0,
                key="perf_cm_model",
                label_visibility="collapsed",
            )
            cm = confusion_matrix(training["y_test"], training["test_predictions"][chosen])
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Tealrose",
                x=[negative, positive],
                y=[negative, positive],
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_style(fig, 330), use_container_width=True)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            pct = (tp / (tp + fn) * 100) if (tp + fn) else 0.0
            st.markdown(
                f"""
                <div class="insight-box">
                    💡 Caught <strong>{tp}</strong> of {tp + fn} {positive.lower()} records
                    (<strong>{pct:.1f}% recall</strong>) with <strong>{fp}</strong> false alarms.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with _card("Benchmark Table"):
        display = results.copy()
        for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            display[column] = (display[column] * 100).round(2).astype(str) + "%"
        st.dataframe(display, use_container_width=True, hide_index=True)
        if training.get("skipped"):
            for note in training["skipped"]:
                st.caption(f"⚠️ {note}")

    dataset_key = getattr(ctx.get("dataset"), "key", "dataset")
    dataset_label = getattr(ctx.get("dataset"), "name", "Uploaded dataset")

    download1, download2 = st.columns(2)
    with download1:
        st.download_button(
            "⬇️ Download metrics (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset_key}_model_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download2:
        buffer = io.BytesIO()
        joblib.dump(
            {
                "pipeline": training["best_pipeline"],
                "best_model_name": best,
                "metrics": metrics,
                "dataset_key": dataset_key,
                "dataset_name": dataset_label,
                "source_file": ctx.get("filename"),
                "feature_names": schema.feature_cols,
                "num_cols": schema.numeric_cols,
                "cat_cols": schema.categorical_cols,
                "target_col": schema.target,
                "positive_label": schema.positive_label,
                "negative_label": schema.negative_label,
                "confusion_matrix": training["confusion_matrix"].tolist(),
            },
            buffer,
        )
        st.download_button(
            "⬇️ Download trained model (.pkl)",
            data=buffer.getvalue(),
            file_name=f"{dataset_key}_churn_model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )
