"""
================================================================================
Dataset Manager: Dataset-Agnostic Churn / Binary Classification Toolkit
================================================================================
File: dataset_manager.py
Purpose:
    Provides everything needed to run the project on ANY tabular binary
    classification dataset (one of two public telecom churn datasets):

      1. Target-column auto-detection (Churn, Exited, Attrition, Target, ...).
      2. Automatic feature typing (numeric vs categorical) with coercion of
         numeric-looking text columns (e.g. Telco's blank "TotalCharges").
      3. Automatic removal of identifier / constant / unusable columns.
      4. Binary target encoding with human-readable class labels.
      5. Leakage-safe preprocessing (imputers + scaler + one-hot encoder are
         fitted strictly on the training split through a Pipeline).
      6. Multi-model benchmarking (Logistic Regression, Decision Tree, KNN,
         Gaussian Naive Bayes) with Accuracy / Precision / Recall / F1.

This module contains NO UI and NO plotting code, so it can be reused by
`app.py` (Streamlit), `train_model.py` (CLI) and `main.py`.
================================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------------------
# 1. CONSTANTS
# ------------------------------------------------------------------------------
#: Column names commonly used for the label column in churn-like datasets.
TARGET_HINTS = [
    "churn",
    "churned",
    "churn_flag",
    "churn_label",
    "is_churn",
    "exited",
    "attrition",
    "target",
    "label",
    "class",
    "outcome",
    "default",
    "fraud",
    "subscribed",
    "converted",
    "y",
]

#: Tokens that identify the positive ("event happened") class.
POSITIVE_TOKENS = {
    "1",
    "1.0",
    "yes",
    "y",
    "true",
    "t",
    "churn",
    "churned",
    "exited",
    "left",
    "default",
    "fraud",
    "subscribed",
    "converted",
    "positive",
    "pos",
    "attrition",
}

#: Tokens that identify the negative ("no event") class.
NEGATIVE_TOKENS = {
    "0",
    "0.0",
    "no",
    "n",
    "false",
    "f",
    "retained",
    "stayed",
    "stays",
    "stayed_customer",
    "not_churned",
    "no_churn",
    "negative",
    "neg",
    "safe",
}

#: Fragments that mark a column as an identifier rather than a predictor.
ID_HINTS = (
    "customerid",
    "customer_id",
    "cust_id",
    "userid",
    "user_id",
    "uuid",
    "guid",
    "rownumber",
    "row_number",
    "rowid",
    "serial",
    "unnamed",
)

#: Categorical columns above this cardinality are excluded from the model.
MAX_CATEGORICAL_LEVELS = 50

#: A text column is treated as numeric when at least this share parses as numbers.
NUMERIC_COERCION_THRESHOLD = 0.90


class DatasetError(ValueError):
    """Raised when a dataset cannot be turned into a trainable problem."""


# ------------------------------------------------------------------------------
# 1b. SUPPORTED DATASET REGISTRY
# ------------------------------------------------------------------------------
def normalize_name(name) -> str:
    """Lower-cases a column name and removes every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass(frozen=True)
class SupportedDataset:
    """
    Metadata for one of the two accepted churn datasets.

    `core_groups` lists groups of column-name variants; a file matches a dataset
    when at least one variant from every group is present. This is deliberately
    tolerant of the small header differences between Kaggle editions
    (e.g. `NumOfProducts` vs `Num Of Products`, `Exited` vs `Churn`).
    """

    key: str
    name: str
    tagline: str
    target_candidates: tuple[str, ...]
    core_groups: tuple[tuple[str, ...], ...]
    signature_columns: tuple[str, ...]
    sources: tuple[tuple[str, str], ...]
    expected_rows: str = ""
    file_name: str = ""

    def matches(self, columns: set[str]) -> bool:
        return all(any(variant in columns for variant in group) for group in self.core_groups)

    def coverage(self, columns: set[str]) -> float:
        if not self.signature_columns:
            return 0.0
        hits = sum(1 for column in self.signature_columns if column in columns)
        return hits / len(self.signature_columns)


SUPPORTED_DATASETS: tuple[SupportedDataset, ...] = (
    SupportedDataset(
        key="telco",
        name="Telco Customer Churn",
        tagline="Telecom subscribers — services, contract and billing behaviour.",
        target_candidates=("Churn",),
        core_groups=(
            ("customerid",),
            ("tenure",),
            ("contract",),
            ("monthlycharges",),
            ("totalcharges",),
            ("churn",),
        ),
        signature_columns=tuple(
            normalize_name(column)
            for column in (
                "customerID",
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
                "Churn",
            )
        ),
        sources=(
            (
                "Kaggle — Telco Customer Churn",
                "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
            ),
            (
                "IBM GitHub — direct CSV",
                "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
            ),
        ),
        expected_rows="7,043 rows × 21 columns",
        file_name="Telco-Customer-Churn.csv",
    ),
    SupportedDataset(
        key="cell2cell",
        name="Cell2Cell Telecom Churn",
        tagline="US wireless subscribers — usage, revenue, handset and retention behaviour.",
        target_candidates=("Churn",),
        core_groups=(
            ("customerid",),
            ("churn",),
            ("monthlyrevenue",),
            ("monthlyminutes",),
            ("overageminutes",),
            ("monthsinservice",),
            ("servicearea",),
        ),
        signature_columns=tuple(
            normalize_name(column)
            for column in (
                "CustomerID",
                "Churn",
                "MonthlyRevenue",
                "MonthlyMinutes",
                "TotalRecurringCharge",
                "DirectorAssistedCalls",
                "OverageMinutes",
                "RoamingCalls",
                "PercChangeMinutes",
                "PercChangeRevenues",
                "DroppedCalls",
                "BlockedCalls",
                "UnansweredCalls",
                "CustomerCareCalls",
                "ThreewayCalls",
                "ReceivedCalls",
                "OutboundCalls",
                "InboundCalls",
                "PeakCallsInOut",
                "OffPeakCallsInOut",
                "DroppedBlockedCalls",
                "CallForwardingCalls",
                "CallWaitingCalls",
                "MonthsInService",
                "UniqueSubs",
                "ActiveSubs",
                "ServiceArea",
                "Handsets",
                "HandsetModels",
                "CurrentEquipmentDays",
                "AgeHH1",
                "AgeHH2",
                "ChildrenInHH",
                "HandsetRefurbished",
                "HandsetWebCapable",
                "TruckOwner",
                "RVOwner",
                "Homeownership",
                "BuysViaMailOrder",
                "RespondsToMailOffers",
                "OptOutMailings",
                "NonUSTravel",
                "OwnsComputer",
                "HasCreditCard",
                "RetentionCalls",
                "RetentionOffersAccepted",
                "NewCellphoneUser",
                "NotNewCellphoneUser",
                "ReferralsMadeBySubscriber",
                "IncomeGroup",
                "OwnsMotorcycle",
                "AdjustmentsToCreditRating",
                "HandsetPrice",
                "MadeCallToRetentionTeam",
                "CreditRating",
                "PrizmCode",
                "Occupation",
                "MaritalStatus",
            )
        ),
        sources=(
            (
                "Kaggle — Cell2Cell Telecom Churn",
                "https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom",
            ),
            (
                "Direct CSV mirror — cell2celltrain.csv (51,047 rows)",
                "https://raw.githubusercontent.com/janebunr/Cell2Cell/master/Data%20files/cell2celltrain.csv",
            ),
        ),
        expected_rows="51,047 rows × 58 columns",
        file_name="cell2celltrain.csv",
    ),
)


class UnsupportedDatasetError(DatasetError):
    """Raised when an uploaded CSV is not one of the two supported datasets."""

    def __init__(self, columns: Sequence[str]):
        self.uploaded_columns = list(columns)
        normalized = {normalize_name(column) for column in columns}
        missing_report = []
        for dataset in SUPPORTED_DATASETS:
            missing = [
                group[0]
                for group in dataset.core_groups
                if not any(variant in normalized for variant in group)
            ]
            if missing:
                missing_report.append(f"{dataset.name}: missing {', '.join(missing)}")
        detail = "; ".join(missing_report) if missing_report else "no matching columns"
        super().__init__(
            "This CSV is not one of the two supported churn datasets. "
            f"Expected signature columns — {detail}."
        )


def identify_dataset(df: pd.DataFrame) -> SupportedDataset | None:
    """Returns the supported dataset the frame belongs to, or None."""
    if df is None or df.empty:
        return None
    normalized = {normalize_name(column) for column in df.columns}
    matches = [dataset for dataset in SUPPORTED_DATASETS if dataset.matches(normalized)]
    if not matches:
        return None
    matches.sort(key=lambda dataset: dataset.coverage(normalized), reverse=True)
    return matches[0]


def ensure_supported_dataset(df: pd.DataFrame) -> SupportedDataset:
    """Returns the matching dataset or raises UnsupportedDatasetError."""
    dataset = identify_dataset(df)
    if dataset is None:
        raise UnsupportedDatasetError(list(df.columns))
    return dataset


def dataset_diagnostics(df: pd.DataFrame) -> list[dict]:
    """Per-dataset match report used by the UI when an upload is rejected."""
    if df is None or df.empty:
        return []
    normalized = {normalize_name(column) for column in df.columns}
    report = []
    for dataset in SUPPORTED_DATASETS:
        missing = [
            group[0]
            for group in dataset.core_groups
            if not any(variant in normalized for variant in group)
        ]
        report.append(
            {
                "Dataset": dataset.name,
                "Status": "Matches" if not missing else "Not matched",
                "Missing columns": ", ".join(missing) if missing else "—",
            }
        )
    return report


# ------------------------------------------------------------------------------
# 2. DATASET SCHEMA
# ------------------------------------------------------------------------------
@dataclass
class DatasetSchema:
    """Describes how a raw DataFrame was turned into an X / y problem."""

    target: str
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    dropped_cols: dict[str, str] = field(default_factory=dict)
    positive_label: str = "1"
    negative_label: str = "0"
    n_rows: int = 0
    n_features: int = 0

    @property
    def feature_cols(self) -> list[str]:
        return list(self.numeric_cols) + list(self.categorical_cols)


# ------------------------------------------------------------------------------
# 3. SCHEMA INFERENCE HELPERS
# ------------------------------------------------------------------------------
def detect_target(df: pd.DataFrame, preferred: Sequence[str] | None = None) -> str | None:
    """
    Guesses the label column:
      0. The dataset's own documented target name (when it is present).
      1. Exact (case-insensitive) match against TARGET_HINTS.
      2. A column whose name contains a hint token.
      3. The last binary column (2 unique values) in the frame.
    Returns None when nothing plausible is found.
    """
    if df is None or df.empty:
        return None

    lowered = {str(c).strip().lower(): c for c in df.columns}

    for candidate in preferred or ():
        if str(candidate).strip().lower() in lowered:
            return lowered[str(candidate).strip().lower()]

    for hint in TARGET_HINTS:
        if hint in lowered:
            return lowered[hint]

    for column in df.columns:
        name = str(column).strip().lower()
        if any(hint in name for hint in TARGET_HINTS if len(hint) > 2):
            if df[column].nunique(dropna=True) <= 10:
                return column

    for column in reversed(list(df.columns)):
        if df[column].nunique(dropna=True) == 2:
            return column

    return None


def suggest_excluded_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """
    Suggests columns that should not be used as model features:
    identifier-looking names, free-text / very high-cardinality columns, and
    all-null or constant columns. Excluding IDs is also what prevents the most
    common form of data leakage in customer-level churn tables.
    """
    suggestions: list[str] = []
    if df is None or df.empty:
        return suggestions

    n_rows = len(df)
    for column in df.columns:
        if column == target:
            continue
        name = str(column).strip().lower()
        series = df[column]

        if any(token in name for token in ID_HINTS) or name in {"id", "index"}:
            suggestions.append(column)
            continue
        if name.endswith("number") and series.nunique(dropna=True) >= 0.9 * n_rows:
            suggestions.append(column)
            continue
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            suggestions.append(column)
            continue
        if series.dtype == object or str(series.dtype).startswith("string"):
            levels = series.nunique(dropna=True)
            if levels > MAX_CATEGORICAL_LEVELS and _coerce_numeric_like(series) is None:
                # Free text such as surnames or addresses: no predictive value.
                suggestions.append(column)

    return suggestions


def _coerce_numeric_like(series: pd.Series) -> pd.Series | None:
    """Returns a numeric version of `series` when it is numeric-in-disguise."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    converted = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    if non_null == 0:
        return None
    if converted.notna().sum() >= NUMERIC_COERCION_THRESHOLD * non_null:
        return converted.astype(float)
    return None


def _format_label(value) -> str:
    """Renders a class value for display."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def encode_binary_target(y_raw: pd.Series) -> tuple[np.ndarray, str, str]:
    """
    Encodes a label column into 0/1.

    Returns (encoded_array, positive_label, negative_label). The positive class
    is the "event" class (churn / exit / default / minority).
    """
    clean = y_raw.dropna()
    unique_values = list(pd.unique(clean))

    if len(unique_values) < 2:
        raise DatasetError(
            "The selected target column has fewer than two distinct values, "
            "so a classification model cannot be trained."
        )
    if len(unique_values) > 2:
        preview = ", ".join(_format_label(v) for v in unique_values[:5])
        raise DatasetError(
            f"The selected target column has {len(unique_values)} distinct values "
            f"({preview}...). This app supports binary targets only. Pick a column "
            "with exactly two classes."
        )

    first, second = unique_values

    def label_for(value) -> str:
        return _format_label(value).strip().lower()

    positive_matches = [v for v in unique_values if label_for(v) in POSITIVE_TOKENS]
    negative_matches = [v for v in unique_values if label_for(v) in NEGATIVE_TOKENS]

    if pd.api.types.is_numeric_dtype(clean) and set(unique_values) != {0, 1}:
        # Numeric encodings such as {1, 2}: the larger value is the event.
        positive = max(unique_values)
    elif len(positive_matches) == 1 and len(negative_matches) <= 1:
        # One class clearly reads as "event happened" (Yes, Churned, 1, ...).
        positive = positive_matches[0]
    elif len(negative_matches) == 1 and positive_matches == []:
        positive = second if negative_matches[0] == first else first
    else:
        # Fall back to the minority class: the event is usually rarer.
        counts = clean.value_counts()
        positive = counts.idxmin()

    negative = second if positive == first else first

    mapping = {positive: 1, negative: 0}
    encoded = y_raw.map(lambda v: np.nan if pd.isna(v) else mapping.get(v, np.nan))
    return encoded.to_numpy(dtype=float), _format_label(positive), _format_label(negative)


@dataclass
class TargetReport:
    """Validation summary for the currently selected target column."""

    column: str
    is_valid: bool
    error: str | None = None
    positive_label: str = ""
    negative_label: str = ""
    n_rows: int = 0
    n_missing: int = 0
    n_positive: int = 0
    n_negative: int = 0

    @property
    def positive_rate(self) -> float:
        valid = self.n_positive + self.n_negative
        return (self.n_positive / valid * 100) if valid else 0.0


#: Column-name fragments that usually mean "recorded after the outcome".
LEAKAGE_NAME_HINTS = (
    "churnreason",
    "reasonforchurn",
    "churndate",
    "cancellationdate",
    "canceldate",
    "exitdate",
    "closuredate",
    "retentionoutcome",
    "churnconfidence",
    "churnscore",
)


@dataclass
class LeakageFinding:
    """One suspicious feature found by `scan_for_leakage`."""

    columns: list[str]
    kind: str
    detail: str


def scan_for_leakage(
    df: pd.DataFrame,
    target: str | None,
    max_levels: int = 20,
) -> list[LeakageFinding]:
    """
    Heuristic leakage investigation run before training.

    Detects:
      1. Columns whose name implies post-outcome information (churn reason/date...).
      2. A feature that reproduces the target 1:1 (duplicate label column).
      3. A low-cardinality feature whose frequent levels are each 100% one class,
         i.e. the feature perfectly separates churners from non-churners.

    Identifier columns are handled by `suggest_excluded_columns` and the
    preprocessing pipeline is always fitted on the training split only.
    """
    findings: list[LeakageFinding] = []
    if df is None or df.empty or not target or target not in df.columns:
        return findings

    try:
        encoded, _, _ = encode_binary_target(df[target])
    except DatasetError:
        return findings

    y = pd.Series(encoded, index=df.index)
    valid = y.notna()
    y = y[valid]
    if y.nunique() < 2:
        return findings

    min_support = max(10, int(0.01 * len(y)))

    for column in df.columns:
        if column == target:
            continue

        normalized = normalize_name(column)
        if any(hint in normalized for hint in LEAKAGE_NAME_HINTS):
            findings.append(
                LeakageFinding(
                    [column],
                    "post_outcome",
                    "The column name suggests information recorded after the outcome.",
                )
            )
            continue

        series = df[column].loc[valid]
        levels = series.nunique(dropna=True)
        if levels < 2 or levels > max_levels:
            continue

        frame = pd.DataFrame({"x": series.astype(str).str.strip(), "y": y})
        stats = frame.groupby("x")["y"].agg(["mean", "count"])
        frequent = stats[stats["count"] >= min_support]
        if len(frequent) < 2:
            continue

        pure = (frequent["mean"] >= 0.999) | (frequent["mean"] <= 0.001)
        if pure.all() and frequent["mean"].nunique() > 1:
            kind = "duplicate_target" if levels == 2 else "perfect_separator"
            findings.append(
                LeakageFinding(
                    [column],
                    kind,
                    "Every frequent value maps to a single class — the feature "
                    "effectively encodes the target.",
                )
            )

    return findings


def validate_target(df: pd.DataFrame, target: str | None) -> TargetReport:
    """
    Checks that `target` exists and holds exactly two classes.
    Never raises: returns a report whose `error` explains any problem.
    """
    if df is None or df.empty:
        return TargetReport(column=str(target), is_valid=False, error="The dataset is empty.")
    if not target or target not in df.columns:
        return TargetReport(
            column=str(target),
            is_valid=False,
            error="Select a target column to predict.",
        )

    raw = df[target]
    is_text = raw.dtype == object or str(raw.dtype).startswith("string")
    if is_text:
        blank = raw.astype(str).str.strip() == ""
        n_missing = int((raw.isna() | blank).sum())
    else:
        n_missing = int(raw.isna().sum())

    try:
        encoded, positive, negative = encode_binary_target(raw)
    except DatasetError as error:
        return TargetReport(
            column=target,
            is_valid=False,
            error=str(error),
            n_rows=len(df),
            n_missing=n_missing,
        )

    values = pd.Series(encoded).dropna()
    return TargetReport(
        column=target,
        is_valid=True,
        positive_label=positive,
        negative_label=negative,
        n_rows=len(df),
        n_missing=n_missing,
        n_positive=int((values == 1).sum()),
        n_negative=int((values == 0).sum()),
    )


# ------------------------------------------------------------------------------
# 4. DATA PREPARATION
# ------------------------------------------------------------------------------
def _unique_column_names(columns: Iterable) -> list[str]:
    """Strips column names and makes them unique (keeps pandas happy)."""
    seen: dict[str, int] = {}
    output: list[str] = []
    for raw in columns:
        name = str(raw).strip() or "column"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        output.append(name)
    return output


def _clean_categorical(series: pd.Series) -> pd.Series:
    """Normalises a categorical column to plain strings while preserving NaN."""
    return series.map(lambda value: np.nan if pd.isna(value) else str(value).strip()).astype(object)


def prepare_dataset(
    df: pd.DataFrame,
    target: str,
    drop_cols: Sequence[str] | None = None,
    max_categorical_levels: int = MAX_CATEGORICAL_LEVELS,
) -> tuple[pd.DataFrame, np.ndarray, DatasetSchema]:
    """
    Turns a raw DataFrame into model-ready (X, y, schema).

    Imputation is intentionally NOT performed here: the Pipeline handles missing
    values so that statistics are learned on the training split only.

    Raises DatasetError with a user-facing message on unsupported input.
    """
    if df is None or len(df) == 0:
        raise DatasetError("The dataset is empty.")
    if target is None or target not in df.columns:
        raise DatasetError(f"Target column '{target}' was not found in the dataset.")

    frame = df.copy()
    frame.columns = _unique_column_names(frame.columns)
    if target not in frame.columns:
        raise DatasetError(
            f"Target column '{target}' is no longer available after cleaning column names."
        )

    # Whitespace-only cells are missing values (e.g. Telco's blank TotalCharges).
    for column in frame.columns:
        if column == target:
            continue
        series = frame[column]
        if series.dtype == object or str(series.dtype).startswith("string"):
            frame[column] = series.map(
                lambda value: np.nan if isinstance(value, str) and value.strip() == "" else value
            )

    # ---- Target ---------------------------------------------------------
    raw_target = frame[target]
    keep_mask = raw_target.notna()
    if raw_target.dtype == object or str(raw_target.dtype).startswith("string"):
        keep_mask &= raw_target.astype(str).str.strip() != ""
    y_encoded, positive_label, negative_label = encode_binary_target(raw_target[keep_mask])

    target_series = pd.Series(y_encoded, index=raw_target[keep_mask].index)
    frame = frame.loc[target_series.index]

    # ---- Feature typing -------------------------------------------------
    drop_cols = [c for c in (drop_cols or []) if c in frame.columns and c != target]
    feature_frame = frame.drop(columns=[target] + drop_cols)

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    dropped: dict[str, str] = {c: "Excluded before training" for c in drop_cols}

    for column in feature_frame.columns:
        series = feature_frame[column]
        if series.isna().all():
            dropped[column] = "All values missing"
            continue

        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_series = series.astype(float)
            feature_frame[column] = numeric_series
            if numeric_series.nunique(dropna=True) <= 1:
                dropped[column] = "Constant value"
            else:
                numeric_cols.append(column)
            continue

        converted = _coerce_numeric_like(series)
        if converted is not None:
            feature_frame[column] = converted
            if converted.nunique(dropna=True) <= 1:
                dropped[column] = "Constant value"
            else:
                numeric_cols.append(column)
            continue

        as_text = _clean_categorical(series)
        feature_frame[column] = as_text
        levels = as_text.nunique(dropna=True)
        if levels <= 1:
            dropped[column] = "Constant value"
        elif levels > max_categorical_levels:
            dropped[column] = f"Too many categories ({levels})"
        else:
            categorical_cols.append(column)

    feature_frame = feature_frame.drop(columns=list(dropped.keys()), errors="ignore")

    if not numeric_cols and not categorical_cols:
        raise DatasetError(
            "No usable feature columns were found after cleaning. "
            "Every column was either constant, empty, or excluded."
        )

    schema = DatasetSchema(
        target=target,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        dropped_cols=dropped,
        positive_label=positive_label,
        negative_label=negative_label,
        n_rows=len(feature_frame),
        n_features=len(numeric_cols) + len(categorical_cols),
    )

    y = target_series.to_numpy(dtype=int)
    return feature_frame, y, schema


# ------------------------------------------------------------------------------
# 5. PREPROCESSING & MODELS
# ------------------------------------------------------------------------------
def build_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    """
    Median-imputes + standardises numeric features and mode-imputes + one-hot
    encodes categorical features. Unknown categories are ignored at inference.
    """
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_cols),
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_cols),
            )
        )

    if not transformers:
        raise DatasetError("There are no numeric or categorical features to preprocess.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def available_models() -> dict[str, callable]:
    """Factory mapping of model display name -> zero-argument constructor."""
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Gaussian Naive Bayes": lambda: GaussianNB(),
    }


def build_pipeline(model_name: str, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> Pipeline:
    """Builds a fresh end-to-end Pipeline (preprocessing + classifier)."""
    factories = available_models()
    if model_name not in factories:
        raise DatasetError(f"Unknown model '{model_name}'.")
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("classifier", factories[model_name]()),
        ]
    )


def train_models(
    X: pd.DataFrame,
    y: np.ndarray,
    schema: DatasetSchema,
    model_names: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Trains and benchmarks the requested models on an 80/20 split.

    Returns a dictionary with the results table, every fitted pipeline, the best
    model by F1 score, test-set predictions and the confusion matrix.
    """
    if len(X) < 20:
        raise DatasetError(
            f"Only {len(X)} usable rows are available. At least 20 are required to train and test a model."
        )

    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < 2:
        raise DatasetError(
            "Each class needs at least two rows to create a train/test split. "
            "The dataset has too few examples of one class."
        )

    stratify = y if class_counts.min() >= 5 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    requested = list(model_names) if model_names else list(available_models().keys())
    skipped: list[str] = []

    results: list[dict] = []
    pipelines: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}

    for name in requested:
        pipe = build_pipeline(name, schema.numeric_cols, schema.categorical_cols)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        pipelines[name] = pipe
        predictions[name] = pred
        if hasattr(pipe, "predict_proba"):
            try:
                probabilities[name] = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                probabilities[name] = np.full(len(y_test), np.nan)

        results.append(
            {
                "Model": name,
                "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "Precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
                "F1 Score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            }
        )

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["F1 Score"].idxmax()]
    best_name = str(best_row["Model"])

    return {
        "results_df": results_df,
        "pipelines": pipelines,
        "best_model_name": best_name,
        "best_pipeline": pipelines[best_name],
        "best_metrics": {k: float(v) for k, v in best_row.to_dict().items() if k != "Model"},
        "best_predictions": predictions[best_name],
        "test_predictions": predictions,
        "test_probabilities": probabilities,
        "confusion_matrix": confusion_matrix(y_test, predictions[best_name]),
        "y_test": y_test,
        "X_test": X_test,
        "X_train": X_train,
        "y_train": y_train,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "schema": schema,
        "skipped": skipped,
    }


# ------------------------------------------------------------------------------
# 6. INFERENCE HELPERS
# ------------------------------------------------------------------------------
def predict_single(pipeline: Pipeline, row: dict) -> tuple[int, float]:
    """
    Predicts the class and positive-class probability for a single row.
    Columns the pipeline expects but the caller omitted are filled with NaN and
    imputed by the pipeline itself, so a partial form never crashes inference.
    """
    frame = pd.DataFrame([row])
    expected = list(pipeline.feature_names_in_) if hasattr(pipeline, "feature_names_in_") else []
    for column in expected:
        if column not in frame.columns:
            frame[column] = np.nan
    if expected:
        frame = frame[expected]

    pred = int(pipeline.predict(frame)[0])
    if hasattr(pipeline, "predict_proba"):
        proba = float(pipeline.predict_proba(frame)[0][1])
    else:
        proba = float(pred)
    return pred, proba


def column_profile(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """
    Builds a compact per-feature profile used by the UI:
    name, kind, missing %, unique values, and the average positive-class rate.
    """
    y = None
    if schema.target in df.columns:
        try:
            encoded, _, _ = encode_binary_target(df[schema.target])
            y = pd.Series(encoded, index=df.index)
        except DatasetError:
            y = None

    rows = []
    for column in schema.feature_cols:
        series = df[column]
        kind = "Numeric" if column in schema.numeric_cols else "Categorical"
        row = {
            "Feature": column,
            "Type": kind,
            "Missing %": round(float(series.isna().mean() * 100), 2),
            "Unique": int(series.nunique(dropna=True)),
        }
        if y is not None and kind == "Categorical":
            grouped = series.astype(str).str.strip().where(series.notna())
            rates = y.groupby(grouped, dropna=True).mean()
            row["Positive rate range"] = (
                f"{rates.min() * 100:.1f}% – {rates.max() * 100:.1f}%" if len(rates) else "n/a"
            )
        elif y is not None:
            values = pd.to_numeric(series, errors="coerce")
            by_class = values.groupby(y).mean()
            row["Positive rate range"] = (
                f"mean {by_class.min():.2f} vs {by_class.max():.2f}"
                if len(by_class) == 2
                else "n/a"
            )
        else:
            row["Positive rate range"] = "n/a"
        rows.append(row)

    return pd.DataFrame(rows)
