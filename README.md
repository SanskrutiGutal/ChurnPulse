# Customer Churn Classification — Telco & Cell2Cell

A clean, upload-only Streamlit application that predicts whether a telecom customer will
**churn or stay**. It supports exactly two public datasets, trains four classifiers on the same
80/20 split, compares them with Accuracy, Precision, Recall and F1 Score, and scores individual
customers from a form generated from the uploaded file's own columns.

**One dataset input method: `📤 Upload Dataset (CSV)`.** There is no built-in dataset button, no
dataset menu and no radio button.

---

## 1. Project at a glance

| Item | Detail |
| :--- | :--- |
| Problem | Binary classification (churn = positive class) |
| Supported datasets | **Telco Customer Churn** and **Cell2Cell Telecom Churn** |
| Input | One CSV at a time, uploaded on the welcome screen |
| Models | Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes |
| Split | 80% train / 20% test, `random_state=42`, stratified when possible |
| Selection | Highest **F1 Score** on the holdout set |
| Preprocessing | Scikit-Learn `Pipeline` (impute → scale → one-hot → model), fitted on train only |
| Outputs | Dataset-specific `.pkl` pipeline and metrics CSV (download on the Performance page) |
| Interface | Streamlit, white background, navy/teal accents, subtle entrance/hover motion, Plotly charts |

### Measured best models (from `outputs/*_model_results.csv`)

| Dataset | Best model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Telco Customer Churn | Logistic Regression | 80.55% | 65.72% | 55.88% | **60.40%** |
| Cell2Cell Telecom Churn | Gaussian Naive Bayes | 59.64% | 35.98% | 51.43% | **42.34%** |

These are genuine holdout results, not tuned or rounded up.

---

## 2. Project structure

```text
D:\program\ML project\
├── app.py                         Streamlit UI: main upload/setup, routing, theme, About page
├── dataset_manager.py             Dataset registry, detection, validation, schema, pipeline, training
├── dataset_views.py               Dashboard, Prediction, Visualization and Performance pages
├── train_model.py                 CLI training + dataset-specific artifacts
├── main.py                        CLI master workflow + sample prediction
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml                Forces the light (white) theme
├── Telco-Customer-Churn.csv       Genuine Telco dataset (workspace copy, tested)
├── cell2celltrain.csv             Genuine Cell2Cell dataset (workspace copy, tested)
├── data/
│   └── Telco-Customer-Churn.csv   Identical bundled Telco copy
├── outputs/
│   ├── telco_model_results.csv
│   ├── telco_accuracy_comparison.png
│   ├── telco_confusion_matrix.png
│   ├── cell2cell_model_results.csv
│   ├── cell2cell_accuracy_comparison.png
│   └── cell2cell_confusion_matrix.png
└── models/
    ├── telco_churn_model.pkl
    └── cell2cell_churn_model.pkl
```

Artifacts are written with a dataset prefix (`telco_…`, `cell2cell_…`), so one dataset never
overwrites another.

---

## 3. Supported datasets

| Dataset | Rows × Columns | Target | Classes | Churn rate |
| :--- | :---: | :---: | :---: | :---: |
| **Telco Customer Churn** | 7,043 × 21 | `Churn` | No / Yes | 26.54% |
| **Cell2Cell Telecom Churn** | 51,047 × 58 | `Churn` | No / Yes | 28.82% |

### 3.1 Telco Customer Churn

- Kaggle: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- Direct CSV (IBM GitHub): <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>
- Tested file: `Telco-Customer-Churn.csv` (project root). An identical copy is in `data/`.

### 3.2 Cell2Cell Telecom Churn

- Kaggle: <https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom>
  (a Kaggle account is required to download).
- Genuine mirror: <https://raw.githubusercontent.com/janebunr/Cell2Cell/master/Data%20files/cell2celltrain.csv>
- Tested file: `cell2celltrain.csv` (project root).
- The Kaggle bundle also contains `cell2cellholdout.csv`, which has **no `Churn` column** and is
  therefore rejected for training (it is a scoring file, not a labelled dataset).

### 3.3 Verified files

Both files were opened with pandas, checked for shape/classes, and hashed before the results were
reported. No synthetic or modified data was used.

| Dataset | Rows | Cols | Class counts | SHA-256 |
| :--- | :---: | :---: | :--- | :--- |
| Telco | 7,043 | 21 | No 5,174 / Yes 1,869 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Cell2Cell | 51,047 | 58 | No 36,336 / Yes 14,711 | `3f6b8ff0b8e8f5975aa0de9f1f54967e6c8cb867a9bec6cbb64f4c8990f9a645` |

---

## 4. Dataset input and navigation

The welcome screen is intentionally minimal: it shows only the project name and one large upload
card in the main panel.

```text
Customer Churn Prediction

📤 Upload Dataset (CSV)
```

It is deliberately **not** hidden in the sidebar, so it stays visible in narrow windows, collapsed
sidebars and embedded previews. The sidebar is still present (expanded by default) and shows the
current dataset and active model once a file is loaded, but the whole workflow works with it closed.

After a file is selected, the setup screen shows:

- the detected dataset name (matched from its columns),
- the uploaded filename, row count and column count,
- the auto-detected target column, editable through a dropdown,
- the two class counts and any empty-target warning,
- an **Advanced · excluded columns** panel (IDs and free text are pre-selected),
- a data preview and the **Train models** button.

The five-page navigation (Dashboard, Prediction, Visualization, Performance, About) is a pill bar at
the top of the main panel, with **Change dataset** beside it. The sidebar also offers **Change
dataset**; either button clears the session and returns to the upload card. This keeps navigation
visible even when the sidebar is collapsed.

If a file is unsupported or the target cannot be used, the screen offers **Use a different CSV** to
return to the upload card immediately.

If the CSV is not one of the two supported datasets, the app stops and lists the signature columns
it expected. Nothing is fabricated or silently substituted.

Motion is deliberately restrained: a one-time fade-up on the title and upload card, a slow animated
teal underline behind the title, hover lifts on cards/buttons, and a `prefers-reduced-motion`
fallback that disables all animation.

Uploaded data stays in the Streamlit session. The original CSV on disk is **never modified**; a file
is only written when you explicitly click a download button or run the CLI.

---

## 5. Pages

After uploading, a **setup screen** appears until the models are trained: it shows the target
selector, suggested exclusions, class balance, dataset summary and the **Train models** button. The
five navigation pages below are then available from the top pill bar.

| Page | Contents |
| :--- | :--- |
| **Dashboard** | Total customers, churned, retained, churn rate, churn distribution donut, dataset summary (file, rows, columns, target, features used, exclusions, leakage check) |
| **Prediction** | Form generated from the uploaded schema (numeric sliders/inputs, categorical dropdowns), predicted class, churn probability, probability gauge and the strongest signals for that record |
| **Visualization** | Churn distribution, churn rate by contract type, tenure distribution by churn, monthly charges by churn, internet service vs churn, payment method vs churn (each only when the column exists), missing-value analysis |
| **Performance** | KPI cards for the best model, comparison of all four algorithms, metric table, per-model confusion matrix, metrics CSV and `.pkl` downloads |
| **About** | Objective, supported datasets and links, algorithms, preprocessing and leakage control, evaluation metrics, limitations |

### How the Visualization page adapts

- **Telco** has `Contract`, `InternetService`, `PaymentMethod`, `tenure` and `MonthlyCharges`, so all
  five requested charts appear.
- **Cell2Cell** has no contract, internet-service or payment-method columns. It shows the charts
  whose concepts exist (`MonthsInService` as tenure, `MonthlyRevenue` as monthly charges), plus a
  feature explorer for any low-cardinality categorical column. The contract chart is replaced by
  this explorer rather than by invented data.

All values, rates and insights are calculated from the uploaded rows.

---

## 6. Machine-learning pipeline

### 6.1 Detection and validation

1. Column headers are stripped; column names are normalised (lowercase, non-alphanumerics removed).
2. The file is matched against a tolerant signature for each supported dataset (accepts common
   casing/spacing variants such as `MonthsInService` / `Months In Service`).
3. The target is auto-detected (`Churn` for both datasets) and can be corrected on the setup screen.
4. The target must contain **exactly two classes**. More or fewer classes block training with a
   clear message. Rows with an empty target are dropped, and the count is reported.

### 6.2 Cleaning and feature handling

- Numeric values stored as text (e.g. Telco `TotalCharges`) are converted with
  `pd.to_numeric(errors="coerce")`.
- Missing numeric values → median imputation; missing categorical values → most-frequent imputation.
- Identifier columns, constants, free-text columns and columns with more than 50 levels are excluded
  automatically (e.g. Telco `customerID`; Cell2Cell `CustomerID` and `ServiceArea`).
- Users can un-exclude any column in **Advanced · excluded columns**.
- Categorical features are one-hot encoded with `handle_unknown="ignore"`.

### 6.3 Leakage prevention

- Imputation, scaling and encoding live **inside** the Scikit-Learn `Pipeline`, fitted on the
  training split only.
- Identifier columns are removed before the split.
- A heuristic leakage scan runs before training and reports:
  - column names that suggest post-outcome information (churn reason/date, cancellation date, …),
  - a feature that reproduces the target 1:1,
  - a low-cardinality feature whose frequent levels each map to a single class (perfect separator).
- **Result for both datasets: no leakage indicators found.** The scan is shown on the About page and
  in the CLI output.

The scan is a safety check, not proof of causality. For example, Cell2Cell retention-call columns
(`RetentionCalls`, `MadeCallToRetentionTeam`, `RetentionOffersAccepted`) are legitimate business
actions recorded around the cancellation decision, not copies of the label; they are kept as
features and are visible in the excluded-columns panel if you choose to remove them.

`TotalCharges` in Telco is derived from tenure × monthly charges, so it is correlated with other
features but is not target leakage.

### 6.4 Models and evaluation

| Model | Baseline configuration |
| :--- | :--- |
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `random_state=42` |
| K-Nearest Neighbors | `n_neighbors=5` (trained on all rows, including Cell2Cell's 40,837 training rows) |
| Gaussian Naive Bayes | default |

- Split: `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` when both classes have
  enough members.
- Metrics: Accuracy, Precision, Recall, F1 Score for the churn (positive) class.
- Best model: highest F1 Score. That pipeline becomes the active model for predictions.

---

## 7. Measured results

### 7.1 Telco Customer Churn — 5,634 train / 1,409 test rows

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | ---: | ---: | ---: | ---: |
| **Logistic Regression** | **80.55%** | **65.72%** | 55.88% | **60.40%** |
| Decision Tree | 79.84% | 63.47% | 56.68% | 59.89% |
| KNN | 76.37% | 55.27% | 57.49% | 56.36% |
| Gaussian Naive Bayes | 69.48% | 45.89% | **83.69%** | 59.28% |

Confusion matrix for the best model (TN / FP / FN / TP): **926 / 109 / 165 / 209**.

### 7.2 Cell2Cell Telecom Churn — 40,837 train / 10,210 test rows

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | ---: | ---: | ---: | ---: |
| Logistic Regression | 71.13% | 48.70% | 3.81% | 7.06% |
| Decision Tree | **71.60%** | **60.00%** | 4.28% | 7.99% |
| KNN | 67.01% | 35.51% | 17.74% | 23.66% |
| **Gaussian Naive Bayes** | 59.64% | 35.98% | **51.43%** | **42.34%** |

Confusion matrix for the best model (TN / FP / FN / TP): **4,576 / 2,692 / 1,429 / 1,513**.

Cell2Cell is noisy and imbalanced. Logistic Regression and the Decision Tree achieve higher accuracy
but miss almost all churners (very low recall), which is why Gaussian Naive Bayes wins on F1. These
numbers are reported exactly as measured — no resampling or tuning was applied.

---

## 8. Installation and run (Windows PowerShell)

```powershell
cd "D:\program\ML project"
python -m pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not on your PATH:

```powershell
python -m streamlit run app.py
```

The app opens at <http://localhost:8501>. If that port is busy:

```powershell
python -m streamlit run app.py --server.port 8502
```

Optional virtual environment:

```powershell
cd "D:\program\ML project"
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Verified with Python 3.14.8, Streamlit 1.56.0, pandas 2.3.3, scikit-learn 1.8.0, NumPy 2.4.0 and
Plotly 7.1.0.

### Demo flow

1. Start the app.
2. In the main upload card, select `Telco-Customer-Churn.csv` or `cell2celltrain.csv`.
3. Confirm the target column (auto-detected as `Churn`) on the setup screen.
4. Click **Train models**.
5. Explore Dashboard → Prediction → Visualization → Performance from the top pill bar.

---

## 9. Command-line usage

Train and export one dataset (writes only dataset-prefixed artifacts):

```powershell
python train_model.py --data Telco-Customer-Churn.csv
python train_model.py --data cell2celltrain.csv
```

Full workflow with a sample prediction from the first row:

```powershell
python main.py --data Telco-Customer-Churn.csv
python main.py --data cell2celltrain.csv
```

Useful options:

```powershell
python train_model.py --help
python train_model.py --data cell2celltrain.csv --drop CustomerID
python train_model.py --data Telco-Customer-Churn.csv --target Churn
```

The README does not require the CLI — the Streamlit app performs the same training in memory and
offers the artifacts as downloads.

---

## 10. Tests actually performed

### 10.1 End-to-end app tests (`streamlit.testing.v1.AppTest`) — 2/2 passed

For **both** datasets, the automated test:

- loaded the welcome screen and confirmed the project title, exactly **1** file uploader and
  **0** radio buttons,
- uploaded the genuine CSV and confirmed the setup screen appeared,
- confirmed the detected dataset, target (`Churn`), row/column counts and exclusions,
- clicked **Train models** and confirmed the active-model card and top navigation bar,
- visited all five pages with no exception,
- verified the four KPI cards and the leakage check on the Dashboard,
- verified the expected Visualization charts (Telco: 5/5 requested column charts; Cell2Cell: the two
  applicable charts plus the feature explorer),
- submitted the Prediction form and confirmed a class + probability were stored,
- found the metrics table with all four models and **2** download buttons.

### 10.2 Negative-path tests — 6/6 passed

| Case | Result |
| :--- | :--- |
| Unsupported bank-churn CSV | Rejected, diagnostics shown, no training offered |
| Removed old Churn 2020 file | Rejected — only the two requested datasets are accepted |
| Telco with a 3-class target | Blocked with an "exactly two classes" message, training disabled |
| Manually selecting `MonthlyCharges` as target | Blocked (1,585 distinct values), training disabled |
| 40 empty target labels | Warning shown; rows dropped; training succeeds |
| 1-row / malformed CSV | Clear "not enough rows" error; no training offered |

### 10.3 CLI tests — 2/2 + 2/2 passed

- `train_model.py` completed for Telco and Cell2Cell and wrote all four artifacts each.
- `main.py` completed for both, including a sample prediction from the first row.
- A live server check returned **HTTP 200 `ok`** from `/_stcore/health`.

### 10.4 Visual checks

Headless Chrome screenshots at 1500×1000 were used to inspect the welcome screen, setup screen,
Dashboard, Visualization, Prediction, Performance and About pages. The main upload card, dataset
cards, top pill navigation, charts and sidebar were verified to render with the white/navy/teal
theme, including with the browser emulating dark mode.

### 10.5 Not tested / known gaps

- The Kaggle-hosted Cell2Cell download requires a logged-in account; the tested file is the genuine
  copy included in the project.
- Phone-width layouts were not tested on a physical device; the stylesheet includes 1100 px and
  640 px breakpoints for the KPI grid, hero and cards.
- Cosmetic Streamlit deprecation warnings about `use_container_width` remain; they do not affect
  behaviour.

---

## 11. Limitations

- Only the two listed telecom churn datasets are accepted; other CSVs are rejected by design.
- Binary targets only (two classes).
- The four algorithms use fixed baseline settings — no hyper-parameter search, class weighting or
  SMOTE.
- Cell2Cell's low precision/recall reflects the data's noise and imbalance; it is not a bug.
- The leakage scan is heuristic; it detects obvious cases, not every possible subtle leak.
- Uploaded data lives only in the Streamlit session. Nothing is persisted unless you download an
  artifact or run a CLI command.
- The app does not modify or overwrite the original CSV files.

---

## 12. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `streamlit` is not recognized | `python -m streamlit run app.py` |
| Port 8501 already in use | `python -m streamlit run app.py --server.port 8502` |
| "This CSV is not one of the two supported churn datasets" | Upload `Telco-Customer-Churn.csv` or `cell2celltrain.csv`; the error lists the expected signature columns |
| Training disabled / target error | Choose a column with exactly two classes (e.g. `Churn`) |
| Upload does nothing | Confirm the file is a CSV/TXT under the 200 MB upload limit |
| Charts look empty after switching pages | Wait for the spinner to finish; large files take a few seconds to redraw |
