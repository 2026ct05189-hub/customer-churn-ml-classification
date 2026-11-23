"""
Telecom Customer Churn - Classification Study
=============================================

What this program does
-----------------------
Given information about a telecom customer (their contract, the services they
use and how they are billed), we try to answer a single yes/no question:

        "Is this customer about to leave us (churn)?"

Because the answer is one of two labels (churn = yes / no), this is a
*binary classification* task. We train five different classifiers on the very
same data and compare them fairly using six evaluation metrics.

The file is organised top-to-bottom as a readable story. I kept the comments
chatty on purpose because I am still learning ML and wanted to explain the
reasoning to myself (and to anyone reading this repo).

Run it with:
        python telco_churn_pipeline.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_curve, confusion_matrix,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="ticks", palette="Set2")   # a cleaner, distinct look

# ---------------------------------------------------------------------------
# Central configuration. Keeping every "knob" in one dictionary means I can
# tweak the experiment without hunting through the code.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CONFIG = {
    "data_file": ROOT / "data" / "telco_customers.csv",
    "model_dir": HERE,
    "report_dir": ROOT / "outputs",
    "holdout_fraction": 0.20,     # keep 20% aside for honest final testing
    "seed": 2026,                 # my fixed seed -> fully repeatable results
    "id_column": "customerID",
    "target_column": "Churn",
    "cv_folds": 5,                # folds for the cross-validation sanity check
}
CONFIG["model_dir"].mkdir(exist_ok=True)
CONFIG["report_dir"].mkdir(exist_ok=True)


# ===========================================================================
#  SECTION 1 - GET TO KNOW THE DATA
# ===========================================================================
def inspect_dataset(frame: pd.DataFrame) -> None:
    """Print a short profile of the dataset so we understand what we're modelling."""
    print("#" * 72)
    print("# SECTION 1 - DATASET OVERVIEW")
    print("#" * 72)

    rows, cols = frame.shape
    print(f"\nCustomers (rows)      : {rows}")
    print(f"Attributes (columns)  : {cols}")

    # Which columns are text vs numbers? That decides how we preprocess them.
    text_cols = frame.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = frame.select_dtypes(include=["number"]).columns.tolist()
    print(f"\nText / categorical columns ({len(text_cols)}): {text_cols}")
    print(f"Numeric columns ({len(num_cols)}): {num_cols}")

    # The label we want to predict.
    target = CONFIG["target_column"]
    label_counts = frame[target].value_counts()
    label_share = frame[target].value_counts(normalize=True).round(4)
    print(f"\nLabel column = '{target}'")
    print("Counts:\n", label_counts.to_string())
    print("Share :\n", label_share.to_string())

    # Is the label lop-sided? If so we must be careful which metrics we trust.
    skew = label_counts.max() / label_counts.min()
    print(f"\nMajority-to-minority ratio = {skew:.2f} : 1")
    print("The 'stay' class dominates, so this is an IMBALANCED problem. We will")
    print("(1) split with stratification, (2) re-weight the classes inside the")
    print("models that support it, and (3) judge with AUC / F1 / MCC, not accuracy.")

    # Hunt for missing values - including the sneaky text-encoded ones.
    print(f"\nExplicit NaN values : {int(frame.isnull().sum().sum())}")
    blanks = (frame["TotalCharges"].astype(str).str.strip() == "").sum()
    print(f"Blank 'TotalCharges': {blanks} (these are missing values in disguise)")
    print(f"Exact duplicate rows: {int(frame.duplicated().sum())}")


# ===========================================================================
#  SECTION 2 - TIDY THE DATA (the steps that do NOT learn from the data)
# ===========================================================================
def tidy_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Do the deterministic clean-up that is safe to run before splitting:
      * drop the ID column,
      * turn 'TotalCharges' into real numbers,
      * delete duplicate rows,
      * turn the Yes/No label into 1/0.

    Anything that *learns* a statistic (a median to fill gaps, a scaling range,
    the list of categories) is deliberately NOT done here - it waits until
    after the split and is fitted on the training part only. That discipline is
    what keeps our test score trustworthy (no "data leakage").
    """
    print("\n" + "#" * 72)
    print("# SECTION 2 - DATA CLEANING")
    print("#" * 72)

    clean = frame.copy()

    clean = clean.drop(columns=[CONFIG["id_column"]])
    print(f"- Removed '{CONFIG['id_column']}' (just a reference number).")

    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    gaps = int(clean["TotalCharges"].isnull().sum())
    print(f"- 'TotalCharges' is numeric now; {gaps} blanks became NaN (imputed later).")

    n0 = len(clean)
    clean = clean.drop_duplicates().reset_index(drop=True)
    print(f"- Dropped {n0 - len(clean)} duplicate rows -> {len(clean)} unique rows left.")

    clean[CONFIG["target_column"]] = clean[CONFIG["target_column"]].map({"Yes": 1, "No": 0})
    print("- Label encoded: churned = 1 (the class of interest), stayed = 0.")

    return clean


# ===========================================================================
#  SECTION 3 - THE TRANSFORMER (learned on training data only)
# ===========================================================================
def assemble_transformer(features: pd.DataFrame) -> ColumnTransformer:
    """
    Describe how every column should be transformed. The actual fitting happens
    later, inside each model's Pipeline, using the training rows only.

    Numeric columns  -> fill blanks with the median, then standardise
                        (so 'TotalCharges' in thousands doesn't bully 'tenure').
    Text columns     -> fill blanks with the most common value, then one-hot
                        encode (each category becomes its own 0/1 flag).
    """
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    text_cols = [c for c in features.columns if c not in numeric_cols]

    numeric_branch = Pipeline([
        ("fill_gaps", SimpleImputer(strategy="median")),
        ("standardise", StandardScaler()),
    ])
    text_branch = Pipeline([
        ("fill_gaps", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformer = ColumnTransformer([
        ("numeric", numeric_branch, numeric_cols),
        ("text", text_branch, text_cols),
    ])
    print(f"\nTransformer ready -> {len(numeric_cols)} numeric + "
          f"{len(text_cols)} categorical columns.")
    return transformer


# ===========================================================================
#  SECTION 4 - THE FIVE CLASSIFIERS
# ===========================================================================
def build_classifier_zoo(seed: int) -> dict:
    """
    Return the five classifiers, each with my own chosen settings.

    Design notes:
      * Where a model supports `class_weight`, I set it to 'balanced' so the
        rarer "churn" class is taken seriously despite the imbalance.
      * kNN and Gaussian Naive Bayes have no class-weight option, so for those
        I lean on the stratified split and the imbalance-robust metrics instead.

    Returned as  display_name -> (short_tag_for_filename, estimator).
    """
    return {
        "Logistic Regression": (
            "logreg",
            LogisticRegression(C=0.3, class_weight="balanced",
                               solver="liblinear", max_iter=500, random_state=seed),
        ),
        "Decision Tree": (
            "dtree",
            DecisionTreeClassifier(criterion="entropy", max_depth=7,
                                   min_samples_leaf=25, class_weight="balanced",
                                   random_state=seed),
        ),
        "kNN": (
            "knn",
            KNeighborsClassifier(n_neighbors=21, weights="distance", p=1),
        ),
        "Naive Bayes": (
            "gnb",
            GaussianNB(var_smoothing=1e-8),
        ),
        "Random Forest (Ensemble)": (
            "rforest",
            RandomForestClassifier(n_estimators=250, max_depth=10,
                                   min_samples_leaf=5, max_features="sqrt",
                                   class_weight="balanced_subsample",
                                   random_state=seed),
        ),
    }


def measure_performance(name: str, fitted_pipe, X_eval, y_eval) -> dict:
    """Compute the six required metrics for one trained model on the hold-out set."""
    predicted = fitted_pipe.predict(X_eval)
    churn_prob = fitted_pipe.predict_proba(X_eval)[:, 1]   # P(churn) -> needed for AUC

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_eval, predicted),
        "AUC": roc_auc_score(y_eval, churn_prob),
        "Precision": precision_score(y_eval, predicted, zero_division=0),
        "Recall": recall_score(y_eval, predicted, zero_division=0),
        "F1": f1_score(y_eval, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_eval, predicted),
    }


# ===========================================================================
#  SECTION 5 - CHARTS
# ===========================================================================
def chart_label_balance(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=CONFIG["target_column"], data=frame,
                       hue=CONFIG["target_column"], legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stayed (0)", "Churned (1)"])
    ax.set_title("How many customers churned?")
    ax.set_xlabel(""); ax.set_ylabel("Customers")
    for bar in ax.patches:
        ax.annotate(f"{int(bar.get_height())}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    sns.despine()
    plt.tight_layout()
    plt.savefig(CONFIG["report_dir"] / "class_distribution.png", dpi=120)
    plt.close()


def chart_metric_bars(scores: pd.DataFrame) -> None:
    metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    long = scores.melt(id_vars="Model", value_vars=metric_names,
                       var_name="Metric", value_name="Value")
    plt.figure(figsize=(13, 6))
    ax = sns.barplot(data=long, x="Metric", y="Value", hue="Model")
    ax.set_title("Five classifiers compared on six metrics")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CONFIG["report_dir"] / "metrics_comparison.png", dpi=120)
    plt.close()


def chart_roc(prob_lookup: dict, y_eval) -> None:
    plt.figure(figsize=(7, 6))
    for name, prob in prob_lookup.items():
        fpr, tpr, _ = roc_curve(y_eval, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_eval, prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Coin flip (0.500)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right", fontsize=8)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CONFIG["report_dir"] / "roc_curves.png", dpi=120)
    plt.close()


def chart_confusions(cm_lookup: dict) -> None:
    n = len(cm_lookup)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6))
    for ax, (name, cm) in zip(axes, cm_lookup.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn", cbar=False, ax=ax,
                    xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Confusion matrices on the hold-out set", y=1.03)
    plt.tight_layout()
    plt.savefig(CONFIG["report_dir"] / "confusion_matrices.png",
                dpi=120, bbox_inches="tight")
    plt.close()


# ===========================================================================
#  MAIN STORY
# ===========================================================================
def run():
    seed = CONFIG["seed"]

    # --- Section 1: load + inspect -----------------------------------------
    churn_df = pd.read_csv(CONFIG["data_file"])
    inspect_dataset(churn_df)
    chart_label_balance(churn_df.assign(
        **{CONFIG["target_column"]: churn_df[CONFIG["target_column"]].map({"Yes": 1, "No": 0})}))

    # --- Section 2: clean ---------------------------------------------------
    data = tidy_dataset(churn_df)

    # Split the table into "inputs" (feature_matrix) and "answer" (churn_label).
    feature_matrix = data.drop(columns=[CONFIG["target_column"]])
    churn_label = data[CONFIG["target_column"]]

    # --- Hold-out split: 80% to learn from, 20% kept unseen for testing -----
    X_fit, X_eval, y_fit, y_eval = train_test_split(
        feature_matrix, churn_label,
        test_size=CONFIG["holdout_fraction"],
        stratify=churn_label,            # keep the churn ratio in both parts
        random_state=seed,               # repeatable split
    )
    print("\n" + "#" * 72)
    print("# SECTION 2b - TRAIN / HOLD-OUT SPLIT (80 / 20, stratified)")
    print("#" * 72)
    print(f"Learn from : {len(X_fit)} rows   |   Test on : {len(X_eval)} rows")
    print(f"Churn share -> learn: {y_fit.mean():.3f}, test: {y_eval.mean():.3f}")

    # Save the untouched test rows so graders can re-score the saved models.
    holdout = X_eval.copy()
    holdout[CONFIG["target_column"]] = y_eval.values
    holdout.to_csv(ROOT / "test_data.csv", index=False)
    print(f"Wrote test_data.csv ({len(holdout)} rows).")

    # --- Section 3: transformer --------------------------------------------
    transformer = assemble_transformer(X_fit)

    # --- Sections 4 & 5: train, cross-check, evaluate, save ----------------
    print("\n" + "#" * 72)
    print("# SECTION 4 - TRAIN, CROSS-VALIDATE AND EVALUATE EACH MODEL")
    print("#" * 72)

    zoo = build_classifier_zoo(seed)
    kfold = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=seed)

    score_rows, prob_lookup, cm_lookup = [], {}, {}

    for name, (tag, estimator) in zoo.items():
        # Glue preprocessing + model so the transformer is fitted on training
        # data only and the exact same steps run again at prediction time.
        pipe = Pipeline([("prep", transformer), ("clf", estimator)])

        # Cross-validation sanity check on the TRAINING data: it trains/tests on
        # 5 internal folds and tells us how stable the AUC is before we ever
        # touch the hold-out set. A quick guard against a lucky/unlucky split.
        cv_auc = cross_val_score(pipe, X_fit, y_fit, cv=kfold, scoring="roc_auc")

        # Now fit on ALL the training data and evaluate once on the hold-out.
        pipe.fit(X_fit, y_fit)
        row = measure_performance(name, pipe, X_eval, y_eval)
        score_rows.append(row)

        prob_lookup[name] = pipe.predict_proba(X_eval)[:, 1]
        cm_lookup[name] = confusion_matrix(y_eval, pipe.predict(X_eval))

        joblib.dump(pipe, CONFIG["model_dir"] / f"clf_{tag}.joblib")

        print(f"\n>> {name}")
        print(f"   5-fold CV AUC (train): {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")
        for metric, value in row.items():
            if metric != "Model":
                print(f"   {metric:<10}: {value:.4f}")

    # --- Comparison table ---------------------------------------------------
    leaderboard = pd.DataFrame(score_rows)
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    leaderboard[metric_cols] = leaderboard[metric_cols].round(4)
    leaderboard.to_csv(CONFIG["report_dir"] / "metrics_comparison.csv", index=False)

    print("\n" + "#" * 72)
    print("# FINAL LEADERBOARD (hold-out set)")
    print("#" * 72)
    print(leaderboard.to_string(index=False))

    champion = leaderboard.loc[leaderboard["AUC"].idxmax(), "Model"]
    print(f"\nBest AUC on this dataset -> {champion}")

    # --- Remaining charts ---------------------------------------------------
    chart_metric_bars(leaderboard)
    chart_roc(prob_lookup, y_eval)
    chart_confusions(cm_lookup)
    print(f"\nCharts + metrics_comparison.csv saved to: {CONFIG['report_dir']}")
    print(f"Five models (clf_*.joblib) saved to: {CONFIG['model_dir']}")


if __name__ == "__main__":
    run()
