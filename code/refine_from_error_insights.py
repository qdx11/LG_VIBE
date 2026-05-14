from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from mining_experiment_suite import DATE_COL, TARGET_REG, build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "mining_process"
RANDOM_STATE = 42
RISK_COL = "high_silica_risk"

INSIGHT_FEATURES = [
    "Flotation Column 07 Air Flow__std",
    "Flotation Column 02 Level__std",
    "Flotation Column 03 Air Flow",
    "Flotation Column 01 Air Flow",
    "Flotation Column 06 Level",
    "% Silica Feed",
    "% Iron Feed",
]


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=7,
        random_state=RANDOM_STATE,
        verbose=0,
    )


def add_rolling_features(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = data.sort_values(DATE_COL).copy()
    valid_features = [col for col in INSIGHT_FEATURES if col in feature_cols]
    for col in valid_features:
        shifted = out[col].shift(1)
        out[f"{col}__rolling3_median_lag1"] = shifted.rolling(3, min_periods=1).median()
        out[f"{col}__rolling6_median_lag1"] = shifted.rolling(6, min_periods=1).median()
        out[f"{col}__delta_vs_rolling3_median"] = out[col] - out[f"{col}__rolling3_median_lag1"]
    return out.bfill().ffill()


def fit_clip(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lower = x_train.quantile(0.01)
    upper = x_train.quantile(0.99)
    return (
        x_train.clip(lower=lower, upper=upper, axis=1),
        x_test.clip(lower=lower, upper=upper, axis=1),
    )


def add_anomaly_flags(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = x_train.copy()
    test = x_test.copy()
    for col in [c for c in INSIGHT_FEATURES if c in train.columns]:
        mean = train[col].mean()
        std = train[col].std(ddof=0)
        if pd.isna(std) or std == 0:
            continue
        train[f"{col}__is_3sigma_anomaly"] = ((train[col] - mean).abs() >= 3 * std).astype(int)
        test[f"{col}__is_3sigma_anomaly"] = ((test[col] - mean).abs() >= 3 * std).astype(int)
    return train, test


def evaluate(y_true: pd.Series, pred: np.ndarray, split: str, experiment: str) -> dict[str, float | str]:
    return {
        "experiment": experiment,
        "split": split,
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
    }


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, proba >= threshold, average="macro", zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def segment_threshold_predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_proba: np.ndarray,
    test_x: pd.DataFrame,
    test_proba: np.ndarray,
    segment_feature: str = "Flotation Column 07 Air Flow__std",
) -> tuple[np.ndarray, pd.DataFrame]:
    if segment_feature not in train_x.columns:
        threshold = best_threshold(train_y, train_proba)
        return (test_proba >= threshold).astype(int), pd.DataFrame()

    bins = pd.qcut(train_x[segment_feature], q=4, duplicates="drop")
    categories = bins.cat.categories
    global_threshold = best_threshold(train_y, train_proba)
    threshold_rows = []
    test_pred = np.zeros(len(test_x), dtype=int)
    assigned = np.zeros(len(test_x), dtype=bool)

    for category in categories:
        train_mask = bins == category
        if train_mask.sum() < 30 or train_y[train_mask].nunique() < 2:
            threshold = global_threshold
        else:
            threshold = best_threshold(train_y[train_mask], train_proba[train_mask])

        left = category.left
        right = category.right
        test_mask = (test_x[segment_feature] > left) & (test_x[segment_feature] <= right)
        if category == categories[0]:
            test_mask = (test_x[segment_feature] >= left) & (test_x[segment_feature] <= right)
        test_pred[test_mask.to_numpy()] = (test_proba[test_mask.to_numpy()] >= threshold).astype(int)
        assigned |= test_mask.to_numpy()
        threshold_rows.append(
            {
                "segment_feature": segment_feature,
                "bin": str(category),
                "train_rows": int(train_mask.sum()),
                "train_positive_rate": float(train_y[train_mask].mean()),
                "threshold": threshold,
            }
        )

    test_pred[~assigned] = (test_proba[~assigned] >= global_threshold).astype(int)
    threshold_rows.append(
        {
            "segment_feature": segment_feature,
            "bin": "fallback_global",
            "train_rows": int(len(train_y)),
            "train_positive_rate": float(train_y.mean()),
            "threshold": global_threshold,
        }
    )
    return test_pred, pd.DataFrame(threshold_rows)


def run_experiment(
    experiment: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    rolling_feature_cols: list[str],
    use_rolling: bool = False,
    use_clip: bool = False,
    use_anomaly_flags: bool = False,
    exclude_suspects: bool = False,
    use_segment_threshold: bool = False,
) -> tuple[list[dict[str, float | str]], pd.DataFrame]:
    selected_features = feature_cols + rolling_feature_cols if use_rolling else feature_cols
    x_train = train_df[selected_features].copy()
    y_train = train_df[RISK_COL].copy()
    x_test = test_df[selected_features].copy()
    y_test = test_df[RISK_COL].copy()

    if exclude_suspects and "expert_review_flag" in train_df.columns:
        keep = ~train_df["expert_review_flag"]
        x_train = x_train.loc[keep]
        y_train = y_train.loc[keep]

    if use_clip:
        x_train, x_test = fit_clip(x_train, x_test)
    if use_anomaly_flags:
        x_train, x_test = add_anomaly_flags(x_train, x_test)

    model = make_model()
    model.fit(x_train, y_train)
    train_proba = model.predict_proba(x_train)[:, 1]
    test_proba = model.predict_proba(x_test)[:, 1]
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    threshold_table = pd.DataFrame()
    if use_segment_threshold:
        test_pred, threshold_table = segment_threshold_predict(x_train, y_train, train_proba, x_test, test_proba)

    rows = [
        evaluate(y_train, train_pred, "Train", experiment),
        evaluate(y_test, test_pred, "Test", experiment),
    ]
    return rows, threshold_table.assign(experiment=experiment) if not threshold_table.empty else threshold_table


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, _, _, base_feature_cols = build_dataset(threshold=4.0, aggregate=True)
    data = add_rolling_features(data, base_feature_cols)
    rolling_feature_cols = [
        c
        for c in data.columns
        if c.endswith("__rolling3_median_lag1")
        or c.endswith("__rolling6_median_lag1")
        or c.endswith("__delta_vs_rolling3_median")
    ]

    label_candidates_path = OUTPUT_DIR / "label_error_candidates.csv"
    suspect_dates: set[pd.Timestamp] = set()
    if label_candidates_path.exists():
        candidates = pd.read_csv(label_candidates_path, parse_dates=[DATE_COL])
        suspect_dates = set(
            candidates.loc[
                candidates["possible_label_delay"] | candidates["possible_sensor_anomaly"],
                DATE_COL,
            ]
        )
    data["expert_review_flag"] = data[DATE_COL].isin(suspect_dates)

    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data[RISK_COL],
        random_state=RANDOM_STATE,
    )

    experiment_settings = [
        {"experiment": "baseline_lightgbm", "use_rolling": False, "use_clip": False, "use_anomaly_flags": False, "exclude_suspects": False, "use_segment_threshold": False},
        {"experiment": "robust_clipping", "use_rolling": False, "use_clip": True, "use_anomaly_flags": False, "exclude_suspects": False, "use_segment_threshold": False},
        {"experiment": "anomaly_flags", "use_rolling": False, "use_clip": False, "use_anomaly_flags": True, "exclude_suspects": False, "use_segment_threshold": False},
        {"experiment": "rolling_features", "use_rolling": True, "use_clip": False, "use_anomaly_flags": False, "exclude_suspects": False, "use_segment_threshold": False},
        {"experiment": "clip_anomaly_flags", "use_rolling": False, "use_clip": True, "use_anomaly_flags": True, "exclude_suspects": False, "use_segment_threshold": False},
        {"experiment": "rolling_clip_anomaly_flags", "use_rolling": True, "use_clip": True, "use_anomaly_flags": True, "exclude_suspects": False, "use_segment_threshold": False},
        {"experiment": "exclude_expert_review_candidates", "use_rolling": True, "use_clip": False, "use_anomaly_flags": True, "exclude_suspects": True, "use_segment_threshold": False},
        {"experiment": "segment_threshold_policy", "use_rolling": True, "use_clip": False, "use_anomaly_flags": True, "exclude_suspects": False, "use_segment_threshold": True},
    ]

    all_rows = []
    threshold_tables = []
    for setting in experiment_settings:
        rows, threshold_table = run_experiment(
            train_df=train_df,
            test_df=test_df,
            feature_cols=base_feature_cols,
            rolling_feature_cols=rolling_feature_cols,
            **setting,
        )
        all_rows.extend(rows)
        if not threshold_table.empty:
            threshold_tables.append(threshold_table)

    results = pd.DataFrame(all_rows)
    pivot = results.pivot(index="experiment", columns="split", values="f1_macro").reset_index()
    pivot["test_minus_train_f1_macro"] = pivot["Test"] - pivot["Train"]
    best_experiment = str(pivot.sort_values("Test", ascending=False).iloc[0]["experiment"])

    results.to_csv(OUTPUT_DIR / "refinement_experiment_metrics.csv", index=False, encoding="utf-8-sig")
    pivot.to_csv(OUTPUT_DIR / "refinement_experiment_summary.csv", index=False, encoding="utf-8-sig")
    if threshold_tables:
        pd.concat(threshold_tables, ignore_index=True).to_csv(
            OUTPUT_DIR / "segment_thresholds.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "best_refinement_experiment": best_experiment,
        "expert_review_flag_count": int(data["expert_review_flag"].sum()),
        "base_feature_count": int(len(base_feature_cols)),
        "rolling_feature_count": int(len(rolling_feature_cols)),
        "feature_count_after_rolling": int(len(base_feature_cols) + len(rolling_feature_cols)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    (OUTPUT_DIR / "refinement_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("고도화 실험 완료!")
    print(f"Best refinement experiment: {best_experiment}")


if __name__ == "__main__":
    main()
