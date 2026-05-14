from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler
except Exception:  # pragma: no cover - optional dependency guard
    RandomOverSampler = None
    RandomUnderSampler = None
    SMOTE = None
    ImbPipeline = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "mining_process" / "MiningProcess_Flotation_Plant_Database.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "mining_process"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_PATH = PROJECT_ROOT / "docs" / "stages" / "mining_experiment_report.md"

TARGET_REG = "% Silica Concentrate"
SECONDARY_TARGET = "% Iron Concentrate"
DATE_COL = "date"
RANDOM_STATE = 42


@dataclass(frozen=True)
class PreprocessConfig:
    name: str
    impute: str = "median"
    missing_indicator: bool = False
    clip: str = "none"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_name: str
    preprocess: PreprocessConfig
    sampling: str = "none"
    class_weight: str = "none"


class OutlierClipper:
    def __init__(self, method: str = "none") -> None:
        self.method = method
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None

    def fit(self, x: pd.DataFrame) -> "OutlierClipper":
        if self.method == "none":
            self.lower_ = pd.Series(-np.inf, index=x.columns)
            self.upper_ = pd.Series(np.inf, index=x.columns)
        elif self.method == "p01_p99":
            self.lower_ = x.quantile(0.01)
            self.upper_ = x.quantile(0.99)
        elif self.method == "p005_p995":
            self.lower_ = x.quantile(0.005)
            self.upper_ = x.quantile(0.995)
        elif self.method == "iqr_1_5":
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            self.lower_ = q1 - 1.5 * iqr
            self.upper_ = q3 + 1.5 * iqr
        else:
            raise ValueError(f"Unknown clipping method: {self.method}")
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("OutlierClipper must be fitted before transform.")
        return x.clip(lower=self.lower_, upper=self.upper_, axis=1)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_mining_data(nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, nrows=nrows, dtype=str)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    for col in df.columns:
        if col == DATE_COL:
            continue
        df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="coerce")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out[DATE_COL]
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["day"] = dt.dt.day
    out["month"] = dt.dt.month
    out["elapsed_hours"] = (dt - dt.min()).dt.total_seconds() / 3600.0
    return out


def aggregate_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c != DATE_COL]
    input_cols = [c for c in numeric_cols if c not in {TARGET_REG, SECONDARY_TARGET}]
    mean_df = df.groupby(DATE_COL, as_index=False)[numeric_cols].mean()
    std_df = df.groupby(DATE_COL)[input_cols].std(ddof=0).add_suffix("__std").reset_index()
    count_df = df.groupby(DATE_COL).size().rename("rows_per_timestamp").reset_index()
    out = mean_df.merge(std_df, on=DATE_COL, how="left").merge(count_df, on=DATE_COL, how="left")
    return out


def build_dataset(
    threshold: float,
    aggregate: bool,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    raw = read_mining_data(nrows=nrows)
    data = aggregate_by_timestamp(raw) if aggregate else raw
    data = add_time_features(data)
    data["high_silica_risk"] = (data[TARGET_REG] >= threshold).astype(int)
    exclude = {DATE_COL, TARGET_REG, SECONDARY_TARGET, "high_silica_risk"}
    feature_cols = [c for c in data.columns if c not in exclude]
    x = data[feature_cols]
    y = data["high_silica_risk"]
    return data, x, y, feature_cols


def profile_dataset(data: pd.DataFrame, feature_cols: list[str], threshold: float) -> None:
    numeric_cols = [c for c in data.columns if c != DATE_COL]
    overview = {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "date_min": str(data[DATE_COL].min()),
        "date_max": str(data[DATE_COL].max()),
        "target": TARGET_REG,
        "classification_threshold": threshold,
        "positive_rate": float(data["high_silica_risk"].mean()),
        "positive_count": int(data["high_silica_risk"].sum()),
        "negative_count": int((1 - data["high_silica_risk"]).sum()),
        "feature_count": len(feature_cols),
    }
    missing = data.isna().sum().rename("missing_count").reset_index().rename(columns={"index": "column"})
    missing["missing_rate"] = missing["missing_count"] / len(data)
    desc = data[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc = desc.reset_index().rename(columns={"index": "column"})
    with (OUTPUT_DIR / "dataset_overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    missing.to_csv(OUTPUT_DIR / "missing_profile.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUTPUT_DIR / "numeric_profile.csv", index=False, encoding="utf-8-sig")


def fit_preprocess(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    config: PreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = x_train.copy()
    valid = x_valid.copy()
    if config.impute == "median":
        imputer = SimpleImputer(strategy="median", add_indicator=config.missing_indicator)
        train_arr = imputer.fit_transform(train)
        valid_arr = imputer.transform(valid)
        names = list(train.columns)
        if config.missing_indicator and getattr(imputer, "indicator_", None) is not None:
            indicator_features = [train.columns[i] + "__missing" for i in imputer.indicator_.features_]
            names += indicator_features
        train = pd.DataFrame(train_arr, columns=names, index=x_train.index)
        valid = pd.DataFrame(valid_arr, columns=names, index=x_valid.index)
    elif config.impute != "none":
        raise ValueError(f"Unknown impute option: {config.impute}")

    clipper = OutlierClipper(config.clip).fit(train)
    return clipper.transform(train), clipper.transform(valid)


def make_model(model_name: str, class_weight: str, y_train: pd.Series) -> Any:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    balanced_weight = "balanced" if class_weight == "balanced" else None

    if model_name == "dummy":
        return DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
    if model_name == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight=balanced_weight,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight=balanced_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight=balanced_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed.")
        params = {
            "n_estimators": 700,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_samples": 25,
            "reg_alpha": 0.05,
            "reg_lambda": 0.3,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        if class_weight == "scale_pos_weight":
            params["scale_pos_weight"] = scale_pos_weight
        elif class_weight == "balanced":
            params["class_weight"] = "balanced"
        return LGBMClassifier(**params)
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed.")
        params = {
            "n_estimators": 500,
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "reg_alpha": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        if class_weight == "scale_pos_weight":
            params["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def wrap_sampler(model: Any, sampling: str) -> Any:
    if sampling == "none":
        return model
    if ImbPipeline is None:
        raise RuntimeError("imbalanced-learn is required for sampling experiments.")
    if sampling == "random_under":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif sampling == "random_over":
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif sampling == "smote":
        sampler = SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown sampling option: {sampling}")
    return ImbPipeline([("sampler", sampler), ("model", model)])


def predict_proba_positive(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return np.repeat(float(model.classes_[0]), len(x))
        return proba[:, 1]
    decision = model.decision_function(x)
    return 1 / (1 + np.exp(-decision))


def metric_dict(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_proba),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
        try:
            out["log_loss"] = log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6))
        except ValueError:
            out["log_loss"] = np.nan
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
        out["log_loss"] = np.nan
    return {k: float(v) for k, v in out.items()}


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, y_proba >= t, zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def experiment_grid(mode: str) -> list[ExperimentConfig]:
    raw = PreprocessConfig("raw_no_clip", impute="median", missing_indicator=False, clip="none")
    indicator = PreprocessConfig("median_indicator", impute="median", missing_indicator=True, clip="none")
    p99 = PreprocessConfig("median_clip_p01_p99", impute="median", missing_indicator=False, clip="p01_p99")
    iqr = PreprocessConfig("median_clip_iqr", impute="median", missing_indicator=False, clip="iqr_1_5")

    configs = [
        ExperimentConfig("dummy_prior", "dummy", raw),
        ExperimentConfig("logistic_balanced", "logistic", raw, class_weight="balanced"),
        ExperimentConfig("random_forest_balanced", "random_forest", raw, class_weight="balanced"),
        ExperimentConfig("extra_trees_balanced", "extra_trees", raw, class_weight="balanced"),
        ExperimentConfig("lightgbm_base", "lightgbm", raw),
        ExperimentConfig("lightgbm_missing_indicator", "lightgbm", indicator),
        ExperimentConfig("lightgbm_clip_p01_p99", "lightgbm", p99),
        ExperimentConfig("lightgbm_clip_iqr", "lightgbm", iqr),
        ExperimentConfig("lightgbm_scale_pos_weight", "lightgbm", raw, class_weight="scale_pos_weight"),
        ExperimentConfig("lightgbm_random_under", "lightgbm", raw, sampling="random_under"),
        ExperimentConfig("lightgbm_random_over", "lightgbm", raw, sampling="random_over"),
        ExperimentConfig("lightgbm_smote", "lightgbm", raw, sampling="smote"),
        ExperimentConfig("xgboost_base", "xgboost", raw),
        ExperimentConfig("xgboost_scale_pos_weight", "xgboost", raw, class_weight="scale_pos_weight"),
    ]
    if mode == "quick":
        return configs
    if mode == "extended":
        extra = [
            ExperimentConfig("xgboost_clip_p01_p99", "xgboost", p99),
            ExperimentConfig("xgboost_random_under", "xgboost", raw, sampling="random_under"),
            ExperimentConfig("xgboost_random_over", "xgboost", raw, sampling="random_over"),
            ExperimentConfig("xgboost_smote", "xgboost", raw, sampling="smote"),
        ]
        return configs + extra
    raise ValueError("mode must be either quick or extended.")


def run_cv(
    data: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    configs: list[ExperimentConfig],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []

    for config in configs:
        print(f"[experiment] {config.name}")
        oof_proba = np.zeros(len(y), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
            x_train_raw = x.iloc[train_idx]
            x_valid_raw = x.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]
            x_train, x_valid = fit_preprocess(x_train_raw, x_valid_raw, config.preprocess)
            model = make_model(config.model_name, config.class_weight, y_train)
            model = wrap_sampler(model, config.sampling)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)
            proba = predict_proba_positive(model, x_valid)
            oof_proba[valid_idx] = proba
            row = {
                "experiment": config.name,
                "fold": fold,
                "model": config.model_name,
                "preprocess": config.preprocess.name,
                "sampling": config.sampling,
                "class_weight": config.class_weight,
                "valid_positive_rate": float(y_valid.mean()),
            }
            row.update(metric_dict(y_valid.to_numpy(), proba))
            fold_rows.append(row)

        threshold, best_f1 = tune_threshold(y.to_numpy(), oof_proba)
        oof_metric = metric_dict(y.to_numpy(), oof_proba, threshold=threshold)
        oof_summary = {
            "experiment": config.name,
            "fold": "oof",
            "model": config.model_name,
            "preprocess": config.preprocess.name,
            "sampling": config.sampling,
            "class_weight": config.class_weight,
            "valid_positive_rate": float(y.mean()),
            "best_f1_threshold": threshold,
            "best_oof_f1": best_f1,
        }
        oof_summary.update({f"oof_{k}": v for k, v in oof_metric.items()})
        fold_rows.append(oof_summary)

        oof_frame = data[[DATE_COL, TARGET_REG, "high_silica_risk"]].copy()
        oof_frame["experiment"] = config.name
        oof_frame["oof_proba"] = oof_proba
        oof_frame["oof_pred_05"] = (oof_proba >= 0.5).astype(int)
        oof_frame["best_f1_threshold"] = threshold
        oof_frame["oof_pred_best_f1"] = (oof_proba >= threshold).astype(int)
        oof_frames.append(oof_frame)

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True)
    return fold_df, oof_df


def summarize_experiments(fold_df: pd.DataFrame) -> pd.DataFrame:
    folds = fold_df[fold_df["fold"] != "oof"].copy()
    metric_cols = ["roc_auc", "pr_auc", "f1", "precision", "recall", "balanced_accuracy", "brier", "log_loss"]
    summaries = []
    for exp, group in folds.groupby("experiment"):
        row: dict[str, Any] = {
            "experiment": exp,
            "model": group["model"].iloc[0],
            "preprocess": group["preprocess"].iloc[0],
            "sampling": group["sampling"].iloc[0],
            "class_weight": group["class_weight"].iloc[0],
        }
        for metric in metric_cols:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_ci95_low"] = values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values))
            row[f"{metric}_ci95_high"] = values.mean() + 1.96 * values.std(ddof=1) / math.sqrt(len(values))
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    return summary.sort_values(["pr_auc_mean", "f1_mean"], ascending=False)


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    indexed = sorted(enumerate(pvalues), key=lambda x: np.inf if pd.isna(x[1]) else x[1])
    adjusted = [np.nan] * len(pvalues)
    m = len(pvalues)
    running = 0.0
    for rank, (idx, pval) in enumerate(indexed, start=1):
        if pd.isna(pval):
            adjusted[idx] = np.nan
            continue
        running = max(running, min((m - rank + 1) * pval, 1.0))
        adjusted[idx] = running
    return adjusted


def statistical_tests(fold_df: pd.DataFrame, baseline: str, metric: str = "pr_auc") -> pd.DataFrame:
    folds = fold_df[fold_df["fold"] != "oof"].copy()
    base = folds[folds["experiment"] == baseline].sort_values("fold")
    rows = []
    for exp, group in folds.groupby("experiment"):
        if exp == baseline:
            continue
        group = group.sort_values("fold")
        merged = base[["fold", metric]].merge(group[["fold", metric]], on="fold", suffixes=("_baseline", "_candidate"))
        diff = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
        if len(diff) >= 3 and np.any(np.abs(diff) > 1e-12):
            try:
                wilcoxon_p = stats.wilcoxon(diff).pvalue
            except ValueError:
                wilcoxon_p = np.nan
            ttest_p = stats.ttest_rel(merged[f"{metric}_candidate"], merged[f"{metric}_baseline"]).pvalue
        else:
            wilcoxon_p = np.nan
            ttest_p = np.nan
        rows.append(
            {
                "baseline": baseline,
                "candidate": exp,
                "metric": metric,
                "mean_diff": float(diff.mean()),
                "median_diff": float(diff.median()),
                "wilcoxon_p": float(wilcoxon_p) if not pd.isna(wilcoxon_p) else np.nan,
                "paired_ttest_p": float(ttest_p) if not pd.isna(ttest_p) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_p_holm"] = holm_bonferroni(out["wilcoxon_p"].tolist())
        out["paired_ttest_p_holm"] = holm_bonferroni(out["paired_ttest_p"].tolist())
    return out.sort_values("mean_diff", ascending=False)


def fit_best_model(
    x: pd.DataFrame,
    y: pd.Series,
    configs: list[ExperimentConfig],
    best_experiment: str,
) -> tuple[Any, pd.DataFrame, ExperimentConfig]:
    config = next(c for c in configs if c.name == best_experiment)
    x_fit, _ = fit_preprocess(x, x, config.preprocess)
    model = wrap_sampler(make_model(config.model_name, config.class_weight, y), config.sampling)
    model.fit(x_fit, y)
    feature_names = list(x_fit.columns)

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
    elif isinstance(model, Pipeline):
        inner = model.named_steps.get("model", model)
    else:
        inner = model

    if hasattr(inner, "feature_importances_"):
        importance = np.asarray(inner.feature_importances_, dtype=float)
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    elif hasattr(inner, "coef_"):
        coef = np.ravel(inner.coef_)
        imp_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)})
    else:
        perm = permutation_importance(model, x_fit, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        imp_df = pd.DataFrame({"feature": feature_names, "importance": perm.importances_mean})
    imp_df = imp_df.sort_values("importance", ascending=False)
    bundle = {"model": model, "config": config, "feature_names": feature_names}
    joblib.dump(bundle, MODEL_DIR / "best_model.joblib")
    return model, imp_df, config


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(np.sum(ai > b) for ai in a)
    lt = sum(np.sum(ai < b) for ai in a)
    return float((gt - lt) / (len(a) * len(b)))


def feature_analysis(data: pd.DataFrame, x: pd.DataFrame, y: pd.Series, importance: pd.DataFrame) -> None:
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    ranked = importance[importance["importance"] > 0].copy()
    selected = pd.concat([ranked.head(15), ranked.tail(10)]).drop_duplicates("feature")
    rows = []
    for feature in selected["feature"]:
        if feature not in x.columns:
            continue
        neg = x.loc[y == 0, feature].dropna().to_numpy()
        pos = x.loc[y == 1, feature].dropna().to_numpy()
        if len(pos) < 3 or len(neg) < 3:
            continue
        u_p = stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue
        rows.append(
            {
                "feature": feature,
                "importance_rank": int(importance.reset_index(drop=True).query("feature == @feature").index[0] + 1),
                "importance": float(importance.loc[importance["feature"] == feature, "importance"].iloc[0]),
                "positive_median": float(np.median(pos)),
                "negative_median": float(np.median(neg)),
                "median_diff_pos_minus_neg": float(np.median(pos) - np.median(neg)),
                "positive_mean": float(np.mean(pos)),
                "negative_mean": float(np.mean(neg)),
                "mannwhitney_p": float(u_p),
                "cliffs_delta": cliffs_delta(pos, neg),
            }
        )
    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["mannwhitney_p_holm"] = holm_bonferroni(stats_df["mannwhitney_p"].tolist())
    stats_df.to_csv(OUTPUT_DIR / "feature_class_stats.csv", index=False, encoding="utf-8-sig")

    corr_cols = list(importance.head(20)["feature"])
    corr_cols = [c for c in corr_cols if c in data.columns]
    corr = data[corr_cols + [TARGET_REG, "high_silica_risk"]].corr(numeric_only=True)
    corr.to_csv(OUTPUT_DIR / "top_feature_correlations.csv", encoding="utf-8-sig")


def misclassification_analysis(
    data: pd.DataFrame,
    oof_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_experiment: str,
) -> None:
    best_oof = oof_df[oof_df["experiment"] == best_experiment].copy()
    threshold = float(best_oof["best_f1_threshold"].iloc[0])
    best_oof["pred"] = (best_oof["oof_proba"] >= threshold).astype(int)
    best_oof["is_correct"] = best_oof["pred"] == best_oof["high_silica_risk"]
    best_oof["margin"] = (best_oof["oof_proba"] - threshold).abs()
    best_oof["error_type"] = np.select(
        [
            (best_oof["high_silica_risk"] == 1) & (best_oof["pred"] == 1),
            (best_oof["high_silica_risk"] == 0) & (best_oof["pred"] == 0),
            (best_oof["high_silica_risk"] == 0) & (best_oof["pred"] == 1),
            (best_oof["high_silica_risk"] == 1) & (best_oof["pred"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="unknown",
    )
    detail = best_oof.merge(data.drop(columns=["high_silica_risk"]), on=[DATE_COL, TARGET_REG], how="left")
    detail.to_csv(OUTPUT_DIR / "best_oof_predictions.csv", index=False, encoding="utf-8-sig")

    wrong = detail[~detail["is_correct"]].copy()
    near_wrong = wrong.sort_values("margin", ascending=True).head(50)
    large_wrong = wrong.sort_values("margin", ascending=False).head(50)
    near_wrong.to_csv(OUTPUT_DIR / "near_threshold_misclassifications.csv", index=False, encoding="utf-8-sig")
    large_wrong.to_csv(OUTPUT_DIR / "large_margin_misclassifications.csv", index=False, encoding="utf-8-sig")

    compare_cols = [c for c in data.columns if c not in {DATE_COL, TARGET_REG, SECONDARY_TARGET, "high_silica_risk"}]
    rows = []
    for col in compare_cols:
        near_vals = near_wrong[col].dropna().to_numpy() if col in near_wrong else np.array([])
        large_vals = large_wrong[col].dropna().to_numpy() if col in large_wrong else np.array([])
        if len(near_vals) < 3 or len(large_vals) < 3:
            continue
        rows.append(
            {
                "feature": col,
                "near_wrong_median": float(np.median(near_vals)),
                "large_wrong_median": float(np.median(large_vals)),
                "median_diff_large_minus_near": float(np.median(large_vals) - np.median(near_vals)),
                "mannwhitney_p": float(stats.mannwhitneyu(large_vals, near_vals, alternative="two-sided").pvalue),
                "cliffs_delta_large_vs_near": cliffs_delta(large_vals, near_vals),
            }
        )
    pd.DataFrame(rows).sort_values("mannwhitney_p").to_csv(
        OUTPUT_DIR / "misclassification_near_vs_large_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(best_oof["high_silica_risk"], best_oof["pred"], labels=[0, 1])
    pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )


def write_report(
    overview: dict[str, Any],
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    best_experiment: str,
) -> None:
    def table_text(df: pd.DataFrame) -> str:
        return "```text\n" + df.to_string(index=False) + "\n```"

    top = summary.head(8)
    significant = tests[(tests["wilcoxon_p_holm"] < 0.05) | (tests["paired_ttest_p_holm"] < 0.05)] if not tests.empty else pd.DataFrame()
    lines = [
        "# Mining Process Experiment Report",
        "",
        "## Problem Definition",
        f"- Regression target observed in EDA: `{TARGET_REG}`",
        f"- Classification target for imbalance/CV/misclassification analysis: `{TARGET_REG} >= {overview['classification_threshold']}`",
        f"- Positive rate: {overview['positive_rate']:.3f} ({overview['positive_count']} / {overview['rows']})",
        "- Default modeling unit: timestamp-level aggregated observations to reduce duplicate timestamp leakage.",
        "",
        "## Best Experiment",
        f"- Best by mean PR-AUC: `{best_experiment}`",
        "",
        "## Top Experiments",
        table_text(top),
        "",
        "## Statistical Validation",
        "- Fold-level paired tests compare each candidate against `dummy_prior` on PR-AUC.",
        "- Holm correction is applied because many model/preprocess/sampling variants are compared.",
    ]
    if significant.empty:
        lines.append("- No Holm-corrected p-value below 0.05 was found against the dummy baseline, or tests were underpowered with 5 folds.")
    else:
        lines.append(table_text(significant.head(10)))
    lines += [
        "",
        "## Next Modeling Ideas",
        "- Add lag/rolling features from previous timestamps to reflect process dynamics.",
        "- Compare timestamp aggregation against grouped row-level CV using `date` as the leakage-control group.",
        "- Tune LightGBM/XGBoost with Optuna around the strongest baseline configuration.",
        "- Calibrate probabilities with isotonic or sigmoid calibration if the dashboard will support risk threshold decisions.",
        "- Ask a domain expert to review large-margin false positives/false negatives as possible label-delay or label-quality issues.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=4.0, help="High-silica risk threshold.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mode", choices=["quick", "extended"], default="quick")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--no-aggregate", action="store_true", help="Use row-level data instead of timestamp aggregation.")
    args = parser.parse_args()

    ensure_dirs()
    data, x, y, feature_cols = build_dataset(args.threshold, aggregate=not args.no_aggregate, nrows=args.nrows)
    profile_dataset(data, feature_cols, args.threshold)
    configs = experiment_grid(args.mode)
    fold_df, oof_df = run_cv(data, x, y, configs, args.folds)
    summary = summarize_experiments(fold_df)
    tests = statistical_tests(fold_df, baseline="dummy_prior", metric="pr_auc")
    best_experiment = str(summary.iloc[0]["experiment"])
    _, importance, _ = fit_best_model(x, y, configs, best_experiment)
    feature_analysis(data, x, y, importance)
    misclassification_analysis(data, oof_df, summary, best_experiment)

    fold_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False, encoding="utf-8-sig")
    oof_df.to_csv(OUTPUT_DIR / "all_oof_predictions.csv", index=False, encoding="utf-8-sig")

    overview = json.loads((OUTPUT_DIR / "dataset_overview.json").read_text(encoding="utf-8"))
    write_report(overview, summary, tests, best_experiment)
    print(f"Saved outputs to {OUTPUT_DIR}")
    print(f"Best experiment: {best_experiment}")


if __name__ == "__main__":
    main()
