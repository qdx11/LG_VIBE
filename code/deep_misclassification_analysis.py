from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mining_experiment_suite import DATE_COL, TARGET_REG, build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "mining_process"
RISK_COL = "high_silica_risk"


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(np.sum(ai > b) for ai in a)
    lt = sum(np.sum(ai < b) for ai in a)
    return float((gt - lt) / (len(a) * len(b)))


def classify_error(row: pd.Series) -> str:
    if row[RISK_COL] == 1 and row["pred"] == 1:
        return "TP"
    if row[RISK_COL] == 0 and row["pred"] == 0:
        return "TN"
    if row[RISK_COL] == 0 and row["pred"] == 1:
        return "FP"
    if row[RISK_COL] == 1 and row["pred"] == 0:
        return "FN"
    return "unknown"


def add_neighbor_context(preds: pd.DataFrame, window: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = preds.sort_values(DATE_COL).reset_index(drop=True).copy()
    for lag in range(1, window + 1):
        preds[f"target_prev_{lag}"] = preds[TARGET_REG].shift(lag)
        preds[f"target_next_{lag}"] = preds[TARGET_REG].shift(-lag)
        preds[f"risk_prev_{lag}"] = preds[RISK_COL].shift(lag)
        preds[f"risk_next_{lag}"] = preds[RISK_COL].shift(-lag)
        preds[f"proba_prev_{lag}"] = preds["oof_proba"].shift(lag)
        preds[f"proba_next_{lag}"] = preds["oof_proba"].shift(-lag)

    preds["neighbor_risk_rate_pm3"] = pd.concat(
        [preds[RISK_COL].shift(i) for i in range(-window, window + 1) if i != 0],
        axis=1,
    ).mean(axis=1)
    preds["neighbor_target_mean_pm3"] = pd.concat(
        [preds[TARGET_REG].shift(i) for i in range(-window, window + 1) if i != 0],
        axis=1,
    ).mean(axis=1)
    preds["target_vs_neighbor_mean"] = preds[TARGET_REG] - preds["neighbor_target_mean_pm3"]
    preds["possible_label_delay"] = (
        ((preds["error_type"] == "FP") & (preds["neighbor_risk_rate_pm3"] >= 0.5))
        | ((preds["error_type"] == "FN") & (preds["neighbor_risk_rate_pm3"] <= 0.5))
    )

    wrong_large = preds[(~preds["is_correct"])].sort_values("margin", ascending=False).head(80)
    context_rows = []
    for _, row in wrong_large.iterrows():
        idx = int(row.name)
        start = max(0, idx - window)
        end = min(len(preds), idx + window + 1)
        block = preds.iloc[start:end][[DATE_COL, TARGET_REG, RISK_COL, "oof_proba", "pred", "error_type", "margin"]].copy()
        block["center_date"] = row[DATE_COL]
        block["relative_step"] = np.arange(start - idx, end - idx)
        block["center_error_type"] = row["error_type"]
        block["center_margin"] = row["margin"]
        context_rows.append(block)
    context = pd.concat(context_rows, ignore_index=True) if context_rows else pd.DataFrame()
    return preds, context


def anomaly_scores(preds: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    baseline = preds[preds["is_correct"]].copy()
    large_wrong = preds[(~preds["is_correct"])].sort_values("margin", ascending=False).head(80).copy()
    rows = []
    for _, row in large_wrong.iterrows():
        feature_z = {}
        max_abs_z = 0.0
        max_feature = None
        for col in feature_cols:
            std = baseline[col].std(ddof=0)
            if pd.isna(std) or std == 0:
                continue
            z = (row[col] - baseline[col].mean()) / std
            feature_z[col] = z
            if abs(z) > max_abs_z:
                max_abs_z = abs(z)
                max_feature = col
        rows.append(
            {
                DATE_COL: row[DATE_COL],
                TARGET_REG: row[TARGET_REG],
                RISK_COL: row[RISK_COL],
                "pred": row["pred"],
                "oof_proba": row["oof_proba"],
                "error_type": row["error_type"],
                "margin": row["margin"],
                "max_abs_z_feature": max_feature,
                "max_abs_z": max_abs_z,
                "feature_values_outside_3sigma": int(sum(abs(v) >= 3 for v in feature_z.values())),
                "possible_sensor_anomaly": bool(max_abs_z >= 3),
                "possible_label_delay": bool(row.get("possible_label_delay", False)),
            }
        )
    return pd.DataFrame(rows)


def feature_error_concentration(preds: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if preds[col].nunique(dropna=True) < 4:
            continue
        try:
            bins = pd.qcut(preds[col], q=4, duplicates="drop")
        except ValueError:
            continue
        grouped = preds.assign(bin=bins).groupby("bin", observed=True)
        for bin_value, group in grouped:
            wrong = group[~group["is_correct"]]
            large_wrong = wrong[wrong["margin"] >= preds.loc[~preds["is_correct"], "margin"].quantile(0.75)]
            rows.append(
                {
                    "feature": col,
                    "bin": str(bin_value),
                    "rows": int(len(group)),
                    "error_rate": float((~group["is_correct"]).mean()),
                    "large_margin_error_rate": float(len(large_wrong) / len(group)),
                    "fp_count": int((wrong["error_type"] == "FP").sum()),
                    "fn_count": int((wrong["error_type"] == "FN").sum()),
                    "positive_rate": float(group[RISK_COL].mean()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["large_margin_error_rate", "error_rate"], ascending=False)


def near_vs_large_tests(preds: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    wrong = preds[~preds["is_correct"]].copy()
    near = wrong.nsmallest(80, "margin")
    large = wrong.nlargest(80, "margin")
    rows = []
    for col in feature_cols + ["neighbor_risk_rate_pm3", "target_vs_neighbor_mean"]:
        if col not in preds:
            continue
        a = large[col].dropna().to_numpy()
        b = near[col].dropna().to_numpy()
        if len(a) < 5 or len(b) < 5:
            continue
        rows.append(
            {
                "feature": col,
                "large_margin_median": float(np.median(a)),
                "near_threshold_median": float(np.median(b)),
                "median_diff_large_minus_near": float(np.median(a) - np.median(b)),
                "mannwhitney_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
                "cliffs_delta": cliffs_delta(a, b),
            }
        )
    return pd.DataFrame(rows).sort_values("mannwhitney_p")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, _, _, feature_cols = build_dataset(threshold=4.0, aggregate=True)
    oof = pd.read_csv(OUTPUT_DIR / "best_oof_predictions.csv", parse_dates=[DATE_COL])
    threshold = float(oof["best_f1_threshold"].iloc[0])
    preds = data.merge(
        oof[[DATE_COL, TARGET_REG, "oof_proba", "best_f1_threshold"]],
        on=[DATE_COL, TARGET_REG],
        how="inner",
    )
    preds["pred"] = (preds["oof_proba"] >= threshold).astype(int)
    preds["is_correct"] = preds["pred"] == preds[RISK_COL]
    preds["margin"] = (preds["oof_proba"] - threshold).abs()
    preds["error_type"] = preds.apply(classify_error, axis=1)

    enriched, context = add_neighbor_context(preds)
    label_candidates = anomaly_scores(enriched, feature_cols)
    concentration = feature_error_concentration(enriched, feature_cols)
    tests = near_vs_large_tests(enriched, feature_cols)

    enriched.to_csv(OUTPUT_DIR / "deep_error_predictions.csv", index=False, encoding="utf-8-sig")
    context.to_csv(OUTPUT_DIR / "large_margin_error_time_context.csv", index=False, encoding="utf-8-sig")
    label_candidates.to_csv(OUTPUT_DIR / "label_error_candidates.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(OUTPUT_DIR / "error_segment_concentration.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUTPUT_DIR / "deep_near_vs_large_error_tests.csv", index=False, encoding="utf-8-sig")

    print("라벨 오류/오분류 심화 분석 완료!")
    print(f"라벨 오류 후보: {len(label_candidates)} rows")
    print(f"시간 인접 구간: {len(context)} rows")
    print(f"오류 집중 세그먼트: {len(concentration)} rows")


if __name__ == "__main__":
    main()
