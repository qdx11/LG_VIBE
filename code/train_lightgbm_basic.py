from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from mining_experiment_suite import build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "mining_process"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "mining_process"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 42


def plot_confusion_matrix(cm: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> None:
    top10 = feature_importance.head(10).sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(top10["feature"], top10["importance"])
    plt.title("LightGBM Feature Importance Top 10")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_split(y_true: pd.Series, y_pred: pd.Series, split: str) -> dict[str, float | str]:
    return {
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    data, x, y, _ = build_dataset(threshold=4.0, aggregate=True)
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    train_df = X_train.copy()
    train_df["high_silica_risk"] = y_train
    test_df = X_test.copy()
    test_df["high_silica_risk"] = y_test
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False, encoding="utf-8-sig")
    print(f"Train CSV 저장 완료: {PROCESSED_DIR / 'train.csv'}")
    print(f"Test CSV 저장 완료: {PROCESSED_DIR / 'test.csv'}")

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=7,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"모델 학습 완료! 소요 시간: {elapsed_time:.2f}초")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)
    print("Train 예측 완료!")
    print("Test 예측 완료!")
    print("Train/Test 확률 예측 완료!")

    metrics_df = pd.DataFrame(
        [
            evaluate_split(y_train, y_train_pred, "Train"),
            evaluate_split(y_test, y_test_pred, "Test"),
        ]
    )
    metrics_df["train_test_gap"] = metrics_df["f1_macro"].iloc[0] - metrics_df["f1_macro"].iloc[1]
    metrics_df["overfitting_check"] = metrics_df["train_test_gap"].apply(
        lambda gap: "주의: 과적합 가능성" if gap > 0.10 else "큰 과적합 신호 없음"
    )
    print(metrics_df.to_string(index=False))

    train_cm = pd.DataFrame(
        confusion_matrix(y_train, y_train_pred),
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )
    test_cm = pd.DataFrame(
        confusion_matrix(y_test, y_test_pred),
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    metrics_df.to_csv(OUTPUT_DIR / "basic_lightgbm_metrics.csv", index=False, encoding="utf-8-sig")
    train_cm.to_csv(OUTPUT_DIR / "basic_lightgbm_train_confusion_matrix.csv", encoding="utf-8-sig")
    test_cm.to_csv(OUTPUT_DIR / "basic_lightgbm_test_confusion_matrix.csv", encoding="utf-8-sig")
    feature_importance.to_csv(OUTPUT_DIR / "basic_lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")

    train_pred_df = pd.DataFrame(
        {
            "split": "Train",
            "y_true": y_train.to_numpy(),
            "y_pred": y_train_pred,
            "proba_0": y_train_proba[:, 0],
            "proba_1": y_train_proba[:, 1],
        },
        index=X_train.index,
    )
    test_pred_df = pd.DataFrame(
        {
            "split": "Test",
            "y_true": y_test.to_numpy(),
            "y_pred": y_test_pred,
            "proba_0": y_test_proba[:, 0],
            "proba_1": y_test_proba[:, 1],
        },
        index=X_test.index,
    )
    pd.concat([train_pred_df, test_pred_df]).to_csv(
        OUTPUT_DIR / "basic_lightgbm_predictions.csv",
        index_label="row_id",
        encoding="utf-8-sig",
    )

    plot_confusion_matrix(train_cm, "Train Confusion Matrix", OUTPUT_DIR / "basic_lightgbm_train_cm.png")
    plot_confusion_matrix(test_cm, "Test Confusion Matrix", OUTPUT_DIR / "basic_lightgbm_test_cm.png")
    plot_feature_importance(feature_importance, OUTPUT_DIR / "basic_lightgbm_feature_importance_top10.png")

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    joblib.dump(model, model_path)
    print(f"모델 저장 완료: {model_path}")
    joblib.dump(list(X_train.columns), feature_names_path)
    print(f"Feature names 저장 완료: {feature_names_path}")

    saved_files = {
        "model": model_path,
        "feature_names": feature_names_path,
    }
    for name, path in saved_files.items():
        print(f"{name} 파일 크기: {os.path.getsize(path):,} bytes")

    summary = {
        "model_type": "LGBMClassifier",
        "target": "% Silica Concentrate >= 4.0",
        "rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "elapsed_time_seconds": elapsed_time,
        "model_path": model_path,
        "feature_names_path": feature_names_path,
        "train_csv_path": str(PROCESSED_DIR / "train.csv"),
        "test_csv_path": str(PROCESSED_DIR / "test.csv"),
        "scaler_path": None,
    }
    (OUTPUT_DIR / "basic_lightgbm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("모델 저장 완료!")


if __name__ == "__main__":
    main()
