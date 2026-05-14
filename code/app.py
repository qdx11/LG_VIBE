from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "mining_process"


st.set_page_config(
    page_title="Mining Process Model Lab",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    path = OUTPUT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def require_outputs() -> bool:
    required = [
        "dataset_overview.json",
        "experiment_summary.csv",
        "statistical_tests.csv",
        "feature_importance.csv",
        "best_oof_predictions.csv",
    ]
    missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
    if missing:
        st.warning("아직 실험 산출물이 없습니다. 먼저 아래 명령을 실행하세요.")
        st.code(r".\lg_vibe\Scripts\python.exe code\mining_experiment_suite.py --mode quick", language="powershell")
        st.caption("누락 파일: " + ", ".join(missing))
        return False
    return True


def fmt_pvalue(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def metric_value(df: pd.DataFrame, experiment: str, column: str) -> str:
    if df.empty or column not in df.columns:
        return "N/A"
    row = df[df["experiment"] == experiment]
    if row.empty:
        return "N/A"
    return f"{float(row[column].iloc[0]):.4f}"


def tab_basic_lightgbm() -> None:
    summary = read_json("basic_lightgbm_summary.json")
    metrics = read_csv("basic_lightgbm_metrics.csv")
    feature_importance = read_csv("basic_lightgbm_feature_importance.csv")
    predictions = read_csv("basic_lightgbm_predictions.csv")

    if not summary or metrics.empty:
        st.warning("기본 LightGBM 학습 결과가 아직 없습니다. 먼저 아래 명령을 실행하세요.")
        st.code(r".\lg_vibe\Scripts\python.exe code\train_lightgbm_basic.py", language="powershell")
        return

    st.subheader("기본 LightGBM 학습 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", summary.get("model_type", ""))
    c2.metric("Train Rows", f"{summary.get('train_rows', 0):,}")
    c3.metric("Test Rows", f"{summary.get('test_rows', 0):,}")
    c4.metric("Training Time", f"{summary.get('elapsed_time_seconds', 0):.2f}s")

    st.caption(f"Target: `{summary.get('target', '')}`")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.markdown(
        """
        **해석**
        - Train/Test 성능을 같이 확인함.
        - Train 성능이 Test보다 높으면 과적합 가능성 봄.
        - 제출용 기준에서는 Test 성능을 더 중요하게 봄.
        """
    )

    if "train_test_gap" in metrics.columns:
        gap = float(metrics["train_test_gap"].iloc[0])
        if gap > 0.10:
            st.warning(f"Train/Test F1 macro gap이 {gap:.3f}로 과적합 가능성이 있습니다.")
        else:
            st.success(f"Train/Test F1 macro gap이 {gap:.3f}로 큰 과적합 신호는 약합니다.")

    st.subheader("Confusion Matrix")
    col1, col2 = st.columns(2)
    train_cm_path = OUTPUT_DIR / "basic_lightgbm_train_cm.png"
    test_cm_path = OUTPUT_DIR / "basic_lightgbm_test_cm.png"
    if train_cm_path.exists():
        col1.image(str(train_cm_path), caption="Train Confusion Matrix")
    if test_cm_path.exists():
        col2.image(str(test_cm_path), caption="Test Confusion Matrix")
    st.markdown(
        """
        **해석**
        - Confusion Matrix로 정상/위험 클래스를 얼마나 맞췄는지 확인함.
        - False Negative는 실제 위험인데 놓친 경우라 운영상 더 주의해서 봄.
        - False Positive는 정상인데 위험으로 예측한 경우라 알람 비용 관점에서 확인함.
        """
    )

    st.subheader("Feature Importance Top 10")
    if not feature_importance.empty:
        top10 = feature_importance.head(10).copy()
        fig = px.bar(top10.sort_values("importance"), x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"""
            **해석**
            - 상위 변수 `{top10.iloc[0]["feature"]}`가 기본 LightGBM에서 가장 크게 작동함.
            - 원료 품질, 공기 유량, 컬럼 레벨, 시간 흐름 관련 변수가 품질 위험 예측에 중요함.
            - 중요도가 낮은 변수는 제거 후보로 보기보다, 다른 변수와의 상호작용 가능성까지 함께 봄.
            """
        )
        st.dataframe(top10, use_container_width=True, hide_index=True)

    if not predictions.empty:
        st.subheader("예측 결과 샘플")
        split = st.radio("Split", ["Test", "Train"], horizontal=True)
        st.dataframe(
            predictions[predictions["split"] == split].head(50),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            """
            **해석**
            - 예측 확률 `proba_1`이 1에 가까울수록 고실리카 위험으로 강하게 판단함.
            - 실제 라벨과 예측이 다른 행은 이후 오분류 분석 탭에서 별도로 확인함.
            """
        )

    st.subheader("저장 파일")
    st.code(f"model: {summary.get('model_path')}\nfeature names: {summary.get('feature_names_path')}", language="text")


def tab_experiments() -> None:
    overview = read_json("dataset_overview.json")
    summary = read_csv("experiment_summary.csv")
    tests = read_csv("statistical_tests.csv")
    fold_metrics = read_csv("fold_metrics.csv")
    refinement_summary = read_json("refinement_summary.json")
    refinement_metrics = read_csv("refinement_experiment_metrics.csv")
    refinement_table = read_csv("refinement_experiment_summary.csv")
    segment_thresholds = read_csv("segment_thresholds.csv")

    st.subheader("실험 개요")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview.get('rows', 0):,}")
    c2.metric("Features", f"{overview.get('feature_count', 0):,}")
    c3.metric("Positive Rate", f"{overview.get('positive_rate', 0):.1%}")
    c4.metric("Risk Threshold", f"{overview.get('classification_threshold', 0):.2f}")

    st.markdown(
        """
        이 대시보드는 `% Silica Concentrate`가 기준치 이상인 고실리카 위험 여부를 예측하는
        분류 실험 결과를 정리합니다. timestamp 중복 누수를 줄이기 위해 기본 실험은
        timestamp 단위 집계 데이터에서 수행됩니다.
        """
    )

    st.subheader("실험 결과 테이블")
    metric = st.selectbox(
        "정렬 기준",
        ["pr_auc_mean", "f1_mean", "roc_auc_mean", "recall_mean", "precision_mean", "balanced_accuracy_mean"],
        index=0,
    )
    show_cols = [
        "experiment",
        "model",
        "preprocess",
        "sampling",
        "class_weight",
        "pr_auc_mean",
        "pr_auc_ci95_low",
        "pr_auc_ci95_high",
        "f1_mean",
        "recall_mean",
        "precision_mean",
        "roc_auc_mean",
    ]
    st.dataframe(
        summary.sort_values(metric, ascending=False)[[c for c in show_cols if c in summary.columns]],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        """
        **해석**
        - 모델, 전처리, 샘플링 조건을 한 표에서 비교함.
        - PR-AUC는 불균형 데이터에서 위험 클래스를 얼마나 잘 구분하는지 보기 위해 우선 확인함.
        - 평균 성능뿐 아니라 CI 범위도 함께 봐서 실험 안정성 확인함.
        """
    )

    st.subheader("모델별 성능 비교")
    plot_df = summary.sort_values(metric, ascending=False).head(15).copy()
    fig = px.bar(
        plot_df,
        x=metric,
        y="experiment",
        color="model",
        orientation="h",
        title=f"Top experiments by {metric}",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        **해석**
        - 막대가 길수록 선택한 기준 지표가 좋은 실험임.
        - 상위권 모델이 특정 계열에 몰리면 해당 모델 구조가 데이터 패턴에 잘 맞는 것으로 봄.
        - 성능 차이가 작을 때는 단순 순위보다 fold 안정성과 오분류 분석까지 같이 확인함.
        """
    )

    st.subheader("실험군별 결과 해석")
    st.markdown(
        f"""
        - **Dummy baseline**: `dummy_prior`는 기준점으로 사용함. PR-AUC `{metric_value(summary, "dummy_prior", "pr_auc_mean")}` 확인함. 이후 모델들이 이 값보다 충분히 개선되는지 봄.
        - **선형 모델**: `logistic_balanced`는 단순하고 해석 가능한 기준 모델로 확인함. 복잡한 비선형 공정 패턴을 모두 잡기에는 한계 있음.
        - **Tree ensemble**: `random_forest_balanced`, `extra_trees_balanced` 비교함. 특히 `extra_trees_balanced`가 PR-AUC `{metric_value(summary, "extra_trees_balanced", "pr_auc_mean")}`로 가장 높게 나와, 비선형 변수 조합을 잘 잡는 것으로 봄.
        - **LightGBM 기본형**: `lightgbm_base` PR-AUC `{metric_value(summary, "lightgbm_base", "pr_auc_mean")}` 확인함. 기본 성능은 안정적이나 recall/precision 균형은 추가 조정 필요함.
        - **불균형 대응**: `scale_pos_weight`, random over/under, SMOTE 비교함. `lightgbm_scale_pos_weight`와 `lightgbm_random_over`는 기본 LightGBM 대비 PR-AUC가 소폭 개선되어 클래스 불균형 대응 효과 확인함.
        - **이상치 처리**: p01/p99 clipping, IQR clipping 비교함. 일부 clipping은 성능이 낮아져, 이 데이터에서는 극단값 일부가 공정 상태를 설명하는 정보일 가능성 있음.
        - **Missing indicator**: 결측이 거의 없는 데이터라 `missing_indicator` 효과는 제한적으로 봄.
        """
    )

    st.subheader("통계검증")
    if tests.empty:
        st.info("통계검정 결과가 없습니다.")
    else:
        tests_view = tests.copy()
        for col in ["wilcoxon_p", "paired_ttest_p", "wilcoxon_p_holm", "paired_ttest_p_holm"]:
            if col in tests_view:
                tests_view[col] = tests_view[col].map(fmt_pvalue)
        st.dataframe(tests_view, use_container_width=True, hide_index=True)
        st.caption(
            "5-fold의 fold별 PR-AUC 차이에 대해 paired test를 수행하고, 여러 실험을 동시에 비교하므로 Holm 보정을 적용했습니다."
        )
        st.markdown(
            """
            **해석**
            - p-value는 모델 개선이 우연인지 확인하기 위한 보조 지표로 봄.
            - 여러 실험을 동시에 비교했기 때문에 Holm 보정값을 함께 확인함.
            - fold 수가 5개라 검정력은 제한적이므로, 효과 크기와 성능 안정성도 같이 봄.
            """
        )

    st.subheader("시도한 방법론과 해석")
    best = summary.sort_values("pr_auc_mean", ascending=False).iloc[0]
    st.markdown(
        f"""
        - 현재 최고 실험은 `{best['experiment']}`이며, 평균 PR-AUC는 `{best['pr_auc_mean']:.4f}`입니다.
        - `sampling`, `class_weight`, `outlier clipping`, `missing indicator`를 분리해 비교하므로 어떤 축이 성능에 기여했는지 추적할 수 있습니다.
        - fold 표준편차와 신뢰구간이 큰 실험은 평균이 좋아도 안정성이 낮을 수 있습니다.
        """
    )

    if not fold_metrics.empty:
        st.subheader("Fold별 안정성")
        folds = fold_metrics[fold_metrics["fold"] != "oof"].copy()
        candidates = summary.sort_values("pr_auc_mean", ascending=False)["experiment"].head(6).tolist()
        fig = px.box(
            folds[folds["experiment"].isin(candidates)],
            x="experiment",
            y="pr_auc",
            points="all",
            color="experiment",
            title="Fold-level PR-AUC dispersion",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
            **해석**
            - fold별 점이 고르게 높으면 안정적인 모델로 봄.
            - 특정 fold에서만 성능이 높거나 낮으면 데이터 구간 특성에 민감한 모델로 봄.
            - 제출용 최종 선택은 평균 성능과 fold 변동성을 같이 고려함.
            """
        )

    st.subheader("오분류 인사이트 기반 고도화 실험")
    if refinement_table.empty:
        st.info("고도화 실험 결과가 아직 없습니다. 아래 명령을 실행하세요.")
        st.code(r".\lg_vibe\Scripts\python.exe code\refine_from_error_insights.py", language="powershell")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Refinement", refinement_summary.get("best_refinement_experiment", ""))
        c2.metric("Expert Review Flags", f"{refinement_summary.get('expert_review_flag_count', 0):,}")
        c3.metric("Rolling Features", f"{refinement_summary.get('rolling_feature_count', 0):,}")
        st.dataframe(
            refinement_table.sort_values("Test", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        fig = px.bar(
            refinement_metrics[refinement_metrics["split"] == "Test"].sort_values("f1_macro"),
            x="f1_macro",
            y="experiment",
            color="recall_macro",
            orientation="h",
            title="Refinement experiments by Test F1 macro",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
            **해석**
            - 오분류 분석에서 얻은 힌트를 실제 feature와 threshold 정책으로 반영함.
            - Test F1 macro가 높을수록 정상/위험 클래스 균형이 좋아진 것으로 봄.
            - `segment_threshold_policy`가 가장 높게 나오면, 전체 공통 threshold보다 구간별 threshold가 더 적합하다고 봄.
            """
        )

        st.markdown(
            f"""
            고도화 실험 해석:
            - `rolling_features`는 직전 timestamp 흐름을 반영함. Test F1 macro `{metric_value(refinement_table.rename(columns={"Test": "test_f1"}), "rolling_features", "test_f1")}` 확인함.
            - `anomaly_flags`는 센서 이상값 방어 목적이었으나 단독 적용 시 개선이 제한적임.
            - `exclude_expert_review_candidates`는 라벨/센서 의심 샘플 제외 재학습 실험임. Test F1 macro `{metric_value(refinement_table.rename(columns={"Test": "test_f1"}), "exclude_expert_review_candidates", "test_f1")}` 확인함.
            - `segment_threshold_policy`는 오류가 집중되는 feature 구간별 threshold를 다르게 적용함. Test F1 macro `{metric_value(refinement_table.rename(columns={"Test": "test_f1"}), "segment_threshold_policy", "test_f1")}`로 가장 좋음.
            - 결론적으로 단순 모델 변경보다, 오분류 분석에서 나온 힌트를 feature와 threshold 정책에 반영한 실험이 더 효과적임.
            """
        )
        if not segment_thresholds.empty:
            st.markdown("**세그먼트별 threshold 정책**")
            st.dataframe(segment_thresholds, use_container_width=True, hide_index=True)
            st.markdown(
                """
                **해석**
                - `Flotation Column 07 Air Flow__std` 구간별로 다른 threshold를 적용함.
                - 공정 변동성이 낮은 구간과 높은 구간의 위험 판단 기준이 다를 수 있음을 확인함.
                - 운영 적용 시에는 구간별 threshold가 과도하게 복잡하지 않은지도 함께 봐야 함.
                """
            )

    st.subheader("향후 추가 실험 제안")
    st.markdown(
        """
        - timestamp 이전 구간의 lag/rolling 평균/표준편차를 추가해 공정 지연 효과를 반영합니다.
        - 현재 집계 실험과 별도로 row-level StratifiedGroupKFold를 비교해 집계로 잃은 정보가 있는지 확인합니다.
        - Optuna로 LightGBM/XGBoost의 `num_leaves`, `max_depth`, `min_child_samples`, `scale_pos_weight`를 정밀 탐색합니다.
        - threshold tuning 결과를 비용 함수 기반으로 재정의합니다. 예를 들어 false negative 비용이 큰 경우 recall 우선 threshold를 선택합니다.
        - 확률 해석이 필요하면 calibration curve, Brier score, isotonic calibration을 추가합니다.
        """
    )


def tab_features() -> None:
    importance = read_csv("feature_importance.csv")
    feature_stats = read_csv("feature_class_stats.csv")
    corr = read_csv("top_feature_correlations.csv")
    numeric_profile = read_csv("numeric_profile.csv")

    st.subheader("Feature Importance")
    if importance.empty:
        st.info("feature importance 산출물이 없습니다.")
        return
    top_n = st.slider("상위 변수 개수", 5, 30, 15)
    top_imp = importance.head(top_n).copy()
    fig = px.bar(top_imp, x="importance", y="feature", orientation="h", title="Top feature importance")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
        **해석**
        - 상위 `{top_n}`개 변수의 상대적 중요도 확인함.
        - 중요도가 높은 변수는 모델이 품질 위험을 판단할 때 자주 사용한 변수로 봄.
        - 중요도가 높다고 원인 변수라고 단정하지 않고, 공정 해석과 함께 봄.
        """
    )

    st.subheader("중요 변수 vs 비중요 변수의 통계적 차이")
    if feature_stats.empty:
        st.info("변수별 통계검정 결과가 없습니다.")
    else:
        st.dataframe(feature_stats, use_container_width=True, hide_index=True)
        sig = feature_stats[feature_stats.get("mannwhitney_p_holm", 1) < 0.05]
        st.markdown(
            f"""
            Holm 보정 후 유의한 변수는 `{len(sig)}`개입니다. `cliffs_delta`의 절댓값이 클수록
            고실리카/정상 집단의 분포 차이가 실질적으로 큽니다.
            """
        )

    st.subheader("상관관계 힌트")
    if not corr.empty:
        corr = corr.set_index(corr.columns[0])
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Top feature correlations")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
            **해석**
            - 붉은색은 양의 상관, 푸른색은 음의 상관으로 봄.
            - 공기 유량/컬럼 레벨 계열 변수끼리 상관이 높으면 중복 정보 가능성 확인함.
            - 상관이 높은 변수는 interaction feature나 차이/비율 feature 후보로 봄.
            """
        )

    st.subheader("심화 EDA 체크포인트")
    if not numeric_profile.empty:
        candidate_cols = ["column", "mean", "std", "min", "1%", "50%", "99%", "max"]
        st.dataframe(numeric_profile[[c for c in candidate_cols if c in numeric_profile.columns]], use_container_width=True, hide_index=True)
        st.markdown(
            """
            **해석**
            - p1, p99, min, max를 비교해 이상치 후보 확인함.
            - 평균과 중앙값 차이가 큰 변수는 분포가 치우친 변수로 봄.
            - 극단값이 많은 변수는 clipping보다 anomaly flag로 남기는 방안도 고려함.
            """
        )
    st.markdown(
        """
        - 공기 유량/컬럼 레벨 변수는 동일 계열 간 상관이 높을 수 있어, 평균/편차/불균형 지표를 추가 feature로 만들 여지가 있습니다.
        - `rows_per_timestamp`와 timestamp 내부 표준편차 feature가 중요하면, 같은 시간대의 공정 변동성이 품질 위험과 연결된다는 힌트입니다.
        - `% Iron Feed`, `% Silica Feed`가 강하게 작동한다면 원료 품질 변화에 대한 보정 feature가 필요합니다.
        """
    )


def tab_misclassification() -> None:
    preds = read_csv("best_oof_predictions.csv")
    near_wrong = read_csv("near_threshold_misclassifications.csv")
    large_wrong = read_csv("large_margin_misclassifications.csv")
    compare = read_csv("misclassification_near_vs_large_stats.csv")
    label_candidates = read_csv("label_error_candidates.csv")
    time_context = read_csv("large_margin_error_time_context.csv")
    segment_concentration = read_csv("error_segment_concentration.csv")
    deep_tests = read_csv("deep_near_vs_large_error_tests.csv")

    if preds.empty:
        st.info("오분류 분석 산출물이 없습니다.")
        return

    st.subheader("최고 모델 OOF 예측 분석")
    default_threshold = float(preds["best_f1_threshold"].iloc[0]) if "best_f1_threshold" in preds else 0.5
    threshold = st.slider("임계치", 0.0, 1.0, default_threshold, 0.01)
    preds = preds.copy()
    preds["pred_at_slider"] = (preds["oof_proba"] >= threshold).astype(int)
    preds["correct_at_slider"] = preds["pred_at_slider"] == preds["high_silica_risk"]
    preds["margin_at_slider"] = (preds["oof_proba"] - threshold).abs()

    tp = int(((preds["high_silica_risk"] == 1) & (preds["pred_at_slider"] == 1)).sum())
    tn = int(((preds["high_silica_risk"] == 0) & (preds["pred_at_slider"] == 0)).sum())
    fp = int(((preds["high_silica_risk"] == 0) & (preds["pred_at_slider"] == 1)).sum())
    fn = int(((preds["high_silica_risk"] == 1) & (preds["pred_at_slider"] == 0)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP", f"{tp:,}")
    c2.metric("TN", f"{tn:,}")
    c3.metric("FP", f"{fp:,}")
    c4.metric("FN", f"{fn:,}")

    cm_df = pd.DataFrame({"Pred 0": [tn, fn], "Pred 1": [fp, tp]}, index=["Actual 0", "Actual 1"])
    st.dataframe(cm_df, use_container_width=True)

    fig = px.histogram(
        preds,
        x="oof_proba",
        color="high_silica_risk",
        nbins=50,
        barmode="overlay",
        title="OOF probability distribution by actual class",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        **해석**
        - 빨간 점선은 현재 임계치임.
        - 실제 위험 클래스의 확률 분포가 오른쪽으로 몰릴수록 모델 구분력이 좋다고 봄.
        - 두 분포가 겹치는 구간은 임계치 근접 오분류가 발생하기 쉬운 구간으로 봄.
        """
    )

    st.subheader("오분류 샘플 직접 검토")
    wrong = preds[~preds["correct_at_slider"]].copy()
    sample_mode = st.radio("샘플 유형", ["임계치 근접 오분류", "큰 마진 오분류", "무작위 오분류"], horizontal=True)
    if sample_mode == "임계치 근접 오분류":
        sample = wrong.sort_values("margin_at_slider", ascending=True).head(30)
    elif sample_mode == "큰 마진 오분류":
        sample = wrong.sort_values("margin_at_slider", ascending=False).head(30)
    else:
        sample = wrong.sample(min(30, len(wrong)), random_state=42) if len(wrong) else wrong
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.markdown(
        """
        **해석**
        - 임계치 근접 오분류는 모델이 애매하게 판단한 케이스로 봄.
        - 큰 마진 오분류는 모델이 강하게 확신했지만 틀린 케이스라 라벨/센서 이상 후보로 봄.
        - 무작위 오분류는 전체 오류 패턴을 빠르게 훑어보기 위해 사용함.
        """
    )

    st.subheader("임계치 근접 오류 vs 큰 마진 오류")
    c1, c2 = st.columns(2)
    c1.markdown("**임계치 근접 오분류**")
    c1.dataframe(near_wrong.head(20), use_container_width=True, hide_index=True)
    c2.markdown("**큰 마진 오분류**")
    c2.dataframe(large_wrong.head(20), use_container_width=True, hide_index=True)

    if not compare.empty:
        st.markdown("**두 오류 유형을 가르는 변수 후보**")
        st.dataframe(compare.head(30), use_container_width=True, hide_index=True)
        st.markdown(
            """
            **해석**
            - p-value가 낮고 `cliffs_delta` 절댓값이 큰 변수는 오류 유형을 가르는 후보로 봄.
            - 큰 마진 오류에서 특정 센서 변동성이 높으면 anomaly flag 또는 rolling feature 후보로 봄.
            """
        )

    st.subheader("라벨 오류 가능성 점검")
    if label_candidates.empty:
        st.info("심화 라벨 오류 점검 결과가 아직 없습니다. 아래 명령을 실행하면 이 섹션이 채워집니다.")
        st.code(r".\lg_vibe\Scripts\python.exe code\deep_misclassification_analysis.py", language="powershell")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("검토 후보", f"{len(label_candidates):,}")
        c2.metric("라벨 지연 의심", f"{int(label_candidates['possible_label_delay'].sum()):,}")
        c3.metric("센서 이상 의심", f"{int(label_candidates['possible_sensor_anomaly'].sum()):,}")
        st.dataframe(
            label_candidates.sort_values(["possible_label_delay", "possible_sensor_anomaly", "margin"], ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "라벨 지연 의심은 주변 +/-3 timestamp의 실제 위험 비율이 현재 라벨과 충돌하는 경우, 센서 이상 의심은 정상 예측 샘플 대비 3-sigma 이상 벗어난 feature가 있는 경우입니다."
        )

    st.subheader("큰 마진 오류의 시간 인접 구간")
    if not time_context.empty:
        center_dates = time_context["center_date"].drop_duplicates().head(30).tolist()
        selected_date = st.selectbox("검토할 큰 마진 오류 시점", center_dates)
        st.dataframe(
            time_context[time_context["center_date"] == selected_date],
            use_container_width=True,
            hide_index=True,
        )
        fig = px.line(
            time_context[time_context["center_date"] == selected_date],
            x="relative_step",
            y=["% Silica Concentrate", "oof_proba"],
            markers=True,
            title="Center error 주변의 품질값과 예측확률",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
            **해석**
            - relative step 0이 큰 마진 오분류 시점임.
            - 주변 시점의 품질값과 예측확률 흐름이 현재 라벨과 다르면 라벨 지연 가능성 봄.
            - 주변 흐름이 급격히 바뀌면 공정 전환 구간 또는 센서 이상 구간 가능성 확인함.
            """
        )

    st.subheader("오류 집중 세그먼트")
    if not segment_concentration.empty:
        st.dataframe(segment_concentration.head(40), use_container_width=True, hide_index=True)
        fig = px.bar(
            segment_concentration.head(20),
            x="large_margin_error_rate",
            y="feature",
            color="error_rate",
            orientation="h",
            hover_data=["bin", "rows", "fp_count", "fn_count", "positive_rate"],
            title="Large-margin error concentration by feature bin",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            """
            **해석**
            - large-margin error rate가 높은 feature bin은 모델이 강하게 틀리는 조건으로 봄.
            - 특정 feature 구간에 FP/FN이 몰리면 구간별 threshold 또는 interaction feature 후보로 봄.
            - 이 결과를 고도화 실험의 segment threshold policy로 연결함.
            """
        )

    st.subheader("근접 오류 vs 큰 마진 오류 차이")
    if not deep_tests.empty:
        st.dataframe(deep_tests.head(30), use_container_width=True, hide_index=True)
        st.markdown(
            """
            **해석**
            - 근접 오류와 큰 마진 오류의 변수 분포 차이를 통계적으로 확인함.
            - 차이가 큰 변수는 오분류 원인을 설명하는 후보로 봄.
            - 이후 feature engineering 또는 라벨 검토 우선순위로 사용함.
            """
        )

    st.subheader("고도화 아이디어")
    st.markdown(
        """
        - `possible_label_delay=True` 후보는 주변 시간대 품질 라벨과 충돌하므로 라벨 측정 지연 또는 timestamp 집계 정책을 재검토합니다.
        - `possible_sensor_anomaly=True` 후보는 센서 이상값 방어를 위해 robust clipping, anomaly flag, rolling median feature를 추가합니다.
        - 오류 집중 세그먼트 상위 feature bin은 interaction feature 또는 세그먼트별 threshold 후보입니다.
        - 라벨 오류 의심 샘플을 expert review flag로 관리한 뒤 제외/보정/가중치 완화 재학습 실험을 비교합니다.
        """
    )


def tab_project_report() -> None:
    overview = read_json("dataset_overview.json")
    basic_summary = read_json("basic_lightgbm_summary.json")
    basic_metrics = read_csv("basic_lightgbm_metrics.csv")
    experiment_summary = read_csv("experiment_summary.csv")
    refinement_summary = read_json("refinement_summary.json")
    refinement_table = read_csv("refinement_experiment_summary.csv")
    feature_importance = read_csv("feature_importance.csv")
    label_candidates = read_csv("label_error_candidates.csv")

    basic_test_f1 = None
    if not basic_metrics.empty and "Test" in set(basic_metrics["split"]):
        basic_test_f1 = float(basic_metrics.loc[basic_metrics["split"] == "Test", "f1_macro"].iloc[0])

    cv_best_name = ""
    cv_best_pr_auc = None
    if not experiment_summary.empty:
        cv_best = experiment_summary.sort_values("pr_auc_mean", ascending=False).iloc[0]
        cv_best_name = str(cv_best["experiment"])
        cv_best_pr_auc = float(cv_best["pr_auc_mean"])

    refinement_best_name = refinement_summary.get("best_refinement_experiment", "")
    refinement_best_f1 = None
    if not refinement_table.empty and refinement_best_name:
        match = refinement_table[refinement_table["experiment"] == refinement_best_name]
        if not match.empty:
            refinement_best_f1 = float(match["Test"].iloc[0])

    top_features = []
    if not feature_importance.empty:
        top_features = feature_importance.head(5)["feature"].tolist()

    st.subheader("프로젝트 진행 보고서")
    feature_text = ", ".join(f"`{feature}`" for feature in top_features)
    st.markdown(
        f"""
        ### 1. 프로젝트 목표
        - 광물 선별 공정 데이터로 최종 품질 위험 예측 모델 구성함.
        - `% Silica Concentrate >= 4.0`을 고실리카 위험으로 정의함.
        - 전체 `{overview.get("rows", 0):,}`개 timestamp 중 위험 클래스 비율 약 `{overview.get("positive_rate", 0):.1%}` 확인함.
        - 불균형 분류 문제로 보고 PR-AUC, F1, Precision, Recall 함께 확인함.
        """
    )

    st.markdown(
        """
        ### 2. 전처리 및 데이터 구성
        - 콤마 소수점 문자열을 실수형 숫자로 변환함.
        - 동일 `date` 반복 구조 확인함.
        - timestamp 단위 집계 적용함.
        - timestamp 내부 표준편차 feature 추가함.
        - 제출용 train/test 데이터 저장함.
        - `data/processed/mining_process/train.csv`
        - `data/processed/mining_process/test.csv`
        - 모델 입력 변수 목록 `models/feature_names.pkl`로 저장함.
        """
    )

    st.markdown(
        f"""
        ### 3. 시도한 모델링
        - 기본 모델로 LightGBM 학습함.
        - 기본 LightGBM Test F1 macro `{basic_test_f1:.4f}` 확인함.
        - Logistic Regression, RandomForest, ExtraTrees, LightGBM, XGBoost 비교함.
        - class weight, over/under sampling, SMOTE, 이상치 clipping, missing indicator 실험함.
        - 5-fold stratified CV 기준 최고 PR-AUC 실험 `{cv_best_name}` 확인함.
        - 평균 PR-AUC `{cv_best_pr_auc:.4f}` 확인함.
        """
    )

    st.markdown(
        f"""
        ### 4. 분석하면서 발견한 점
        - 주요 변수 {feature_text} 등 확인함.
        - 원료 품질, 약품 투입, 컬럼 공기 유량/레벨, 시간 흐름 중요함.
        - 큰 마진 오분류 일부에서 주변 timestamp 라벨 흐름과 충돌 확인함.
        - 일부 오분류에서 센서 값이 정상 범위에서 크게 벗어난 상태 확인함.
        - 라벨 측정 지연, 센서 이상값, timestamp 집계 방식 영향 의심함.
        """
    )

    st.markdown(
        f"""
        ### 5. 최적화하면서 고려한 내용
        - 라벨 오류 의심 샘플 `expert_review_flag`로 표시함.
        - 의심 샘플 제외 후 재학습 비교함.
        - 센서 이상값 방어용 3-sigma anomaly flag 추가함.
        - 공정 흐름 반영용 rolling median, delta feature 추가함.
        - p01/p99 robust clipping 비교함.
        - 오류 집중 feature 구간별 threshold 차등 적용함.
        - 최종 고도화 실험 `{refinement_best_name}` 확인함.
        - Test F1 macro `{refinement_best_f1:.4f}` 확인함.
        """
    )

    st.markdown(
        f"""
        ### 6. 최종 산출물 및 한계
        - 전처리 데이터: `train.csv`, `test.csv`
        - 학습 모델: `models/best_model.pkl`
        - 대시보드: 기본 학습, 실험 비교, 변수 분석, 오분류 분석, 프로젝트 보고서 통합함.
        - 라벨 오류 가능성은 데이터 기반 추정으로 정리함.
        - 실제 공정 전문가 검토 필요함.
        - 향후 긴 rolling window, 시간 기준 검증, 확률 보정, 비용 기반 threshold 추가 필요함.
        """
    )

    if not label_candidates.empty:
        st.caption(
            f"참고: 라벨/센서 검토 후보 {len(label_candidates):,}건 확인함. 향후 expert review 대상으로 분리 관리 필요함."
        )


st.title("Mining Process Quality Prediction Lab")
st.caption("실험 결과, 입력 변수 분석, 오분류 심화 분석을 한 화면에서 검토합니다.")

if require_outputs():
    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["기본 LightGBM", "실험 결과", "입력 변수 분석", "오분류 데이터 심화 분석", "프로젝트 보고서"]
    )
    with tab0:
        tab_basic_lightgbm()
    with tab1:
        tab_experiments()
    with tab2:
        tab_features()
    with tab3:
        tab_misclassification()
    with tab4:
        tab_project_report()
