# Mining Process Experiment Report

## Problem Definition
- Regression target observed in EDA: `% Silica Concentrate`
- Classification target for imbalance/CV/misclassification analysis: `% Silica Concentrate >= 4.0`
- Positive rate: 0.116 (475 / 4097)
- Default modeling unit: timestamp-level aggregated observations to reduce duplicate timestamp leakage.

## Best Experiment
- Best by mean PR-AUC: `extra_trees_balanced`

## Top Experiments
```text
                experiment         model          preprocess    sampling     class_weight  roc_auc_mean  roc_auc_std  roc_auc_ci95_low  roc_auc_ci95_high  pr_auc_mean  pr_auc_std  pr_auc_ci95_low  pr_auc_ci95_high  f1_mean   f1_std  f1_ci95_low  f1_ci95_high  precision_mean  precision_std  precision_ci95_low  precision_ci95_high  recall_mean  recall_std  recall_ci95_low  recall_ci95_high  balanced_accuracy_mean  balanced_accuracy_std  balanced_accuracy_ci95_low  balanced_accuracy_ci95_high  brier_mean  brier_std  brier_ci95_low  brier_ci95_high  log_loss_mean  log_loss_std  log_loss_ci95_low  log_loss_ci95_high
      extra_trees_balanced   extra_trees         raw_no_clip        none         balanced      0.885760     0.016925          0.870924           0.900595     0.613185    0.044587         0.574102          0.652267 0.489748 0.026064     0.466901      0.512594        0.733769       0.035252            0.702869             0.764670     0.368421    0.028828         0.343153          0.393689                0.675375               0.013747                    0.663325                     0.687425    0.071429   0.002879        0.068906         0.073952       0.255522      0.007542           0.248912            0.262133
 lightgbm_scale_pos_weight      lightgbm         raw_no_clip        none scale_pos_weight      0.847746     0.022873          0.827698           0.867795     0.567667    0.055079         0.519388          0.615945 0.462775 0.040317     0.427436      0.498114        0.721238       0.067872            0.661746             0.780731     0.341053    0.031226         0.313682          0.368424                0.661829               0.017071                    0.646865                     0.676793    0.075638   0.006825        0.069655         0.081621       0.314811      0.034319           0.284729            0.344893
      lightgbm_random_over      lightgbm         raw_no_clip random_over             none      0.847674     0.022099          0.828303           0.867045     0.566722    0.049398         0.523422          0.610021 0.453682 0.035029     0.422978      0.484386        0.727791       0.076863            0.660417             0.795164     0.330526    0.028441         0.305597          0.355456                0.656980               0.014823                    0.643987                     0.669974    0.076415   0.005757        0.071368         0.081461       0.314243      0.030561           0.287455            0.341032
             lightgbm_base      lightgbm         raw_no_clip        none             none      0.846978     0.020733          0.828805           0.865151     0.564040    0.056433         0.514575          0.613506 0.394492 0.044005     0.355920      0.433063        0.829824       0.067650            0.770527             0.889122     0.258947    0.032101         0.230810          0.287085                0.626023               0.017200                    0.610946                     0.641099    0.080025   0.005565        0.075147         0.084902       0.353823      0.034231           0.323818            0.383828
lightgbm_missing_indicator      lightgbm    median_indicator        none             none      0.846978     0.020733          0.828805           0.865151     0.564040    0.056433         0.514575          0.613506 0.394492 0.044005     0.355920      0.433063        0.829824       0.067650            0.770527             0.889122     0.258947    0.032101         0.230810          0.287085                0.626023               0.017200                    0.610946                     0.641099    0.080025   0.005565        0.075147         0.084902       0.353823      0.034231           0.323818            0.383828
     lightgbm_clip_p01_p99      lightgbm median_clip_p01_p99        none             none      0.842504     0.023942          0.821518           0.863490     0.558812    0.059235         0.506890          0.610733 0.375610 0.053651     0.328583      0.422637        0.823669       0.078678            0.754705             0.892634     0.244211    0.041039         0.208238          0.280183                0.618654               0.021032                    0.600219                     0.637090    0.080710   0.005875        0.075560         0.085859       0.357407      0.036538           0.325380            0.389434
            lightgbm_smote      lightgbm         raw_no_clip       smote             none      0.845994     0.014286          0.833472           0.858517     0.551188    0.022612         0.531368          0.571009 0.453702 0.025464     0.431382      0.476022        0.649933       0.032549            0.621402             0.678463     0.349474    0.029209         0.323871          0.375077                0.662312               0.013608                    0.650384                     0.674240    0.076864   0.002064        0.075055         0.078674       0.301507      0.016370           0.287158            0.315856
    random_forest_balanced random_forest         raw_no_clip        none         balanced      0.856860     0.015755          0.843050           0.870670     0.549057    0.021609         0.530116          0.567997 0.301133 0.049774     0.257504      0.344762        0.794090       0.079885            0.724067             0.864112     0.187368    0.038962         0.153217          0.221520                0.590372               0.018580                    0.574085                     0.606658    0.077810   0.001370        0.076609         0.079011       0.275755      0.004831           0.271521            0.279990
```

## Statistical Validation
- Fold-level paired tests compare each candidate against `dummy_prior` on PR-AUC.
- Holm correction is applied because many model/preprocess/sampling variants are compared.
```text
   baseline                  candidate metric  mean_diff  median_diff  wilcoxon_p  paired_ttest_p  wilcoxon_p_holm  paired_ttest_p_holm
dummy_prior       extra_trees_balanced pr_auc   0.497246     0.509504      0.0625        0.000015           0.8125             0.000154
dummy_prior  lightgbm_scale_pos_weight pr_auc   0.451728     0.471515      0.0625        0.000052           0.8125             0.000261
dummy_prior       lightgbm_random_over pr_auc   0.450783     0.453309      0.0625        0.000034           0.8125             0.000224
dummy_prior              lightgbm_base pr_auc   0.448102     0.474363      0.0625        0.000059           0.8125             0.000261
dummy_prior lightgbm_missing_indicator pr_auc   0.448102     0.474363      0.0625        0.000059           0.8125             0.000261
dummy_prior      lightgbm_clip_p01_p99 pr_auc   0.442873     0.463336      0.0625        0.000075           0.8125             0.000261
dummy_prior             lightgbm_smote pr_auc   0.435250     0.425848      0.0625        0.000002           0.8125             0.000021
dummy_prior     random_forest_balanced pr_auc   0.433118     0.439577      0.0625        0.000001           0.8125             0.000019
dummy_prior          lightgbm_clip_iqr pr_auc   0.418339     0.441832      0.0625        0.000028           0.8125             0.000224
dummy_prior               xgboost_base pr_auc   0.394449     0.400803      0.0625        0.000018           0.8125             0.000158
```

## Next Modeling Ideas
- Add lag/rolling features from previous timestamps to reflect process dynamics.
- Compare timestamp aggregation against grouped row-level CV using `date` as the leakage-control group.
- Tune LightGBM/XGBoost with Optuna around the strongest baseline configuration.
- Calibrate probabilities with isotonic or sigmoid calibration if the dashboard will support risk threshold decisions.
- Ask a domain expert to review large-margin false positives/false negatives as possible label-delay or label-quality issues.