# Model Results

## Dataset Summary

- Input dataset: `data/features/final`
- Total rows: 5720777
- Feature count: 19
- Label column: `is_likely_fraud`
- Model: Spark MLlib Random Forest
- Model output path: `models/rf_v1`

## Evaluation Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9870 |
| Weighted Precision | 0.9871 |
| Weighted Recall | 0.9870 |
| Weighted F1 | 0.9870 |
| ROC AUC | 0.9917 |
| Area Under PR | 0.9676 |

## Confusion Matrix

| Actual Label | Predicted Label | Count |
|---:|---:|---:|
| 0.0 | 0.0 | 1554633 |
| 0.0 | 1.0 | 11690 |
| 1.0 | 0.0 | 10591 |
| 1.0 | 1.0 | 138807 |

## Feature Importance

| Feature | Importance |
|---|---:|
| rev_max_per_day | 0.488322 |
| rev_total_reviews | 0.306323 |
| rev_rating_stddev | 0.080964 |
| xf_rating_vs_reviewer | 0.051213 |
| rev_days_active | 0.028554 |
| txt_word_count | 0.011281 |
| xf_rating_vs_product | 0.007910 |
| txt_body_length | 0.007833 |
| prod_pct_5star | 0.005164 |
| rev_pct_verified | 0.004802 |
| prod_avg_rating | 0.003557 |
| txt_sentiment | 0.001650 |
| prod_pct_verified | 0.000907 |
| xf_extreme_review_neutral_product | 0.000429 |
| prod_total_reviews | 0.000373 |
| prod_days_active | 0.000285 |
| txt_caps_ratio | 0.000223 |
| txt_caps_word_count | 0.000121 |
| txt_exclamation_count | 0.000088 |

## Strict Methodology Note

This model predicts weak-supervision labels generated from behavioral and review-pattern heuristics. 
It should be described as suspicious-review detection, not confirmed real-world fake-review detection.

If model performance is extremely high, the likely cause is label leakage from heuristic-derived features.
