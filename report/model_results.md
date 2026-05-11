# Model Results

## Dataset Summary

- Input dataset: `data/features/final`
- Total rows: 8176737
- Feature count: 20
- Label column: `is_likely_fraud`
- Model: Spark MLlib Random Forest
- Model output path: `models/rf_v1`

## Evaluation Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9890 |
| Weighted Precision | 0.9891 |
| Weighted Recall | 0.9890 |
| Weighted F1 | 0.9890 |
| ROC AUC | 0.9926 |
| Area Under PR | 0.9730 |

## Confusion Matrix

| Actual Label | Predicted Label | Count |
|---:|---:|---:|
| 0.0 | 0.0 | 2224001 |
| 0.0 | 1.0 | 15361 |
| 1.0 | 0.0 | 11701 |
| 1.0 | 1.0 | 201224 |

## Feature Importance

| Feature | Importance |
|---|---:|
| rev_max_per_day | 0.441138 |
| rev_total_reviews | 0.330016 |
| rev_avg_rating | 0.086433 |
| rev_rating_stddev | 0.041821 |
| rev_days_active | 0.031872 |
| xf_rating_vs_reviewer | 0.029028 |
| txt_word_count | 0.011050 |
| txt_body_length | 0.009436 |
| txt_sentiment | 0.007969 |
| rev_pct_verified | 0.003577 |
| xf_rating_vs_product | 0.001784 |
| txt_caps_ratio | 0.001508 |
| prod_pct_5star | 0.001208 |
| prod_pct_verified | 0.001187 |
| prod_avg_rating | 0.000994 |
| xf_extreme_review_neutral_product | 0.000450 |
| prod_total_reviews | 0.000262 |
| prod_days_active | 0.000132 |
| txt_caps_word_count | 0.000079 |
| txt_exclamation_count | 0.000054 |

## Strict Methodology Note

This model predicts weak-supervision labels generated from behavioral and review-pattern heuristics. 
It should be described as suspicious-review detection, not confirmed real-world fake-review detection.

If model performance is extremely high, the likely cause is label leakage from heuristic-derived features.
