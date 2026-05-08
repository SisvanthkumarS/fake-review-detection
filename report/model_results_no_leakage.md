# Leakage-Controlled Model Results

## Dataset Summary

- Input dataset: `data/features/final`
- Total rows: 5720777
- Feature count: 12
- Label column: `is_likely_fraud`
- Model: Spark MLlib Random Forest
- Model output path: `models/rf_no_leakage_v1`

## Removed Leakage-Prone Features

- `rev_days_active`
- `rev_max_per_day`
- `rev_pct_verified`
- `rev_rating_stddev`
- `rev_total_reviews`
- `xf_extreme_review_neutral_product`
- `xf_rating_vs_reviewer`

## Evaluation Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9131 |
| Weighted Precision | 0.8976 |
| Weighted Recall | 0.9131 |
| Weighted F1 | 0.8720 |
| ROC AUC | 0.7098 |
| Area Under PR | 0.2058 |

## Confusion Matrix

| Actual Label | Predicted Label | Count |
|---:|---:|---:|
| 0.0 | 0.0 | 1566144 |
| 0.0 | 1.0 | 179 |
| 1.0 | 0.0 | 148902 |
| 1.0 | 1.0 | 496 |

## Feature Importance

| Feature | Importance |
|---|---:|
| txt_word_count | 0.377724 |
| txt_body_length | 0.284815 |
| prod_avg_rating | 0.073267 |
| txt_sentiment | 0.053895 |
| prod_total_reviews | 0.052221 |
| prod_pct_5star | 0.043849 |
| prod_pct_verified | 0.032261 |
| xf_rating_vs_product | 0.026542 |
| txt_caps_word_count | 0.018742 |
| txt_exclamation_count | 0.015473 |
| prod_days_active | 0.014775 |
| txt_caps_ratio | 0.006437 |

## Interpretation

This model intentionally excludes reviewer-behavior features that are likely to overlap with weak-label generation. 
The result is a more conservative estimate of whether product-level, text-level, and rating-deviation features can generalize beyond the labeling heuristic.
