# Methodology

This section describes the pipeline architecture, data scope, labeling
approach, feature engineering, and model training methodology used to build
the fake review detection system.

## Pipeline architecture

The system follows a two-tier design that mirrors how production fraud
detection systems are deployed in industry: an offline batch tier for model
training and an online serving tier for real-time scoring.

The **batch tier** ingests raw Amazon review data, filters and cleans it into
a curated dataset, generates weak-supervision fraud labels, engineers
behavioral and textual features, and trains a Random Forest classifier using
Apache Spark MLlib. The output of this tier is a serialized model artifact.

The **serving tier** uses Apache Kafka to simulate a live event stream of
incoming reviews. A Spark Structured Streaming consumer reads from Kafka,
joins each event against precomputed reviewer-history and product-history
lookup tables (broadcast at startup), applies the saved model, and emits
fraud predictions. A windowed aggregation flags products receiving bursts
of likely-fraudulent reviews in real time.

The saved model artifact is the contract between tiers — trained once
offline, loaded into the streaming consumer at startup, never modified
during serving. This separation reflects production patterns where model
training and inference run on different infrastructure with different
latency requirements.

## Data scope

We use the Amazon Customer Reviews PDS dataset hosted on the public
ClickHouse mirror at
`datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/`. The
2015 yearly Parquet shard contains 41.9M reviews spanning December 2014
through August 2015 — an 8-month window despite the file's name suggesting
a full calendar year. We document this scope honestly throughout the report.

From the 41.9M raw rows, we filter to three product categories chosen for
their distinct review domains:

- **Wireless** (3.0M reviews) — proxy for consumer electronics, a documented
  major target in opinion-spam literature
- **Books** (2.8M reviews) — long-form text reviews, distinct fraud patterns
  involving discussion of content rather than purchase experience
- **Apparel** (2.4M reviews) — high verified-purchase ratio, sizing-related
  review patterns

The filtered working dataset contains 8.2M reviews, partitioned by category
for efficient downstream operations. Filtering is performed via Spark's
predicate pushdown directly on the source Parquet, demonstrating distributed
data lake patterns over a multi-category corpus.

## Labeling approach

Ground-truth fraud labels for Amazon reviews are not publicly available. We
apply a weak-supervision approach using three independent heuristics
combined via OR logic. Full details of each heuristic, the threshold tuning
process, and the resulting positive class rate (8.7%) are documented in
`labeling_methodology.md`.

The choice of OR-combination over AND-combination prioritizes recall over
precision in the label generator: a review is flagged if any one signal
fires. This produces a positive class rate within the 5–15% target band
recommended for binary classification with weak supervision.

## Feature engineering

We engineer 19 features across three categories from the labeled training
data. Feature engineering operates on the 70% training split (5.7M rows)
to prevent test-set information from contaminating reviewer-level and
product-level aggregates.

**Behavioral features** (10 features) — reviewer-level and product-level
aggregates computed via Spark Window functions:

- Reviewer-level: total reviews, rating standard deviation, percent
  verified, days active, max reviews per single day
- Product-level: total reviews, average rating, percent 5-star, percent
  verified, days active

**Textual features** (6 features) — computed from review_body text:

- Body length (characters and words)
- Exclamation count
- All-caps word count and ratio
- Sentiment polarity (TextBlob, range [-1, 1])

**Cross features** (3 features) — capture how unusual a review is relative
to its context:

- Rating deviation from product's mean rating
- Rating deviation from reviewer's historical mean
- Boolean: extreme rating (1 or 5) on a neutrally-rated product (avg 2.5–4.0)

### Label leakage exclusions

A critical methodological concern with weak supervision is that the model
will trivially relearn the labeling heuristics if features overlap with the
heuristic inputs. We applied an explicit exclusion audit:

| Excluded feature | Reason |
|---|---|
| `helpful_vote_ratio` | Direct H1 input |
| `total_votes` | H1 threshold variable |
| `rolling_24h_count` | Direct H2 output |
| `rev_avg_rating` | Direct H3 input |
| H1/H2/H3 boolean flags | Direct heuristic outputs |

We retained `rev_total_reviews` (general activity proxy, weakly correlated
with H3 threshold) and `prod_avg_rating` (product-level signal, not used in
any heuristic). The retained features do not give the model the ingredients
to reconstruct any single heuristic by threshold.

## Train/test split

The 8.2M labeled rows are split 70/30 into train and test sets, **by
reviewer rather than by row**. Splitting randomly by row would place the
same reviewer in both splits, so reviewer-level aggregates computed in
training would leak into test predictions through shared customer_id. The
by-reviewer split keeps each customer fully in one or the other.

Random seed 42 is used for reproducibility. The positive class rate is
8.70% in train (497,844 of 5,720,777 rows) and 8.71% in test (213,808 of
2,455,960 rows), confirming the random split preserved class balance.

## Model training

[Placeholder — completed Day 4]

A Random Forest classifier is trained on the 19 engineered features using
Spark MLlib's distributed implementation. The training pipeline (placeholder
for Day 4):

- VectorAssembler combines the 19 feature columns into a single vector
- StandardScaler normalizes feature scales
- RandomForestClassifier with 100 trees, max depth 10
- Evaluation on the held-out 30% test split using precision, recall, F1,
  AUC, and confusion matrix per category

## Limitations

The labels are weak-supervision proxies, not ground truth. Model performance
metrics measure how well the classifier learns the heuristic patterns
beyond the explicit features the heuristics use, not absolute fraud
detection accuracy. We document this constraint in the Limitations section
of the final business report.

The 8-month dataset window represents one slice of one year; behavioral
patterns evolve over time and the model's signal would shift if applied to
other time periods. For this project the timeframe was constrained by what
the public mirror provided.

The single-node Spark deployment was used for development efficiency. The
architecture scales horizontally to a multi-node cluster with no code
changes — see the Cluster Validation section once that work is completed
on Day 7.