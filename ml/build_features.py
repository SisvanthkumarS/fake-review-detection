"""Build the feature dataset used by both training scripts.

Input:  data/labeled/      (from ml/generate_labels.py)
Output: data/features/final

Computes reviewer-level aggregates, product-level aggregates, text
features, and cross-features directly from the full labeled dataset.
The streaming pipeline uses separate lookup tables built from training
data only; this script is for batch model training and may use the
full dataset for aggregation without train/test leakage concerns
(the model training scripts do their own hold-out split).
"""
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELED_DIR  = PROJECT_ROOT / "data" / "labeled"
FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "final"


def main():
    if not LABELED_DIR.exists():
        raise SystemExit("No data/labeled/. Run ml/generate_labels.py first.")

    spark = (
        SparkSession.builder
        .appName("BuildFeatures")
        .master("local[*]")
        .config("spark.driver.memory", "12g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(LABELED_DIR))
    n = df.count()
    print(f"Labeled rows: {n:,}\n")

    # ── Reviewer-level aggregates ─────────────────────────────────────────
    daily_counts = (
        df.groupBy("customer_id", "review_date")
          .agg(F.count("*").alias("_daily"))
    )
    rev_max = (
        daily_counts.groupBy("customer_id")
                    .agg(F.max("_daily").alias("rev_max_per_day"))
    )
    rev_agg = (
        df.groupBy("customer_id")
          .agg(
              F.count("*").alias("rev_total_reviews"),
              F.avg("star_rating").alias("rev_avg_rating"),
              F.stddev("star_rating").alias("rev_rating_stddev"),
              F.avg(F.col("verified_purchase").cast("int")).alias("rev_pct_verified"),
              F.datediff(F.max("review_date"), F.min("review_date")).alias("rev_days_active"),
          )
          .join(rev_max, on="customer_id", how="left")
    )
    print(f"Reviewer aggregates: {rev_agg.count():,} unique reviewers")

    # ── Product-level aggregates ──────────────────────────────────────────
    prod_agg = (
        df.groupBy("product_id")
          .agg(
              F.count("*").alias("prod_total_reviews"),
              F.avg("star_rating").alias("prod_avg_rating"),
              F.avg((F.col("star_rating") == 5).cast("int")).alias("prod_pct_5star"),
              F.avg(F.col("verified_purchase").cast("int")).alias("prod_pct_verified"),
              F.datediff(F.max("review_date"), F.min("review_date")).alias("prod_days_active"),
          )
    )
    print(f"Product aggregates:  {prod_agg.count():,} unique products\n")

    # Join aggregates back onto labeled rows
    df = (
        df.join(F.broadcast(rev_agg),  on="customer_id", how="left")
          .join(F.broadcast(prod_agg), on="product_id",  how="left")
    )

    # ── Text features ─────────────────────────────────────────────────────
    body  = F.coalesce(F.col("review_body"), F.lit(""))
    words = F.split(body, r"\s+")

    df = (
        df
        .withColumn("txt_body_length",
                    F.length(body))
        .withColumn("txt_word_count",
                    F.size(words))
        .withColumn("txt_exclamation_count",
                    F.length(F.regexp_replace(body, r"[^!]", "")))
        .withColumn("txt_caps_word_count",
                    F.size(F.filter(words, lambda w: w.rlike(r"^[A-Z]{2,}$"))))
        .withColumn("txt_caps_ratio",
                    F.length(F.regexp_replace(body, r"[^A-Z]", ""))
                    / F.greatest(F.lit(1.0), F.length(body).cast("double")))
        # Sentiment proxy: maps 1★→-1.0, 3★→0.0, 5★→+1.0
        .withColumn("txt_sentiment",
                    (F.col("star_rating").cast("double") - 3.0) / 2.0)
    )

    # ── Cross-features ────────────────────────────────────────────────────
    df = (
        df
        .withColumn("xf_rating_vs_reviewer",
                    F.col("star_rating").cast("double")
                    - F.coalesce(F.col("rev_avg_rating"), F.lit(3.0)))
        .withColumn("xf_rating_vs_product",
                    F.col("star_rating").cast("double")
                    - F.coalesce(F.col("prod_avg_rating"), F.lit(3.0)))
        # Extreme individual review on an otherwise neutral product
        .withColumn("xf_extreme_review_neutral_product",
                    (
                        ((F.col("star_rating") == 5) | (F.col("star_rating") == 1))
                        & F.col("prod_avg_rating").between(3.5, 4.5)
                    ).cast("int"))
    )

    # ── Final select ──────────────────────────────────────────────────────
    feature_cols = [
        "rev_total_reviews", "rev_avg_rating", "rev_rating_stddev",
        "rev_pct_verified", "rev_days_active", "rev_max_per_day",
        "prod_total_reviews", "prod_avg_rating", "prod_pct_5star",
        "prod_pct_verified", "prod_days_active",
        "txt_body_length", "txt_word_count", "txt_exclamation_count",
        "txt_caps_word_count", "txt_caps_ratio", "txt_sentiment",
        "xf_rating_vs_reviewer", "xf_rating_vs_product",
        "xf_extreme_review_neutral_product",
    ]

    out = (
        df.select("review_id", "product_category", "is_likely_fraud", *feature_cols)
          .fillna(0, subset=feature_cols)
    )

    print(f"Feature columns: {len(feature_cols)}")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out.write.mode("overwrite").parquet(str(FEATURES_DIR))

    written = out.count()
    print(f"Wrote {written:,} rows to {FEATURES_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()
