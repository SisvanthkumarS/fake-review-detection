"""Generate weak-supervision fraud labels for the curated dataset.

Three heuristics combined via OR:
  H1: helpful_vote_ratio < 0.20 AND total_votes >= 5
  H2: reviewer posted >= 5 reviews in any 24-hour window
  H3: reviewer has >= 10 reviews AND avg_rating >= 4.8 OR <= 1.2

Target positive class rate: 5-15%.
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CURATED_DIR = Path("data/curated")
LABELED_DIR = Path("data/labeled")


def main():
    if not CURATED_DIR.exists():
        raise SystemExit("No curated dir. Run batch/build_curated.py first.")

    spark = (SparkSession.builder
             .appName("GenerateLabels")
             .master("local[*]")
             .config("spark.driver.memory", "12g")
             .config("spark.sql.shuffle.partitions", "64")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(CURATED_DIR))
    n_total = df.count()
    print(f"Curated rows loaded: {n_total:,}\n")

    # H1: low helpful-vote ratio (only meaningful with >= 5 votes)
    df = df.withColumn(
        "h1_low_helpful",
        F.when(
            (F.col("total_votes") >= 5) &
            ((F.col("helpful_votes") / F.col("total_votes")) < 0.20),
            True
        ).otherwise(False)
    )

    # H2: burst activity - reviewer posted >= 5 reviews in any 24h window
    burst_window = (Window
                    .partitionBy("customer_id")
                    .orderBy(F.col("review_date").cast("timestamp").cast("long"))
                    .rangeBetween(-86400, 0))
    df = df.withColumn("rolling_24h_count", F.count("*").over(burst_window))
    df = df.withColumn("h2_burst_activity", F.col("rolling_24h_count") >= 10)

    # H3: reviewer-level rating extremes (>=10 reviews + avg rating in extremes)
    reviewer_window = Window.partitionBy("customer_id")
    df = (df
          .withColumn("reviewer_total_reviews",
                      F.count("*").over(reviewer_window))
          .withColumn("reviewer_avg_rating",
                      F.avg("star_rating").over(reviewer_window)))
    df = df.withColumn(
        "h3_extreme_bias",
        (F.col("reviewer_total_reviews") >= 10) &
        ((F.col("reviewer_avg_rating") >= 4.8) | (F.col("reviewer_avg_rating") <= 1.2))
    )

    # Combine via OR
    df = df.withColumn(
        "is_likely_fraud",
        F.col("h1_low_helpful") | F.col("h2_burst_activity") | F.col("h3_extreme_bias")
    )

    # Sanity check counts
    df.cache()
    print("=== Per-heuristic positive counts ===")
    counts = df.agg(
        F.sum(F.col("h1_low_helpful").cast("int")).alias("H1"),
        F.sum(F.col("h2_burst_activity").cast("int")).alias("H2"),
        F.sum(F.col("h3_extreme_bias").cast("int")).alias("H3"),
        F.sum(F.col("is_likely_fraud").cast("int")).alias("ANY"),
    ).collect()[0]
    print(f"  H1 (low helpful):     {counts.H1:>9,}  ({100*counts.H1/n_total:.2f}%)")
    print(f"  H2 (burst activity):  {counts.H2:>9,}  ({100*counts.H2/n_total:.2f}%)")
    print(f"  H3 (extreme bias):    {counts.H3:>9,}  ({100*counts.H3/n_total:.2f}%)")
    print(f"  ANY (final label):    {counts.ANY:>9,}  ({100*counts.ANY/n_total:.2f}%)\n")

    rate = counts.ANY / n_total
    if rate < 0.05 or rate > 0.15:
        print(f"WARNING: positive class rate {100*rate:.1f}% outside 5-15% target.")
        print("Consider tuning H1/H2/H3 thresholds before training.\n")

    # Drop intermediate columns we don't want in the labeled output
    out = df.drop(
        "rolling_24h_count", "reviewer_total_reviews", "reviewer_avg_rating"
    )

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    (out.write
        .mode("overwrite")
        .partitionBy("product_category")
        .parquet(str(LABELED_DIR)))

    print(f"Wrote labeled dataset to {LABELED_DIR}/")
    spark.stop()


if __name__ == "__main__":
    main()