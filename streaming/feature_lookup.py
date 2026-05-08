"""Pre-compute reviewer-history and product-history lookup tables.

The streaming consumer (Day 5) cannot recompute reviewer-level or product-level
features per incoming event. Instead, we precompute these aggregates from the
TRAINING split only, save as small Parquet tables, and broadcast them at
streaming startup.

IMPORTANT: built from train data only. Using full data would leak test-set
information into the model at inference time.
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

TRAIN_DIR = Path("data/train")
LOOKUP_DIR = Path("data/lookups")


def build_reviewer_history(df):
    """Per-reviewer aggregates safe for use as features.

    Excluded (would leak labels): rolling 24h burst counts, reviewer avg-rating
    extremes-flag. Kept: total reviews, avg rating (continuous, not flagged),
    rating stddev, % verified, days active.
    """
    return (df.groupBy("customer_id").agg(
        F.count("*").alias("rh_total_reviews"),
        F.avg("star_rating").alias("rh_avg_rating"),
        F.stddev("star_rating").alias("rh_rating_stddev"),
        F.avg(F.col("verified_purchase").cast("int")).alias("rh_pct_verified"),
        F.datediff(F.max("review_date"), F.min("review_date")).alias("rh_days_active"),
    ))


def build_product_history(df):
    """Per-product aggregates safe for use as features.

    Excluded: nothing label-derived (labels are reviewer-level, not product-level).
    Kept: total reviews, avg rating, % 5-star, % verified, review velocity proxy.
    """
    return (df.groupBy("product_id").agg(
        F.count("*").alias("ph_total_reviews"),
        F.avg("star_rating").alias("ph_avg_rating"),
        F.avg((F.col("star_rating") == 5).cast("int")).alias("ph_pct_5star"),
        F.avg(F.col("verified_purchase").cast("int")).alias("ph_pct_verified"),
        F.datediff(F.max("review_date"), F.min("review_date")).alias("ph_days_active"),
    ))


def main():
    if not TRAIN_DIR.exists():
        raise SystemExit("No data/train/ dir. Run streaming/build_train_test_split.py first.")

    spark = (SparkSession.builder
             .appName("BuildLookups")
             .master("local[*]")
             .config("spark.driver.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    train = spark.read.parquet(str(TRAIN_DIR))
    print(f"Train rows: {train.count():,}\n")

    LOOKUP_DIR.mkdir(parents=True, exist_ok=True)

    print("Building reviewer history...")
    rh = build_reviewer_history(train)
    rh_count = rh.count()
    print(f"  {rh_count:,} unique reviewers")
    rh.write.mode("overwrite").parquet(str(LOOKUP_DIR / "reviewer_history.parquet"))

    print("Building product history...")
    ph = build_product_history(train)
    ph_count = ph.count()
    print(f"  {ph_count:,} unique products")
    ph.write.mode("overwrite").parquet(str(LOOKUP_DIR / "product_history.parquet"))

    print(f"\nLookup tables written to {LOOKUP_DIR}/")
    spark.stop()


if __name__ == "__main__":
    main()
