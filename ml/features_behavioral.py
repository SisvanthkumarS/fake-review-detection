"""Behavioral features: reviewer-level and product-level aggregates.

EXCLUDED for label leakage (would let the model just relearn the heuristics):
  - helpful_vote_ratio (H1 signal)
  - rolling_24h_count (H2 signal)
  - h3_extreme_bias_flag (H3 boolean)

INCLUDED (continuous proxies that generalize beyond the heuristics):
  Reviewer-level:
    - rev_total_reviews         (count, not the boolean threshold)
    - rev_rating_stddev         (variance signal)
    - rev_pct_verified
    - rev_days_active
    - rev_max_per_day           (DAILY max, NOT 24h rolling like H2)
  Product-level:
    - prod_total_reviews
    - prod_avg_rating
    - prod_pct_5star
    - prod_pct_verified
    - prod_days_active
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

TRAIN_DIR = Path("data/train")
OUTPUT_DIR = Path("data/features/behavioral")


def main():
    if not TRAIN_DIR.exists():
        raise SystemExit("No data/train/ dir.")

    spark = (SparkSession.builder
             .appName("BehavioralFeatures")
             .master("local[*]")
             .config("spark.driver.memory", "10g")
             .config("spark.sql.shuffle.partitions", "64")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(TRAIN_DIR))

    # Reviewer-level aggregates via window
    rev_w = Window.partitionBy("customer_id")
    df = df.withColumn("rev_total_reviews", F.count("*").over(rev_w))
    df = df.withColumn("rev_rating_stddev",
                       F.stddev("star_rating").over(rev_w))
    df = df.withColumn("rev_pct_verified",
                       F.avg(F.col("verified_purchase").cast("int")).over(rev_w))
    df = df.withColumn(
        "rev_days_active",
        F.datediff(F.max("review_date").over(rev_w),
                   F.min("review_date").over(rev_w)),
    )

    # Reviewer's max reviews in a single day -- DIFFERENT from H2 (rolling 24h)
    daily = (df.groupBy("customer_id", "review_date").count()
             .withColumnRenamed("count", "_daily_count"))
    daily_max = (daily.groupBy("customer_id")
                 .agg(F.max("_daily_count").alias("rev_max_per_day")))
    df = df.join(F.broadcast(daily_max), "customer_id", "left")

    # Product-level aggregates
    prod_w = Window.partitionBy("product_id")
    df = df.withColumn("prod_total_reviews", F.count("*").over(prod_w))
    df = df.withColumn("prod_avg_rating", F.avg("star_rating").over(prod_w))
    df = df.withColumn("prod_pct_5star",
                       F.avg((F.col("star_rating") == 5).cast("int")).over(prod_w))
    df = df.withColumn("prod_pct_verified",
                       F.avg(F.col("verified_purchase").cast("int")).over(prod_w))
    df = df.withColumn(
        "prod_days_active",
        F.datediff(F.max("review_date").over(prod_w),
                   F.min("review_date").over(prod_w)),
    )

    # Keep only features + identifiers + label
    feature_cols = [
        "review_id",
        "rev_total_reviews", "rev_rating_stddev",
        "rev_pct_verified", "rev_days_active", "rev_max_per_day",
        "prod_total_reviews", "prod_avg_rating", "prod_pct_5star",
        "prod_pct_verified", "prod_days_active",
        "is_likely_fraud", "product_category",
    ]
    out = df.select(*feature_cols)

    print(f"Behavioral features: {len(feature_cols) - 3} feature columns")
    out.show(5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write.mode("overwrite").partitionBy("product_category").parquet(str(OUTPUT_DIR))

    print(f"Wrote to {OUTPUT_DIR}/")
    spark.stop()


if __name__ == "__main__":
    main()