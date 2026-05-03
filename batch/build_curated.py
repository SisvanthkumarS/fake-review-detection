"""Filter raw multi-year data to one product_category and write curated Parquet.

Day 2 task. Default category: Electronics.
Override:  CATEGORY=Apparel python batch/build_curated.py
"""
import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CATEGORY = os.environ.get("CATEGORY", "Electronics")


def main():
    files = sorted(Path("data/raw").glob("amazon_reviews_*.snappy.parquet"))
    if not files:
        raise SystemExit("No files in data/raw/. Run scripts/download_data.py first.")

    print(f"Filtering to category: {CATEGORY}")
    print(f"Source files: {len(files)}\n")

    spark = (SparkSession.builder
             .appName(f"BuildCurated-{CATEGORY}")
             .master("local[*]")
             .config("spark.driver.memory", "8g")
             .config("spark.sql.shuffle.partitions", "64")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(*[str(f) for f in files])

    cleaned = (df.filter(F.col("product_category") == CATEGORY)
        .withColumn("star_rating", F.col("star_rating").cast("int"))
        .withColumn("helpful_votes", F.col("helpful_votes").cast("int"))
        .withColumn("total_votes", F.col("total_votes").cast("int"))
        .withColumn("review_date",
                    F.to_date(F.col("review_date").cast("string"), "yyyy-MM-dd"))
        .withColumn("verified_purchase",
                    (F.col("verified_purchase") == "Y").cast("boolean"))
        .withColumn("review_headline",
                    F.coalesce(F.col("review_headline"), F.lit("")))
        .withColumn("review_body",
                    F.coalesce(F.col("review_body"), F.lit("")))
        .filter(F.col("customer_id").isNotNull())
        .filter(F.col("product_id").isNotNull())
        .filter(F.col("star_rating").between(1, 5)))

    n = cleaned.count()
    print(f"Curated row count: {n:,}")

    if n < 100_000:
        print(f"\nWARNING: Only {n:,} rows after filtering.")
        print(f"  '{CATEGORY}' may be too small. Run inspect_raw.py to pick another.")

    out = Path("data/curated")
    out.mkdir(parents=True, exist_ok=True)

    (cleaned
        .withColumn("year", F.year("review_date"))
        .write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(str(out)))

    print(f"\nWrote {n:,} rows to {out}/ partitioned by year.")
    spark.stop()


if __name__ == "__main__":
    main()