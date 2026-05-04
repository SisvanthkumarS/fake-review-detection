"""Filter raw 2015 data to three target categories and write curated Parquet.

Filters 41.9M raw rows to ~8.2M rows across Wireless, Books, Apparel.
Partitions output by category for efficient downstream reads.
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CATEGORIES = ["Wireless", "Books", "Apparel"]
RAW_DIR = Path("data/raw")
CURATED_DIR = Path("data/curated")


def main():
    files = sorted(RAW_DIR.glob("amazon_reviews_*.snappy.parquet"))
    if not files:
        raise SystemExit(f"No files in {RAW_DIR}/. Run scripts/download_data.py first.")

    print(f"Filtering to categories: {CATEGORIES}")
    print(f"Source files: {len(files)}\n")

    spark = (SparkSession.builder
             .appName("BuildCurated-3Categories")
             .master("local[*]")
             .config("spark.driver.memory", "8g")
             .config("spark.sql.shuffle.partitions", "64")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(*[str(f) for f in files])

    # Cast binary string columns to actual strings (PDS schema quirk)
    binary_cols = ["marketplace", "review_id", "product_id", "product_title",
                   "product_category", "review_headline", "review_body"]
    for col in binary_cols:
        df = df.withColumn(col, F.col(col).cast("string"))

    # Filter to our three target categories
    filtered = df.filter(F.col("product_category").isin(CATEGORIES))

    # Clean types and drop null/invalid rows
    cleaned = (filtered
        .withColumn("star_rating", F.col("star_rating").cast("int"))
        .withColumn("helpful_votes", F.col("helpful_votes").cast("int"))
        .withColumn("total_votes", F.col("total_votes").cast("int"))
        .withColumn("review_date",
                    F.from_unixtime(F.col("review_date").cast("long") * 86400)
                    .cast("date"))
        .withColumn("review_headline",
                    F.coalesce(F.col("review_headline"), F.lit("")))
        .withColumn("review_body",
                    F.coalesce(F.col("review_body"), F.lit("")))
        .filter(F.col("customer_id").isNotNull())
        .filter(F.col("product_id").isNotNull())
        .filter(F.col("star_rating").between(1, 5)))

    # Force evaluation, count by category for sanity-check log
    cleaned.cache()
    n = cleaned.count()
    print(f"Total curated rows: {n:,}\n")

    print("=== Per-category counts ===")
    cleaned.groupBy("product_category").count().orderBy(F.col("count").desc()).show()

    # Write partitioned by category for fast downstream reads
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    (cleaned.write
        .mode("overwrite")
        .partitionBy("product_category")
        .parquet(str(CURATED_DIR)))

    print(f"\nWrote curated dataset to {CURATED_DIR}/")
    print("Partitioned by product_category. Downstream reads should filter on this column.")

    spark.stop()


if __name__ == "__main__":
    main()