"""Inspect the downloaded raw Parquet: schema, row count, category breakdown.

The PDS schema stores string columns as binary, so we cast them to strings
before any group-by or display operations. Without the cast, category names
appear as hex byte arrays.
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    files = sorted(Path("data/raw").glob("amazon_reviews_*.snappy.parquet"))
    if not files:
        raise SystemExit("No files in data/raw/. Run scripts/download_data.py first.")

    print(f"Reading {len(files)} file(s):")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.0f} MB)")
    print()

    spark = (SparkSession.builder
             .appName("InspectRaw")
             .master("local[*]")
             .config("spark.driver.memory", "6g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(*[str(f) for f in files])

    # Cast binary string columns to actual strings (PDS schema quirk)
    binary_cols = ["marketplace", "review_id", "product_id", "product_title",
                   "product_category", "review_headline", "review_body"]
    for col in binary_cols:
        df = df.withColumn(col, F.col(col).cast("string"))

    print("=== Schema (after string cast) ===")
    df.printSchema()
    print()

    n = df.count()
    print(f"=== Row count: {n:,} ===\n")

    print("=== Top 20 categories ===")
    (df.groupBy("product_category")
       .count()
       .orderBy(F.col("count").desc())
       .show(20, truncate=False))

    print("=== Star rating distribution ===")
    df.groupBy("star_rating").count().orderBy("star_rating").show()

    print("=== Verified purchase breakdown ===")
    df.groupBy("verified_purchase").count().show()

    print("=== Sample rows ===")
    (df.select("product_category", "star_rating", "helpful_votes",
               "total_votes", "verified_purchase",
               F.substring("review_headline", 1, 60).alias("headline"))
       .show(5, truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()