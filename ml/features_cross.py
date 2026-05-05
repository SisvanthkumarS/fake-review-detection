"""Cross-features: rating deviations from product/reviewer means.

Captures 'how unusual is this review's rating relative to its context?'
A 5-star review for a generally low-rated product is more suspicious than
a 5-star review for a generally high-rated product.

EXCLUDED for label leakage: nothing label-derived. Reviewer-level avg
rating is allowed because the label uses the *boolean flag* derived from
extreme avg rating, not the continuous value itself.
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

TRAIN_DIR = Path("data/train")
OUTPUT_DIR = Path("data/features/cross")


def main():
    if not TRAIN_DIR.exists():
        raise SystemExit("No data/train/ dir.")

    spark = (SparkSession.builder
             .appName("CrossFeatures")
             .master("local[*]")
             .config("spark.driver.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(TRAIN_DIR))

    prod_w = Window.partitionBy("product_id")
    rev_w = Window.partitionBy("customer_id")

    # Rating deviation from product's mean (signed)
    df = df.withColumn("xf_rating_vs_product",
                       F.col("star_rating") - F.avg("star_rating").over(prod_w))

    # Rating deviation from reviewer's historical mean
    df = df.withColumn("xf_rating_vs_reviewer",
                       F.col("star_rating") - F.avg("star_rating").over(rev_w))

    # Extreme rating (1 or 5) when product avg is middling
    df = df.withColumn(
        "xf_extreme_review_neutral_product",
        ((F.col("star_rating").isin([1, 5])) &
         (F.avg("star_rating").over(prod_w).between(2.5, 4.0))).cast("int"),
    )

    feature_cols = [
        "review_id",
        "xf_rating_vs_product", "xf_rating_vs_reviewer",
        "xf_extreme_review_neutral_product",
        "is_likely_fraud", "product_category",
    ]
    out = df.select(*feature_cols)

    print(f"Cross features: {len(feature_cols) - 3} feature columns")
    out.show(5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write.mode("overwrite").partitionBy("product_category").parquet(str(OUTPUT_DIR))

    print(f"Wrote to {OUTPUT_DIR}/")
    spark.stop()


if __name__ == "__main__":
    main()