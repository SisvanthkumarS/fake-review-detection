"""Join behavioral, textual, and cross feature DataFrames into final feature set.

Output is the input to ml/train.py on Day 4.
"""
from pathlib import Path
from pyspark.sql import SparkSession

BEH_DIR = Path("data/features/behavioral")
TXT_DIR = Path("data/features/textual")
XF_DIR = Path("data/features/cross")
OUT_DIR = Path("data/features/final")


def main():
    spark = (SparkSession.builder
             .appName("AssembleFeatures")
             .master("local[*]")
             .config("spark.driver.memory", "10g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    beh = spark.read.parquet(str(BEH_DIR))
    txt = spark.read.parquet(str(TXT_DIR)).drop("is_likely_fraud", "product_category")
    xf = spark.read.parquet(str(XF_DIR)).drop("is_likely_fraud", "product_category")

    print(f"Behavioral rows:  {beh.count():,}  cols: {len(beh.columns)}")
    print(f"Textual rows:     {txt.count():,}  cols: {len(txt.columns)}")
    print(f"Cross rows:       {xf.count():,}  cols: {len(xf.columns)}")

    # Join on review_id (the only identifier present in all three)
    joined = (beh
              .join(txt, "review_id", "inner")
              .join(xf, "review_id", "inner"))

    n = joined.count()
    print(f"\nFinal feature DataFrame: {n:,} rows, {len(joined.columns)} columns")
    joined.printSchema()

    print("\n=== 5-row sample ===")
    joined.show(5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined.write.mode("overwrite").partitionBy("product_category").parquet(str(OUT_DIR))

    print(f"\nWrote to {OUT_DIR}/")
    spark.stop()


if __name__ == "__main__":
    main()