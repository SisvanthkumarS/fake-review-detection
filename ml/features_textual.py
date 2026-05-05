"""Textual features: review length, exclamation count, all-caps ratio, sentiment.

Sanitizes review_body in SparkSQL before the Pandas UDF to handle non-UTF-8
bytes that would otherwise crash PyArrow during JVM <-> Python transport.
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd
from textblob import TextBlob

TRAIN_DIR = Path("data/train")
OUTPUT_DIR = Path("data/features/textual")


@F.pandas_udf(DoubleType())
def sentiment_polarity(text_series: pd.Series) -> pd.Series:
    """TextBlob polarity in [-1.0, 1.0]. Empty/null returns 0.0."""
    return text_series.fillna("").apply(
        lambda s: TextBlob(s).sentiment.polarity if s else 0.0
    )


def main():
    if not TRAIN_DIR.exists():
        raise SystemExit("No data/train/ dir.")

    spark = (SparkSession.builder
             .appName("TextualFeatures")
             .master("local[*]")
             .config("spark.driver.memory", "10g")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(TRAIN_DIR))

    # CRITICAL: sanitize bad UTF-8 bytes in SparkSQL before the Pandas UDF.
    # encode -> decode forces a round-trip; invalid bytes get replaced with
    # the Unicode replacement char, leaving valid UTF-8 for Arrow transport.
    df = df.withColumn("review_body_clean",
                       F.decode(F.encode(F.col("review_body"), "UTF-8"), "UTF-8"))

    # Length features (use the original column - encoding doesn't affect length)
    df = df.withColumn("txt_body_length", F.length("review_body"))
    df = df.withColumn("txt_word_count",
                       F.size(F.split("review_body", r"\s+")))
    df = df.withColumn("txt_exclamation_count",
                       F.size(F.split("review_body", "!")) - 1)

    df = df.withColumn(
        "txt_caps_word_count",
        F.size(F.expr(
            r"filter(split(review_body, '\\s+'), "
            r"x -> length(x) > 2 AND upper(x) == x AND x rlike '[A-Z]')"
        ))
    )
    df = df.withColumn(
        "txt_caps_ratio",
        F.when(F.col("txt_word_count") > 0,
               F.col("txt_caps_word_count") / F.col("txt_word_count"))
         .otherwise(0.0),
    )

    # Sentiment uses the sanitized column
    print("Computing sentiment polarity (slow step, 15-30 min expected)...")
    df = df.withColumn("txt_sentiment", sentiment_polarity(F.col("review_body_clean")))

    feature_cols = [
        "review_id",
        "txt_body_length", "txt_word_count", "txt_exclamation_count",
        "txt_caps_word_count", "txt_caps_ratio", "txt_sentiment",
        "is_likely_fraud", "product_category",
    ]
    out = df.select(*feature_cols)

    print(f"Textual features: {len(feature_cols) - 3} feature columns")
    out.show(5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write.mode("overwrite").partitionBy("product_category").parquet(str(OUTPUT_DIR))

    print(f"Wrote to {OUTPUT_DIR}/")
    spark.stop()


if __name__ == "__main__":
    main()