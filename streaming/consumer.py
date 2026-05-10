"""
Day 5: Spark Structured Streaming Consumer
Reads from Kafka topic reviews-stream, applies saved RF model,
outputs predictions and windowed fraud alerts.
Topic in:  reviews-stream
Topic out: console (demo)
"""

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    BooleanType, LongType
)
from pyspark.ml import PipelineModel

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
MODEL_PATH    = str(PROJECT_ROOT / "models" / "rf_v1")
KAFKA_BROKER  = "localhost:9092"
INPUT_TOPIC   = "reviews-stream"
LOOKUPS_PATH  = str(PROJECT_ROOT / "data" / "lookups")

REVIEW_SCHEMA = StructType([
    StructField("review_id",         StringType(),  True),
    StructField("customer_id",       StringType(),  True),
    StructField("product_id",        StringType(),  True),
    StructField("product_category",  StringType(),  True),
    StructField("star_rating",       IntegerType(), True),
    StructField("helpful_votes",     IntegerType(), True),
    StructField("total_votes",       IntegerType(), True),
    StructField("verified_purchase", BooleanType(), True),
    StructField("review_body",       StringType(),  True),
    StructField("review_date",       StringType(),  True),
    StructField("is_likely_fraud",   BooleanType(), True),
])


def main():
    spark = (
        SparkSession.builder
        .appName("FakeReviewConsumer")
        .master("local[3]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("Loading saved model...")
    model = PipelineModel.load(MODEL_PATH)

    print("Loading reviewer and product lookup tables...")
    reviewer_raw = spark.read.parquet(f"{LOOKUPS_PATH}/reviewer_history.parquet")
    product_raw  = spark.read.parquet(f"{LOOKUPS_PATH}/product_history.parquet")

    # Rename prefixed columns to match feature names the model expects
    reviewer_lookup = reviewer_raw.toDF(
        "customer_id", "rev_total_reviews", "rev_avg_rating",
        "rev_rating_stddev", "rev_pct_verified", "rev_days_active", "rev_max_per_day"
    )
    product_lookup = product_raw.toDF(
        "product_id", "prod_total_reviews", "prod_avg_rating",
        "prod_pct_5star", "prod_pct_verified", "prod_days_active"
    )

    print(f"Connecting to Kafka at {KAFKA_BROKER}, topic '{INPUT_TOPIC}'...")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw_stream
        .select(
            F.from_json(F.col("value").cast("string"), REVIEW_SCHEMA).alias("data"),
            F.col("timestamp")
        )
        .select("data.*", "timestamp")
    )

    enriched = (
        parsed
        .join(F.broadcast(reviewer_lookup), on="customer_id", how="left")
        .join(F.broadcast(product_lookup),  on="product_id",  how="left")
        .fillna(0)
        # Missing from lookup — add as placeholder
        # Text features
        .withColumn("txt_body_length",              F.length(F.col("review_body")))
        .withColumn("txt_word_count",               F.size(F.split(F.col("review_body"), " ")))
        .withColumn("txt_exclamation_count",        F.size(F.split(F.col("review_body"), "!")) - 1)
        .withColumn("txt_caps_word_count",          F.lit(0).cast(IntegerType()))
        .withColumn("txt_caps_ratio",               F.lit(0.0))
        .withColumn("txt_sentiment",                F.lit(0.0))
        # Cross features
        .withColumn("xf_rating_vs_product",
                    (F.col("star_rating") - F.col("prod_avg_rating")))
        .withColumn("xf_rating_vs_reviewer",
                    (F.col("star_rating") - F.col("rev_avg_rating")))
        .withColumn("xf_extreme_review_neutral_product",
                    ((F.col("star_rating") >= 4) &
                     (F.col("prod_avg_rating").between(2.5, 3.5))).cast(IntegerType()))
        # Required placeholder for pipeline schema
        .withColumn("label", F.lit(0.0))
    )

    predictions = model.transform(enriched)

    # Extract fraud probability from sparse vector safely
    fraud_prob = vector_to_array(F.col("probability")).getItem(1)

    # ── Output 1: per-review predictions ─────────────────────────────────────
    review_query = (
        predictions
        .select(
            "review_id", "product_id", "product_category",
            "star_rating", "prediction",
            fraud_prob.alias("fraud_probability"),
            "timestamp"
        )
        .writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", False)
        .option("numRows", 20)
        .trigger(processingTime="5 seconds")
        .start()
    )

    # ── Output 2: windowed fraud alert aggregation ────────────────────────────
    alert_query = (
        predictions
        .withColumn("fraud_probability", fraud_prob)
        .filter(F.col("prediction") == 1.0)
        .withWatermark("timestamp", "10 minutes")
        .groupBy(
            F.window("timestamp", "10 minutes", "5 minutes"),
            F.col("product_id"),
            F.col("product_category")
        )
        .agg(
            F.count("*").alias("fraud_count"),
            F.avg("fraud_probability").alias("avg_fraud_prob")
        )
        .filter(F.col("fraud_count") >= 3)
        .writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", False)
        .trigger(processingTime="10 seconds")
        .start()
    )

    print("\nStreaming started. Waiting for reviews from producer...")
    print("Per-review predictions will appear every 5 seconds.")
    print("Fraud alerts will appear when a product gets 3+ fraud flags in 10 minutes.")
    print("Press Ctrl+C to stop.\n")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()