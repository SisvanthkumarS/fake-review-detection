"""
Day 5: Kafka Producer
Reads held-out test data and publishes one review per second to Kafka.
Topic: reviews-stream
"""

import json
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from kafka import KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH    = str(PROJECT_ROOT / "data" / "test")
KAFKA_BROKER = "localhost:9092"
TOPIC        = "reviews-stream"


def main():
    spark = (
        SparkSession.builder
        .appName("FakeReviewProducer")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading test data from {TEST_PATH}...")
    df = (
        spark.read.parquet(TEST_PATH)
        .select(
            "review_id", "customer_id", "product_id",
            "product_category", "star_rating", "helpful_votes",
            "total_votes", "verified_purchase", "review_body",
            "review_date", "is_likely_fraud"
        )
        .filter(F.col("product_category") == "Wireless")
        .limit(5000)
    )

    rows = df.toJSON().collect()
    spark.stop()

    print(f"Loaded {len(rows)} reviews. Connecting to Kafka at {KAFKA_BROKER}...")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: v.encode("utf-8"),
    )

    print(f"Publishing to topic '{TOPIC}' at 1 review/sec. Press Ctrl+C to stop.\n")

    for i, row in enumerate(rows):
        producer.send(TOPIC, value=row)
        producer.flush()
        print(f"  Sent review {i+1}/{len(rows)}")
        time.sleep(1)

    print("\nAll reviews published.")
    producer.close()


if __name__ == "__main__":
    main()