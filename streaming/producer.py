import time
import json
from pathlib import Path
import pyarrow.parquet as pq
from kafka import KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH   = str(PROJECT_ROOT / "data" / "test")
KAFKA_BROKER = "localhost:9092"
TOPIC        = "reviews-stream"

COLS = [
    "review_id", "customer_id", "product_id",
    "product_category", "star_rating", "helpful_votes",
    "total_votes", "verified_purchase", "review_body",
    "review_date", "is_likely_fraud"
]


def main():
    print("Reading data...")
    dataset = pq.ParquetDataset(INFERENCE_PATH)
    table = dataset.read(columns=COLS)

    print(f"Loaded {table.num_rows:,} total rows")
    print("Category breakdown:")
    categories = table.column("product_category")
    from collections import Counter
    counts = Counter(categories.to_pylist())
    for cat, count in counts.items():
        print(f"  {cat}: {count:,}")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: v.encode("utf-8"),
    )

    print(f"\nPublishing to '{TOPIC}'. Press Ctrl+C to stop.\n")

    for i in range(table.num_rows):
        try:
            row = {col: table.column(col)[i].as_py() for col in COLS}
            for k, v in row.items():
                if hasattr(v, 'isoformat'):
                    row[k] = v.isoformat()
                elif v is None:
                    row[k] = None

            producer.send(TOPIC, value=json.dumps(row, default=str))
            producer.flush()
            if i % 1000 == 0:
                print(f"  Sent {i+1:,}/{table.num_rows:,}")
            time.sleep(0)
        except Exception as e:
            print(f"  Skipping row {i}: {e}")
            continue

    print("\nAll reviews published.")
    producer.close()


if __name__ == "__main__":
    main()