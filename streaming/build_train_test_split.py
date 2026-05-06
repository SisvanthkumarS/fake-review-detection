"""Split labeled dataset into train (70%) and test (30%) by REVIEWER, not by row.

Splitting by row would put the same reviewer in both train and test, leaking
behavioral features. Splitting by reviewer keeps the same person fully in
one or the other.

Output:
  data/train/  — partitioned by product_category (same as data/labeled)
  data/test/   — partitioned by product_category
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

LABELED_DIR = Path("data/labeled")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
SEED = 42


def main():
    if not LABELED_DIR.exists():
        raise SystemExit("No data/labeled/. Run ml/generate_labels.py first.")

    spark = (SparkSession.builder
             .appName("TrainTestSplit")
             .master("local[*]")
             .config("spark.driver.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(LABELED_DIR))
    n_total = df.count()
    print(f"Labeled rows: {n_total:,}")

    # Assign each unique customer_id to train or test (random 70/30)
    customers = df.select("customer_id").distinct()
    train_customers, test_customers = customers.randomSplit([0.7, 0.3], seed=SEED)
    print(f"Train customers: {train_customers.count():,}")
    print(f"Test customers:  {test_customers.count():,}")

    train = df.join(F.broadcast(train_customers), "customer_id", "inner")
    test = df.join(F.broadcast(test_customers), "customer_id", "inner")

    n_train = train.count()
    n_test = test.count()
    print(f"\nTrain rows: {n_train:,} ({100*n_train/n_total:.1f}%)")
    print(f"Test rows:  {n_test:,} ({100*n_test/n_total:.1f}%)")

    # Sanity: positive class rate similar in both splits
    print("\nPositive class rate:")
    print(f"  Train: {train.filter('is_likely_fraud').count() / n_train * 100:.2f}%")
    print(f"  Test:  {test.filter('is_likely_fraud').count() / n_test * 100:.2f}%")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    train.write.mode("overwrite").partitionBy("product_category").parquet(str(TRAIN_DIR))
    test.write.mode("overwrite").partitionBy("product_category").parquet(str(TEST_DIR))

    print(f"\nWrote train/ and test/ to data/")
    spark.stop()


if __name__ == "__main__":
    main()
