"""Hello-world Spark job. Verifies PySpark works end-to-end."""
from pyspark.sql import SparkSession


def main():
    spark = (SparkSession.builder
             .appName("HelloSpark")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.createDataFrame(
        [(1, "great product", 5),
         (2, "terrible", 1),
         (3, "okay", 3)],
        ["id", "review", "rating"])

    print("\n=== Spark is working ===")
    df.show()
    print(f"Spark version: {spark.version}")
    spark.stop()


if __name__ == "__main__":
    main()