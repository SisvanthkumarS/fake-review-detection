"""Run six EDA queries on the curated dataset and save chart PNGs.

Charts produced (in report/figures/):
  1. rating_distribution.png       - bar chart of star ratings, faceted by category
  2. helpful_vote_ratio.png        - histogram of helpful_votes / total_votes
  3. reviewer_activity.png         - distribution of reviews-per-reviewer
  4. verified_vs_unverified.png    - bar chart by category
  5. review_velocity_timeseries.png - reviews per day across 2015
  6. top_reviewers.png             - top-20 most prolific reviewers
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend - works without display
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CURATED_DIR = Path("data/curated")
FIG_DIR = Path("report/figures")


def save_chart(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved: {path}")
    plt.close(fig)


def main():
    spark = (SparkSession.builder
             .appName("EDA")
             .master("local[*]")
             .config("spark.driver.memory", "6g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    df = spark.read.parquet(str(CURATED_DIR))
    df.cache()

    # Chart 1 - rating distribution by category
    print("Chart 1: rating distribution")
    pdf = df.groupBy("product_category", "star_rating").count().toPandas()
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = pdf.pivot(index="star_rating", columns="product_category", values="count")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Star rating distribution by category (2015)")
    ax.set_xlabel("Star rating"); ax.set_ylabel("Number of reviews")
    save_chart(fig, "rating_distribution.png")

    # Chart 2 - helpful vote ratio histogram
    print("Chart 2: helpful vote ratio")
    rated = (df
        .filter(F.col("total_votes") >= 1)
        .withColumn("helpful_ratio", F.col("helpful_votes") / F.col("total_votes")))
    sample = rated.select("helpful_ratio").sample(0.01).toPandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample["helpful_ratio"], bins=20, edgecolor="black")
    ax.set_title("Helpful-vote ratio distribution (1% sample)")
    ax.set_xlabel("helpful_votes / total_votes"); ax.set_ylabel("Frequency")
    save_chart(fig, "helpful_vote_ratio.png")

    # Chart 3 - reviewer activity distribution
    print("Chart 3: reviewer activity")
    activity = (df
        .groupBy("customer_id").count()
        .withColumnRenamed("count", "n_reviews"))
    pdf = activity.groupBy("n_reviews").count().orderBy("n_reviews").limit(50).toPandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pdf["n_reviews"], pdf["count"])
    ax.set_yscale("log")
    ax.set_title("Distribution of reviews-per-reviewer (log scale)")
    ax.set_xlabel("Reviews per customer"); ax.set_ylabel("Number of customers (log)")
    save_chart(fig, "reviewer_activity.png")

    # Chart 4 - verified vs unverified by category
    print("Chart 4: verified breakdown")
    pdf = df.groupBy("product_category", "verified_purchase").count().toPandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = pdf.pivot(index="product_category", columns="verified_purchase", values="count")
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Verified vs unverified purchases by category")
    ax.set_xlabel("Category"); ax.set_ylabel("Reviews")
    save_chart(fig, "verified_vs_unverified.png")

    # Chart 5 - review velocity over time
    print("Chart 5: review velocity")
    daily = df.groupBy("review_date").count().orderBy("review_date").toPandas()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["review_date"], daily["count"])
    ax.set_title("Reviews per day, Dec 2014 to Aug 2015")
    ax.set_xlabel("Date"); ax.set_ylabel("Reviews submitted")
    fig.autofmt_xdate()
    save_chart(fig, "review_velocity_timeseries.png")

    # Chart 6 - top 20 reviewers
    print("Chart 6: top reviewers")
    top = (df
        .groupBy("customer_id").count()
        .orderBy(F.col("count").desc()).limit(20).toPandas())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top["count"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"#{i+1}" for i in range(len(top))])
    ax.set_title("Top 20 most prolific reviewers (2015, 3 categories)")
    ax.set_xlabel("Number of reviews")
    ax.invert_yaxis()
    save_chart(fig, "top_reviewers.png")

    print("\nAll 6 charts saved to report/figures/")
    spark.stop()


if __name__ == "__main__":
    main()