from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = str(PROJECT_ROOT / "data" / "features" / "final")
MODEL_PATH = str(PROJECT_ROOT / "models" / "rf_v1")
RESULTS_PATH = PROJECT_ROOT / "report" / "model_results.md"

LABEL_BOOL_COL = "is_likely_fraud"
LABEL_COL = "label"
RAW_FEATURES_COL = "raw_features"
FEATURES_COL = "features"


def main():
    spark = (
        SparkSession.builder
        .appName("TrainFakeReviewRandomForest")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading feature dataset from: {FEATURES_PATH}")
    df = spark.read.parquet(FEATURES_PATH)

    df = df.withColumn(
        LABEL_COL,
        when(col(LABEL_BOOL_COL) == True, 1.0).otherwise(0.0)
    )

    excluded_cols = {
        "review_id",
        "product_category",
        LABEL_BOOL_COL,
        LABEL_COL,
    }

    numeric_types = {
        "int",
        "bigint",
        "double",
        "float",
        "long",
        "smallint",
        "tinyint",
    }

    feature_cols = [
        field.name
        for field in df.schema.fields
        if field.name not in excluded_cols
        and field.dataType.simpleString() in numeric_types
    ]

    print("Rows:", df.count())
    print("Feature count:", len(feature_cols))

    print("\nUsing features:")
    for feature in feature_cols:
        print(f"  - {feature}")

    df = df.fillna(0, subset=feature_cols)

    print("\nLabel distribution:")
    df.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

    _, test_df = df.randomSplit([0.7, 0.3], seed=42)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=RAW_FEATURES_COL,
        handleInvalid="keep",
    )

    scaler = StandardScaler(
        inputCol=RAW_FEATURES_COL,
        outputCol=FEATURES_COL,
        withMean=False,
        withStd=True,
    )

    rf = RandomForestClassifier(
        labelCol=LABEL_COL,
        featuresCol=FEATURES_COL,
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=100,
        maxDepth=10,
        seed=42,
        subsamplingRate=0.8,
        featureSubsetStrategy="sqrt",
    )

    pipeline = Pipeline(stages=[assembler, scaler, rf])

    from pyspark.ml import PipelineModel
    print("\nLoading saved model...")
    model = PipelineModel.load(MODEL_PATH)

    print(f"\nSaving model to: {MODEL_PATH}")
    model.write().overwrite().save(MODEL_PATH)

    print("\nEvaluating model...")
    predictions = model.transform(test_df).cache()

    binary_auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)

    area_pr = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    ).evaluate(predictions)

    accuracy = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="accuracy",
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="weightedPrecision",
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="weightedRecall",
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="f1",
    ).evaluate(predictions)

    print("\nMetrics:")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1:        {f1:.4f}")
    print(f"ROC AUC:            {binary_auc:.4f}")
    print(f"Area Under PR:      {area_pr:.4f}")

    print("\nConfusion Matrix:")
    confusion = (
        predictions
        .groupBy(LABEL_COL, "prediction")
        .count()
        .orderBy(LABEL_COL, "prediction")
    )
    confusion.show()

    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray().tolist()

    feature_importances = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nTop Feature Importances:")
    for name, score in feature_importances[:15]:
        print(f"{name}: {score:.6f}")

    total_rows = df.count()

    confusion_rows = confusion.collect()
    confusion_md = "\n".join(
        f"| {row[LABEL_COL]} | {row['prediction']} | {row['count']} |"
        for row in confusion_rows
    )

    importances_md = "\n".join(
        f"| {name} | {score:.6f} |"
        for name, score in feature_importances
    )

    RESULTS_PATH.write_text(
        f"""# Model Results

## Dataset Summary

- Input dataset: `data/features/final`
- Total rows: {total_rows}
- Feature count: {len(feature_cols)}
- Label column: `{LABEL_BOOL_COL}`
- Model: Spark MLlib Random Forest
- Model output path: `models/rf_v1`

## Evaluation Metrics

| Metric | Value |
|---|---:|
| Accuracy | {accuracy:.4f} |
| Weighted Precision | {precision:.4f} |
| Weighted Recall | {recall:.4f} |
| Weighted F1 | {f1:.4f} |
| ROC AUC | {binary_auc:.4f} |
| Area Under PR | {area_pr:.4f} |

## Confusion Matrix

| Actual Label | Predicted Label | Count |
|---:|---:|---:|
{confusion_md}

## Feature Importance

| Feature | Importance |
|---|---:|
{importances_md}

## Strict Methodology Note

This model predicts weak-supervision labels generated from behavioral and review-pattern heuristics. 
It should be described as suspicious-review detection, not confirmed real-world fake-review detection.

If model performance is extremely high, the likely cause is label leakage from heuristic-derived features.
""",
        encoding="utf-8",
    )

    print(f"\nSaved report to: {RESULTS_PATH}")

    predictions.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
