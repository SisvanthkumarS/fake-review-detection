# Fake Review Detection at Scale

CS-GY-6513 Big Data, Spring 2026 Assignment 3.
A scalable big-data pipeline for detecting fraudulent Amazon product reviews using Apache Spark, MLlib, and Kafka.

## Dataset

Original Amazon Customer Reviews PDS (1995-2015, ~150M rows) via the public ClickHouse mirror. Default category: Electronics.

## Repo layout

- batch/ — Workload 1: SparkSQL batch jobs
- ml/ — Workload 2: feature engineering and MLlib training
- streaming/ — Workload 3: Kafka producer and Structured Streaming consumer
- dashboard/ — Streamlit fraud-alerts UI
- docker/ — Local Kafka container
- data/ — Parquet artifacts (gitignored)
- models/ — Saved MLlib models (gitignored)
- report/ — Final business report and figures
- slides/ — Presentation deck
- scripts/ — Setup, data download, demo orchestration

## Team

Person A: Infrastructure, streaming.
Person B: Features, MLlib.
Person C: EDA, report, slides, dashboard.