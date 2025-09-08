from pyspark.sql import SparkSession
import os

CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kindle_reviews.csv")
)

spark = (
    SparkSession.builder.appName("KindleReviews")
    .config("spark.driver.memory", "1g")
    .config("spark.executor.memory", "1g")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

# Read CSV
df = spark.read.csv(
    CSV_PATH,
    header=True,
    inferSchema=True,
    quote='"',
    escape='"',
    multiLine=True,
)

print("Header:", ", ".join(df.columns))
print("Schema:", ", ".join([f.name for f in df.schema.fields]))

first_col = df.columns[0]
if first_col == "" or first_col == "_c0":
    df = df.drop(first_col)
    print("Dropped leading column:", repr(first_col))
    print("New Header:", ", ".join(df.columns))

expected = [
    "asin",
    "helpful",
    "overall",
    "reviewText",
    "reviewTime",
    "reviewerID",
    "reviewerName",
    "summary",
    "unixReviewTime",
]
if len(df.columns) == len(expected) + 1 and df.columns[0] not in expected:
    df = df.drop(df.columns[0])
    print("Dropped first column to match expected schema")

print("Final Header:", ", ".join(df.columns))
print("Final Schema:", df.schema.simpleString())

spark.stop()
