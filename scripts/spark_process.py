"""Обработка kindle_reviews.csv c оптимизациями.

Изменения/особенности:
* Автоматическое удаление искусственного индексного столбца (leading comma / _c0)
* Балансировка по классам (звёздам) с настраиваемым лимитом по классу
* Минимизация числа shuffle (агрегации пользователя и товара в одном проходе каждая)
* Контролируемое число shuffle партиций для малых/средних объёмов
"""

from pathlib import Path
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    length,
    split,
    count,
    avg,
    row_number,
    size,
    lower,
    when,
)
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark import StorageLevel
from config import (
    FORCE_PROCESS,
    CSV_PATH,
    PROCESSED_DATA_DIR,
    PER_CLASS_LIMIT,
    HASHING_TF_FEATURES,
    SHUFFLE_PARTITIONS,
)

TRAIN_PATH = PROCESSED_DATA_DIR + "/train.parquet"
VAL_PATH = PROCESSED_DATA_DIR + "/val.parquet"
TEST_PATH = PROCESSED_DATA_DIR + "/test.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if (
    not FORCE_PROCESS
    and Path(TRAIN_PATH).exists()
    and Path(VAL_PATH).exists()
    and Path(TEST_PATH).exists()
):
    log.warning(
        "Обработанные данные уже существуют в %s. Для форсированной обработки установите флаг FORCE_PROCESS = True в config.py.",
        PROCESSED_DATA_DIR,
    )
else:
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

    # Снижаем число shuffle партиций для средних объёмов
    try:
        spark.conf.set("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
    except Exception:
        pass

    log.info(
        "Конфигурация: driverMemory=%s, executorMemory=%s, shuffle.partitions=%s",
        spark.sparkContext.getConf().get("spark.driver.memory"),
        spark.sparkContext.getConf().get("spark.executor.memory"),
        spark.conf.get("spark.sql.shuffle.partitions"),
    )

    # Чтение CSV с возможной пустой первой колонкой (индекс) из-за начальной запятой
    df = spark.read.csv(
        CSV_PATH,
        header=True,
        inferSchema=True,
        quote='"',
        escape='"',
        multiLine=True,
    )

    log.info("Удаление индексного столбца")
    df = df.drop(df.columns[0])

    # Оставляем только нужные колонки
    cols = [
        "reviewerID",
        "asin",
        "reviewText",
        "overall",
        "unixReviewTime",
        "reviewTime",
    ]
    df = df.select(*cols)

    log.info("Строк после селекта колонок: %s", df.count())

    # Чистим данные: валидные тексты и оценки
    clean = df.filter((col("reviewText").isNotNull()) & (col("overall").isNotNull()))
    log.info("После фильтрации null: %s", clean.count())

    # Балансировка: берём последние n по каждому классу
    window = Window.partitionBy("overall").orderBy(col("unixReviewTime").desc())
    clean = (
        clean.withColumn("row_num", row_number().over(window))
        .filter(col("row_num") <= PER_CLASS_LIMIT)
        .drop("row_num")
    )
    balanced_count = clean.count()
    log.info(
        "После балансировки (<= %d на класс) строк: %d", PER_CLASS_LIMIT, balanced_count
    )

    clean = clean.withColumn("text_len", length(col("reviewText")))
    clean = clean.withColumn("word_count", size(split(col("reviewText"), " ")))
    # Частота слова 'kindle' в отзыве
    clean = clean.withColumn(
        "kindle_freq", size(split(lower(col("reviewText")), "kindle")) - 1
    )

    # Sentiment (простая эвристика: +1 если есть 'good', 'excellent', -1 если 'bad', 'poor', 0 иначе)
    clean = clean.withColumn(
        "sentiment",
        when(lower(col("reviewText")).contains("good"), 1)
        .when(lower(col("reviewText")).contains("excellent"), 1)
        .when(lower(col("reviewText")).contains("bad"), -1)
        .when(lower(col("reviewText")).contains("poor"), -1)
        .otherwise(0),
    )

    # Делим на выборки
    train, val, test = clean.randomSplit([0.7, 0.15, 0.15], seed=42)
    tr_c, v_c, te_c = train.count(), val.count(), test.count()
    log.info(
        "Размеры выборок: train=%d, val=%d, test=%d (total=%d)",
        tr_c,
        v_c,
        te_c,
        tr_c + v_c + te_c,
    )

    # TF-IDF: фитим IDF только на train и применяем к val/test тем же моделям
    tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
    hashingTF = HashingTF(
        inputCol="words", outputCol="rawFeatures", numFeatures=HASHING_TF_FEATURES
    )
    idf = IDF(inputCol="rawFeatures", outputCol="tfidfFeatures")

    train_words = tokenizer.transform(train)
    train_feat = hashingTF.transform(train_words)
    idfModel = idf.fit(train_feat)
    train = idfModel.transform(train_feat)

    val = idfModel.transform(hashingTF.transform(tokenizer.transform(val)))
    test = idfModel.transform(hashingTF.transform(tokenizer.transform(test)))

    train = train.persist(StorageLevel.MEMORY_AND_DISK)
    val = val.persist(StorageLevel.MEMORY_AND_DISK)
    test = test.persist(StorageLevel.MEMORY_AND_DISK)
    log.info("Признаков HashingTF: %d", HASHING_TF_FEATURES)

    # Агрегации на train
    user_stats = train.groupBy("reviewerID").agg(
        avg("text_len").alias("user_avg_len"),
        count("reviewText").alias("user_review_count"),
    )
    item_stats = train.groupBy("asin").agg(
        avg("text_len").alias("item_avg_len"),
        count("reviewText").alias("item_review_count"),
    )

    # Присоединяем агрегаты к каждому датасету
    train = train.join(user_stats, on="reviewerID", how="left").join(
        item_stats, on="asin", how="left"
    )
    val = val.join(user_stats, on="reviewerID", how="left").join(
        item_stats, on="asin", how="left"
    )
    test = test.join(user_stats, on="reviewerID", how="left").join(
        item_stats, on="asin", how="left"
    )

    log.info(
        "После добавления агрегатов кол-во колонок в train: %d", len(train.columns)
    )

    train.write.mode("overwrite").parquet(TRAIN_PATH)
    val.write.mode("overwrite").parquet(VAL_PATH)
    test.write.mode("overwrite").parquet(TEST_PATH)

    log.info(
        "Обработка завершена. Данные сохранены в %s",
        str(Path(PROCESSED_DATA_DIR).resolve()),
    )
    spark.stop()
