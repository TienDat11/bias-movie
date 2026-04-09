from typing import List, Optional, Tuple, Union
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Word2Vec
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, explode, split, regexp_extract, avg, count, udf, when, max as max_func, \
    collect_set, array, lit, array_intersect, size
import logging
import numpy as np
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Spark
spark: SparkSession = SparkSession.builder \
    .appName("OptimizedMovieLensRecommendation") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.default.parallelism", "100") \
    .config("spark.driver.memory", "5000m") \
    .config("spark.executor.memory", "5000m") \
    .config("spark.driver.extraJavaOptions", "-Xss8m") \
    .config("spark.executor.extraJavaOptions", "-Xss8m") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "30s") \
    .config("spark.python.worker.reuse", "true") \
    .getOrCreate()


def spark_info(df: DataFrame):
    """
    Hiển thị thông tin cơ bản về DataFrame, bao gồm số hàng, số cột và số lượng giá trị null.
    Lý do: Giúp kiểm tra dữ liệu để đảm bảo không có lỗi trước khi xử lý.
    """
    print("=" * 40)
    print(f"Total Rows: {df.count()}")
    print(f"Total Columns: {len(df.columns)}")
    print("=" * 40)
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0]
    schema_info = df.dtypes
    print(f"{'Column Name':<25} {'Data Type':<15} {'Null Count':<10}")
    print("-" * 60)
    for (col_name, col_type), null_count in zip(schema_info, null_counts):
        print(f"{col_name:<25} {col_type:<15} {null_count:<10}")
    print("=" * 40)


# Tải dữ liệu
logger.info("🔹 Đang tải dữ liệu...")
df_ratings: DataFrame = spark.read.csv("data/ml-1m/ratings.dat", sep="::", inferSchema=True) \
    .toDF("userId", "movieId", "rating", "timestamp").cache()
df_movies: DataFrame = spark.read.csv("data/ml-1m/movies.dat", sep="::", inferSchema=True) \
    .toDF("movieId", "title", "genres").cache()
df_users: DataFrame = spark.read.csv("data/ml-1m/users.dat", sep="::", inferSchema=True) \
    .toDF("userId", "gender", "age", "occupation", "zip_code").cache()

# Tiền xử lý dữ liệu
logger.info("🔹 Đang tiền xử lý dữ liệu...")
df_users = df_users.withColumn("gender", col("gender").cast("string"))
df_movies = df_movies.withColumn("genres", split(col("genres"), "\\|")) \
    .withColumn("year", regexp_extract(col("title"), "\\((\\d{4})\\)", 1).cast("int"))

# Mã hóa đặc trưng người dùng
# Lý do: Chuyển đổi dữ liệu phân loại (gender, occupation) thành dạng số để sử dụng trong mô hình học máy.
indexer = StringIndexer(inputCol="gender", outputCol="gender_idx")
encoder = OneHotEncoder(inputCols=["gender_idx", "occupation"], outputCols=["gender_vec", "occupation_vec"])
pipeline = Pipeline(stages=[indexer, encoder])
pipeline_model: PipelineModel = pipeline.fit(df_users)
df_users = pipeline_model.transform(df_users).cache()

# Tạo embeddings cho phim dựa trên thể loại
# Lý do: Word2Vec tạo ra biểu diễn số của các thể loại phim, giúp so sánh sở thích người dùng và phim mới.
word2vec = Word2Vec(vectorSize=5, minCount=0, inputCol="genres", outputCol="movie_embedding")
model_w2v = word2vec.fit(df_movies)
df_movies = model_w2v.transform(df_movies).cache()

# Phân cụm người dùng
# Lý do: Phân cụm giúp nhóm người dùng có đặc điểm tương tự, hỗ trợ đề xuất cho người dùng mới.
logger.info("🔹 Đang phân cụm người dùng...")
vec_assembler = VectorAssembler(
    inputCols=["age", "gender_vec", "occupation_vec"],
    outputCol="features",
    handleInvalid="skip"
)
df_users = vec_assembler.transform(df_users).cache()
kmeans = KMeans(k=5, featuresCol="features", predictionCol="cluster")
model_kmeans = kmeans.fit(df_users)
df_users = model_kmeans.transform(df_users).cache()

# Ghép dữ liệu
# Lý do: Kết hợp dữ liệu người dùng, phim và đánh giá để tạo tập dữ liệu tổng hợp cho các phân tích tiếp theo.
df_merged: DataFrame = df_ratings.join(df_users, "userId").join(df_movies, "movieId").cache()

# Hiển thị thông tin dữ liệu
spark_info(df_merged)

# Tính toán profile embedding cho từng người dùng
# Lý do: Profile embedding biểu diễn sở thích thể loại của người dùng dựa trên trung bình embedding của các phim họ đã đánh giá.
logger.info("🔹 Đang tính toán profile embedding cho người dùng...")
embedding_size = 5  # Kích thước embedding từ Word2Vec
df_with_emb = df_ratings.join(df_movies.select("movieId", "movie_embedding"), "movieId")
for i in range(embedding_size):
    df_with_emb = df_with_emb.withColumn(f"emb_{i}", col("movie_embedding").getItem(i))
user_profiles = df_with_emb.groupBy("userId").agg(
    *[avg(f"emb_{i}").alias(f"avg_emb_{i}") for i in range(embedding_size)])
vector_assembler = VectorAssembler(inputCols=[f"avg_emb_{i}" for i in range(embedding_size)],
                                   outputCol="profile_embedding")
user_profiles = vector_assembler.transform(user_profiles).select("userId", "profile_embedding").cache()

# Tính toán thể loại yêu thích của người dùng
# Lý do: Lưu trữ các thể loại từ các phim được đánh giá cao để sử dụng trong giải thích đề xuất.
logger.info("🔹 Đang tính toán thể loại yêu thích của người dùng...")
high_ratings = df_ratings.filter(col("rating") >= 4)
high_ratings_with_genres = high_ratings.join(df_movies.select("movieId", "genres"), "movieId")
exploded = high_ratings_with_genres.withColumn("genre", explode(col("genres"))).select("userId", "genre")
user_preferred_genres = exploded.groupBy("userId").agg(collect_set("genre").alias("preferred_genres")).cache()

# FP-Growth cho thể loại phim
# Lý do: Tìm các mẫu thể loại phổ biến để hỗ trợ phân tích xu hướng, nhưng không sử dụng trực tiếp cho đề xuất phim mới.
logger.info("🔹 Đang áp dụng FP-Growth...")
df_transactions = df_movies.select("movieId", "genres").withColumnRenamed("genres", "items")
popular_movies = df_ratings.groupBy("movieId").agg(count("userId").alias("num_ratings"))
df_filtered_movies = df_transactions.join(popular_movies, "movieId").filter(col("num_ratings") > 50)
fp_growth = FPGrowth(itemsCol="items", minSupport=0.05, minConfidence=0.8)
model_fp = fp_growth.fit(df_filtered_movies)
rules = model_fp.associationRules.cache()

# ALS Collaborative Filtering
# Lý do: Huấn luyện mô hình ALS để sử dụng trong đề xuất kết hợp, nhưng không áp

logger.info("🔹 Đang huấn luyện ALS...")
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
          rank=50, maxIter=20, regParam=0.1, nonnegative=True, coldStartStrategy="drop")
train, test = df_ratings.randomSplit([0.8, 0.2], seed=42)
model_als = als.fit(train)
predictions = model_als.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse: float = evaluator.evaluate(predictions)
logger.info(f"ALS RMSE: {rmse}")


# Hàm đề xuất kết hợp
def recommend_movies_hybrid(user_id: int, n: int = 5, alpha: float = 0.7, diversity_weight: float = 0.3) -> Optional[
    DataFrame]:
    """
    Đề xuất phim cho người dùng hiện tại bằng cách kết hợp collaborative filtering (ALS) và content-based filtering.
    Lý do: Kết hợp cả hai phương pháp để cân bằng giữa sở thích cá nhân và xu hướng chung, đồng thời đa dạng hóa đề xuất.
    """
    cf_candidates = None
    movie_candidates = None
    hybrid_candidates = None
    try:
        user_df = spark.createDataFrame([(user_id,)], ["userId"])
        cf_recs = model_als.recommendForUserSubset(user_df, 50)
        cf_candidates = cf_recs.select(explode("recommendations").alias("rec")) \
            .selectExpr("rec.movieId as movieId", "rec.rating as cf_rating").cache()

        user_ratings = df_ratings.filter(col("userId") == user_id)
        user_movies = user_ratings.join(df_movies.select("movieId", "movie_embedding"), "movieId")
        user_movies = user_movies.withColumn("embedding_array", vector_to_array(col("movie_embedding")))
        embedding_size = 5
        avg_embedding = [avg(col("embedding_array")[i]).alias(f"avg_embedding_{i}") for i in range(embedding_size)]
        user_profile_df = user_movies.groupBy().agg(*avg_embedding).collect()[0]
        user_profile = [user_profile_df[f"avg_embedding_{i}"] for i in range(embedding_size)]
        user_profile_broadcast = spark.sparkContext.broadcast(user_profile)

        movie_candidates = df_movies.select("movieId", "movie_embedding")
        cosine_udf = udf(lambda x: float(np.dot(np.array(x.toArray()), np.array(user_profile_broadcast.value)) /
                                         (np.linalg.norm(x.toArray()) * np.linalg.norm(user_profile_broadcast.value))),
                         FloatType())
        movie_candidates = movie_candidates.withColumn("sim", cosine_udf(col("movie_embedding"))).cache()

        hybrid_candidates = cf_candidates.join(movie_candidates, "movieId") \
            .join(df_movies.select("movieId", "genres"), "movieId")
        hybrid_candidates = hybrid_candidates.withColumn(
            "hybrid_score",
            alpha * col("cf_rating") + (1 - alpha) * col("sim")
        ).cache()

        top_n = hybrid_candidates.orderBy(col("hybrid_score").desc()).limit(n * 2).collect()
        diversified_recs = diversify_recommendations(top_n, n, diversity_weight)
        return spark.createDataFrame(diversified_recs).join(df_movies.select("movieId", "title"), "movieId")
    except Exception as e:
        logger.error(f"Error in hybrid recommendation for user {user_id}: {str(e)}")
        return None
    finally:
        if cf_candidates is not None:
            cf_candidates.unpersist()
        if movie_candidates is not None:
            movie_candidates.unpersist()
        if hybrid_candidates is not None:
            hybrid_candidates.unpersist()


def diversify_recommendations(recs: List, n: int, diversity_weight: float) -> List:
    """
    Đa dạng hóa danh sách đề xuất để tránh trùng lặp thể loại.
    Lý do: Đảm bảo người dùng nhận được các đề xuất phong phú, không chỉ tập trung vào một thể loại.
    """
    selected = []
    remaining = list(recs)
    while len(selected) < n and remaining:
        if not selected:
            selected.append(remaining.pop(0))
        else:
            best_score = -float('inf')
            best_rec = None
            for rec in remaining:
                diversity_score = min(
                    [1 - len(set(rec["genres"]).intersection(set(sel["genres"]))) / len(rec["genres"]) for sel in
                     selected])
                score = (1 - diversity_weight) * rec["hybrid_score"] + diversity_weight * diversity_score
                if score > best_score:
                    best_score = score
                    best_rec = rec
            selected.append(best_rec)
            remaining.remove(best_rec)
    return selected


# Hàm đề xuất cho người dùng mới
def recommend_for_new_user(age: int, gender: str, occupation: int, n: int = 5) -> Optional[DataFrame]:
    """
    Đề xuất phim cho người dùng mới dựa trên cụm của họ.
    Lý do: Sử dụng cụm để tìm nhóm người dùng tương tự, từ đó đề xuất phim dựa trên sở thích chung của nhóm.
    """
    try:
        user_features = spark.createDataFrame([(age, gender, occupation)], ["age", "gender", "occupation"])
        user_features = pipeline_model.transform(user_features)
        user_features = vec_assembler.transform(user_features)
        cluster = model_kmeans.transform(user_features).select("cluster").collect()[0]["cluster"]
        cluster_ratings = df_merged.filter(col("cluster") == cluster).groupBy("movieId").agg(
            avg("rating").alias("avg_rating"))
        return cluster_ratings.join(df_movies.select("movieId", "title"), "movieId").orderBy(
            col("avg_rating").desc()).limit(n)
    except Exception as e:
        logger.error(f"Error in cold-start recommendation: {str(e)}")
        return None


# Hàm đề xuất bộ phim mới
def recommend_new_movie(new_title: str, new_genres_str: str, N: int = 10) -> List[tuple]:
    """
    Đề xuất một bộ phim mới cho các người dùng dựa trên content-based filtering.
    Cung cấp giải thích dựa trên các thể loại chung với các phim mà người dùng đã đánh giá cao.
    Lý do: Content-based filtering phù hợp với phim mới không có đánh giá, và giải thích dựa trên thể loại giúp người dùng hiểu rõ lý do đề xuất.
    """
    new_genres = new_genres_str.split("|")
    max_movieId = df_movies.select(max_func("movieId")).first()[0]
    new_movieId = max_movieId + 1
    new_movie_data = [(new_movieId, new_title, new_genres)]
    new_movie_df = spark.createDataFrame(new_movie_data, ["movieId", "title", "genres"])
    new_movie_with_embedding = model_w2v.transform(new_movie_df)
    new_movie_embedding = new_movie_with_embedding.select("movie_embedding").first()["movie_embedding"].toArray()
    broadcast_new_emb = spark.sparkContext.broadcast(new_movie_embedding)

    def cosine_sim(vec):
        vec_array = vec.toArray()
        new_emb = broadcast_new_emb.value
        dot_product = np.dot(vec_array, new_emb)
        norm_a = np.linalg.norm(vec_array)
        norm_b = np.linalg.norm(new_emb)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0.0

    cosine_udf = udf(cosine_sim, FloatType())
    user_similarities = user_profiles.withColumn("similarity", cosine_udf(col("profile_embedding")))
    top_users_df = user_similarities.orderBy(col("similarity").desc()).limit(N).select("userId", "similarity")
    top_users = top_users_df.collect()

    new_genres_array = array([lit(g) for g in new_genres])
    recommendations = []
    for user in top_users:
        userId = user["userId"]
        similarity = user["similarity"]
        user_high_ratings = df_ratings.filter((col("userId") == userId) & (col("rating") >= 4))
        user_high_movies = user_high_ratings.join(df_movies.select("movieId", "title", "genres"), "movieId")
        shared_genre_movies = user_high_movies.withColumn("common_genres",
                                                          array_intersect(col("genres"), new_genres_array)).filter(
            size("common_genres") > 0)
        top_shared = shared_genre_movies.orderBy(col("rating").desc()).limit(2).select("title").collect()
        top_titles = [row["title"] for row in top_shared]

        if top_titles:
            explanation = f"vì bạn đã thích các phim như {', '.join(top_titles)} có cùng thể loại"
        else:
            explanation = "dựa trên sở thích chung của bạn"

        recommendations.append((userId, similarity, explanation))

    return recommendations


# Hiển thị kết quả
def show_hybrid_recommendations(user_id: int, n: int = 5, alpha: float = 0.7) -> None:
    """
    Hiển thị đề xuất kết hợp cho người dùng.
    Lý do: Cung cấp giao diện thân thiện để người dùng xem kết quả đề xuất.
    """
    logger.info(f"🔹 Đề xuất kết hợp cho người dùng {user_id}:")
    recs = recommend_movies_hybrid(user_id, n, alpha)
    if recs:
        recs.select("movieId", "title", "hybrid_score").show(truncate=False)


def show_cold_start_recommendations(age: int, gender: str, occupation: int, n: int = 5) -> None:
    """
    Hiển thị đề xuất cho người dùng mới.
    Lý do: Giúp người dùng mới nhận được đề xuất dựa trên đặc điểm nhân khẩu học.
    """
    logger.info(f"🔹 Đề xuất cho người dùng mới (age={age}, gender={gender}, occupation={occupation}):")
    recs = recommend_for_new_user(age, gender, occupation, n)
    if recs:
        recs.show(truncate=False)


# Kiểm tra hệ thống
show_hybrid_recommendations(1, n=5, alpha=0.7)
show_cold_start_recommendations(25, "M", 12, n=5)

# Ví dụ đề xuất phim mới
new_title = "Inception 2"
new_genres_str = "Action|Sci-Fi|Thriller"
recs = recommend_new_movie(new_title, new_genres_str, N=3)
for rec in recs:
    print(f"Người dùng {rec[0]} với độ tương đồng {rec[1]}: Đề xuất {rec[2]}")

logger.info("✅ Tối ưu hóa hoàn tất!")
spark.stop()