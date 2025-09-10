# -*- coding: utf-8 -*-
"""price_prediction_pipeline.py

Avocado Price Prediction using PySpark MLlib
"""

# Import Spark libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, to_date, day, month, year, dayofweek, when

# ML imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ------------------------------
# 1. Initialize Spark Session
# ------------------------------
spark = SparkSession.builder \
    .appName("Avocado Price Prediction") \
    .getOrCreate()

# ------------------------------
# 2. Load dataset
# ------------------------------
df = spark.read.csv('/kaggle/input/avocado-prices/avocado.csv', header=True, inferSchema=True)
df.show(5)  # preview first 5 rows
df.printSchema()  # check data types

# ------------------------------
# 3. Check missing values
# ------------------------------
df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()
print(f"Total rows: {df.count()}")

# ------------------------------
# 4. Data Cleaning
# ------------------------------
df = df.drop('_c0')  # remove unnecessary column
# Rename PLU columns for clarity
df = df.withColumnRenamed("4046", "PLU_4046") \
       .withColumnRenamed("4225", "PLU_4225") \
       .withColumnRenamed("4770", "PLU_4770")

# ------------------------------
# 5. Feature Engineering
# ------------------------------
# Convert Date to separate features
df = df.withColumn("Date", to_date("Date", "yyyy-MM-dd"))
df = df.withColumn('Day', day(col('Date'))) \
       .withColumn('Month', month(col('Date'))) \
       .withColumn('year', year(col('Date'))) \
       .withColumn('Day_of_week', dayofweek(col('Date')))
df = df.drop('Date')

# Create ratio features
df = df.withColumn("PLU_4046_Ratio", col("PLU_4046") / col("Total Volume")) \
       .withColumn("PLU_4225_Ratio", col("PLU_4225") / col("Total Volume")) \
       .withColumn("PLU_4770_Ratio", col("PLU_4770") / col("Total Volume")) \
       .withColumn("SmallBag_Ratio", col("Small Bags") / col("Total Bags")) \
       .withColumn("LargeBag_Ratio", col("Large Bags") / col("Total Bags")) \
       .withColumn("XLargeBag_Ratio", col("XLarge Bags") / col("Total Bags"))

# Create Season feature
df = df.withColumn("Season",
    when(col("Month").isin(12, 1, 2), "Winter")
    .when(col("Month").isin(3, 4, 5), "Spring")
    .when(col("Month").isin(6, 7, 8), "Summer")
    .otherwise("Fall")
)

# Drop rows with missing values
df = df.dropna()

# ------------------------------
# 6. Encode categorical variables
# ------------------------------
categorical_cols = ['type', 'region', 'Season']
indexers = [StringIndexer(inputCol=col, outputCol=col + '_encoded') for col in categorical_cols]
encoded_cols = [col + "_encoded" for col in categorical_cols]

# ------------------------------
# 7. Assemble features
# ------------------------------
numeric_cols = [
    'Total Volume', 'PLU_4046', 'PLU_4225', 'PLU_4770',
    'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags',
    'year', 'Day', 'Month', 'Day_of_week',
    'PLU_4046_Ratio', 'PLU_4225_Ratio', 'PLU_4770_Ratio',
    'SmallBag_Ratio', 'LargeBag_Ratio', 'XLargeBag_Ratio'
]

assembler_input = numeric_cols + encoded_cols
assembler = VectorAssembler(inputCols=assembler_input, outputCol='assembled_features')

# Scale features between 0 and 1
scaler = MinMaxScaler(inputCol='assembled_features', outputCol='features')

# ------------------------------
# 8. Train-Test Split
# ------------------------------
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1)

# ------------------------------
# 9. Random Forest Regressor
# ------------------------------
rf = RandomForestRegressor(
    featuresCol='features', 
    labelCol='AveragePrice', 
    numTrees=50, 
    maxDepth=10
)

# ------------------------------
# 10. Create Pipeline
# ------------------------------
pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])

# Fit model
model = pipeline.fit(train_data)

# Make predictions
predictions = model.transform(test_data)
predictions.select('AveragePrice', 'prediction').show(10)

# ------------------------------
# 11. Evaluate Model
# ------------------------------
evaluator_rmse = RegressionEvaluator(labelCol='AveragePrice', predictionCol='prediction', metricName='rmse')
evaluator_mae = RegressionEvaluator(labelCol='AveragePrice', predictionCol='prediction', metricName='mae')
evaluator_r2 = RegressionEvaluator(labelCol='AveragePrice', predictionCol='prediction', metricName='r2')

rmse_rf = evaluator_rmse.evaluate(predictions)
mae_rf = evaluator_mae.evaluate(predictions)
r2_rf = evaluator_r2.evaluate(predictions)

print(f"RF RMSE: {rmse_rf:.4f}")
print(f"RF MAE: {mae_rf:.4f}")
print(f"RF R2: {r2_rf:.4f}")
