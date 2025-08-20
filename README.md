# ML Pipeline by Apache Spark

## Overview

This project demonstrates building a scalable and reproducible machine learning pipeline using Apache Spark MLlib. The pipeline automates the full workflow from raw data preprocessing to model training and evaluation, ensuring consistent and maintainable results.

---

## Pipeline Components

- **Data Preprocessing:**  
  Encodes categorical variables using `StringIndexer`, and scales features with `MinMaxScaler`.

- **Feature Engineering:**  
  Combines numeric and encoded categorical features into a single vector using `VectorAssembler`.

- **Model Training:**  
  Trains a regression model (Random Forest Regressor) to predict the target variable.

- **Model Evaluation:**  
  Evaluates model performance using RMSE, MAE, and R² metrics to measure accuracy and goodness of fit.

---
## Key aspects include:
- Clear separation of preprocessing and modeling steps:
Encoding categorical variables into numeric form, assembling features, scaling numerical values, and applying machine learning models are organized as distinct pipeline stages. This modularity improves maintainability and reusability.

- Robust data preprocessing:
Proper handling of categorical features and numerical scaling prepares the data effectively for learning algorithms, reducing bias and improving convergence.

- Feature engineering:
Combining both original numeric features and encoded categorical features into a single feature vector suitable for Spark ML models.

- Model training and evaluation:
The pipeline facilitates training regression models and generating predictions consistently on new data. Evaluation metrics can be applied to assess model performance objectively.

- Scalability and reproducibility:
- Leveraging Spark’s distributed processing and pipeline abstraction supports large datasets and repeatable workflows, ideal for production environments or collaborative projects.

## Example Pipeline Code

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor

categorical_cols = ['type', 'region', 'Season']
indexers = [StringIndexer(inputCol=col, outputCol=col + '_encoded') for col in categorical_cols]

numeric_cols = ['Total Volume', 'PLU_4046', 'PLU_4225', 'PLU_4770', 'Total Bags']

assembler = VectorAssembler(
    inputCols=numeric_cols + [col + '_encoded' for col in categorical_cols],
    outputCol='assembled_features'
)

scaler = MinMaxScaler(inputCol='assembled_features', outputCol='features')

rf = RandomForestRegressor(featuresCol='features', labelCol='AveragePrice')

pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])

model = pipeline.fit(train_data)
predictions = model.transform(test_data)
```
