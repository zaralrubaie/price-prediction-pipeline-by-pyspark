# Avocado Price Prediction Pipeline

## Overview

This project demonstrates building a scalable and reproducible machine learning pipeline using Apache Spark MLlib. The pipeline automates the full workflow from raw data preprocessing to model training and evaluation, ensuring consistent and maintainable results.

The goal is to predict avocado prices based on historical sales data, seasonal information, and various bag and PLU ratios.

---

## Dataset

The dataset includes the following key columns:

- `Total Volume`, `PLU_4046`, `PLU_4225`, `PLU_4770` — Sales volume by PLU
- `Total Bags`, `Small Bags`, `Large Bags`, `XLarge Bags` — Bag size quantities
- `type` — Avocado type (conventional or organic)
- `region` — Sales region
- `AveragePrice` — Target variable
- `Date` — Transaction date (used to create day, month, year, and season features)

---

## Pipeline Components

1. **Data Preprocessing**
   - Handles missing values
   - Renames columns and converts `Date` to day, month, year, and day-of-week
   - Creates bag and PLU ratios
   - Assigns seasonal labels based on the month

2. **Feature Engineering**
   - Encodes categorical variables using `StringIndexer`
   - Combines numeric and encoded categorical features into a single vector using `VectorAssembler`
   - Scales features with `MinMaxScaler`

3. **Model Training**
   - Trains a `RandomForestRegressor` to predict `AveragePrice`
   - Uses a PySpark pipeline to ensure modularity and reproducibility

4. **Model Evaluation**
   - Metrics: RMSE, MAE, R²
   - Produces predictions for test data
   - Outputs can be saved as CSV for further analysis

---

## Project Structure

```text
avocado_price_pipeline/
│
├── price_prediction_pipeline.py      # Main PySpark script with full workflow
├── model_summary.txt                 # Summary of evaluation metrics (RMSE, MAE, R²)
├── README.md                         # Project documentation and pipeline explanation
├── requirements.txt                  # Required Python libraries
````
## How to Run
1. Clone the repository:
````
git clone https://github.com/zaralrubaie/avocado_price_pipeline.git
````
2. Install dependencies:
````
pip install -r requirements.txt
````
## Results

The pipeline generates:

- Predictions on test data (predictions.csv)
- Model evaluation metrics (model_summary.txt) including RMSE, MAE, and R²
- Scalable and reproducible workflow for large datasets

## License

This project is licensed under the MIT License. See the LICENSE file for details.


