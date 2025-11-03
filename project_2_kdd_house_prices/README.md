# Project 2: Discovering House Price Drivers with the KDD Process

This project uses the **Knowledge Discovery in Databases (KDD)** methodology to analyze the Ames Housing dataset. The goal is not just to build a model, but to "discover" and "interpret" the key factors that influence a home's final sale price.

Video tutorial: https://drive.google.com/drive/folders/1iBKyp4gBYIiL5SZmKTl_jErzjbv4tXNZ?usp=sharing

### 1. Selection

The first step is to **select** the target data and variables for our knowledge discovery task.

* **Data Source:** We are using the "Ames Housing Dataset" from a 2017 Kaggle competition. We will focus on the `train.csv` file, which contains 1460 property listings.
* **Goal:** To discover knowledge about what features most significantly impact a home's `SalePrice`.
* **Target Variable:** Our target for discovery is the `SalePrice` column.
* **Initial Feature Set:** We will initially **select all 80 available features** (e.g., `OverallQual`, `GrLivArea`, `Neighborhood`, `GarageCars`) to cast a wide net, allowing the data mining step to find non-obvious patterns.

### 2. Pre-processing

This phase involves "cleaning" the raw selected data to handle imperfections. The Ames dataset is famously "messy" and requires significant pre-processing.

* **Handling Missing Data (Categorical):** Many features (like `PoolQC`, `MiscFeature`, `Alley`, `Fence`) have a large number of `NaN` (Not a Number) values. In the data dictionary, `NaN` for these features actually means "No Pool" or "No Alley." We will **impute** these `NaN` values with the string "None".
* **Handling Missing Data (Numerical):** Other features (like `LotFrontage` or `GarageYrBlt`) have true missing values. We will **impute** these using the **median** value of their respective columns to avoid skewing the data.
* **Handling Outliers:** We will check for and remove a few extreme outliers (e.g., houses with over 4000 sq. ft. but an unusually low price) that could negatively impact the data mining step.

### 3. Transformation

With a clean dataset, we now **transform** it into a format suitable for the data mining algorithms.

* **Feature Engineering:** We will create new, more powerful features from existing ones. For example:
    * `TotalSqFoot` = `1stFlrSF` + `2ndFlrSF` + `TotalBsmtSF`
    * `TotalBath` = `FullBath` + (0.5 * `HalfBath`) + `BsmtFullBath` + (0.5 * `BsmtHalfBath`)
    * `HouseAge` = `YrSold` - `YearBuilt`
* **Data Type Conversion:** Many "categorical" features are stored as numbers (e.g., `MSSubClass`). We will convert these to strings so they are treated as categories.
* **Encoding Categorical Data:** We will use **One-Hot Encoding** to convert text-based categorical features (like `Neighborhood` or `Condition1`) into a binary (0/1) format that our algorithms can understand.
* **Normalizing Numerical Data:** We will **log-transform** our target variable `SalePrice`, as its distribution is highly skewed. This transformation makes the pattern linear and easier for the model to learn.

### 4. Data Mining

This is the core "discovery" step where we apply algorithms to our transformed data to find patterns. Our task is **regression** (predicting a continuous value).

* **Algorithm 1 (Baseline):** We will use **Ridge Regression**. This is a fast, linear model that is good at handling a large number of features and preventing overfitting.
* **Algorithm 2 (Advanced):** We will use a **Random Forest Regressor**. This powerful "ensemble" model is excellent at finding complex, non-linear patterns. Its key benefit for KDD is that it can **rank all features by their importance**, which is central to our "knowledge discovery" goal.
* **Training/Test Split:** We will split our data (80% training, 20% testing) to ensure we are testing our patterns on unseen data.

### 5. Interpretation & Evaluation

The final step is to **evaluate** the patterns we found and **interpret** them into human-readable "knowledge."

* **Evaluation:** We will "grade" our data mining models using two key metrics:
    * **R-squared ($R^2$):** How much of the change in `SalePrice` can our model's features explain? (e.g., 0.90 = 90% explained).
    * **Root Mean Squared Error (RMSE):** The average "error" of our model's price prediction, in dollars.
* **Interpretation (The "Knowledge"):** This is the most important part of KDD. By looking at the `feature_importance_` attribute of our trained Random Forest model, we can extract the "knowledge" we were looking for.
    * **Discovered Knowledge:** We will find that the top 5 most important drivers of a home's sale price are:
        1.  **`OverallQual`:** The overall quality and finish of the house.
        2.  **`TotalSqFoot`:** The total square footage of the house.
        3.  **`Neighborhood`:** The location of the house.
        4.  **`GarageCars`:** The size of the garage in car capacity.
        5.  **`HouseAge`:** How recently the home was built.

This discovered knowledge is the final, actionable output of the KDD process.
