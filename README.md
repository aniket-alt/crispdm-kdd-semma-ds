# Data Science Methodologies (CRISP-DM, KDD, SEMMA)

This repository contains the project artifacts for an assignment comparing three core data science methodologies. The project focuses on a single use case, **Predicting Customer Churn**, and analyzes it through the lens of CRISP-DM, KDD, and SEMMA.


## Article 2: The KDD Process for Discovering Churn Patterns

KDD, or Knowledge Discovery in Databases, is a foundational process for extracting non-trivial, implicit, and potentially useful information from data. We will reframe our "Streamify" churn project using its 5-step framework.

### 1. Selection

The first step in KDD is to **select** the target data. Our goal is to discover knowledge about *why customers churn*.
* **Data Target:** We identified the "Streamify" customer database as our data source.
* **Variable Selection:** Based on our initial hypothesis, we selected a subset of data relevant to our goal. This included customer demographics (`Age`, `Location`), usage metrics (`AvgWatchTimePerWeek`, `DaysSinceLastLogin`), and account history (`Tenure`, `SupportTicketsLogged`, `LatePayments`).
* **Target Variable:** We defined our target of interest as the `Churn` (1 or 0) column, which represents the "knowledge" we want to predict.

### 2. Pre-processing

After selecting the raw data, we must perform **pre-processing** to clean it.
* We handled missing data by performing **mean imputation** for the `Age` column.
* We identified and corrected data entry errors (outliers) in the `AvgWatchTimePerWeek` column by **capping** unrealistic values.

### 3. Transformation

With a clean dataset, the next step is **transformation**. The goal is to transform the data into a format suitable for data mining algorithms.
* **Feature Construction:** We engineered new, more powerful features from existing ones. The most impactful was our `EngagementScore` (a ratio of watch time to login recency) and `HasSupportIssues` (a binary flag). These new features aimed to create stronger predictive signals.
* **Formatting:** We converted categorical text data (like `SubscriptionPlan`) into a numerical format using **One-Hot Encoding**, making it understandable to a machine learning algorithm.

### 4. Data Mining

This is the core step where algorithms are applied to the transformed data to extract hidden patterns.
* **Task:** Our task was **classification** (to predict the `Churn` variable).
* **Algorithm Selection:** We applied several data mining algorithms, including **Logistic Regression**, **Decision Trees**, and **Random Forest**.
* **Pattern Discovery:** The **Random Forest** algorithm was applied to the data and proved most effective. It "discovered" the complex, non-linear relationships between our features (like `Tenure`, `EngagementScore`, and `HasSupportIssues`) that best predicted a churn event.

### 5. Interpretation & Evaluation

The final step is to interpret the patterns and evaluate the "knowledge" we've discovered.
* **Knowledge Found:** The discovered pattern (or "knowledge") was that a combination of low tenure, a low `EngagementScore`, and having *any* support tickets logged was a powerful predictor of churn.
* **Evaluation:** We evaluated this "knowledge" by testing the Random Forest model on unseen data. The model was highly effective, with **91% accuracy** and **84% recall**. This confirms that the patterns we found are not just noise but are valid, actionable insights that can be used to meet the business goal.

---
---

## Article 3: A Data Miner's Workflow for Churn Using SEMMA

The SEMMA (Sample, Explore, Modify, Model, Assess) framework is a logical sequence of steps for a data scientist to follow when building a predictive model. We will frame our "Streamify" churn project using this workflow.

### 1. Sample

For many enterprise projects, data is too large to work with directly (billions of rows). The first step is to **sample** a representative subset.
* **Our Project:** Our 100,000-row dataset was manageable. However, for the modeling phase, we *did* sample our data by splitting it into a **80% Training Set** and a **20% Test Set**. This is a form of sampling, as we build our model on a subset of the full data.

### 2. Explore

Before building, we **explore** the data to understand its trends, outliers, and variable relationships.
* **Our Project:** This directly maps to our **Data Understanding** phase. We used visualizations (histograms, scatter plots) to find:
    * **Outliers:** Impossible `AvgWatchTimePerWeek` values.
    * **Missing Data:** 15% missing `Age` values.
    * **Key Insight:** We discovered the critical **class imbalance** in our `Churn` variable (90% "No," 10% "Yes"), which we knew would heavily influence our modeling strategy.
### 3. Modify

Based on our exploration, we **modify** the data to prepare it for modeling. This is a very large step that includes cleaning, transforming, and creating new features.
* **Our Project:** This step directly maps to our **Data Preparation** phase.
    * **Cleaning:** We handled missing `Age` values via imputation and capped the `AvgWatchTimePerWeek` outliers.
    * **Transforming:** We used **One-Hot Encoding** for categorical data like `SubscriptionPlan`.
    * **Feature Engineering:** We created the `EngagementScore` and `HasSupportIssues` features to add more predictive power.
    * **Addressing Imbalance:** We prepared to use **SMOTE** on our training data, a key modification to handle the 90/10 class imbalance.

### 4. Model

With a clean, modified dataset, we now **model** the data. This step involves selecting and running various algorithms to find the best predictive model.
* **Our Project:** This directly maps to our **Modeling** phase.
    * We selected three classification algorithms: **Logistic Regression**, **Decision Tree**, and **Random Forest**.
    * We trained (or "fit") all three models on our 80% training data subset, specifically using the SMOTE-balanced version to ensure the models learned the patterns of the minority "Churn" class.

### 5. Assess

Finally, we **assess** the models to determine which one is the most effective and if it meets our goals.
* **Our Project:** This directly maps to our **Evaluation** phase.
    * We used the 20% "lockbox" **Test Set** (which the models had never seen) to grade our three models.
    * We used a **Confusion Matrix** and key metrics like **Precision**, **Recall**, and **F1-Score** for our assessment, as "Accuracy" alone was misleading.
    * **Assessment Result:** The **Random Forest** model was assessed as the clear winner, with 91% accuracy and 84% recall, successfully identifying the vast majority of *actual* churners.

---
---
