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
