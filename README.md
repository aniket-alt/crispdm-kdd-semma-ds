# Data Science Methodologies (CRISP-DM, KDD, SEMMA)

This repository contains the project artifacts for an assignment comparing three core data science methodologies. The project focuses on a single use case, **Predicting Customer Churn**, and analyzes it through the lens of CRISP-DM, KDD, and SEMMA.

---

## Article 1: A Step-by-Step Guide to Predicting Customer Churn with CRISP-DM

The CRISP-DM (Cross-Industry Standard Process for Data Mining) model is a 6-phase cyclical process that provides a robust framework for managing data science projects.

### 1. Business Understanding

**The Problem:** Our project begins with a common but costly business problem. Our client, a fictional subscription-based streaming service called "Streamify," has seen a 20% increase in customer churn (customers canceling their memberships) over the last quarter. This is expensive, as acquiring a new customer costs 5x more than retaining an existing one.

**The Business Goal:** The company's goal is straightforward: **reduce customer churn by at least 15%** in the next six months.

**Our Project's Role:** It's impossible to stop all churn, but we can target *preventable* churn. Our project's objective is to build a system that can **identify which *current* customers are at a high risk of churning** within the next 30 days.

**Defining Success:**
* **Business Success:** The marketing team successfully uses our tool to launch a retention campaign (e.g., offering discounts to high-risk users) that results in a measurable decrease in the churn rate.
* **Data Mining Success:** We will build a predictive model (a binary classifier) that achieves at least **85% accuracy**. More importantly, we must have a high **recall** (also called sensitivity), meaning we want to correctly identify as many of the *actual* churners as possible, even if we accidentally mislabel a few safe customers.

### 2. Data Understanding

With our business goal set, we now need to acquire and examine the raw data. This phase is about understanding what we have *before* we start cleaning and modeling.

**Data Collection:**
For this project, we'll assume we have been granted read-only access to the "Streamify" production database. We run an SQL query to pull a dataset of all customers who have been active in the last 2 years.

**Data Description (Data Dictionary):**
Our dataset contains 100,000 rows (one per customer) and the following 13 columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | String | Unique identifier for each customer. |
| `Age` | Integer | Customer's age in years. |
| `Gender` | String | "Male", "Female", "Non-binary", "Prefer not to say". |
| `Location` | String | Customer's registered city/state. |
| `SubscriptionPlan` | String | "Basic", "Standard", "Premium". |
| `MonthlyCost` | Float | The amount ($) the customer is billed each month. |
| `Tenure` | Integer | How many months the customer has been with "Streamify". |
| `AvgWatchTimePerWeek` | Float | Average hours watched per week in the last 3 months. |
| `DaysSinceLastLogin` | Integer | Number of days since the customer's last login. |
| `SupportTicketsLogged` | Integer | Number of support tickets filed in the last 6 months. |
| `PaymentMethod` | String | "Credit Card", "PayPal", "Gift Card". |
| `LatePayments` | Integer | Number of late payments in the last 12 months. |
| `Churn` | Integer | **(Our Target Variable)** 1 = Churned, 0 = Not Churned. |

**Initial Data Exploration & Quality Report:**
After loading the data, we perform an initial exploratory data analysis (EDA) and find several key quality issues that must be addressed:

* **Missing Values:** `Age` has 15% missing values. `Gender` has 8% "Prefer not to say" (which is a form of missing data).
* **Outliers:** `AvgWatchTimePerWeek` has some impossible values, including several users with over 200 hours/week.
* **Inconsistent Data:** The `Location` column is a free-text field, resulting in entries like "NY," "New York," and "NYC" that all mean the same thing.
* **Target Variable Imbalance:** We find that our dataset is highly **imbalanced**. 90% of the rows are `Churn = 0` (Not Churned) and only 10% are `Churn = 1` (Churned). This is a critical discovery, as it will make it hard for a naive model to learn what a "churner" looks like.

### 3. Data Preparation

This phase (also called "data wrangling" or "data munging") is where we execute the cleanup plan informed by our Data Understanding. This is often the most time-consuming step (up to 80% of a project) but is critical for building an accurate model.

Our goal is to create a final, clean "feature table." Here are the "recipes" we used:

**1. Handling Missing Values:**
* **`Age` (15% missing):** Deleting 15% of our data would be wasteful. Instead, we use **mean imputation**. We calculate the mean age of all *other* customers (e.g., 38.5 years) and fill in the missing `Age` values with this mean.
* **`Gender` (8% "Prefer not to say"):** This isn't "missing," but it's not a useful category. We group "Prefer not to say" and any other null values into a new category: "Other."

**2. Correcting Outliers:**
* **`AvgWatchTimePerWeek` (impossible values):** We found values over 168 (24*7). These are data entry errors. We handle this by **capping** the outliers. We decide that any value over 80 hours/week (the 99th percentile) is unrealistic and will be set to 80.

**3. Standardizing Categorical Data:**
* **`Location` (inconsistent entries):** This column is too messy to fix manually (e.g., "NY", "NYC", "New York"). To make it useful, we create a mapping to group major markets (e.g., "New York," "California," "Texas") and consolidate all other entries into a single "Other" category.
* **`SubscriptionPlan` & `PaymentMethod`:** Machines don't understand text like "Basic" or "PayPal." We use **One-Hot Encoding** to convert these columns. This process creates new binary (0 or 1) columns for each category (e.g., `Plan_Basic`, `Plan_Standard`, `Plan_Premium`).

**4. Feature Engineering (The "Magic"):**
This is where we use our domain knowledge to create *new* features that will give our model better predictive signals:
* **`TenureInYears`:** We convert `Tenure` (in months) to years to make it more interpretable.
* **`EngagementScore`:** This is a powerful new feature. We create a score by dividing `AvgWatchTimePerWeek` by (`DaysSinceLastLogin` + 1). A high score means they watch a lot *and* were active recently. A low score means they haven't logged in for a while, even if they used to watch a lot—a strong churn indicator.
* **`HasSupportIssues`:** We convert `SupportTicketsLogged` into a simple binary (0 or 1) feature. We hypothesize that *logging any ticket at all* (1 or more) is a key predictor of churn, more so than the specific *number* of tickets.

**5. Handling Class Imbalance (For Modeling):**
* **`Churn` (90% "No," 10% "Yes"):** If we train a model on this, it will just learn to "always predict No" and be 90% accurate. To fix this, we will use a technique called **SMOTE (Synthetic Minority Over-sampling TEchnique)** *only on our training data*. This process intelligently creates new, "synthetic" examples of churners to give the model a balanced 50/50 dataset to learn from.

After these steps, we have a clean, complete, and feature-rich dataset ready for the next phase.

### 4. Modeling

Now that our data is prepared, we can begin the modeling phase. Our business problem is to predict a "Yes/No" outcome (Churn/No Churn), which is a **binary classification** task.

**1. Data Splitting:**
First, we split our entire clean dataset into two parts:
* **Training Set (80% of data):** The model will learn the patterns of churn from this data.
* **Test Set (20% of data):** This data is kept in a "lockbox" and is *only* used at the very end to grade the model's performance on unseen data. This prevents the model from "cheating."

```python
# We use scikit-learn's train_test_split function
from sklearn.model_selection import train_test_split

X = clean_data.drop('Churn', axis=1) # All columns except our target
y = clean_data['Churn']              # Only the target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 5. Evaluation

In the modeling phase, we built three models. Now, we must evaluate their performance on the unseen `X_test` data to see which one is best and if it meets our business goals.

**Key Evaluation Metrics:**
Our dataset is imbalanced (90% "No Churn," 10% "Churn"). This means **Accuracy** is a misleading metric. (A model that just "guesses No" every time would be 90% accurate but 100% useless).

We must use a **Confusion Matrix** to see the *types* of errors our model makes.



Based on this, our key metrics are:
* **Precision:** Of all the customers our model *predicted* would churn, what percentage *actually* churned? (We don't want to waste money giving discounts to "safe" customers).
* **Recall (Our Top Priority):** Of all the customers who *actually* churned, what percentage did our model successfully *catch*? (This was our business goal: find as many at-risk users as possible).
* **F1-Score:** The harmonic mean of Precision and Recall. A good all-around score for imbalanced datasets.

**Model Performance Results:**
We ran all three models on the `X_test` set. Here are their scores:

| Model | Accuracy | Precision | Recall (Our Goal) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 88% | 65% | 70% | 67% |
| **Decision Tree** | 86% | 61% | 72% | 66% |
| **Random Forest** | **91%** | **78%** | **84%** | **81%** |

**Evaluation & Model Selection:**
The **Random Forest** is the clear winner.
* It has the highest overall accuracy (91%).
* It has good Precision (78%), meaning when it flags someone, it's right 3 out of 4 times.
* Most importantly, it has an excellent **Recall of 84%**. This means it successfully "catches" 84% of all customers who were *actually* about to churn.

**Reviewing Business Goals:**
Let's check back with our goals from Step 1.
* **Data Mining Success (Goal: >85% accuracy):** We achieved **91% accuracy**. **(Met)**
* **Data Mining Success (Goal: High Recall):** We achieved **84% Recall**, successfully identifying the vast majority of at-risk users. **(Met)**
* **Business Success (Goal: Reduce churn):** Our model is a strong candidate to help the business achieve this goal. It provides the marketing team with a highly accurate list of users to target for their retention campaign.

The model is approved. It is effective, accurate, and directly aligned with the business objective.

### 6. Deployment

We have an evaluated, high-performing model (our Random Forest) that is approved by the business. The final step is to integrate it into the company's operations.

**Deployment Plan:**
We propose a **phased batch-scoring system**.
1.  **Initial Rollout (Simple Batch):** We will not start with a complex, real-time API. Our model will be packaged and deployed on a secure cloud server.
2.  **Nightly Scoring:** A script will run every night at 1 AM. This script will:
    * Pull the latest data for all *current* subscribers (e.g., `DaysSinceLastLogin`, `AvgWatchTimePerWeek`).
    * Use our saved Random Forest model to generate a fresh "churn_risk_score" (a probability from 0.0 to 1.0) for every single user.
    * Save these scores to a new table in the company database.
3.  **Actionable Dashboard:** This new score table will power a dashboard for the Marketing team. They can log in each morning and see a list of the "Top 500 Highest-Risk Customers."
4.  **Integration with Marketing Tools:** This list will then be automatically fed into the company's email marketing tool, which will send a targeted "We miss you! Here's a 25% discount" offer to that high-risk group.

**Monitoring & Maintenance (The "Cycle"):**
Deployment is *not* the end. The CRISP-DM model is a cycle for a reason.
* **Model Monitoring:** We must continuously monitor our model's performance. Is its accuracy (Recall) dropping over time?
* **Concept Drift:** Customer behavior changes. The patterns that predict churn today might not be the same patterns that predict churn in six months.
* **Retraining Plan:** We will schedule our model to be automatically retrained on the *newest* customer data (e.g., the last 3 months) on a quarterly basis. This ensures the model stays "fresh" and adapts to new user trends, starting the CRISP-DM cycle all over again.

---
---

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
