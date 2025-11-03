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

* ### 3. Data Preparation

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
