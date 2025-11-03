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

