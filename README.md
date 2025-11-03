# Data Science Methodologies: A 3-Project Comparison

This repository is an academic project demonstrating and comparing three foundational data science methodologies: **CRISP-DM**, **KDD**, and **SEMMA**.

To properly showcase the unique focus of each framework, this repository contains three distinct, end-to-end projects. Each project is self-contained in its own directory, complete with a detailed "Medium-style" article (in its `README.md`) and a full Google Colab notebook (`.ipynb`) implementation.

## 🚀 The Projects

This repository is organized into three subdirectories:

| Methodology | Project & Dataset | Problem Type | Directory |
| :--- | :--- | :--- | :--- |
| **CRISP-DM** | 1. Customer Churn Prediction (Telco Churn) | **Binary Classification** | [`./project_1_crisp_dm_churn/`](./project_1_crisp_dm_churn/) |
| **KDD** | 2. House Price Driver Discovery (Ames Housing) | **Regression** | [`./project_2_kdd_house_prices/`](./project_2_kdd_house_prices/) |
| **SEMMA** | 3. Dry Bean Classification (Dry Bean Dataset) | **Multi-Class Classification** | [`./project_3_semma_bean_classification/`](./project_3_semma_bean_classification/) |

---

## 🧭 Methodology Overview

A brief summary of the three frameworks explored:

* ### 1. CRISP-DM (Cross-Industry Standard Process for Data Mining)
    * **Focus:** **Business-centric.** It is a 6-phase cyclical process that begins with "Business Understanding" and ends with "Deployment," ensuring the final model delivers tangible business value.
    * **Project:** We used it to solve a classic business problem: identifying at-risk customers to reduce churn.



* ### 2. KDD (Knowledge Discovery in Databases)
    * **Focus:** **Knowledge-centric.** It is a 5-step foundational process for "discovering" non-trivial, useful knowledge. Its goal is to *find and interpret patterns*.
    * **Project:** We used it as a research tool to discover the key factors that influence a home's sale price.



[Image of the KDD process flow diagram]


* ### 3. SEMMA (Sample, Explore, Modify, Model, Assess)
    * **Focus:** **Technique-centric.** It is a 5-step "workbench" workflow for the data scientist. It provides a logical, hands-on sequence for building a model.
    * **Project:** We used it as a technical guide to build a robust multi-class classifier from raw data.



[Image of the SEMMA process flow diagram]


---

## 🛠️ Technology Stack

All projects were developed in Python 3 using Google Colab.

* **Core Libraries:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (`sklearn`), Imbalanced-learn (`imblearn`)
* **Visualization:** Matplotlib, Seaborn
* **Data Handling:** `openpyxl` (for Excel files)
