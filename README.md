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
