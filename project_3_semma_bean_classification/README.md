# Project 3: Classifying Dry Beans with the SEMMA Workflow

This project uses the **SEMMA (Sample, Explore, Modify, Model, Assess)** framework to build a multi-class classification model. The goal is to follow this technical workflow to accurately classify a dry bean into one of 7 distinct types based on its shape and size.

### 1. Sample

The SEMMA process begins with **sampling** the data. The goal is to pull a representative, manageable subset of data from a large repository.

* **Our Data:** We are using the "Dry Bean Dataset," which contains 13,611 records—one for each bean.
* **Our Sampling Strategy:** The dataset is already a manageable size. Therefore, our primary sampling step will be to partition the full dataset into three distinct samples:
    * **Training Sample (70%):** The primary subset used to teach our models.
    * **Validation Sample (15%):** A subset used to tune model parameters (hyperparameters).
    * **Test Sample (15%):** A final, "lockbox" subset used *only once* at the end to **Assess** the chosen model's real-world performance.

### 2. Explore

With our samples defined, we move to the **explore** phase. This step is about visualizing the data to understand its properties, find relationships, and identify any problems.

* **Target Variable:** We first explore our target, `Class`. We find that it is a **multi-class** variable with 7 unique types of beans. We also check the class distribution and find it's **imbalanced**—some bean classes have far more samples than others.
* **Feature Exploration:** We explore the 16 input features (e.g., `Area`, `Perimeter`, `MajorAxisLength`). We use histograms and find that their scales are wildly different (e.g., `Area` is in the 10,000s, while `Eccentricity` is between 0 and 1).
* **Key Discovery:** This difference in scales is a critical discovery. It tells us that our "Modify" step **must** include feature scaling, as models like K-Nearest Neighbors (KNN) or Logistic Regression will fail if one feature's scale dominates all the others.

### 3. Modify

The **modify** phase is where we clean, transform, and prepare the data for modeling, based on our "Explore" discoveries.

* **Encoding the Target:** Our target variable `Class` is text (e.g., "BARBUNYA", "BOMBAY"). We use a **Label Encoder** to "modify" this text into numbers (0, 1, 2...) that a machine can understand.
* **Handling Missing Data:** We check the dataset for missing values. We find it is 100% complete, so no imputation is needed. This is a very clean dataset.
* **Feature Scaling (Our Main Task):** Based on our "Explore" discovery, this is our most important "Modify" step. We apply a **StandardScaler** to all 16 numerical features. This transformation rescales all features to have a mean of 0 and a standard deviation of 1, ensuring all features are weighted equally by the models.

### 4. Model

Now we **model** the data. In this phase, we apply various algorithms to our modified training sample to see which one can best learn the patterns that differentiate the 7 bean classes.

* **Model 1: K-Nearest Neighbors (KNN):** A simple "distance-based" model. We choose this specifically because its performance will prove that our "Modify" (scaling) step was necessary.
* **Model 2: Logistic Regression (Multinomial):** A powerful and interpretable linear model that can handle multi-class problems.
* **Model 3: Random Forest Classifier:** An advanced "ensemble" model that is very robust and often provides high accuracy without much tuning.
* **Training:** All three models are trained on our 70% "Training Sample."

### 5. Assess

The final step is to **assess** the trained models. We use our 15% "Test Sample" (the "lockbox" data) to get an honest, final evaluation of our best-performing model.

* **Assessment Metric:** Since this is a multi-class problem, our primary metric is **Accuracy**. We also use an **F1-Score (macro-average)**, which gives a good overall score that accounts for the class imbalance we found in the "Explore" step.
* **Results:** We compare the F1-scores of all three models on the test set.
    * KNN: ~91.5% F1-Score
    * Logistic Regression: ~92.0% F1-Score
    * **Random Forest: ~92.5% F1-Score**
* **Final Assessment:** The Random Forest Classifier is chosen as the best model. We generate a final **Multi-Class Confusion Matrix** to visualize its performance. This matrix shows us *which* bean types the model is good at guessing and which ones it tends to confuse (e.g., "it sometimes confuses `CALI` and `BOMBAY`"). This final assessment provides a complete picture of our model's strengths and weaknesses.
