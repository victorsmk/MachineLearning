# Model Comparison: RF vs. XGBoost vs. MLP on Pima Diabetes Dataset

This project compares the performance of three different classification models on a small dataset:
1.  **Random Forest (RF)**
2.  **XGBoost (XGB)**
3.  **Multi-Layer Perceptron (MLP)**

The goal is to see how these models perform in a binary classification task with a limited amount of data and an imbalanced class distribution.

---

## Dataset

The dataset used is the **Pima Indians Diabetes Database** from Kaggle.
* **Link:** [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Description:** This is a binary classification dataset used to predict whether or not a patient has diabetes based on 8 diagnostic medical predictor variables.
* **Outcome:** 0 (No Diabetes) or 1 (Diabetes)
* **Size:** 768 rows × 9 columns
* **Class Imbalance:** The dataset is imbalanced, with significantly more non-diabetic (0) entries than diabetic (1) entries.

---

## Methodology

1.  **Load Data:** The `diabetes.csv` file is loaded into a pandas DataFrame.
2.  **Pre-processing:**
    * Features (`x`) and the target variable (`y`) are separated.
    * The data is split into training (80%) and testing (20%) sets.
    * The features are scaled using `StandardScaler`. This is crucial for the MLP and can also benefit the other models.
3.  **Model Training & Evaluation:**
    * Three models are trained on the scaled training data.
    * Each model is evaluated on the test set.
    * Performance is measured using accuracy, a confusion matrix, and a detailed classification report (precision, recall, f1-score).
    * For the tree-based models (RF and XGB), feature importance is also plotted.

---

## Model 1: Random Forest (Scikit-learn)

* **Implementation:** `RandomForestClassifier` from `sklearn.ensemble`.
* **Hyperparameters:** `n_estimators=100`, `oob_score=True`, `random_state=42`.

### Results

* **Training Accuracy:** 100.0%
* **Test Accuracy:** 74.68%
* **OOB Score:** 74.92%

**Classification Report (Test Set):**
| | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| **negative (0)** | 0.75 | 0.88 | 0.81 | 95 |
| **positive (1)** | 0.74 | 0.53 | 0.61 | 59 |
| **accuracy** | | | **0.75** | 154 |
| **macro avg** | 0.74 | 0.70 | 0.71 | 154 |
| **weighted avg** | 0.75 | 0.75 | 0.74 | 154 |

**Feature Importance:**
`Glucose`, `BMI`, and `Age` were identified as the most important features.

---

## Model 2: XGBoost

* **Implementation:** `XGBClassifier` from `xgboost`.
* **Hyperparameters:** `n_estimators=100`, `learning_rate=1`, `max_depth=6`, `objective='binary:logistic'`.

### Results

* **Test Accuracy:** 69.48%

**Classification Report (Test Set):**
| | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| **negative (0)** | 0.72 | 0.83 | 0.77 | 95 |
| **positive (1)** | 0.64 | 0.47 | 0.54 | 59 |
| **accuracy** | | | **0.69** | 154 |
| **macro avg** | 0.68 | 0.65 | 0.66 | 154 |
| **weighted avg** | 0.69 | 0.69 | 0.68 | 154 |

**Feature Importance:**
`Glucose`, `BMI`, and `DiabetesPedigreeFunction` were identified as the most important features. `SkinThickness` had zero importance.

---

## Model 3: Multi-Layer Perceptron (Keras)

* **Implementation:** Keras `Sequential` API.
* **Architecture:**
    * Input (8 features)
    * Dense Layer (32 units, ReLU)
    * Dense Layer (8 units, ReLU)
    * Output Layer (1 unit, Sigmoid)
* **Training:**
    * Optimizer: `adam`
    * Loss: `binary_crossentropy`
    * Epochs: 100 (with `EarlyStopping` patience of 10 on `val_loss`)
    * Batch Size: 32

### Results

* **Test Accuracy:** 74.68%

**Classification Report (Test Set):**
| | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| **negative (0)** | 0.77 | 0.84 | 0.80 | 95 |
| **positive (1)** | 0.70 | 0.59 | 0.64 | 59 |
| **accuracy** | | | **0.75** | 154 |
| **macro avg** | 0.73 | 0.72 | 0.72 | 154 |
| **weighted avg** | 0.74 | 0.75 | 0.74 | 154 |

---

## Conclusions

1.  **Class Imbalance is Key:** All three models struggled to correctly classify the positive (1) class, as seen by the lower recall for "positive" (0.53 for RF, 0.47 for XGB, 0.59 for MLP). This is a classic symptom of an imbalanced dataset where the model becomes biased toward the majority class.
2.  **Comparable Performance:** On this small dataset, the **Random Forest** and the **MLP** performed almost identically, achieving ~75% accuracy. The **XGBoost** model, with the chosen hyperparameters, performed slightly worse.
3.  **Feature Importance:** It was interesting to note the difference in feature importance. Both RF and XGB agreed that `Glucose` and `BMI` were top predictors. However, RF prioritized `Age` while XGB prioritized `DiabetesPedigreeFunction`.
4.  **Future Work:** It would be interesting to re-run this comparison on a much larger and more balanced dataset to see if the performance differences between the models become more pronounced. Techniques to handle class imbalance (e.g., SMOTE, class weights) would also likely improve the recall for the positive class.
