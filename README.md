# Match Outcome Prediction with Machine Learning

First project for the course **Machine Learning Techniques** at Pontifical Xavierian University (Bogotá, 2024-3). The project compares three supervised classification approaches to predict the outcome (win/loss) of high-level Clash Royale matches.

## Table of Contents

- [Description](#description)
- [Data](#data)
- [Implemented Models](#implemented-models)
- [Results](#results)
- [Technologies](#technologies)
- [Installation and Execution](#installation-and-execution)
- [Repository Structure](#repository-structure)
- [Conclusions](#conclusions)
- [Authors](#authors)

## Description

Clash Royale is a real-time strategy game with a complex ecosystem of cards, troops, and mechanics that influence the outcome of each battle. The goal of this project was to answer the following question:

> How can we use different machine learning models to accurately predict battle outcomes in Clash Royale?

To answer this question, three binary classification models were implemented, trained, and evaluated by comparing their performance using standard metrics (precision, recall, F1-score, confusion matrix, and ROC/AUC curve).

## Data

Two datasets were used:

| Dataset | Dimensions | Description |
|---------|-------------|-------------|
| `cardlist.csv` | 106 x 3 | List of cards and their IDs |
| `data_ord.csv` | 718,886 x 20 | Detailed battle records |

The target variable is `outcome` (binary result: Player 1 win/loss). The predictor variables correspond to the 8 deck cards of each player (`p1card1`..`p1card8`, `p2card1`..`p2card8`) and other match-related variables.

### Common Preprocessing

- One-hot encoding (dummy variables) for card columns using `pd.get_dummies`.
- 70/30 train-test split with `train_test_split` (`random_state=42`).
- Normalization using `StandardScaler` (mean 0, standard deviation 1).
- Dimensionality reduction using **PCA with 34 components** for Models 1 and 2.

## Implemented Models

### 1. Logistic Regression

- `LogisticRegression` with `class_weight='balanced'` to compensate for class imbalance.
- Decision threshold adjusted to **0.4** to improve sensitivity for the minority class.

### 2. Feed-Forward Neural Network

Sequential architecture implemented in Keras/TensorFlow:

| Layer | Neurons | Activation |
|------|----------|------------|
| Input (Dense) | 28 | ReLU |
| Hidden (Dense) | 14 | ReLU |
| Output (Dense) | 1 | Sigmoid |

- Total trainable parameters: **1,401**
- Optimizer: Adam (`learning_rate = 0.001`)
- Loss function: `binary_crossentropy`
- 25 epochs, batch size = 64
- Decision threshold: 0.56

### 3. Hybrid Model with LightGBM

Pipeline combining advanced preprocessing and an efficient classifier:

1. Random sampling of 10% of the dataset to handle data volume.
2. Missing value imputation using `SimpleImputer`.
3. Feature scaling with `StandardScaler`.
4. Feature selection with `SelectFromModel` using LightGBM.
5. Final classification with `LGBMClassifier` (`n_estimators=100`, `learning_rate=0.1`).

During development, Random Forest, SVM with different kernels, XGBoost, and stacking techniques were also tested, but LightGBM provided the best balance between computational cost and performance.

## Results

### Logistic Regression

```text
              precision    recall  f1-score   support
           0       0.52      0.00      0.01     95434
           1       0.56      1.00      0.72    120232
    accuracy                           0.56    215666
Feed-Forward Neural Network
              precision    recall  f1-score   support
           0       0.60      0.48      0.54    119471
           1       0.49      0.61      0.54     96195
    accuracy                           0.54    215666
Hybrid Model (LightGBM)
              precision    recall  f1-score   support
           0       0.51      0.18      0.26      9513
           1       0.57      0.87      0.69     12054
    accuracy                           0.56     21567
```
Conclusions

Predicting battle outcomes in Clash Royale proved to be a complex task. All three models achieved moderate performance, only slightly better than random guessing, suggesting that the available variables (deck composition) do not fully capture the factors that determine the outcome of a match, such as player skill, game timing, synergies, and tactical patterns not present in the dataset.

Main findings:

Logistic regression failed to discriminate between classes (AUC = 0.50) and showed a strong bias toward the majority class.
The neural network improved class balance but showed signs of overfitting.
The LightGBM hybrid model provided the best balance between performance and computational cost, although its predictive capability remained modest.
Possible future work includes feature engineering based on card synergies, embeddings, sequential models considering play order, and explicit balancing techniques (SMOTE, undersampling).

This project illustrates common challenges when applying machine learning in real-world scenarios: feature selection, class imbalance handling, and the importance of iterative model refinement to improve predictive performance.

Authors
Diego Caballero Sarmiento
Liseth Lozano
Aura Atuesta
