# Bank Marketing Classifier Comparison

## Project Overview

This project, "Practical Application III: Comparing Classifiers," focuses on evaluating the performance of four machine learning classifiers—K Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM)—on a bank marketing dataset. The dataset, sourced from a Portuguese banking institution, contains data from 17 telephone marketing campaigns conducted between May 2008 and November 2010, with 79,354 contacts and an 8% success rate (6,499 subscriptions to a long-term deposit). The goal is to predict whether a client will subscribe to a term deposit based on various features, addressing the challenge of an imbalanced dataset where the positive class (subscriptions) is a minority.

The analysis is implemented in a Jupyter Notebook, which includes data loading, preprocessing, model training, hyperparameter tuning, and performance evaluation. The project emphasizes metrics like F1-score and Recall due to the imbalanced nature of the dataset, ensuring the identification of potential subscribers while minimizing false positives and negatives.

## Dataset

The dataset, `bank-additional-full.csv`, is obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). It includes 21 features, such as:

- **Client Data**: Age, job, marital status, education, default status, housing loan, and personal loan.
- **Campaign Data**: Contact type, month, day of the week, duration, number of contacts (campaign), days since last contact (pdays), previous contacts, and previous outcome.
- **Economic Indicators**: Employment variation rate, consumer price index, consumer confidence index, Euribor 3-month rate, and number of employees.
- **Target Variable**: `y` (binary: "yes" or "no" for subscription to a term deposit).

The dataset is read using pandas with a semicolon separator (`sep=';'`). The accompanying paper, [CRISP-DM-BANK.pdf](module17_starter/CRISP-DM-BANK.pdf), provides additional context on the data collection and features.

## Summary of Findings

### Data Understanding
- The dataset represents **17 marketing campaigns** with 79,354 contacts, of which 6,499 (8%) resulted in a subscription.
- The target variable is highly imbalanced, with a baseline accuracy of approximately 88.73% for predicting the majority class ("no" subscription). This necessitates the use of metrics like Precision, Recall, and F1-score to evaluate model performance on the minority class ("yes" subscription).

### Model Performance
Four classifiers were trained and evaluated: KNN, Logistic Regression, Decision Trees, and SVM (with both default `SVC` and optimized `LinearSVC` or `cuLinearSVC` for GPU acceleration). Key observations include:

1. **Class Imbalance Impact**: All models achieved test accuracies near or above the baseline (88.73%), but accuracy alone is insufficient due to the imbalance. F1-score and Recall are critical for identifying subscribers.
2. **Initial Model Performance**:
   - **Logistic Regression**: Best initial F1-score (0.5920) and Recall (0.5177), balancing precision and recall effectively.
   - **Decision Tree**: Perfect training accuracy (1.0), indicating overfitting, with decent test F1-score but lower Recall.
   - **KNN**: Good performance but computationally expensive.
   - **SVM (`SVC`)**: High training time with default parameters and no significant performance advantage.
3. **Hyperparameter Tuning**:
   - Tuning via GridSearchCV improved F1-scores and Recall for most models, though it was computationally intensive.
   - **Linear SVC** (or `cuLinearSVC` with GPU) reduced training time compared to default `SVC` and achieved competitive performance.
4. **GPU Acceleration**: Using `cuML` for `cuLinearSVC` provided faster training times, particularly beneficial for large datasets.
5. **Tuned Model Comparison** (from the provided results):
   - **Logistic Regression**: F1-score: 0.6145, Recall: 0.5395
   - **KNN**: F1-score: 0.6275, Recall: 0.5402
   - **Decision Tree**: F1-score: 0.6149, Recall: 0.5398
   - **Linear SVC**: F1-score: 0.6275, Recall: 0.5516

### Best Classifier
The **Tuned Linear SVC** is the best classifier for this task due to:
- **Highest Recall (0.5516)**: Maximizes identification of potential subscribers, critical for marketing campaigns to avoid missing opportunities.
- **Highest F1-score (0.6275, tied with KNN)**: Balances precision and recall effectively.
- **Efficiency**: Faster training with `LinearSVC` (or `cuLinearSVC` on GPU) compared to non-linear `SVC`.
- **Interpretability**: Linear models like Linear SVC and Logistic Regression provide insights into feature importance, unlike KNN or complex Decision Trees.

**Tuned Logistic Regression** is a close second, offering slightly lower Recall (0.5395) but similar F1-score (0.6145) and greater simplicity/interpretability, making it a viable alternative if computational resources or model transparency are priorities.

## Repository Structure

- **Directory and File Naming**:
  - `bank_marketing_classifier/`: Root directory for clarity.
  - `data/`: Stores the dataset (`bank-additional-full.csv`).
  - `notebooks/`: Contains the Jupyter Notebook (`prompt_III.ipynb`).
  - `CRISP-DM-BANK.pdf`: Included for reference, as provided in the original context.
  - `README.md`: Project overview and summary.
- **No Unnecessary Files**: Only essential files are included, ensuring a clean repository.

## Jupyter Notebook
The Jupyter Notebook (`notebooks/prompt_III.ipynb`) is well-structured with:
- **Headings**: Clear section headings for each problem (e.g., "Problem 1: Understanding the Data," "Problem 2: Read in the Data").
- **Formatted Text**: Markdown cells provide explanations, answers, and analysis, with code cells for implementation.
- **Content**: Includes data loading, feature analysis, model training, tuning, and comparison, culminating in a detailed analysis of classifier performance.

## Prerequisites
To run the notebook, ensure the following dependencies are installed:
- Python 3.8+
- pandas
- scikit-learn
- numpy
- (Optional) cuML (for GPU-accelerated Linear SVC)
- Jupyter Notebook

Install dependencies via:
```bash
pip install pandas scikit-learn numpy
```

For GPU support (if available):
```bash
pip install cuml
```
How to Run
Clone the repository:
```bash
git clone https://github.com/nabiharaza/Comparing-Classifiers-Assignment---UC-Berkeley-Extension.git
```
Navigate to the repository directory:
```bash
cd bank-marketing-classifier
```
Run the Jupyter Notebook:
```bash
jupyter notebook prompt_III.ipynb
```
Ensure the dataset (bank-additional-full.csv) is in the data/ directory.
Launch Jupyter Notebook:
```bash
jupyter notebook prompt_III.ipynb
```
Run all cells to reproduce the analysis.

##Conclusion
- The project successfully compares four classifiers on the bank marketing dataset, with the Tuned Linear SVC emerging as the best model due to its high Recall and F1-score, balancing the identification of subscribers with computational efficiency. 
- The Tuned Logistic Regression is a strong alternative for simpler, interpretable models. 
- The analysis underscores the importance of hyperparameter tuning and appropriate metrics (F1-score, Recall) for imbalanced datasets in marketing applications.

