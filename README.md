# Breast Cancer Detection using Ensemble Classifiers

Detecting malignant tumours in the Breast Cancer Wisconsin (Diagnostic) dataset with a PCA/NMF
feature-extraction ensemble and majority-vote classifiers.

## Overview

An ensemble classifier for the Breast Cancer Wisconsin (Diagnostic) dataset that combines two
feature-extraction methods (PCA and NMF) with two classifiers (Logistic Regression and SVM) and
merges their predictions by majority voting. The number of extracted components is tuned on a
validation set, and the best model is evaluated on a held-out test set.

- Notebook: [`breast_cancer_ensemble.ipynb`](breast_cancer_ensemble.ipynb)
- Report: [`report.md`](report.md)

## Method

```
model_k = { (PCA_k, LR), (NMF_k, LR), (PCA_k, SVM), (NMF_k, SVM) }  ->  majority vote
```

1. Stratified 60 / 20 / 20 split into train / validation / test.
2. Min-Max scaling fitted on the training set (`clip=True`, so values stay in `[0, 1]` for NMF).
3. For each `k` in {2, 4, 6, 8, 10}: fit PCA and NMF with `k` components, train the four base
   learners (assessed with 5-fold cross-validation), and majority-vote on the validation set.
4. Select the `k` with the best validation F-score.
5. Apply the best model to the test set (scaled with the training scaler).

Ties in the four-way vote are resolved in favour of the malignant class.

## Results

Validation F-score rose with the number of components (0.914 at k=2 up to 0.951 at k=10), so
**model_10** was selected. On the test set:

| Metric | Value |
|--------|-------|
| F-score | 0.976 |
| Accuracy | 0.982 |
| Precision | 1.000 |
| Recall | 0.952 |

The confusion matrix had no false positives and two false negatives.

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook breast_cancer_ensemble.ipynb
```

## Files

| File | Description |
|------|-------------|
| `breast_cancer_ensemble.ipynb` | Full implementation and analysis |
| `report.md` | Written report |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
| `breast-cancer.csv` | Breast Cancer Wisconsin (Diagnostic) dataset |
