# Breast Cancer Detection using an Ensemble of PCA/NMF Features with Logistic Regression and SVM

## Author

Jibril Hussaini

## Abstract

This project builds an ensemble classifier for the Breast Cancer Wisconsin (Diagnostic) dataset. The
ensemble pairs two feature-extraction methods, Principal Component Analysis (PCA) and Non-negative
Matrix Factorization (NMF), with two classifiers, Logistic Regression and a Support Vector Machine,
giving four base learners whose predictions are merged by majority voting. The number of extracted
components is treated as the main hyperparameter and is tuned on a validation set over the values 2,
4, 6, 8 and 10. The configuration with ten components gave the best validation F-score and was kept
as the final model. On the held-out test set it reached an F-score of 0.976 and an accuracy of
0.982, with perfect precision and only two missed malignant cases. The results show that combining
complementary feature representations and classifiers produces a reliable detector on this dataset.

## 1. Introduction

Breast cancer is one of the most common cancers, and early detection from diagnostic imaging
improves outcomes. The Wisconsin Diagnostic dataset turns this into a supervised learning problem:
each tumour is described by thirty numeric measurements and labelled benign or malignant. The aim is
to predict that label as accurately as possible, with particular attention to not missing malignant
cases.

A single model and a single view of the data can be brittle. This project instead combines several
views. PCA and NMF summarise the thirty features in different ways, and Logistic Regression and SVM
learn different decision boundaries. By training all four pairings and letting them vote, the
ensemble can stay accurate even when one member is weak on a given example. The number of components
controls how much information each feature extractor keeps, so it is tuned carefully and the choice
is validated before the model ever sees the test data.

## 2. Proposed method

The data is split into three stratified parts: 60% for training, 20% for validation and 20% for
testing. Stratification keeps the benign/malignant ratio the same in every part. All features are
scaled to the range [0, 1] with a Min-Max scaler that is fitted on the training set only and then
applied to the validation and test sets. Min-Max scaling is required here because NMF cannot accept
negative values; the scaler also clips unseen values into [0, 1] so that validation and test points
lying outside the training range stay non-negative.

For a given number of components `k`, the method fits PCA with `k` components and NMF with `k`
components on the scaled training data. Four base learners are then trained on these features:
(PCA, Logistic Regression), (NMF, Logistic Regression), (PCA, SVM) and (NMF, SVM). Each classifier
uses its default settings and is assessed with 5-fold cross-validation on the training set. The four
trained learners predict the validation labels, and a majority vote combines them into one
prediction. Because there are four voters, a 2-2 tie is possible; ties are resolved in favour of the
malignant class, since a false negative is the more dangerous error in this setting.

This whole procedure is repeated for `k` in {2, 4, 6, 8, 10}, producing five candidate models
named model_2 through model_10. Each `model_k` is the ensemble

    model_k = { (PCA_k, LR), (NMF_k, LR), (PCA_k, SVM), (NMF_k, SVM) }.

The model with the highest validation F-score is selected as the best model. It is then applied once
to the test set, which is scaled with the training scaler, to give an unbiased estimate of
performance.

## 3. Dataset description

The Breast Cancer Wisconsin (Diagnostic) dataset has 569 samples and 30 numeric features. The
features come in three groups of ten (the mean, standard error and worst value of measurements such
as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry
and fractal dimension). The target is the diagnosis: 357 samples are benign (62.7%) and 212 are
malignant (37.3%). There are no missing values. The identifier column is removed and the diagnosis
is encoded as 1 for malignant and 0 for benign.

## 4. Data exploration

The class distribution is moderately imbalanced, with benign cases outnumbering malignant ones by
roughly two to one. This is mild enough that the F-score, which balances precision and recall on the
malignant class, is a fair way to measure performance.

The features are strongly correlated with one another. The size measurements are almost
interchangeable: radius, perimeter and area (in both their mean and worst forms) correlate above
0.99, and concave points, concavity and compactness form another tightly linked group. This heavy
redundancy is the key observation from the exploration. It means the data really lives in far fewer
than thirty dimensions, which is exactly the situation where PCA and NMF help: a handful of
components can capture most of the signal. It also explains why size and shape features, which track
how irregular and large a tumour is, carry most of the information that separates malignant from
benign cases.

## 5. Results

The table below reports the validation F-score of each candidate model, together with the F-score of
its four individual base learners on the validation set.

| Model | Components | PCA+LR | NMF+LR | PCA+SVM | NMF+SVM | Ensemble |
|-------|-----------|--------|--------|---------|---------|----------|
| model_2  | 2  | 0.914 | 0.853 | 0.914 | 0.914 | 0.914 |
| model_4  | 4  | 0.927 | 0.842 | 0.914 | 0.938 | 0.927 |
| model_6  | 6  | 0.938 | 0.771 | 0.938 | 0.938 | 0.938 |
| model_8  | 8  | 0.951 | 0.872 | 0.938 | 0.925 | 0.938 |
| model_10 | 10 | 0.951 | 0.822 | 0.951 | 0.897 | 0.951 |

The ensemble F-score rises steadily with the number of components, from 0.914 at two components to
0.951 at ten. **model_10 is the best model.** Two patterns stand out. First, the PCA-based learners
are consistently strong, and the SVM pairings are usually the best single members. Second, NMF with
Logistic Regression is the weakest learner by a wide margin (its cross-validated training F-score
dips into the 0.5-0.8 range), because the non-negative, parts-based components NMF produces are not
as linearly separable as PCA components. The value of the ensemble is visible here: majority voting
lets the three stronger members outvote the weak NMF+LR learner, so the combined F-score stays at or
above the best individual learner at every component count.

Applying model_10 to the test set gives:

| Metric | Test value |
|--------|-----------|
| F-score | 0.976 |
| Accuracy | 0.982 |
| Precision | 1.000 |
| Recall | 0.952 |

The confusion matrix shows 72 benign and 40 malignant cases classified correctly, with no false
positives and two false negatives. Precision of 1.0 means every tumour the model flagged as
malignant truly was malignant. Recall of 0.952 means it caught 40 of the 42 malignant cases. In a
screening context the two missed malignant cases are the result that matters most, and they are the
natural place to focus any further work, for example by lowering the decision threshold to trade a
little precision for higher recall.

## 6. Test-set data distribution

Projecting the test set onto the first two components of each method shows why the classifiers do so
well. Along the first PCA component the benign cases sit in a tight cluster at low values and the
malignant cases stretch out toward high values, with only a narrow overlap in the middle. The first
NMF component separates the classes in the same direction but with a bit more overlap, which fits the
weaker performance of the NMF-only learners. In both projections the second component adds little
separation on its own. The clear, mostly one-dimensional split confirms that the dataset is highly
separable and that a few components are enough to carry the signal, which is consistent with the
model reaching a high F-score and with performance improving as more components are added.

## 7. Conclusion

An ensemble of PCA and NMF feature extractors combined with Logistic Regression and SVM, merged by
majority voting, classifies the Wisconsin Diagnostic tumours very accurately. Tuning the number of
components on a validation set selected a ten-component model, which reached a test F-score of 0.976
and an accuracy of 0.982 with perfect precision. The strong, redundant feature set makes the data
well suited to dimensionality reduction, and the ensemble's main benefit is robustness: it absorbs
the weak NMF plus Logistic Regression learner without losing accuracy. The clearest avenue for
improvement is the two false negatives, which could be addressed by threshold tuning or by adding
classifiers that are more sensitive to the malignant class.
