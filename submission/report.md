# Giskard - Data Scientist Intern - Technical Exercise Report
## Problem Description
### Goal
Improve performance of a vanilla logistic regression using data-augmentation techniques.
### Steps to be made
1. Generate new data-samples;
2. Retrain same model using augmented dataset;
3. Obtain higher score than model with basic data.
### Requirements
- Propose at least 2 data augmentation techniques;
- Take into account observations, which model under-performs on;
- Obtain well-generalized model with no over-fitting behaviour;
- Use augmentation techniques based on domain knowledge/heuristics.

## Data analysis
### Dataset
- Name: *German Credit Scoring*;
- Link: 'https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/credit_scoring_classification_model_dataset/german_credit_prepared.csv';
- Shape: 1000 observations, with 21 features and 1 target;
- Features: 14 categorical and 7 numerical;
- Target: 1 binary target - "Default" (give credit), "Not default" (don't give credit);
### EDA
* Dataset moderately imbalanced, with 70% "Not default" and 30% "Default" targets:

![classes_distribution.png](..%2Fdocs%2Fimages%2Fmiscs%2Fclasses_distribution.png)

* Numerical features have both continuous and discrete nature, which must be taken into account: 

![numerical_features_distribution.png](..%2Fdocs%2Fimages%2Fmiscs%2Fnumerical_features_distribution.png)

* Discrete features distribution:

![discrete_features_distribution.png](..%2Fdocs%2Fimages%2Fmiscs%2Fdiscrete_features_distribution.png)

## Metrics and Validation process
Taking into account data-set imbalance and naive, but adequate assumption, that FP and FN have similar importance, next metrics were used:
- F1-Macro score - Average between per-class F1-score;
- Balanced accuracy - Average of per-class recall;

In order to prevent over-fitting on the test data, cross-validation scoring was used to get less biased estimate of the generalization loss. 
Main indicator of model performance was not test F1-macro, but F1-macro cross-validation score, calculated on train dataset, 
but if two models have similar cross-val F1, model with higher test F1 considered 'better'.  
Due to imbalanced data regime, stratification by class during train-test split and cross-validation splits was used to preserve 
same classes distribution in each set.  
Based on the fact, that data set is small, repeated-cross-validation was used to get less-biased estimate of the generalization loss.  
For cross-validation, I used 10 splits with 10 repeats. 

## Experiments

### Baseline results
| Cross F1 | Acc.  | F1    | Balanced Acc. |
|----------|-------|-------|---------------|
| 0.675    | 0.755 | 0.698 | 0.692         |

![baseline_confusion_matrix.png](..%2Fdocs%2Fimages%2Fconfusion_matrices%2Fbaseline_confusion_matrix.png)  

As we can see, model performs better on samples with label 1 "Not default" and it is natural, because it makes up 70% of 
labels distribution. Thus, the task will be to balance the number of both classes to let model perform equally on observations 
with different labels.

### Over-sampling methods
First we are going to consider over-sampling approaches. *imblearn* library was used, because it provides list of re-sampling
methods, which are perfectly integrated with scikit-learn, making it possible to build ML-Pipelines, which helped with
code re-usability and avoiding logical errors (data-leakage, incorrect resampling etc.)

#### Random Over-Sampling
Simple approach, which randomly clones observations from minority class.  

| Cross F1 | Acc.  | F1    | Balanced Acc. |
|----------|-------|-------|---------------|
| 0.689    | 0.755 | 0.712 | 0.715         |

![random_oversampling_confusion_matrix.png](..%2Fdocs%2Fimages%2Fconfusion_matrices%2Frandom_oversampling_confusion_matrix.png)

As we can see, we were able to increase all possible metrics, including test and cross-validation. On the confusion matrix 
it is notable, that TN number decreased, although model performance increased. That is, because we are optimizing model performance
on all classes, not only on class 1 ("Not default").

#### SMOTE
This approach is known as a 'standard' for over-sampling techniques, because of its simplicity and reasonable performance.
To generate new samples, two points from minority class are selected, then the new point selected, which lies on the line
between these two points.

| Cross F1 | Acc.  | F1    | Balanced Acc. |
|----------|-------|-------|---------------|
| 0.688    | 0.740 | 0.718 | 0.748         |

![smote_confusion_matrix.png](..%2Fdocs%2Fimages%2Fconfusion_matrices%2Fsmote_confusion_matrix.png)

We got Cross-Validation F1 score quite similar to the one with previous method, but metrics on test set are reasonably better.

#### ADASYN
This approach is also can be considered as a 'standard' and is very similar to the SMOTE, except it makes an additional 
preprocessing step to define 'hard' observations to focus over-sampling on them, while SMOTE do not make any distinctions
between 'hard' and 'easy' samples.

| Cross F1 | Acc.  | F1    | Balanced Acc. |
|----------|-------|-------|---------------|
| 0.686    | 0.730 | 0.699 | 0.717         |

As we can see, obtained results are worse, than the one, obtained with SMOTE. One of possible reasons, could be that by 
generating more 'hard' samples, which will be located near decision boundary, ADASYN enriches dataset with outliers, i.e. 
points, whose neighbourhood has a lot of points from another class, thus negatively impacting the building of regression's decision boundary.     

![adasyn_confusion_matrix.png](..%2Fdocs%2Fimages%2Fconfusion_matrices%2Fadasyn_confusion_matrix.png)

#### SMOTE-NC
There is one important point to be considered, when using SMOTE and ADASYN. If dataset contains categorical features or 
numerical, but discrete, than these two method can generate non-reasonable samples. For example, if we draw a point between 
samples, where feature 'age' has values 47 and 48, it is real to get feature 47.34 for the new point, which makes no sense.
Thus post-processing is needed to transform such features or it is possible to use over-sampling methods, which handle categorical 
and discrete features. One of these methods is SMOTE-NC, which for categorical features picks the most frequent values across
its neighbourhood.

| Cross F1 | Acc.  | F1    | Balanced Acc. |
|----------|-------|-------|---------------|
| 0.688    | 0.730 | 0.697 | 0.712         |

![smotenc_confusion_matrix.png](..%2Fdocs%2Fimages%2Fconfusion_matrices%2Fsmotenc_confusion_matrix.png)

In terms of Cross-Validation F1, we got result, comparable with previous methods, but metrics on the test dataset are worse.
Truly speaking, during experiments, it was usual, when two different configurations of hyperparameters brought similar
cross-validation score, but very different test scores. I suppose, this is due to small data-set size, which makes non-averaged
test score very unstable. This is one of the reasons, why I decided to focus much more on cross-validation F1.


### Under-sampling methods
Next set of algorithms aims to remove samples from majority class. Advantage of these methods are that they mostly work with
categorical features, although the main disadvantage is that we can remove useful information from the dataset, especially 
during small-data regime.

#### Random Under-Sampling
Consider opposite to the Random Over-Sampling method, which in contrast, randomly removes samples from majority class.

| Cross F1 | Acc.  | F1    | Balanced Acc. |
|----------|-------|-------|---------------|
| 0.685    | 0.720 | 0.698 | 0.729         |

![random_undersampling_confusion_matrix.png](..%2Fdocs%2Fimages%2Fconfusion_matrices%2Frandom_undersampling_confusion_matrix.png)

In terms of cross-validation F1, this resampler brought worse model, than previous over-sampling methods, but, surprisingly, 
test balanced accuracy is better than all but one, obtained with the SMOTE.

#### Instance Hardness Threshold
Other method to consider is Instance Hardness Threshold. This algorithm first trains a classifier, and then filters samples,
with low probabilities. Thus, it is possible to reduce noisy samples near decision boundary and those, which are far from 
decision boundary, but on the incorrect side (possible outliers). Disadvantage could be the fact, that developer needs to
additionally fine-tune classifier.

| Cross F1 | Acc.  | F1    | Balanced Acc. |
|----------|-------|-------|---------------|
| 0.686    | 0.730 | 0.701 | 0.721         |

This method did not bring notable performance increase, nor decrease across other methods.

### Custom Over-Sampling
Finally, I decided to take into account model performance on specific data slices, specifically on those, where model under-performed.
I decided to use quite straightforward approach, which turned out to be very effective. It is consisted of two steps. First, 
random oversampling, mostly, but not only on minority class was performed. We oversampled those data-slices, which the model under-performs on. On the second step SMOTE-NC was used.

To define 'hard' data-slices, I used a mixture of heuristics:
- Feature importance (contribution), discovered through Giskard Inspector;

![giskard_inspector.png](..%2Fdocs%2Fimages%2Fconfusion_matrices%2Fgiskard_inspector.png)

- Analysis of features distribution on correct and misclassified samples. Those features were chosen, where 
proportion of correct and misclassified observations was more than ~50% or the number of incorrect points was higher than 20.

![baseline_errors_distributions_categoricals_common_norm_trunc.png](..%2Fdocs%2Fimages%2Fcategoricals%2Fbaseline_errors_distributions_categoricals_common_norm_trunc.png)

| Cross F1 | Acc.  | F1    | Balanced Acc. |
|----------|-------|-------|---------------|
| 0.697    | 0.740 | 0.718 | 0.748         |

![custom_oversampling_confusion_matrix.png](..%2Fdocs%2Fimages%2Fconfusion_matrices%2Fcustom_oversampling_confusion_matrix.png)

Eventually, this algorithm turned out to be the best. All metrics, except standard accuracy (which is not important), 
turned out to be the best. Specifically, Cross-Validation F1 increased significantly comparing to previous methods. Also 
test metrics are also the highest (although, they are equal to the SMOTE metrics).  
I defined next features and related values, which were oversampled to reduce the number of incorrect predictions:

| account_check_status | credit_history                          | personal_status | savings      |
|----------------------|-----------------------------------------|-----------------|--------------|
| 0 <= ... < 200 DM    | all credits at this bank paid back duly | divorced        | ... < 100 DM |
| < 0 DM               |                                         |                 |              |

What should be noted, is the fact, that this approach takes into account most problem requirements:
- Focus on features, which model under-performs on;
- Does not create data-bias during over-sampling process, because generated samples are drawn from real distribution, i.e. categorical features does not have continuous values;
- Model was not over-fitted on the test set, because hyperparameters selection was performed using repeated-cross-validation.

## Conclusion
This report provides an overview of the given problem, dataset and results of performed experiments with the data augmentation.  
All requirements have been met:
- Higher score was obtained using only data-augmentation preprocessing step;
- Proposed more than 2 data augmentation techniques;
- Models are well-generalized, because were selected through repeated-cross-validation;
- Data augmentation approaches based on domain-knowledge/heuristics were used;
- Giskard Inspector was used.

## Possible improvements
1) Variational auto-encoder to sample observations which are close in latent spase to misclassified points;
2) Use more granular oversampling ratio in custom oversampling, i.e. use specific oversampling ratio for each feature value.
3) Develop automated heuristic to distill 'bad' feature values;
4) Use more feature values, which model under-performs on.

------------------------------------
Author: Mykyta Alekseiev, 05.04.2023
