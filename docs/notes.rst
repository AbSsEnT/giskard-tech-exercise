Giskard technical exercise
====

Description
----

* Goal - To improve the performance of logistic regression model using data augmentation.
* Steps to be made:
    1) Generate new data samples;
    2) Retrain same model using augmented dataset;
    3) Obtain higher score than model with basic data.
* Requirements:
    - To propose **at least 2** data augmentation techniques;
    - To **take into account** observations, which model under-performs on;
    - To obtain **well-generalized** model with no over-fitting behaviour;
    - To use **augmentation** techniques based on **domain knowledge/heuristics**.
* Option - Use AI Inspector of Giskard to get intuition about model behaviour.

Notes
----
- People under maintenance feature creates 2 distinct clusters after T-SNE dimensionality reduction.

Attributes
~~~~
1) [CATEGORY] Status of existing checking account, in Deutsche Mark.
2) [NUMERIC / DISCRETE] Duration in months
3) [CATEGORY] Credit history (credits taken, paid back duly, delays, critical accounts)
4) [CATEGORY] Purpose of the credit (car, television,...)
5) [NUMERIC / CONTINUOUS] Credit amount
6) [CATEGORY] Status of savings account/bonds, in Deutsche Mark.
7) [CATEGORY] Present employment, in number of years.
8) [NUMERIC / CONTINUOUS *(but in dataset behaves like discrete)*] Installment rate in percentage of disposable income
9) [CATEGORY] Sex
10) [CATEGORY] Personal status (married, single,...)
11) [CATEGORY] Other debtors / guarantors
12) [NUMERIC / DISCRETE] Present residence since X years
13) [CATEGORY] Property (e.g. real estate)
14) [NUMERIC / DISCRETE] Age in years
15) [CATEGORY] Other installment plans (banks, stores)
16) [CATEGORY] Housing (rent, own,...)
17) [NUMERIC / DISCRETE] Number of existing credits at this bank
18) [CATEGORY] Job
19) [NUMERIC / DISCRETE] Number of people being liable to provide maintenance for
20) [CATEGORY] Telephone (yes,no)
21) [CATEGORY] Foreign worker (yes,no)


Baseline confusion matrix
~~~~
.. image:: docs/images/baseline_confusion_matrix.png
  :width: 600
  

Ideas and Reasoning
~~~~
- We need to somehow determine 'rare' or 'outlier' samples, on which model underperformes. Thus, we can apply different 'anomaly detection' techniques to mine such samples. Also we can use hard negative mining. Then, we will sample more examples from collected outliers. Possible techniques are: clustering, auto-encoder..
- As far as we build logistic regression model, maybe the problem with 'bad' data-points, is that they lie near decision boundary..
- The problem could arise, if we interpolate between values of discrete variables.
- If model confused sample with high probability, then this sample lies far away from decision boundary in the incorrect area.
- If model confused sample with probability ~ 0.5, that this sample is near decision boundary.
- We can cluster the records of the majority class and do the under-sampling by removing records from each cluster, thus seeking to preserve information.
- Heuristics:
    1) Simple:
        1) First step - up-sample misclassified observations;
        2) Second step - further up-sampling using SMOTE-NC or down-sampling;
        Note: Risk of over-fitting.
    2) Better:
        1) First step - up-sample observations, which have feature values with high percentage of errors. But we need to choose those features, which have the highest impact on model and have high relative quantity. For example: high shap-importance + high number of misclassified observations + high ratio between correct and error answers;
        2) Second step - further up-sampling using SMOTE-NC or down-sampling;
- I think removing outliers will help. Despite their existence in the test data-set, we will get better prediction for normal samples.

Resources
~~~~
- https://www.kaggle.com/code/lazygene/german-bank-clients-clusterisation

