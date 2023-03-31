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

Attributes
~~~~
1) [CATEGORY] Status of existing checking account, in Deutsche Mark.
2) [NUMERIC] Duration in months
3) [CATEGORY] Credit history (credits taken, paid back duly, delays, critical accounts)
4) [CATEGORY] Purpose of the credit (car, television,...)
5) [NUMERIC] Credit amount
6) [CATEGORY] Status of savings account/bonds, in Deutsche Mark.
7) [CATEGORY] Present employment, in number of years.
8) [NUMERIC] Installment rate in percentage of disposable income
9) [CATEGORY] Sex
10) [CATEGORY] Personal status (married, single,...)
11) [CATEGORY] Other debtors / guarantors
12) [NUMERIC] Present residence since X years
13) [CATEGORY] Property (e.g. real estate)
14) [NUMERIC] Age in years
15) [CATEGORY] Other installment plans (banks, stores)
16) [CATEGORY] Housing (rent, own,...)
17) [NUMERIC] Number of existing credits at this bank
18) [CATEGORY] Job
19) [NUMERIC] Number of people being liable to provide maintenance for
20) [CATEGORY] Telephone (yes,no)
21) [CATEGORY] Foreign worker (yes,no)


Baseline confusion matrix
~~~~
.. image:: docs/images/baseline_confusion_matrix.png
  :width: 600
  

Ideas
~~~~
- We need to somehow determine 'rare' or 'outlier' samples, on which model underperformes. Thus, we can apply different 'anomaly detection' techniques to mine such samples. Also we can use hard negative mining. Then, we will sample more examples from collected outliers. Possible techniques are: clustering, auto-encoder..


Resources
~~~~
- https://www.mphasis.com/content/dam/mphasis-com/global/en/home/innovation/next-lab/Mphasis_Data-Augmentation-for-Tabular-Data_Whitepaper.pdf
- https://github.com/analyticalmindsltd/smote_variants
- https://www.kaggle.com/code/residentmario/undersampling-and-oversampling-imbalanced-data
