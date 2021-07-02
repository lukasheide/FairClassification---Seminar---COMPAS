from fairness.data.objects.Data import Data
from fairness.data.objects.list import DATASETS
from fairness.data.objects.ProcessedData import ProcessedData
from sklearn.linear_model import LogisticRegression as SKLearn_LR, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import make_column_transformer
from Scripts.metrics.fairnessMetrics import FairnessMetrics, CausalDiscrimination, compute_metrics

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas
from aif360.sklearn.inprocessing import AdversarialDebiasing
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.sklearn.datasets import fetch_adult, fetch_compas
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr, generalized_fnr, difference
from aif360.sklearn.preprocessing import ReweighingMeta

import tensorflow as tf

import pandas as pd


def pipeline():
    """
    FairClassificationPipeline consisting of three steps:
    1) Preprocessing:
    - We start by fetching the COMPAS dataset from the AIF360 library
    - We then clean the data and rop columns that are not used in training
    - We define race as the sensitive attribute and choose African-american as the unprivileged group
    - We also split the dataset into training and test data choosing recidivism within two years as the target variable
    - and normalize the input data to ensure that all attributes are equally important during training

    2) Modelling + Evaluation:
    - We then create four models which we train on the trainset:
        - Logistic Regression (Baseline model)
        - Reweighing (Pre-processing)
        - Adversarial Debiasing (In-processing)
        - Calibrated Equalized Odds (Post-processing)
    - We use the models for which the AIF360-Scikit API had already been implemented before July 2021.

    - We then evaluate the results on the following metrics:
        - Correctness:
            - Accuracy
            - Precision
            - Recall
            - F1-Score
            - AUC
        - Fairness:
            - Disparate Impact
            - True Positive Rate Balance
            - True Negative Rate Balance
            - Causal Discrimination

    - We use the SKLearn Library to compute the correctness metrics and implemented the fairness notions ourself.
    - Unfortunately the methods we wanted to use are not available in the AIF360-Sklearn API or the results do not fit
    the results of the benchmarking studies used in our thesis.
    - We therefore suggest to wait for the execution of our experiment until the AIF360-Sklearn API is in a more mature
    state and includes the approaches and fairness metrics that we want to use for our experiment.

    :return:
    """

    ### https://github.com/Trusted-AI/AIF360/blob/master/examples/sklearn/demo_new_features.ipynb

    seed = 12345
    sensitive_attr = 'race'

    # Import Dataset:
    X, y = fetch_compas(binary_race=True)
    # usecols=['sex', 'race', 'age_cat', 'priors_count', 'c_charge_degree']

    ## Cleaning:
    # Drop column:
    # c_charge_desc (389 unique textual descriptions -> can not be properly used in quantitative analysis)
    X = X.drop(['c_charge_desc'], axis=1)


    # Set multiclass-indizes (requirement of the AIF360 library in later steps):
    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

    # Convert y to pd.Series:
    y = pd.Series(y.factorize(sort=True)[0], index=y.index)

    # Split dataset:
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=seed)

    # Encode race: (0 if african-american, 1 otherwise)
    X_train.race = 1 - (X_train['race'] == 'African-American').astype(int)
    X_test.race = 1 - (X_test['race'] == 'African-American').astype(int)

    # Change column order so that sensitive attribute is in first column. This is important for computing the metrics later on.
    cols = X_train.columns.tolist()
    cols = ['race'] + [col for col in cols if col!='race']
    X_train = X_train[cols]
    X_test = X_test[cols]


    # One-Hot-Encode categorical data:
    ohe = make_column_transformer(
        (OneHotEncoder(sparse=False, drop='first'), X_train.dtypes == 'category'),
        remainder='passthrough')

    X_train = pd.DataFrame(ohe.fit_transform(X_train), index=X_train.index)
    X_test = pd.DataFrame(ohe.transform(X_test), index=X_test.index)


    # Normalize data:
    normalizer = MinMaxScaler()
    X_train_normalized = normalizer.fit_transform(X_train)
    X_test_normalized = normalizer.transform(X_test)

    # Transform back into Pandas DataFrames:
    X_train = pd.DataFrame(X_train_normalized, index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_normalized, index=X_test.index, columns=X_test.columns)


    ### Baseline model ###
    log_reg = LogisticRegression(solver='lbfgs')
    log_reg.name = 'Logistic Regression'

    y_pred_baseline = pd.DataFrame(log_reg.fit(X_train, y_train).predict(X_test), index=y_test.index, columns=['y_pred_test'])

    compute_metrics(y_pred=y_pred_baseline, y_actual=y_test, x_test=X_test,
                    model=log_reg, sensitive_attr=sensitive_attr, verbose=True)


    ### Preprocessing Algorithm ###
    rew = ReweighingMeta(estimator=LogisticRegression(solver='lbfgs'))

    params = {'estimator__C': [1,3,5,7,10], 'reweigher__prot_attr': ['race']}

    clf = GridSearchCV(rew, params, scoring='accuracy', cv=5)
    clf.name = 'Reweighing Algorithm'

    clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))  
    print(clf.best_params_)

    y_pred_rew = pd.DataFrame(clf.predict(X_test), index=y_test.index, columns=['y_pred_test'])

    compute_metrics(y_pred=y_pred_rew, y_actual=y_test, x_test=X_test,
                    model=clf, sensitive_attr=sensitive_attr, verbose=True)

    ### Fair in-processing algorithm ###
    adv_deb = AdversarialDebiasing(prot_attr=sensitive_attr, random_state=seed)
    adv_deb.name = 'Adversarial Debiasing'

    adv_deb.fit(X_train, y_train)

    y_pred_adv_deb = pd.DataFrame(adv_deb.predict(X_test), index=y_test.index, columns=['y_pred_test'])

    compute_metrics(y_pred=y_pred_adv_deb, y_actual=y_test, x_test=X_test,
                    model=adv_deb, sensitive_attr=sensitive_attr, verbose=True)

    # close tensor-flow board
    adv_deb.sess_.close()



    ### Fair post-processing algorithm ###
    cal_eq_odds = CalibratedEqualizedOdds(sensitive_attr, cost_constraint='fnr', random_state=seed)
    log_reg = LogisticRegression(solver='lbfgs')
    postproc = PostProcessingMeta(estimator=log_reg, postprocessor=cal_eq_odds, random_state=seed)
    postproc.name = 'Calibrated Equalized Odds'

    postproc.fit(X_train, y_train)
    y_pred_eq_odds = pd.DataFrame(postproc.predict(X_test), index=y_test.index, columns=['y_pred_test'])

    compute_metrics(y_pred=y_pred_eq_odds, y_actual=y_test, x_test=X_test,
                    model=postproc, sensitive_attr=sensitive_attr, verbose=True)

    print('pipeline finished')


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    pipeline()
