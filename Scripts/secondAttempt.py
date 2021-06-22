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

    ### https://github.com/Trusted-AI/AIF360/blob/master/examples/sklearn/demo_new_features.ipynb

    seed = 12345

    # Import Dataset:
    X, y = fetch_compas(usecols=['sex', 'race', 'age_cat', 'priors_count', 'c_charge_degree'], binary_race=True)

    # Set multiclass-indizes:
    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

    # Convert y to pd.Series:
    y = pd.Series(y.factorize(sort=True)[0], index=y.index)

    # Split dataset:
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=seed)

    # One-Hot-Encode categorical data:
    ohe = make_column_transformer(
        (OneHotEncoder(sparse=False, drop='first'), X_train.dtypes == 'category'),
        remainder='passthrough')

    X_train = pd.DataFrame(ohe.fit_transform(X_train), index=X_train.index)
    X_test = pd.DataFrame(ohe.transform(X_test), index=X_test.index)




    ### Baseline model ###
    y_pred_baseline = LogisticRegression(solver='lbfgs').fit(X_train, y_train).predict(X_test)

    # Calculate Accuracy and DI:
    acc_log_reg = accuracy_score(y_test, y_pred_baseline)
    di_log_reg = disparate_impact_ratio(y_test, y_pred_baseline, prot_attr='race')

    print('accuracy: ' + str(acc_log_reg))
    print('disparate_impact: ' + str(di_log_reg))


    ### Preprocessing Algorithm ###
    rew = ReweighingMeta(estimator=LogisticRegression(solver='lbfgs'))

    params = {'estimator__C': [1, 10], 'reweigher__prot_attr': ['race']}

    clf = GridSearchCV(rew, params, scoring='accuracy', cv=5)
    clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    print(clf.best_params_)

    y_pred_rew = clf.predict(X_test)

    # Calculate Accuracy and DI:
    acc_rew = accuracy_score(y_test, y_pred_rew)
    di_rew = disparate_impact_ratio(y_test, y_pred_rew, prot_attr='race')

    print('accuracy: ' + str(acc_rew))
    print('disparate_impact: ' + str(di_rew))


    ### Fair in-processing algorithm ###
    adv_deb = AdversarialDebiasing(prot_attr='race', random_state=seed)
    adv_deb.fit(X_train, y_train)

    y_pred_adv_deb = adv_deb.predict(X_test)

    # Calculate Accuracy and DI:
    acc_adv = accuracy_score(y_test, y_pred_adv_deb)
    di_adv = disparate_impact_ratio(y_test, y_pred_adv_deb, prot_attr='race')

    print('accuracy: ' + str(acc_adv))
    print('disparate_impact: ' + str(di_adv))

    # close tensor-flow board
    adv_deb.sess_.close()



    ### Fair post-processing algorithm ###
    cal_eq_odds = CalibratedEqualizedOdds('race', cost_constraint='fnr', random_state=seed)
    log_reg = LogisticRegression(solver='lbfgs')
    postproc = PostProcessingMeta(estimator=log_reg, postprocessor=cal_eq_odds, random_state=seed)

    postproc.fit(X_train, y_train)
    y_pred_eq_odds = postproc.predict(X_test)

    # Calculate Accuracy and DI:
    acc_eq_odds = accuracy_score(y_test, y_pred_eq_odds)
    di_eq_odds = disparate_impact_ratio(y_test, y_pred_eq_odds, prot_attr='race')

    print('accuracy: ' + str(acc_eq_odds))
    print('disparate_impact: ' + str(di_eq_odds))


    print('pipeline finished')


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    pipeline()
