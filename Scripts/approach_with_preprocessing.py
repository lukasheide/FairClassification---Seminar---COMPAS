from fairness.data.objects.Data import Data
from fairness.data.objects.list import DATASETS
from fairness.data.objects.ProcessedData import ProcessedData
from sklearn.linear_model import LogisticRegression as SKLearn_LR, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from Scripts.metrics.fairnessMetrics import FairnessMetrics, CausalDiscrimination


### Import COMPAS data from Friedler repository: https://github.com/algofairness/fairness-comparison/blob/master/fairness/preprocess.py


import pandas as pd


def pipeline():
    url = 'https://raw.githubusercontent.com/algofairness/fairness-comparison/80b34d25bb9b0387691c6cb8651f0e40edd262c8/fairness/data/preprocessed/propublica-recidivism_processed.csv'

    data_raw = pd.read_csv(url)
    import aif360.algorithms.preprocessing.disparate_impact_remover
    import aif360.algorithms.preprocessing.lfr
    from aif360 import datasets
    data_raw = pd.read_csv(url)
    dis_im_rem = aif360.algorithms.preprocessing.disparate_impact_remover.DisparateImpactRemover(
        repair_level=1.0, sensitive_attribute='race')


    data_raw = datasets.CompasDataset(label_name='two_year_recid', favorable_classes=[0], protected_attribute_names=['race'], privileged_classes=[['Caucasian']], instance_weights_name=None, categorical_features=['age_cat', 'c_charge_degree', 'c_charge_desc'], features_to_keep=['sex','age', 'age_cat', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc', 'two_year_recid'], features_to_drop=['sex'], na_values=[], metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}], 'protected_attribute_maps': [{1.0: 'Caucasian', 0.0: 'American'}]})
    #

    #apply disparateImpactRemover
    data_raw = dis_im_rem.fit_transform(data_raw)
    data_raw = data_raw.convert_to_dataframe()[0]
    ### Cleaning:
    # Drop columns:
    # 1) c_charge_desc (389 unique textual descriptions -> can not be used in quantitative analysis)
    # 2) decile_score, score_text -> both are result columns of what the actual COMPAS algorithm predicted
    # 3) sex-race -> combination of sex and race which is not used in our analysis
    #df = data_raw[['age', 'race', 'juv_fel_count', 'two_year_recid', 'juv_misd_count', 'juv_other_count','priors_count']].copy()
    df=data_raw
    df.drop(['two_year_recid'],axis=1)

    # Target variable: two_year_recid
    target = 'two_year_recid'
    # Input variables:
    #features = [f for f in df.columns if f not in target]
    # Sensitive attribute: race
    sensitive_attr = 'race'

    ### Normalization:
    # normalize using sklearn scaler:
    normalizer = MinMaxScaler()
    df_normalized = normalizer.fit_transform(df)

    # transform result back into dataframe:
    df_normalized = pd.DataFrame(df_normalized, index=df.index, columns=df.columns)

    ### Data Segregation:
    # split into train/test set:
    seed = 42
    train_set = df_normalized.sample(frac=0.8, random_state=seed)
    test_set = df_normalized.drop(train_set.index)

    # split into X (input variables) and Y (target variable)
    # 1) training:
    x_train = train_set
    y_train = pd.DataFrame(train_set.pop(target), columns=[target])

    # 2) testing:
    x_test = test_set
    y_test = pd.DataFrame(test_set.pop(target), columns=[target])


    ##### Modelling Phase #####

    ### Baseline model: Logistic regression
    log_reg = LogisticRegression()

    # train model on training data:
    log_reg.fit(x_train, y_train)

    # get predictions:
    y_pred_test = pd.DataFrame(log_reg.predict(x_test), index=y_test.index, columns=['y_pred_test'])

    # compute model accuracy:
    result = {'Train_Score': log_reg.score(x_train, y_train),
              'Held_Out_Score': log_reg.score(x_test, y_test)}


    ### Compute metrics:
    log_reg.metrics = {'correctness': {}, 'fairness': {}, 'efficiency': {}}
    ## Correctness:

    log_reg.metrics['correctness'] = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test),
        'auc': roc_auc_score(y_test, y_pred_test)
    }
    print(log_reg.metrics)

    ##  Fairness:
    # 1) DI
    fair_metrics = FairnessMetrics(y_pred=y_pred_test, y_actual=y_test, x_test=x_test, sensitive_attr=sensitive_attr)

    di = fair_metrics.disparate_impact()
    tprb = fair_metrics.true_positive_rate_balance()
    tnrb = fair_metrics.true_negative_rate_balance()

    log_reg.metrics['fairness'] = {
        'di':di,
        'tprb':tprb,
        'tnrb':tnrb
    }

    cd = CausalDiscrimination(y_pred=y_pred_test, y_test=y_test, x_test=x_test, sensitive_attr=sensitive_attr, model=log_reg)
    cd.compute_causal_discrimination()

    import aif360.sklearn.metrics.metrics
    #di = aif360.sklearn.metrics.disparate_impact_ratio()
    #y_true=df['two_year_recid'], y_pred=y_pred_test,
    #print('di '+ str(di))
    #print('end reached')


if __name__ == '__main__':
    pipeline()
