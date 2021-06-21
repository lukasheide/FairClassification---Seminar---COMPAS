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
    # import ProRepublica Recidivism Data:
    url = 'https://raw.githubusercontent.com/algofairness/fairness-comparison/80b34d25bb9b0387691c6cb8651f0e40edd262c8/fairness/data/preprocessed/propublica-recidivism_processed.csv'

    # to be precise the data is not the raw data but already preprocessed by ProRepublica. However, some preprocessing
    # steps are still missing for our analysis, which we will add in the following.
    data_raw = pd.read_csv(url)

    ### Column descriptions:
    # juv_fel_count = number of juvenile felonies
    # juv_misd_count = number of juvenile misdeavors
    # juv_other_count = number of other juvenile wrongdoing
    # priors count = number of priors
    # c_charge degree = current charge degree (f = felony, m = misdemeanor)
    # c_charge desc = description of charge
    # decile_score = prediction by COMPAS algorithm (risk of recidivism on a scale from 1 (low risk) to 10 (high risk))
    # score_text: 1-4 = low risk, 5-7 = medium risk, 8-10 = high risk

    ### Cleaning:
    # Drop columns:
    # 1) c_charge_desc (389 unique textual descriptions -> can not be used in quantitative analysis)
    # 2) decile_score, score_text -> both are result columns of what the actual COMPAS algorithm predicted
    # 3) sex-race -> combination of sex and race which is not used in our analysis
    df = data_raw.drop(['c_charge_desc', 'decile_score', 'score_text', 'sex-race'], axis=1)

    # Target variable: two_year_recid
    target = 'two_year_recid'
    # Input variables:
    features = [f for f in df.columns if f not in target]
    # Sensitive attribute: race
    sensitive_attr = 'race'

    ### Transformation
    # Encoding:
    # 1) sex (1 if male, 0 otherwise)
    df.sex = (df['sex'] == 'Male').astype(int)

    # 2) age-cat (one-hot encode: Greater than 45, 25-45, Less than 25)
    df = pd.get_dummies(df, prefix=['age_'], columns=['age_cat'], drop_first=True)

    # 3) race (0 if african-american, 1 otherwise)
    df.race = (df['race'] == 'African-American').astype(int)
    df.race = 1 - df.race

    # 4) c_charge_degree (1 if felony, 0 if misdemeanor)
    df.c_charge_degree = (df['c_charge_degree'] == 'F').astype(int)

    # 5) transform target variable so that positive prediction implies that person does not recidivate and vice versa
    # (1 if does not recidivate, 0 if recidivates)
    df.two_year_recid = 1 - df.two_year_recid

    ##
    # y = 1 -> positive label -> does not recidivate
    # y = 0 -> negative label -> does recidivate
    # s = 1 -> privileged class = unprotected class
    # s = 0 -> unprivileged class = protected class

    # => expectation: false negative rate higher in unprotected class compared to protected class
    # -> P(Y_hat = 0 | Y = 1, S = 0) > P(Y_hat = 0 | Y = 1, S = 1)


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

    print('end reached')


if __name__ == '__main__':
    pipeline()
