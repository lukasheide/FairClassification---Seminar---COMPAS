from fairness.data.objects.Data import Data
from fairness.data.objects.list import DATASETS
from fairness.data.objects.ProcessedData import ProcessedData
from sklearn.linear_model import LogisticRegression as SKLearn_LR, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from Scripts.metrics.fairnessMetrics import FairnessMetrics, CausalDiscrimination
#import aif360.algorithms.preprocessing.disparate_impact_remover
#import aif360.sklearn.preprocessing.
import aif360.algorithms.preprocessing.disparate_impact_remover

import pandas as pd
import time
import numpy as np
import dataset
from aif360 import datasets

url = 'https://raw.githubusercontent.com/algofairness/fairness-comparison' \
      '/80b34d25bb9b0387691c6cb8651f0e40edd262c8/fairness/data/preprocessed/propublica' \
      '-recidivism_processed.csv'

data_raw = pd.read_csv(url)
dis_im_rem = aif360.algorithms.preprocessing.disparate_impact_remover.DisparateImpactRemover(repair_level=1.0, sensitive_attribute='race')


time.sleep(0.1)
pre_dataset = datasets.CompasDataset()
#print(pre_dataset)
#dataset = datasets.StandardDataset(data_raw, label_name= 'race' , favorable_classes= '1' , protected_attribute_names= ['race'], privileged_classes='American')
df = dis_im_rem.fit_transform(pre_dataset)
#df = dis_im_rem.fit_predict(data_raw.head(3))
print(df)
print(type(df))
df= pd.DataFrame(df.convert_to_dataframe())
print(type(df))
print(df())
#print(df.columns())
#data_raw.drop('c_jail_in',axis=0)
#pd.DataFrame(df)
#print(type(df))
