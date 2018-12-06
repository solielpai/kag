# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.model_selection import KFold
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils
import os
#a=os.listdir("../input")
path=r'C:\Users\Administrator\Desktop\akk'
#print(a)
train=pd.read_csv(path+r'/train.csv')
tra=['first_active_month', 'card_id', 'feature_1', 'feature_2', 'feature_3','target']
#merchants=pd.read_csv('../input/merchants.csv')
#mer=['merchant_id', 'merchant_group_id', 'merchant_category_id',
#       'subsector_id', 'numerical_1', 'numerical_2', 'category_1',
#       'most_recent_sales_range', 'most_recent_purchases_range',
#       'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
#       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
#       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12',
#       'category_4', 'city_id', 'state_id', 'category_2']

test=pd.read_csv(path+r'/test.csv')
te=['first_active_month', 'card_id', 'feature_1', 'feature_2', 'feature_3']
#historical_transactions=pd.read_csv('../input/historical_transactions.csv')
#
#his=['authorized_flag', 'card_id', 'city_id', 'category_1', 'installments',
#       'category_3', 'merchant_category_id', 'merchant_id', 'month_lag',
#       'purchase_amount', 'purchase_date', 'category_2', 'state_id',
#       'subsector_id']
#Data_Dictionary=pd.read_excel('../input/Data_Dictionary.xlsx')
#print(Data_Dictionary)
#print(Data_Dictionary.columns)
#new_merchant_transactions=pd.read_csv('../input/new_merchant_transactions.csv')
#
#ne=['authorized_flag', 'card_id', 'city_id', 'category_1', 'installments',
#       'category_3', 'merchant_category_id', 'merchant_id', 'month_lag',
#       'purchase_amount', 'purchase_date', 'category_2', 'state_id',
#       'subsector_id']
#sam=pd.read_csv('../input/sample_submission.csv')
tar=test['card_id']
ta=train['target'].values
a=[max(ta),min(ta)]
print(a)
test['target']=19
data=pd.concat([train,test],axis=0)
print(data.columns)
print(data.shape,train.shape,test.shape)
import re
t=[]

m=[]
data['first_active_month']=data['first_active_month'].fillna('1-13')
for x in data['first_active_month']:
    
    t.append(int(str(x).split('-')[0]))
    m.append(int(str(x).split('-')[1]))
        
data['year']=t
data['mon']=m
a=['first_active_month', 'feature_1', 'feature_2', 'feature_3','year','mon']
for x in a:
    data[x]=data[x].astype('category')
train=data[data['target']<18.5]
test=data[data['target']>18.5]
label=train['target']
sub=pd.DataFrame()
sub['card_id']=test['card_id']
train.drop(['card_id','target'],axis=1,inplace=True)
test.drop(['card_id','target'],axis=1,inplace=True)
param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, label.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=label.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=label.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    oof[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
sub['target']= predictions/5
sub.to_csv(path+'submission.csv',index=False)  