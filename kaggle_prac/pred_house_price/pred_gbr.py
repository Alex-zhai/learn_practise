# coding:utf-8

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from scipy.stats import skew

import xgboost as xgb

def load_file(is_test=False):
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    data_df = pd.concat((train.loc[:, cols], test.loc[:, cols]))
    numeric_feats = data_df.dtypes[data_df.dtypes != 'object'].index  # 取的是num类型属性名称
    skewed_feats = data_df[numeric_feats].apply(lambda x: skew(x.dropna()))
    #print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    print(skewed_feats)
    skewed_feats = skewed_feats.index
    #print(skewed_feats)
    data_df[skewed_feats] = np.log1p(data_df[skewed_feats])
    print(data_df.head())
    data_df = pd.get_dummies(data_df)
    data_df = data_df.fillna(data_df.mean())

    return data_df, train, test

all_data, train, test = load_file()
x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
#y_train = np.expand_dims(np.log1p(train.SalePrice), axis=1)
y_train = np.log1p(train.SalePrice)

print(x_train.shape, y_train.shape)
print(all_data.head())

gbr = GradientBoostingClassifier()
gbr.fit(x_train, y_train)
test_price = gbr.predict(x_test)
test_price = np.expm1(test_price)

ids = pd.read_csv("test.csv").Id

submission = ['Id,SalePrice']

for id, prediction in zip(ids, test_price):
    submission.append('{0},{1}'.format(id, prediction))

submission = '\n'.join(submission)

with open('submission1.csv', 'w') as outfile:
    outfile.write(submission)