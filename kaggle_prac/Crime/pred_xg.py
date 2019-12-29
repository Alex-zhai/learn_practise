# -*- coding: utf-8 -*-

import xgboost as xgb
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing

train_date = pd.read_csv('train.csv')
test_date = pd.read_csv('test.csv')

print(train_date.info())  # 没有缺失值
print(train_date.head(2))


# 首先处理特征  date中抽出年、月、日、时、分
def parse_date(date):
    dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    hour = abs(hour - 30)
    return minute, hour, day, month, year


train_date['minute'], train_date['hour'], train_date['day'], train_date['month'], train_date['year'] = \
    zip(*train_date['Dates'].apply(func=parse_date))

test_date['minute'], test_date['hour'], test_date['day'], test_date['month'], test_date['year'] = \
    zip(*test_date['Dates'].apply(func=parse_date))


# 处理Address
def address(addr):
    if addr.find("/") != -1:
        return 1
    else:
        return 0


train_date['Address'] = train_date['Address'].apply(func=address)
test_date['Address'] = test_date['Address'].apply(func=address)

# 处理X和Y
x = preprocessing.scale(train_date['X'])
train_date = pd.DataFrame(data=x, columns=['xx']).join(train_date)
y = preprocessing.scale(train_date['Y'])
train_date = pd.DataFrame(data=y, columns=['yy']).join(train_date)
x = preprocessing.scale(test_date['X'])
test_date = pd.DataFrame(data=x, columns=['xx']).join(test_date)
y = preprocessing.scale(test_date['Y'])
test_date = pd.DataFrame(data=y, columns=['yy']).join(test_date)

# 处理 PdDistrict DayOfWeek
train_date = pd.get_dummies(data=train_date['PdDistrict']).join(train_date)
train_date = pd.get_dummies(data=train_date['DayOfWeek']).join(train_date)
test_date = pd.get_dummies(data=test_date['PdDistrict']).join(test_date)
test_date = pd.get_dummies(data=test_date['DayOfWeek']).join(test_date)

print(train_date.info())

# label
labelset = pd.DataFrame(data=train_date['Category'], columns=['Category'])
le = preprocessing.LabelEncoder()

le.fit(labelset['Category'].unique().tolist())
labelData = le.transform(labelset['Category'].values)
cList = le.classes_.tolist()
testId = test_date['Id']
print(cList)

train_date.drop(['Dates', 'Descript', 'DayOfWeek', 'PdDistrict', 'X', 'Y', 'Resolution', 'Category'],
                axis=1, inplace=True)
test_date.drop(['Dates', 'X', 'Y', 'DayOfWeek', 'PdDistrict', 'Id'], axis=1, inplace=True)

train_date = train_date.iloc[:, :].values
test_date = test_date.iloc[:, :].values

offset = 600000
xgtrain = xgb.DMatrix(train_date[:offset, :], label=labelData[:offset])
xgeval = xgb.DMatrix(train_date[offset:, :], label=labelData[offset:])
xgtest = xgb.DMatrix(test_date)

params = {"booster": "gbtree", "objective": "multi:softprob", "num_class": 39, "max_delta_step": 1, "max_depth": 6}
watchlist = [(xgtrain, 'train'), (xgeval, 'val')]
xgb_model = xgb.train(params, xgtrain, num_boost_round=150, evals=watchlist, early_stopping_rounds=2)
preds = np.column_stack((testId, xgb_model.predict(xgtest, ntree_limit=xgb_model.best_iteration)))
preds = [[int(i[0])] + i[1:] for i in preds]
print(preds[0])

cList.insert(0, "Id")
with open("submission_xg.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(cList)
    writer.writerow(preds)
