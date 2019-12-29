# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# print(train.info())
# print(test.info())

# choose useful features by hand
selected_feat = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
x_train = train[selected_feat]
x_test = test[selected_feat]

y_train = train['Survived']

print(x_train['Embarked'].value_counts())
print(x_test['Embarked'].value_counts())

# 对于类别形特征，使用频率最高的特征值来填充缺失值
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

# 对于数值型特征，使用平均值来填充缺失值
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

# print(x_train.info())
# print(x_test.info())

# 特征向量化
dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
print(dict_vec.feature_names_)

x_test = dict_vec.transform(x_test.to_dict(orient='record'))


params = {'max_depth': list(range(2, 7)), 'n_estimators': list(range(100, 1100, 200)), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, verbose=1)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

pred_values = gs.predict(x_test)

xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_values})
xgbc_submission.to_csv("submission_xgbc_best.csv", index=False)