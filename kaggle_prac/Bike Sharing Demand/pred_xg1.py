import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def MinMaxScale(data):
    temp1 = data - np.min(data, axis=0)
    temp2 = np.max(data, axis=0) - np.min(data, axis=0)
    return temp1 / (temp2 + 1e-7)


def load_data(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv")
        data = data_df.values[:, 1:]
        labels = data_df["datetime"].values
    else:
        data_df = pd.read_csv("train.csv")
        data = data_df.values[:, 1:-3]
        labels = data_df["count"].values
    return data, labels


x_train, y_train = load_data(0)
print(x_train.shape, y_train.shape)

train_len = len(x_train)

x_test, test_id = load_data(1)
print(x_test.shape)
x_all = np.vstack((x_train, x_test))
print(x_all.shape)

x_all_scale = MinMaxScale(x_all)

x_train = x_all_scale[:train_len]
x_test = x_all_scale[train_len:]

params = {'max_depth': list(range(2, 7)), 'n_estimators': list(range(100, 1100, 200)),
          'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgb = XGBClassifier()
xgb_cv = GridSearchCV(xgb, params, verbose=1)
xgb_cv.fit(x_train, y_train)
print(xgb_cv.best_score_)
print(xgb_cv.best_params_)

preds = xgb_cv.predict(x_test)

submission = ['datetime,count']
for id, pred in zip(test_id, preds):
    if int(pred) < 0:
        pred = 0
    submission.append('{0},{1}'.format(id, int(pred)))
submission = '\n'.join(submission)

with open("submission_xg.csv", "w") as outfile:
    outfile.write(submission)
