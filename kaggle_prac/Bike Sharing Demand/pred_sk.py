import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

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

gbr = GradientBoostingClassifier()
gbr.fit(x_train, y_train)
preds = gbr.predict(x_test)

submission = ['datetime,count']
for id, pred in zip(test_id, preds):
    if int(pred) < 0:
        pred = 0
    submission.append('{0},{1}'.format(id, int(pred)))
submission = '\n'.join(submission)

with open("submission_sk.csv", "w") as outfile:
    outfile.write(submission)