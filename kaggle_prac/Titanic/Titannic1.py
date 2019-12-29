import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def MinMaxScale(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def load_file(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv")
    else:
        data_df = pd.read_csv("train.csv")

    cols = ["Pclass", "Sex", "Age", "Fare", "Embarked_0", "Embarked_1", "Embarked_2"]
    data_df["Sex"] = data_df["Sex"].map({'female': 0, 'male': 1}).astype(int)

    # handle missing values of 'Age'
    data_df["Age"] = data_df["Age"].fillna(data_df["Age"].mean())
    data_df["Fare"] = data_df["Fare"].fillna(data_df["Fare"].mean())

    data_df["Embarked"] = data_df["Embarked"].fillna('S')
    data_df["Embarked"] = data_df["Embarked"].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    data_df = pd.concat([data_df, pd.get_dummies(data_df["Embarked"], prefix="Embarked")], axis=1)
    data = data_df[cols].values
    if is_test:
        sing_col = data_df["PassengerId"].values
    else:
        sing_col = data_df["Survived"].values
    return sing_col, data

y_train, x_train = load_file(0)
y_train = np.expand_dims(y_train, 1)
train_len = len(x_train)
pred_pass_id, x_test = load_file(1)
x_all_data = np.vstack((x_train, x_test))
scale_data = MinMaxScale(x_all_data)
x_train = scale_data[:train_len]
x_test = scale_data[train_len:]
print(x_train.shape, x_test.shape)

dt_model = GradientBoostingClassifier()
dt_model = dt_model.fit(x_train, y_train)
pred_value = dt_model.predict(x_test)
print(pred_value)

submission = ['PassengerId,Survived']
for id, prediction in zip(pred_pass_id, pred_value):
    submission.append('{0},{1}'.format(id, int(prediction)))
submission = '\n'.join(submission)
with open('submission1.csv', 'w') as outfile:
    outfile.write(submission)