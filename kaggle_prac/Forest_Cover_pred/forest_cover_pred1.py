import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score

def MinMaxScale(data):
    temp1 = data - np.min(data, axis=0)
    temp2 = np.max(data, axis=0) - np.max(data, axis=0)
    return temp1 / (temp2 + 1e-7)

def get_data(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv.zip", compression='zip')
        data = data_df.values[:, 1:]  # ignore ID
        lables = data_df["Id"].values
    else:
        data_df = pd.read_csv("train.csv.zip", compression='zip')
        data = data_df.values[:, 1:-1] # ignore Id and cover_type
        lables = data_df["Cover_Type"].values
    #print(data_df.head(n=1))
    return lables, data

y_train, x_train = get_data(0)
y_train -= 1
#y_train = np.expand_dims(y_train, 1)
train_set_len = len(x_train)

test_id, x_test = get_data(1)
print(x_train.shape, x_test.shape)

x_all = np.vstack((x_train, x_test))
print(x_all.shape)

scale_x = MinMaxScale(x_all)

x_train = scale_x[:train_set_len]
x_test = scale_x[train_set_len:]

print(x_train.shape, x_test.shape)

gbc_model = GradientBoostingClassifier()

# use feature selection
percentiles = range(1, 100, 2)
# results = []
# for i in percentiles:
#     fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
#     x_train_fs = fs.fit_transform(x_train, y_train)
#     scores = cross_val_score(gbc_model, x_train_fs, y_train, cv=5)
#     results = np.append(results, scores.mean())
# print(results)

#opt = np.where(results == results.max())[0]
print('Optimal number of feature %d' % percentiles[25])

fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=51)
x_train_fs = fs.fit_transform(x_train, y_train)
gbc_model.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
pred_values = gbc_model.predict(x_test_fs)
print(pred_values)
pred_values += 1


submission = ['Id,Cover_Type']
for i, c in zip(test_id, pred_values):
    submission.append('{0},{1}'.format(i, int(c)))
submission = '\n'.join(submission)
with open("submission1.csv", "w") as outfile:
    outfile.write(submission)