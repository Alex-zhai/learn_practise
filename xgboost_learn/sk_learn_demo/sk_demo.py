import pickle
import xgboost as xgb
import numpy as np

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
from sklearn.datasets import load_iris, load_digits, load_boston

rng = np.random.RandomState(31337)

print("Zeros and Ones from the Digits dataset: binary classification")

digits = load_digits(2)
x_data = digits['data']
y_data = digits['target']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)

for train_index, test_index in kf.split(x_data):
    train_x = x_data[train_index]
    train_y = y_data[train_index]
    test_x = x_data[test_index]
    test_y = y_data[test_index]
    xgb_model = xgb.XGBClassifier().fit(train_x, train_y)
    preds = xgb_model.predict(test_x)
    print(classification_report(test_y, preds))
    print("Accuracy is: %.2f" % (xgb_model.score(test_x, test_y)))
    print(confusion_matrix(test_y, preds))

print("Iris: multiclass classification")

iris = load_iris()
x_data = iris['data']
y_data = iris['target']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)

for train_index, test_index in kf.split(x_data):
    train_x = x_data[train_index]
    train_y = y_data[train_index]
    test_x = x_data[test_index]
    test_y = y_data[test_index]
    xgb_model = xgb.XGBClassifier().fit(train_x, train_y)
    preds = xgb_model.predict(test_x)
    print(classification_report(test_y, preds))
    print("Accuracy is: %.2f" % (xgb_model.score(test_x, test_y)))
    print(confusion_matrix(test_y, preds))

print("Boston Housing: regression")
boston = load_boston()
x_data = boston['data']
y_data = boston['target']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)

for train_index, test_index in kf.split(x_data):
    train_x = x_data[train_index]
    train_y = y_data[train_index]
    test_x = x_data[test_index]
    test_y = y_data[test_index]
    xgb_model = xgb.XGBRegressor().fit(train_x, train_y)
    preds = xgb_model.predict(test_x)
    print(mean_squared_error(test_y, preds))

print("Parameter optimization")
x_data = boston['data']
y_data = boston['target']
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                               'n_estimators': [50, 100, 200]}, verbose=1)
clf.fit(x_data, y_data)
print(clf.best_params_)
print(clf.best_score_)

print("Pickling sklearn API models")
pickle.dump(clf, open("best_boston.pkl", "wb"))
clf2 = pickle.load(open("best_boston.pkl", "rb"))
print(np.allclose(clf.predict(x_data), clf2.predict(x_data)))

# early_stop
x = digits['data']
y = digits['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

clf = xgb.XGBClassifier()
clf.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=[(x_test, y_test)])
