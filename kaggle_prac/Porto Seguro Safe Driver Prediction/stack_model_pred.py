import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

train = pd.read_csv("D:/kaggle数据集/Porto Seguro Safe Driver Prediction/train.csv")
test = pd.read_csv("D:/kaggle数据集/Porto Seguro Safe Driver Prediction/test.csv")

id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target', 'id'], axis=1)
test = test.drop(['id'], axis=1)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)
test = test.drop(col_to_drop, axis=1)

train = train.fillna(train.mean())
test = test.fillna(test.mean())

cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
    temp = pd.get_dummies(pd.Series(train[column]))
    train = pd.concat([train, temp], axis=1)
    train = train.drop([column], axis=1)

for column in cat_features:
    temp = pd.get_dummies(pd.Series(test[column]))
    test = pd.concat([test, temp], axis=1)
    test = test.drop([column], axis=1)

print(train.values.shape, test.values.shape)

train_x, train_y, test_x = train.values, target_train, test.values


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)


def get_models():
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    linear_svc = LinearSVC()
    knn = KNeighborsClassifier(n_neighbors=3)
    rfg = RandomForestClassifier(n_estimators=50, random_state=123)
    gbc = GradientBoostingClassifier(n_estimators=400, random_state=123)
    ada = AdaBoostClassifier(random_state=123)
    dtr = DecisionTreeClassifier(random_state=123)
    lr = LogisticRegression(random_state=123)
    xgb = XGBClassifier()
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
    models = {
        'knn': knn, 'random_forest': rfg, 'gradient_boost': gbc,
        'nb': nb, 'decision_tree': dtr, 'ada_tree': ada, 'lr': lr,
        'svm': svc, 'linear_svm': linear_svc, 'xgboost': xgb, 'mlp': mlp
    }
    return models


def train_predict(models):
    models_result_data = np.zeros((y_test.shape[0], len(models)))
    models_result_df = pd.DataFrame(models_result_data)
    cols = []
    for i, (model_name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        models_result_df.iloc[:, i] = model.predict_proba(X_test)[:, 1]
        cols.append(model_name)
    models_result_df.columns = cols
    return models_result_df


def score_models(models_result_df, y):
    for m in models_result_df.columns:
        score = roc_auc_score(y, models_result_df.loc[:, m])
        print("model %s result is %.3f" % (m, score))


models = get_models()
models_result_df = train_predict(models)
score_models(models_result_df, y_test)

print("simple avg ensemble score is %.3f " % (roc_auc_score(models_result_df.mean(axis=1), y_test)))

# step1: define base models
base_models = get_models()

# step2: define meta model
meta_model = GradientBoostingClassifier(
    n_estimators=1000,
    max_features=5,
    max_depth=4,
    subsample=0.5,
    learning_rate=0.005,
    random_state=123
)

# step3: generate train and test data for base models
x_train_base, x_pred_base, y_train_base, y_test_base = train_test_split(X_train, y_train,
                                                                        test_size=0.5, random_state=123)


# step4: train base models on data
def train_base_models(base_models, train_x, train_y):
    for i, (model_name, model) in enumerate(base_models.items()):
        model.fit(train_x, train_y)


train_base_models(base_models, x_train_base, y_train_base)


# step5: generate predictions of base models
def predict_base_models(base_models, test_x):
    result_data = np.zeros((test_x.shape[0], len(base_models)))
    for i, (model_name, model) in enumerate(base_models.items()):
        preds = model.predict_proba(test_x)
        result_data[:, i] = preds[:, 1]
    return result_data


base_models_preds = predict_base_models(base_models, x_pred_base)

# step6: train meta model
meta_model.fit(base_models_preds, y_test_base)


def ensemble_predict(base_models, meta_model, inp):
    preds = predict_base_models(base_models, inp)
    return meta_model.predict_proba(preds)[:, 1]


meta_preds = ensemble_predict(base_models, meta_model, X_test)
print("meta ensemble score is %.3f " % (roc_auc_score(meta_preds, y_test)))

print("******************cv ensemble**********************")


def stacking(base_models, meta_model, X, y, generator):
    print("Fitting final base learners...", end="")
    train_base_models(base_models, X, y)
    # Generate predictions for training meta learners
    cv_preds, cv_y = [], []
    for i, (train_idx, test_idx) in enumerate(generator.split(X)):
        fold_xtrain, fold_ytrain = X[train_idx, :], y[train_idx]
        fold_xtest, fold_ytest = X[test_idx, :], y[test_idx]
        fold_base_models = {name: clone(model) for name, model in base_models.items()}
        train_base_models(fold_base_models, fold_xtrain, fold_ytrain)
        fold_preds_base = predict_base_models(fold_base_models, fold_xtest)
        cv_preds.append(fold_preds_base)
        cv_y.append(fold_ytest)
    print("CV-predictions done")
    cv_preds = np.vstack(cv_preds)
    cv_y = np.hstack(cv_y)
    # Train meta learner
    meta_model.fit(cv_preds, cv_y)
    return base_models, meta_model

cv_base_models, cv_meta_model = stacking(get_models(), clone(meta_model), X_train, y_train, KFold(5))
preds = ensemble_predict(cv_base_models, cv_meta_model, X_test)
print("meta cv ensemble score is %.3f " % (roc_auc_score(preds, y_test)))

# get final submission
test_preds = ensemble_predict(cv_base_models, cv_meta_model, test_x)
result = pd.DataFrame({"loss": test_preds})
result.to_csv("kaggle_xgb_submission_2018_1_19.csv", index=False, header=None)