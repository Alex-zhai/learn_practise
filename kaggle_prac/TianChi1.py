# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.model_selection import KFold
# from mlens.ensemble import SuperLearner
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv("D:/tianchiData/d_train_20180102.csv", encoding='gb2312')
test_df = pd.read_csv("D:/tianchiData/d_test_A_20180102.csv", encoding='gb2312')
train_y = train_df['血糖'].values
train_df.drop('血糖', axis=1, inplace=True)
data_df = pd.concat([train_df, test_df])

data_df["day"] = data_df['体检日期'].apply(lambda x: x.split("/")[0])
data_df["month"] = data_df['体检日期'].apply(lambda x: x.split("/")[1])
data_df["year"] = data_df['体检日期'].apply(lambda x: x.split("/")[2])
data_df = data_df.drop("体检日期", axis=1)
data_df['day'] = data_df['day'].astype(np.int32)
data_df['month'] = data_df['month'].astype(np.int32)
data_df['year'] = data_df['year'].astype(np.int32)
data_df['性别'] = data_df['性别'].map({'男': 1, '女': 0})
data_df = data_df.fillna(data_df.mean())
train_id = train_df.id.values.copy()
test_id = test_df.id.values.copy()
train_x = data_df[data_df.id.isin(train_id)].values
test_x = data_df[data_df.id.isin(test_id)].values

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat - y) ** 2)) / 2


def get_models():
    # # we can also use cv model  example:
    # cross_val_score(KNeighborsRegressor)
    knn = KNeighborsRegressor(n_neighbors=3)
    rfg = RandomForestRegressor(n_estimators=20, random_state=123)
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=123)
    br = BaggingRegressor(random_state=123)
    dtr = DecisionTreeRegressor(random_state=123)
    etr = ExtraTreeRegressor(random_state=123)
    lr = LinearRegression()
    ridge = Ridge(random_state=123)
    lasso = Lasso(random_state=123)
    hr = HuberRegressor()
    xgb = XGBRegressor()
    models = {
        'knn': knn, 'random_forest': rfg, 'gradient_boost': gbr,
        'bagging': br, 'decision_tree': dtr, 'extra_tree': etr, 'lr': lr, 'ridge': ridge,
        'lasso': lasso, 'huber': hr, 'xgboost': xgb
    }
    return models


def train_predict(models):
    models_result_data = np.zeros((y_test.shape[0], len(models)))
    models_result_df = pd.DataFrame(models_result_data)
    cols = []
    for i, (model_name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        models_result_df.iloc[:, i] = model.predict(X_test)
        cols.append(model_name)
    models_result_df.columns = cols
    return models_result_df


def score_models(models_result_df, y):
    for m in models_result_df.columns:
        score = rmspe(y, models_result_df.loc[:, m])
        print("model %s result is %.3f" % (m, score))


models = get_models()
models_result_df = train_predict(models)
score_models(models_result_df, y_test)

print("simple avg ensemble score is %.3f " % (rmspe(models_result_df.mean(axis=1), y_test)))

# step1: define base models
base_models = get_models()

# step2: define meta model
meta_model = GradientBoostingRegressor(
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
        preds = model.predict(test_x)
        result_data[:, i] = preds
    return result_data


base_models_preds = predict_base_models(base_models, x_pred_base)

# step6: train meta model
meta_model.fit(base_models_preds, y_test_base)


def ensemble_predict(base_models, meta_model, inp):
    preds = predict_base_models(base_models, inp)
    return meta_model.predict(preds)


meta_preds = ensemble_predict(base_models, meta_model, X_test)
print("meta ensemble score is %.3f " % (rmspe(meta_preds, y_test)))

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
print("meta cv ensemble score is %.3f " % (rmspe(preds, y_test)))

# get final submission
test_preds = ensemble_predict(cv_base_models, cv_meta_model, test_x)
result = pd.DataFrame({"loss": test_preds})
result.to_csv("xgb_submission_2018_1_19.csv", index=False, header=None)

print("******************use mlen**********************")

# if __name__ == '__main__':
#     super_learner = SuperLearner(folds=10, random_state=123, verbose=2)
#     super_learner.add(list(base_models.values()))
#     super_learner.add_meta(meta_model)
#     super_learner.fit(X_train, y_train)
#     preds = super_learner.predict(X_test)
#     print("meta cv ensemble score of mlen is %.3f " % (rmspe(preds, y_test)))


