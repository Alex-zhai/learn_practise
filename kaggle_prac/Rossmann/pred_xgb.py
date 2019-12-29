# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import operator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def build_features(features, data):
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])
    # 类别型特征
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
                              (data.Month - data.CompetitionOpenSinceMonth)

    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
                        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    features.append('IsPromoMonth')
    month2str = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', \
        7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1
    return data


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)


types = {
    'CompetitionOpenSinceYear': np.dtype(int),
    'CompetitionOpenSinceMonth': np.dtype(int),
    'StateHoliday': np.dtype(str),
    'Promo2SinceWeek': np.dtype(int),
    'SchoolHoliday': np.dtype(float),
    'PromoInterval': np.dtype(str)
}
train = pd.read_csv('train.csv.zip', compression='zip', parse_dates=[2], dtype=types)
test = pd.read_csv('test.csv.zip', compression='zip', parse_dates=[3], dtype=types)
store = pd.read_csv('store.csv.zip', compression='zip')
# print(train.info())
# print(test.info())
# print(store.info())
# # 发现test数据集上open有一个缺失值
# print(test['Open'].value_counts())
# # 使用出现频率最高的特征值来填充缺失值
train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

train = train[train["Open"] != 0]
train = train[train["Sales"] > 0]

train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

# print(train.info())
# print(test.info())
features = []
build_features(features, train)
build_features([], test)
# print(features)

print(train.info())
print(test.info())

params = {
    'objective': 'reg:linear',
    'booster': 'gbtree',
    'eta': 0.3,
    'max_depth': 10,
    'subsample': 0.9,  # 用于训练模型的子样本占整个样本集合的比例
    'colsample_bytree': 0.7,  # 在建立树时对特征采样的比例
    'silent': 1,  # 取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息
    'seed': 1301
}
nums_round = 300

# prepare x_train, y_train, x_test
x_train, x_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(x_train.Sales)
y_valid = np.log1p(x_valid.Sales)

# train
dtrain = xgb.DMatrix(x_train[features], y_train)
dvalid = xgb.DMatrix(x_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
xgb_model = xgb.train(params, dtrain, nums_round, evals=watchlist,
                      early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

xgb_model.save_model('xgb.model')

# validating
yhat = xgb_model.predict(xgb.DMatrix(x_valid[features]))
error = rmspe(x_valid.Sales.values, np.expm1(yhat))
print("RMSPE: {:.6f}".format(error))

# get preds
dtest = xgb.DMatrix(test[features])
test_preds = xgb_model.predict(dtest)

result = pd.DataFrame({"Id": test["Id"], "Sales": np.expm1(test_preds)})
result.to_csv("xgb_submission.csv", index=False)

# get feat importance
create_feature_map(features)
importance = xgb_model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['features', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

# featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# fig_featp = featp.get_figure()
# plt.show()
# fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
