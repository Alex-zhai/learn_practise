import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

dtrain = xgb.DMatrix('../agaricus.txt.train')
dtest = xgb.DMatrix('../agaricus.txt.test')


watchlist = [(dtest, 'eval'), (dtrain, 'train')]

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

gsearch1 = GridSearchCV(estimator=XGBClassifier(max_depth=6, learning_rate=0.05, min_child_weight=0.5, subsample=0.8,
                                               objective='reg:linear', gamma=0.1, scale_pos_weight=1, seed=27),
                        param_grid=param_test1, scoring='neg_mean_squared_error', n_jobs=4, iid=False, cv=5)

bst = xgb.train(param, dtrain, num_boost_round=1, evals=watchlist)
gsearch1.fit()
ptrain = bst.predict(dtrain, output_margin=True)
ptest = bst.predict(dtest, output_margin=True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)

bst = xgb.train(param, dtrain, 1, watchlist)