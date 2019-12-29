# -*- coding:utf-8 -*-

import xgboost as xgb
import scipy.sparse
import pickle
import numpy as np

dtrain = xgb.DMatrix("agaricus.txt.train")
dtest = xgb.DMatrix("agaricus.txt.test")

# set xgboost parameters
param = {
    'max_depth': 2,
    'eta': 1,
    'silent': 1,
    'objective': 'binary:logistic'
}
num_round = 2
watchlist = [(dtest, 'eval'), (dtrain, 'train')]

bst = xgb.train(param, dtrain, num_round, evals=watchlist)

preds = bst.predict(dtest)

print(preds)

y = dtest.get_label()
print(y)

error_count = sum(y != (preds > 0.5))
error_rate = float(error_count) / len(preds)

print("Total nums: %d" % (len(y)))
print("pred error num is", error_count)
print("pred error rate is %.2f%%" % (100 * error_rate))

bst.save_model("0001.model")
bst.dump_model("dump.raw.txt")

dtest.save_binary('dtest.buffer')

# save model
bst.save_model('xgb.model')

bst2 = xgb.Booster(model_file="xgb.model")

dtest2 = xgb.DMatrix("dtest.buffer")
pred2 = bst2.predict(dtest2)

assert np.sum(np.abs(preds - pred2)) == 0

pks = pickle.dumps(bst2)
bst3 = pickle.loads(pks)
pred3 = bst3.predict(dtest2)

assert np.sum(np.abs(preds - pred3)) == 0