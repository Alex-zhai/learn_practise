import numpy as np
import xgboost as xgb

dtrain = xgb.DMatrix('../agaricus.txt.train')
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 2

print('running cross validation')
#  metric_name:mean_value+std_value
xgb.cv(param, dtrain, num_round, nfold=5, metrics={'error'}, seed=0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

# [iteration]  metric_name:mean_value

res = xgb.cv(param, dtrain, num_boost_round=10, nfold=5, metrics={'error'},
             seed=0, callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                                xgb.callback.early_stop(3)])
print(res)

print('running cross validation, with preprocessing function')

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

xgb.cv(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=0,
       fpreproc=fpreproc)

print('running cross validation, with cutomsized loss function')
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', float(sum(labels!=(preds>0.0)))/len(labels)

param = {'max_depth': 2, 'eta': 1, 'silent': 1}
xgb.cv(param, dtrain, num_round, nfold=5, seed=0, obj=logregobj, feval=evalerror)