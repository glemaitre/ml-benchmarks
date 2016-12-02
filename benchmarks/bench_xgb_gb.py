import sys, misc
import numpy as np
from datetime import datetime

N_ESTIMATORS = 1
LEARNING_RATE = 0.1
MIN_IMPURITY_SPLIT = 1e-7
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 1
SUBSAMPLES = 1.
N_THREADS = 1
RND_SEED = 42


def bench_xgb(X, y, T, valid):
#
#       .. xgboost ..
#
    import xgboost as xgb

    start = datetime.now()

    # Create the data matrix
    xgb_training = xgb.DMatrix(X, label=y, missing=None, weight=None,
                               silent=False, feature_names=None,
                               feature_types=None)

    xgb_testing = xgb.DMatrix(T, label=valid, missing=None, weight=None,
                              silent=False, feature_names=None,
                              feature_types=None)

    # Setup the parameters
    params = {}
    params['booster'] = 'gbtree'
    params['nthread'] = N_THREADS
    params['eta'] = LEARNING_RATE
    params['gamma'] = MIN_IMPURITY_SPLIT
    params['max_depth'] = MAX_DEPTH
    params['min_child_weight'] = MIN_SAMPLES_LEAF
    params['max_delta_step'] = 0
    params['subsample'] = SUBSAMPLES
    params['colsample_bytree'] = 1.
    params['colsample_bylevel'] = 1.
    params['alpha'] = 0.
    params['delta'] = 0.
    params['tree_method'] = 'exact'
    params['scale_pos_weight'] = 1.
    params['objective'] = 'binary:logistic'
    params['seed'] = RND_SEED

    bst = xgb.train(params, xgb_training, N_ESTIMATORS)
    pred = bst.predict(xgb_testing)
    pred[np.nonzero(pred >= 0.5)] = 1
    pred[np.nonzero(pred < 0.5)] = 0

    score = np.mean(pred == valid)

    return score, datetime.now() - start


dataset = sys.argv[1]
data = misc.load_data(dataset)

print '... Change the label to 0 and 1'
data[1][np.nonzero(data[1] == -1)] = 0
data[3][np.nonzero(data[3] == -1)] = 0

print 'Done, %s samples with %s features loaded into ' \
    ' memory' % data[0].shape

score, res_xgb = misc.bench(bench_xgb, data)
print 'xgboost: mean %.2f, std %.2f\n' % (res_xgb.mean(),
                                          res_xgb.std())
print 'Score: %.2f' % score

