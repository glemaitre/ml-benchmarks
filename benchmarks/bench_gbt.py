"""Various libraries classifying on gradient boosting trees"""
from __future__ import print_function

import numpy as np
from datetime import datetime

# Define a set of common parameters
N_ESTIMATORS = 5
LEARNING_RATE = 0.1
MIN_IMPURITY_SPLIT = 1e-7
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 1
SUBSAMPLES = 1.
N_THREADS = 1
RND_SEED = 42


def bench_skl(X, y, T, valid):
    """Scikit-learn"""
    from sklearn.ensemble import GradientBoostingClassifier

    start = datetime.now()

    clf = GradientBoostingClassifier(
        max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        loss='deviance', min_weight_fraction_leaf=0.,
        subsample=SUBSAMPLES, max_features=None,
        min_samples_split=2,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_impurity_split=MIN_IMPURITY_SPLIT,
        max_leaf_nodes=None,
        presort='auto', init=None, warm_start=False,
        verbose=0, random_state=42,
        criterion='friedman_mse')

    clf.fit(X, y)

    score = np.mean(clf.predict(T) == valid)

    return score, datetime.now() - start


def bench_xgb(X, y, T, valid):
    """XGBoost"""
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

    # Train
    bst = xgb.train(params, xgb_training, N_ESTIMATORS)
    # Predict
    pred = bst.predict(xgb_testing)
    # Binarize
    pred[np.nonzero(pred >= 0.5)] = 1
    pred[np.nonzero(pred < 0.5)] = 0

    score = np.mean(pred == valid)

    return score, datetime.now() - start


if __name__ == '__main__':
    import sys
    import misc

    # don't bother me with warnings
    import warnings
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print(__doc__ + '\n')
    if not len(sys.argv) == 2:
        print(misc.USAGE % __file__)
        sys.exit(-1)
    else:
        dataset = sys.argv[1]

    print('Loading data ...')
    data = misc.load_data(dataset)

    print('... Change the label to 0 and 1')
    data[1][np.nonzero(data[1] == -1)] = 0
    data[3][np.nonzero(data[3] == -1)] = 0

    print('Done, %s samples with %s features loaded into '
          'memory' % data[0].shape)

    print('Scikit-learn ...')

    score, res_skl = misc.bench(bench_skl, data)
    print('scikit-learn: mean %.2f, std %.2f\n' % (res_skl.mean(),
                                                   res_skl.std()))
    print('Score: %.2f' % score)

    print('XGBoost ...')

    score, res_xgb = misc.bench(bench_xgb, data)
    print('xgboost: mean %.2f, std %.2f\n' % (res_xgb.mean(),
                                              res_xgb.std()))
    print('Score: %.2f' % score)
