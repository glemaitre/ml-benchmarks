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


def bench_skl(X, y, T, valid):
#
#       .. scikit-learn ..
#
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


dataset = sys.argv[1]
data = misc.load_data(dataset)

print '... Change the label to 0 and 1'
data[1][np.nonzero(data[1] == -1)] = 0
data[3][np.nonzero(data[3] == -1)] = 0

print 'Done, %s samples with %s features loaded into ' \
    ' memory' % data[0].shape

print 'Scikit-learn ...'
score, res_skl = misc.bench(bench_skl, data)
print 'scikit-learn: mean %.2f, std %.2f\n' % (res_skl.mean(),
                                               res_skl.std())
print 'Score: %.2f' % score
