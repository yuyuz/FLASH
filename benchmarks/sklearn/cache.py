from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.cross_validation import KFold

import logging
import numpy as np
import hashlib
import os.path
import re
import time

logging.basicConfig(filename='cache.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s)')
logger = logging.getLogger("flash.cache")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

_n_fold = 3  # hard code cross validated folds
_random_state = 41


def cached_run(steps, X, y):
    step_identifier = ''

    # split data
    n = len(y)
    kf = KFold(n, _n_fold, random_state=_random_state)
    folded_data = [(X[train_index], y[train_index], X[test_index], y[test_index]) for train_index, test_index in kf]

    # last step is estimator, handle separately
    for step in steps[:-1]:
        step_identifier += "/%s" % _step_identifier(step)
        logger.info("Processing %s", step_identifier)
        folded_data = run_step_on_demand(step_identifier, step, folded_data)

    scores = []
    estimator = steps[-1]
    step_identifier += "/%s" % _step_identifier(estimator)
    for (X_train, y_train, X_test, y_test) in folded_data:
        estimator.fit(X_train, y_train)
        scores.append(estimator.score(X_test, y_test))

    score = np.mean(scores)
    logger.info("score of %s is %r", step_identifier, score)
    return score


def run_step_on_demand(step_identifier, step, folded_data): #X_train, y_train, X_test, y_test
    # Checkout from cache first
    should_cache = _step_should_cache(step_identifier)
    logger.info("Step %s should cache = %r", step_identifier, should_cache)
    if should_cache:
        res_cached = check_step_res_exist(step_identifier)
        if res_cached:
            logger.info("Cached hit for step %s", step_identifier)
            starttime = time.time()
            res = [load_step_res(_step_fold_identifier(step_identifier, fold)) for fold in range(_n_fold)]
            duration = time.time() - starttime
            logger.info("load cache with %f seconds", duration)
            return res

    logger.info("Cache missed for step %s. Calculating...", step_identifier)
    starttime = time.time()
    res = []
    for (fold, (X_train, y_train, X_test, y_test)) in enumerate(folded_data):
        X_train, y_train, X_test, y_test = run_step_fold(step, X_train, y_train, X_test, y_test)
        if should_cache:
            save_step_res(_step_fold_identifier(step_identifier, fold), X_train, y_train, X_test, y_test)
        res.append((X_train, y_train, X_test, y_test))
    duration = time.time() - starttime
    logger.info("finished step %s running in %f seconds", step_identifier, duration)
    return res


def run_step_fold(step, X_train, y_train, X_test, y_test):
    X_train = step.fit_transform(X_train, y_train)
    X_test = step.transform(X_test)
    return (X_train, y_train, X_test, y_test)


def save_step_res(step_fold_identifier, X_train, y_train, X_test, y_test):

    file_name_base = hashlib.sha224(step_fold_identifier).hexdigest()
    logger.info("Saving [%s] to [%s]", step_fold_identifier, file_name_base)

    with open(file_name_base + ".train", "wb") as f:
        dump_svmlight_file(X_train, y_train, f)

    with open(file_name_base + ".test", "wb") as f:
        dump_svmlight_file(X_test, y_test, f)


def load_step_res(step_fold_identifier):
    file_name_base = hashlib.sha224(step_fold_identifier).hexdigest()
    logger.info("loading [%s] from [%s]", step_fold_identifier, file_name_base)

    with open(file_name_base + ".train", "rb") as f:
        X_train, y_train = load_svmlight_file(f)
        X_train = X_train.toarray()

    with open(file_name_base + ".test", "rb") as f:
        X_test, y_test = load_svmlight_file(f)
        X_test = X_test.toarray()

    return (X_train, y_train, X_test, y_test)


def check_step_res_exist(step_identifier):
    return all(check_step_fold_res_exist(_step_fold_identifier(step_identifier, fold)) for fold in range(_n_fold))


def check_step_fold_res_exist(step_fold_identifier):
    file_name_base = hashlib.sha224(step_fold_identifier).hexdigest()
    logger.debug("checking %s", file_name_base)

    existence = os.path.isfile(file_name_base + ".test")

    logger.debug("%s existence = %r", file_name_base, existence)

    return existence


def _step_fold_identifier(step_identifier, fold):
    return '/' + str(fold) + step_identifier


def _step_identifier(step):
    def param_value_to_string(value):
        if hasattr(value, '__call__'):
            return value.__name__
        return v

    return type(step).__name__ + '=' + '&'.join(['%s:%s' %(k, param_value_to_string(v)) for k,v in step.get_params().items()])


def _step_should_cache(step_identifier):
    # TODO: check param and decide, a smarter way should be employed here
    def step_cache(name):
        logger.info("checking %s", name)
        return name in ['MaxAbsScaler', 'MinMaxScaler', 'StandardScaler', 'Normalizer', 'PolynomialFeatures']

    return all(step_cache(name) for name in re.findall(r'(\w+)=', step_identifier))


def main():
    from sklearn import svm
    from sklearn.datasets import samples_generator
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.preprocessing import MinMaxScaler

    X, y = samples_generator.make_classification(n_samples=1000, n_informative=5, n_redundant=4, random_state=_random_state)
    anova_filter = SelectKBest(f_regression, k=5)
    scaler = MinMaxScaler()
    clf = svm.SVC(kernel='linear')

    steps = [scaler, anova_filter, clf]
    cached_run(steps, X, y)

if __name__ == '__main__':
    main()
