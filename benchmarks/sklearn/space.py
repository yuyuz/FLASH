from math import log
from hyperopt import hp


rescaling_list = ['None', 'MinMax', 'Standardize', 'Normalize']  # default: 'None'

rescaling = hp.choice('rescaling', rescaling_list)


balancing_list = ['None', 'weighting']

balancing = hp.choice('balancing', balancing_list)  # default: 'None'


feat_pre_list = [
    {'feat_pre': 'ExtraTreesClassifier',
        'fp0:criterion': hp.choice('fp0:criterion', ['gini', 'entropy']),  # default: 'gini'
        'fp0:max_features': hp.uniform('fp0:max_features', 1, 5),  # default: 1
        'fp0:min_samples_split': hp.uniform('fp0:min_samples_split', 2, 20),  # default: 2
        'fp0:min_samples_leaf': hp.uniform('fp0:min_samples_leaf', 1, 20),  # default: 1
        'fp0:bootstrap': hp.choice('fp0:bootstrap', ['True', 'False'])},  # default: 'False'

    {'feat_pre': 'FastICA',
        'fp1:n_components': hp.uniform('fp1:n_components', 10, 2000),  # default: 100
        'fp1:algorithm': hp.choice('fp1:algorithm', ['parallel', 'deflation']),  # default: 'parallel'
        'fp1:whiten': hp.choice('fp1:whiten', ['True', 'False']),  # default: 'False'
        'fp1:fun': hp.choice('fp1:fun', ['logcosh', 'exp', 'cube'])},  # default: 'logcosh'

    {'feat_pre': 'FeatureAgglomeration',
        'fp2:n_clusters': hp.uniform('fp2:n_clusters', 2, 400),  # default: 25
        'fp2:linkage+affinity': hp.choice('fp2:linkage+affinity', ['ward+euclidean',  # default: 'ward+euclidean'
                                                                   'complete+euclidean',
                                                                   'complete+manhattan',
                                                                   'complete+cosine',
                                                                   'average+euclidean',
                                                                   'average+manhattan',
                                                                   'average+cosine']),
        'fp2:pooling_func': hp.choice('fp2:pooling_func', ['mean', 'median', 'max'])},  # default: 'mean'

    {'feat_pre': 'KernelPCA',
        'fp3:n_components': hp.uniform('fp3:n_components', 10, 2000),  # default: 100
        'fp3:kernel': hp.choice('fp3:kernel', ['cosine',  # default: 'rbf'
            {'fp3:rbf.gamma': hp.loguniform('fp3:rbf.gamma', log(3.0517578125e-05), log(8))},  # default: log(1)
            {'fp3:sigmoid.coef0': hp.uniform('fp3:sigmoid.coef0', -1, 1)},  # default: 0
            {'fp3:poly.degree': hp.uniform('fp3:poly.degree', 2, 5),  # default: 3
             'fp3:poly.coef0': hp.uniform('fp3:poly.coef0', -1, 1),  # default: 0
             'fp3:poly.gamma': hp.loguniform('fp3:poly.gamma', log(3.0517578125e-05), log(8))}])},  # default: log(1)

    {'feat_pre': 'RBFSampler',
        'fp4:gamma': hp.uniform('fp4:gamma', 0.3, 2),  # default: 1.0
        'fp4:n_components': hp.loguniform('fp4:n_components', log(50), log(10000))},  # default: log(100)

    {'feat_pre': 'LinearSVC',
        'fp5:tol': hp.loguniform('fp5:tol', log(1e-5), log(1e-1)),  # default: log(1e-4)
        'fp5:C': hp.loguniform('fp5:C', log(0.03125), log(32768))},  # default: log(1)

    {'feat_pre': 'None'},

    {'feat_pre': 'Nystroem',
        'fp7:n_components': hp.loguniform('fp7:n_components', log(50), log(10000)),  # default: log(100)
        'fp7:kernel': hp.choice('fp7:kernel', ['cosine',  # default: 'rbf'
            {'fp7:rbf.gamma': hp.loguniform('fp7:rbf.gamma', log(3.0517578125e-05), log(8))},  # default: log(0.1)
            {'fp7:chi2.gamma': hp.loguniform('fp7:chi2.gamma', log(3.0517578125e-05), log(8))},  # default: log(0.1)
            {'fp7:sigmoid.coef0': hp.uniform('fp7:sigmoid.coef0', -1, 1),  # default: 0
             'fp7:sigmoid.gamma': hp.loguniform('fp7:sigmoid.gamma', log(3.0517578125e-05), log(8))},  # default: log(0.1)
            {'fp7:poly.degree': hp.uniform('fp7:poly.degree', 2, 5),  # default: 3
             'fp7:poly.coef0': hp.uniform('fp7:poly.coef0', -1, 1),  # default: 0
             'fp7:poly.gamma': hp.loguniform('fp7:poly.gamma', log(3.0517578125e-05), log(8))}])},  # default: log(0.1)

    {'feat_pre': 'PCA',
        'fp8:n_components': hp.uniform('fp8:n_components', 0.5, 0.9999),  # default: 0.9999
        'fp8:whiten': hp.choice('fp8:whiten', ['True', 'False'])},  # default: 'False'

    {'feat_pre': 'PolynomialFeatures',
        'fp9:degree': hp.uniform('fp9:degree', 2, 3),  # default: 2
        'fp9:interaction_only': hp.choice('fp9:interaction_only', ['True', 'False']),  # default: 'False'
        'fp9:include_bias': hp.choice('fp9:include_bias', ['True', 'False'])},  # default: 'True'

    {'feat_pre': 'RandomTreesEmbedding',
        'fp10:n_estimators': hp.uniform('fp10:n_estimators', 10, 100),  # default: 10
        'fp10:max_depth': hp.uniform('fp10:max_depth', 2, 10),  # default: 5
        'fp10:min_samples_split': hp.uniform('fp10:min_samples_split', 2, 20),  # default: 2
        'fp10:min_samples_leaf': hp.uniform('fp10:min_samples_leaf', 1, 20)},  # default: 1

    {'feat_pre': 'SelectPercentile',
        'fp11:percentile': hp.uniform('fp11:percentile', 1, 99),  # default: 50
        'fp11:score_func': hp.choice('fp11:score_func', ['chi2', 'f_classif'])},  # default: 'chi2'

    {'feat_pre': 'GenericUnivariateSelect',
        'fp12:param': hp.uniform('fp12:param', 0.01, 0.5),  # default: 0.1
        'fp12:score_func': hp.choice('fp12:score_func', ['chi2', 'f_classif']),  # default: 'chi2'
        'fp12:mode': hp.choice('fp12:mode', ['fpr', 'fdr', 'fwe'])}]  # default: 'fpr'

feat_pre = hp.choice('feat_pre', feat_pre_list)


classifier_list = [
    {'classifier': 'AdaBoostClassifier',
        'clf0:n_estimators': hp.uniform('clf0:n_estimators', 50, 500),  # default: 50
        'clf0:learning_rate': hp.loguniform('clf0:learning_rate', log(0.0001), log(2)),  # default: log(0.1)
        'clf0:algorithm': hp.choice('clf0:algorithm', ['SAMME', 'SAMME.R']),  # default: 'SAMME.R'
        'clf0:max_depth': hp.uniform('clf0:max_depth', 1, 10)},  # default: 1

    {'classifier': 'DecisionTreeClassifier',
        'clf1:criterion': hp.choice('clf1:criterion', ['gini', 'entropy']),  # default: 'gini'
        'clf1:max_depth': hp.uniform('clf1:max_depth', 0, 2),  # default: 0.5
        'clf1:min_samples_split': hp.uniform('clf1:min_samples_split', 2, 20),  # default: 2
        'clf1:min_samples_leaf': hp.uniform('clf1:min_samples_leaf', 1, 20)},  # default: 1

    {'classifier': 'ExtraTreesClassifier',
        'clf2:criterion': hp.choice('clf2:criterion', ['gini', 'entropy']),  # default: 'gini'
        'clf2:max_features': hp.uniform('clf2:max_features', 1, 5),  # default: 1
        'clf2:min_samples_split': hp.uniform('clf2:min_samples_split', 2, 20),  # default: 2
        'clf2:min_samples_leaf': hp.uniform('clf2:min_samples_leaf', 1, 20),  # default: 1
        'clf2:bootstrap': hp.choice('clf2:bootstrap', ['True', 'False'])},  # default: 'False'

    {'classifier': 'GaussianNB'},

    {'classifier': 'GradientBoostingClassifier',
        'clf4:learning_rate': hp.loguniform('clf4:learning_rate', log(0.0001), log(1)),  # default: log(0.1)
        'clf4:max_depth': hp.uniform('clf4:max_depth', 1, 10),  # default: 3
        'clf4:min_samples_split': hp.uniform('clf4:min_samples_split', 2, 20),  # default: 2
        'clf4:min_samples_leaf': hp.uniform('clf4:min_samples_leaf', 1, 20),  # default: 1
        'clf4:subsample': hp.uniform('clf4:subsample', 0.01, 1),  # default: 1
        'clf4:max_features': hp.uniform('clf4:max_features', 1, 5)},  # default: 1

    {'classifier': 'KNeighborsClassifier',
        'clf5:n_neighbors': hp.uniform('clf5:n_neighbors', 1, 100),  # default: 1
        'clf5:weights': hp.choice('clf5:weights', ['uniform', 'distance']),  # default: 'uniform'
        'clf5:p': hp.choice('clf5:p', ['l1', 'l2'])},  # default: 'l2'

    {'classifier': 'LDA',
        'clf6:shrinkage': hp.choice('clf6:shrinkage', ['None', 'auto',  # default: 'None'
            {'clf6:shrinkage.factor': hp.uniform('clf6:shrinkage.factor', 0, 1)}]),  # default: 0.5
        'clf6:n_components': hp.uniform('clf6:n_components', 1, 250),  # default: 10
        'clf6:tol': hp.loguniform('clf6:tol', log(1e-5), log(1e-1))},  # default: log(1e-4)

    {'classifier': 'LinearSVC',
        'clf7:tol': hp.loguniform('clf7:tol', log(1e-5), log(1e-1)),  # default: log(1e-4)
        'clf7:C': hp.loguniform('clf7:C', log(0.03125), log(32768))},  # default: log(1)

    {'classifier': 'SVC',
        'clf8:C': hp.loguniform('clf8:C', log(0.03125), log(32768)),  # default: log(1)
        'clf8:kernel': hp.choice('clf8:kernel', ['rbf', 'sigmoid',  # default: 'rbf'
            {'clf8:poly.degree': hp.uniform('clf8:poly.degree', 1, 5)}]),  # default: 3
        'clf8:gamma': hp.loguniform('clf8:gamma', log(3.0517578125e-05), log(8)),  # default: log(0.1)
        'clf8:coef0': hp.uniform('clf8:coef0', -1, 1),  # default: 0
        'clf8:shrinking': hp.choice('clf8:shrinking', ['True', 'False']),  # default: 'True'
        'clf8:tol': hp.loguniform('clf8:tol', log(1e-5), log(1e-1))},  # default: log(1e-4)

    {'classifier': 'MultinomialNB',
        'clf9:alpha': hp.loguniform('clf9:alpha', log(1e-2), log(100)),  # default: log(1)
        'clf9:fit_prior': hp.choice('clf9:fit_prior', ['True', 'False'])},  # default: 'True'

    {'classifier': 'PassiveAggressiveClassifier',
        'clf10:loss': hp.choice('clf10:loss', ['hinge', 'squared_hinge']),  # default: 'hinge'}
        'clf10:n_iter': hp.uniform('clf10:n_iter', 5, 1000),  # default: 20
        'clf10:C': hp.loguniform('clf10:C', log(1e-5), log(10))},  # default: log(1)

    {'classifier': 'QDA',
        'clf11:reg_param': hp.uniform('clf11:reg_param', 0, 10)},  # default: 0.5

    {'classifier': 'RandomForestClassifier',
        'clf12:criterion': hp.choice('clf12:criterion', ['gini', 'entropy']),  # default: 'gini'
        'clf12:max_features': hp.uniform('clf12:max_features', 1, 5),  # default: 1
        'clf12:min_samples_split': hp.uniform('clf12:min_samples_split', 2, 20),  # default: 2
        'clf12:min_samples_leaf': hp.uniform('clf12:min_samples_leaf', 1, 20),  # default: 1
        'clf12:bootstrap': hp.choice('clf12:bootstrap', ['True', 'False'])},  # default: 'True'

    {'classifier': 'SGDClassifier',
        'clf13:loss': hp.choice('clf13:loss', ['hinge', 'log', 'squared_hinge', 'perceptron',  # default: 'hinge'
            {'clf13:modified_huber.epsilon': hp.loguniform('clf13:modified_huber.epsilon', log(1e-5), log(1e-1))}]),  # default: log(1e-4)
        'clf13:penalty': hp.choice('clf13:penalty', ['l1', 'l2',  # default: 'l2'
            {'clf13:elasticnet.l1_ratio': hp.uniform('clf13:elasticnet.l1_ratio', 0, 1)}]),  # default: 0.15
        'clf13:alpha': hp.loguniform('clf13:alpha', log(1e-6), log(1e-1)),  # default: log(0.0001)
        'clf13:n_iter': hp.uniform('clf13:n_iter', 5, 1000),  # default: 20
        'clf13:learning_rate': hp.choice('clf13:learning_rate', ['optimal', 'constant',  # default: 'optimal'
            {'clf13:invscaling.power_t': hp.uniform('clf13:invscaling.power_t', 1e-5, 1)}]),  # default: 0.25
        'clf13:eta0': hp.uniform('clf13:eta0', 1e-6, 1e-1),  # default: 0.01
        'clf13:average': hp.choice('clf13:average', ['True', 'False'])}]  # default: 'False'

classifier = hp.choice('classifier', classifier_list)


space = {
    'rescaling': rescaling,
    'balancing': balancing,
    'feat_pre': feat_pre,
    'classifier': classifier}

d_rescaling = dict()
for i in range(len(rescaling_list)):
    d_rescaling[rescaling_list[i]] = i

d_balancing = dict()
for i in range(len(balancing_list)):
    d_balancing[balancing_list[i]] = i

d_feat_pre = dict()
for i in range(len(feat_pre_list)):
    d_feat_pre[feat_pre_list[i]['feat_pre']] = i

d_clf = dict()
for i in range(len(classifier_list)):
    d_clf[classifier_list[i]['classifier']] = i

layer_dict_list = [d_rescaling, d_balancing, d_feat_pre, d_clf]
