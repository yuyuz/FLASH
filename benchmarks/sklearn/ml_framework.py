import time
import os
import arff
import numpy as np
import HPOlib.benchmark_util as benchmark_util

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

from MultilabelClassifier import MultilabelClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from sklearn.decomposition import FastICA, KernelPCA, PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.preprocessing import PolynomialFeatures
from sklearn import feature_selection

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, RandomTreesEmbedding
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn import svm, naive_bayes
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.qda import QDA

from importlib import import_module
from copy import deepcopy


def load_arff_data(filename):
    with open(filename) as f:
        decoder = arff.ArffDecoder()
        arff_obj = decoder.decode(f, encode_nominal=True)
        # feat_num = len([v for v in arff_obj['attributes'] if v[0] != 'class'])
        data = np.array(arff_obj['data'])
        X = data[:,:-1]
        y = data[:,-1]

        return X, y


def get_data_preprocessor_rescaling(params):
    dpr = None
    d_rescaling = params['layer_dict_list'][0]

    if params['rescaling'] == str(d_rescaling['None']) or params['rescaling'] == 'None':
        dpr = None
    elif params['rescaling'] == str(d_rescaling['MinMax']) or params['rescaling'] == 'MinMax':
        dpr = MinMaxScaler()
    elif params['rescaling'] == str(d_rescaling['Standardize']) or params['rescaling'] == 'Standardize':
        dpr = StandardScaler()
    elif params['rescaling'] == str(d_rescaling['Normalize']) or params['rescaling'] == 'Normalize':
        dpr = Normalizer()

    return dpr


def get_data_preprocessor_balancing(params, y):
    d_balancing = params['layer_dict_list'][1]

    if params['balancing'] == str(d_balancing['None']) or params['balancing'] == 'None':
        # for fp: ['ExtraTreesClassifier', 'LinearSVC'] + clf: ['DecisionTreeClassifier', 'ExtraTreesClassifier', 'LinearSVC', 'SVC', 'RandomForestClassifier', 'SGDClassifier']
        params['class_weight'] = None
        # for clf: ['Adasample_weightBoostClassifier', 'GradientBoostingClassifier']
        params['sample_weight'] = None
    elif params['balancing'] == str(d_balancing['weighting']) or params['balancing'] == 'weighting':
        # for fp: ['ExtraTreesClassifier', 'LinearSVC'] + clf: ['DecisionTreeClassifier', 'ExtraTreesClassifier', 'LinearSVC', 'SVC', 'RandomForestClassifier', 'SGDClassifier']
        params['class_weight'] = 'auto'
        # for clf: ['AdaBoostClassifier', 'GradientBoostingClassifier']
        if len(y.shape) > 1:
            offsets = [2 ** i for i in range(y.shape[1])]
            y_ = np.sum(y * offsets, axis=1)
        else:
            y_ = y
        unique, counts = np.unique(y_, return_counts=True)
        cw = 1. / counts
        cw = cw / np.mean(cw)
        sample_weight = np.ones(y_.shape)
        for i, ue in enumerate(unique):
            mask = y_ == ue
            sample_weight[mask] *= cw[i]
        params['sample_weight'] = sample_weight

    return params


def get_feature_preprocessor(params):
    fp = None
    d_feat_pre = params['layer_dict_list'][2]

    if params['feat_pre'] == str(d_feat_pre['ExtraTreesClassifier']) or params['feat_pre'] == 'ExtraTreesClassifier':
        if params['fp0:criterion'] == '0' or params['fp0:criterion'] == 'gini':
            criterion = 'gini'
        elif params['fp0:criterion'] == '1' or params['fp0:criterion'] == 'entropy':
            criterion = 'entropy'
        max_features = int(float(params['fp0:max_features']))
        min_samples_split = int(float(params['fp0:min_samples_split']))
        min_samples_leaf = int(float(params['fp0:min_samples_leaf']))
        if params['fp0:bootstrap'] == '0' or params['fp0:bootstrap'] == 'True':
            bootstrap = True
        elif params['fp0:bootstrap'] == '1' or params['fp0:bootstrap'] == 'False':
            bootstrap = False
        fp = ExtraTreesClassifier(n_estimators=100,
                                  criterion=criterion,
                                  max_features=max_features,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  min_weight_fraction_leaf=0.0,
                                  bootstrap=bootstrap,
                                  class_weight=params['class_weight'])

    elif params['feat_pre'] == str(d_feat_pre['FastICA']) or params['feat_pre'] == 'FastICA':
        n_components = int(float(params['fp1:n_components']))
        if params['fp1:algorithm'] == '0' or params['fp1:algorithm'] == 'parallel':
            algorithm = 'parallel'
        elif params['fp1:algorithm'] == '1' or params['fp1:algorithm'] == 'deflation':
            algorithm = 'deflation'
        if params['fp1:whiten'] == '0' or params['fp1:whiten'] == 'True':
            whiten = True
        elif params['fp1:whiten'] == '1' or params['fp1:whiten'] == 'False':
            whiten = False
        if params['fp1:fun'] == '0' or params['fp1:fun'] == 'logcosh':
            fun = 'logcosh'
        elif params['fp1:fun'] == '1' or params['fp1:fun'] == 'exp':
            fun = 'exp'
        elif params['fp1:fun'] == '2' or params['fp1:fun'] == 'cube':
            fun = 'cube'
        fp = FastICA(n_components=n_components,
                     algorithm=algorithm,
                     whiten=whiten,
                     fun=fun)

    elif params['feat_pre'] == str(d_feat_pre['FeatureAgglomeration']) or params['feat_pre'] == 'FeatureAgglomeration':
        n_clusters = int(float(params['fp2:n_clusters']))
        if params['fp2:linkage+affinity'] == '0' or params['fp2:linkage+affinity'] == 'ward+euclidean':
            linkage = 'ward'
            affinity = 'euclidean'
        elif params['fp2:linkage+affinity'] == '1' or params['fp2:linkage+affinity'] == 'complete+euclidean':
            linkage = 'complete'
            affinity = 'euclidean'
        elif params['fp2:linkage+affinity'] == '2' or params['fp2:linkage+affinity'] == 'complete+manhattan':
            linkage = 'complete'
            affinity = 'manhattan'
        elif params['fp2:linkage+affinity'] == '3' or params['fp2:linkage+affinity'] == 'complete+cosine':
            linkage = 'complete'
            affinity = 'cosine'
        elif params['fp2:linkage+affinity'] == '4' or params['fp2:linkage+affinity'] == 'average+euclidean':
            linkage = 'average'
            affinity = 'euclidean'
        elif params['fp2:linkage+affinity'] == '5' or params['fp2:linkage+affinity'] == 'average+manhattan':
            linkage = 'average'
            affinity = 'manhattan'
        elif params['fp2:linkage+affinity'] == '6' or params['fp2:linkage+affinity'] == 'average+cosine':
            linkage = 'average'
            affinity = 'cosine'
        if params['fp2:pooling_func'] == '0' or params['fp2:pooling_func'] == 'mean':
            pooling_func = np.mean
        elif params['fp2:pooling_func'] == '1' or params['fp2:pooling_func'] == 'median':
            pooling_func = np.median
        elif params['fp2:pooling_func'] == '2' or params['fp2:pooling_func'] == 'max':
            pooling_func = np.max
        fp = FeatureAgglomeration(n_clusters=n_clusters,
                                  linkage=linkage,
                                  affinity=affinity,
                                  pooling_func=pooling_func)

    elif params['feat_pre'] == str(d_feat_pre['KernelPCA']) or params['feat_pre'] == 'KernelPCA':
        n_components = int(float(params['fp3:n_components']))
        degree = 3
        coef0 = 1
        gamma = None
        if 'fp3:rbf.gamma' in params:
            kernel = 'rbf'
            gamma = float(params['fp3:rbf.gamma'])
        elif 'fp3:sigmoid.coef0' in params:
            kernel = 'sigmoid'
            coef0 = float(params['fp3:sigmoid.coef0'])
        elif 'fp3:poly.degree' in params and 'fp3:poly.coef0' in params and 'fp3:poly.gamma' in params:
            kernel = 'poly'
            degree = int(float(params['fp3:poly.degree']))
            coef0 = float(params['fp3:poly.coef0'])
            gamma = float(params['fp3:poly.gamma'])
        elif params['fp3:kernel'] == '0' or params['fp3:kernel'] == 'cosine':
            kernel = 'cosine'
        fp = KernelPCA(n_components=n_components,
                       kernel=kernel,
                       degree=degree,
                       coef0=coef0,
                       gamma=gamma)

    elif params['feat_pre'] == str(d_feat_pre['RBFSampler']) or params['feat_pre'] == 'RBFSampler':
        gamma = float(params['fp4:gamma'])
        n_components = int(float(params['fp4:n_components']))
        fp = RBFSampler(gamma=gamma,
                        n_components=n_components)

    elif params['feat_pre'] == str(d_feat_pre['LinearSVC']) or params['feat_pre'] == 'LinearSVC':
        tol = float(params['fp5:tol'])
        C = float(params['fp5:C'])
        fp = svm.LinearSVC(penalty='l1',
                           loss='squared_hinge',
                           dual=False,
                           tol=tol,
                           C=C,
                           multi_class='ovr',
                           fit_intercept=True,
                           intercept_scaling=1,
                           class_weight=params['class_weight'])

    elif params['feat_pre'] == str(d_feat_pre['None']) or params['feat_pre'] == 'None':
        fp = None

    elif params['feat_pre'] == str(d_feat_pre['Nystroem']) or params['feat_pre'] == 'Nystroem':
        n_components = int(float(params['fp7:n_components']))
        degree = 3
        coef0 = 1
        gamma = None
        if 'fp7:rbf.gamma' in params:
            kernel = 'rbf'
            gamma = float(params['fp7:rbf.gamma'])
        elif 'fp7:chi2.gamma' in params:
            kernel = 'chi2'
            gamma = float(params['fp7:chi2.gamma'])
        elif 'fp7:sigmoid.coef0' in params and 'fp7:sigmoid.gamma' in params:
            kernel = 'sigmoid'
            coef0 = float(params['fp7:sigmoid.coef0'])
            gamma = float(params['fp7:sigmoid.gamma'])
        elif 'fp7:poly.degree' in params and 'fp7:poly.coef0' in params and 'fp7:poly.gamma' in params:
            kernel = 'poly'
            degree = int(float(params['fp7:poly.degree']))
            coef0 = float(params['fp7:poly.coef0'])
            gamma = float(params['fp7:poly.gamma'])
        elif params['fp7:kernel'] == '0' or params['fp7:kernel'] == 'cosine':
            kernel = 'cosine'
        fp = Nystroem(n_components=n_components,
                      kernel=kernel,
                      degree=degree,
                      coef0=coef0,
                      gamma=gamma)

    elif params['feat_pre'] == str(d_feat_pre['PCA']) or params['feat_pre'] == 'PCA':
        n_components = float(params['fp8:n_components'])
        if params['fp8:whiten'] == '0' or params['fp8:whiten'] == 'True':
            whiten = True
        elif params['fp8:whiten'] == '1' or params['fp8:whiten'] == 'False':
            whiten = False
        fp = PCA(n_components=n_components,
                 whiten=whiten)

    elif params['feat_pre'] == str(d_feat_pre['PolynomialFeatures']) or params['feat_pre'] == 'PolynomialFeatures':
        degree = int(float(params['fp9:degree']))
        if params['fp9:interaction_only'] == '0' or params['fp9:interaction_only'] == 'True':
            interaction_only = True
        elif params['fp9:interaction_only'] == '1' or params['fp9:interaction_only'] == 'False':
            interaction_only = False
        if params['fp9:include_bias'] == '0' or params['fp9:include_bias'] == 'True':
            include_bias = True
        elif params['fp9:include_bias'] == '1' or params['fp9:include_bias'] == 'False':
            include_bias = False
        fp = PolynomialFeatures(degree=degree,
                                interaction_only=interaction_only,
                                include_bias=include_bias)

    elif params['feat_pre'] == str(d_feat_pre['RandomTreesEmbedding']) or params['feat_pre'] == 'RandomTreesEmbedding':
        n_estimators = int(float(params['fp10:n_estimators']))
        max_depth = int(float(params['fp10:max_depth']))
        min_samples_split = int(float(params['fp10:min_samples_split']))
        min_samples_leaf = int(float(params['fp10:min_samples_leaf']))
        fp = RandomTreesEmbedding(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  min_weight_fraction_leaf=0,
                                  sparse_output=False)

    elif params['feat_pre'] == str(d_feat_pre['SelectPercentile']) or params['feat_pre'] == 'SelectPercentile':
        percentile = int(float(params['fp11:percentile']))
        if params['fp11:score_func'] == '0' or params['fp11:score_func'] == 'chi2':
            score_func = feature_selection.chi2
        elif params['fp11:score_func'] == '1' or params['fp11:score_func'] == 'f_classif':
            score_func = feature_selection.f_classif
        fp = feature_selection.SelectPercentile(score_func=score_func,
                                                percentile=percentile)

    elif params['feat_pre'] == str(d_feat_pre['GenericUnivariateSelect']) or params['feat_pre'] == 'GenericUnivariateSelect':
        param = float(params['fp12:param'])
        if params['fp12:score_func'] == '0' or params['fp12:score_func'] == 'chi2':
            score_func = feature_selection.chi2
        elif params['fp12:score_func'] == '1' or params['fp12:score_func'] == 'f_classif':
            score_func = feature_selection.f_classif
        if params['fp12:mode'] == '0' or params['fp12:mode'] == 'fpr':
            mode = 'fpr'
        elif params['fp12:mode'] == '1' or params['fp12:mode'] == 'fdr':
            mode = 'fdr'
        elif params['fp12:mode'] == '2' or params['fp12:mode'] == 'fwe':
            mode = 'fwe'
        fp = feature_selection.GenericUnivariateSelect(param=param,
                                                       score_func=score_func,
                                                       mode=mode)

    return fp


def get_classifier(params):
    clf = None
    d_clf = params['layer_dict_list'][3]

    if params['classifier'] == str(d_clf['AdaBoostClassifier']) or params['classifier'] == 'AdaBoostClassifier':
        n_estimators = int(float(params['clf0:n_estimators']))
        learning_rate = float(params['clf0:learning_rate'])
        if params['clf0:algorithm'] == '0' or params['clf0:algorithm'] == 'SAMME':
            algorithm = 'SAMME'
        elif params['clf0:algorithm'] == '1' or params['clf0:algorithm'] == 'SAMME.R':
            algorithm = 'SAMME.R'
        max_depth = int(float(params['clf0:max_depth']))
        estimator = AdaBoostClassifier(n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       algorithm=algorithm,
                                       base_estimator=DecisionTreeClassifier(max_depth=max_depth))
        clf = MultilabelClassifier(estimator, params['sample_weight'])

    elif params['classifier'] == str(d_clf['DecisionTreeClassifier']) or params['classifier'] == 'DecisionTreeClassifier':
        if params['clf1:criterion'] == '0' or params['clf1:criterion'] == 'gini':
            criterion = 'gini'
        elif params['clf1:criterion'] == '1' or params['clf1:criterion'] == 'entropy':
            criterion = 'entropy'
        max_depth = float(params['clf1:max_depth'])
        min_samples_split = int(float(params['clf1:min_samples_split']))
        min_samples_leaf = int(float(params['clf1:min_samples_leaf']))
        clf = DecisionTreeClassifier(criterion=criterion,
                                     splitter='best',
                                     max_features=1.0,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     min_weight_fraction_leaf=0.0,
                                     class_weight=params['class_weight'])

    elif params['classifier'] == str(d_clf['ExtraTreesClassifier']) or params['classifier'] == 'ExtraTreesClassifier':
        if params['clf2:criterion'] == '0' or params['clf2:criterion'] == 'gini':
            criterion = 'gini'
        elif params['clf2:criterion'] == '1' or params['clf2:criterion'] == 'entropy':
            criterion = 'entropy'
        max_features = int(float(params['clf2:max_features']))
        min_samples_split = int(float(params['clf2:min_samples_split']))
        min_samples_leaf = int(float(params['clf2:min_samples_leaf']))
        if params['clf2:bootstrap'] == '0' or params['clf2:bootstrap'] == 'True':
            bootstrap = True
        elif params['clf2:bootstrap'] == '1' or params['clf2:bootstrap'] == 'False':
            bootstrap = False
        clf = ExtraTreesClassifier(n_estimators=100,
                                   criterion=criterion,
                                   max_features=max_features,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   min_weight_fraction_leaf=0.0,
                                   bootstrap=bootstrap,
                                   class_weight=params['class_weight'])

    elif params['classifier'] == str(d_clf['GaussianNB']) or params['classifier'] == 'GaussianNB':
        clf = naive_bayes.GaussianNB()

    elif params['classifier'] == str(d_clf['GradientBoostingClassifier']) or params['classifier'] == 'GradientBoostingClassifier':
        learning_rate = float(params['clf4:learning_rate'])
        max_depth = int(float(params['clf4:max_depth']))
        min_samples_split = int(float(params['clf4:min_samples_split']))
        min_samples_leaf = int(float(params['clf4:min_samples_leaf']))
        subsample = float(params['clf4:subsample'])
        max_features = int(float(params['clf4:max_features']))
        estimator = GradientBoostingClassifier(loss='deviance',
                                               learning_rate=learning_rate,
                                               n_estimators=100,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,
                                               min_weight_fraction_leaf=0.0,
                                               subsample=subsample,
                                               max_features=max_features)
        clf = MultilabelClassifier(estimator, params['sample_weight'])

    elif params['classifier'] == str(d_clf['KNeighborsClassifier']) or params['classifier'] == 'KNeighborsClassifier':
        n_neighbors = int(float(params['clf5:n_neighbors']))
        if params['clf5:weights'] == '0' or params['clf5:weights'] == 'uniform':
            weights = 'uniform'
        elif params['clf5:weights'] == '1' or params['clf5:weights'] == 'distance':
            weights = 'distance'
        if params['clf5:p'] == '0' or params['clf5:p'] == 'l1':
            p = 1
        elif params['clf5:p'] == '1' or params['clf5:p'] == 'l2':
            p = 2
        clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                   weights=weights,
                                   p=p)

    elif params['classifier'] == str(d_clf['LDA']) or params['classifier'] == 'LDA':
        if 'clf6:shrinkage.factor' in params:
            shrinkage = float(params['clf6:shrinkage.factor'])
            solver = 'lsqr'
        elif params['clf6:shrinkage'] == '0' or params['clf6:shrinkage'] == 'None':
            shrinkage = None
            solver = 'svd'
        elif params['clf6:shrinkage'] == '1' or params['clf6:shrinkage'] == 'auto':
            shrinkage = 'auto'
            solver = 'lsqr'
        n_components = int(float(params['clf6:n_components']))
        tol = float(params['clf6:tol'])
        clf = LDA(shrinkage=shrinkage,
                  n_components=n_components,
                  tol=tol,
                  solver=solver)

    elif params['classifier'] == str(d_clf['LinearSVC']) or params['classifier'] == 'LinearSVC':
        tol = float(params['clf7:tol'])
        C = float(params['clf7:C'])
        clf = svm.LinearSVC(penalty='l2',
                            loss='squared_hinge',
                            dual=False,
                            tol=tol,
                            C=C,
                            multi_class='ovr',
                            fit_intercept=True,
                            intercept_scaling=1,
                            class_weight=params['class_weight'])

    elif params['classifier'] == str(d_clf['SVC']) or params['classifier'] == 'SVC':
        C = float(params['clf8:C'])
        degree = 3
        if 'clf8:poly.degree' in params:
            kernel = 'poly'
            degree = int(float(params['clf8:poly.degree']))
        elif params['clf8:kernel'] == '0' or params['clf8:kernel'] == 'rbf':
            kernel = 'rbf'
        elif params['clf8:kernel'] == '1' or params['clf8:kernel'] == 'sigmoid':
            kernel = 'sigmoid'
        gamma = float(params['clf8:gamma'])
        coef0 = float(params['clf8:coef0'])
        if params['clf8:shrinking'] == '0' or params['clf8:shrinking'] == 'True':
            shrinking = True
        elif params['clf8:shrinking'] == '1' or params['clf8:shrinking'] == 'False':
            shrinking = False
        tol = float(params['clf8:tol'])
        clf = svm.SVC(C=C,
                      kernel=kernel,
                      degree=degree,
                      gamma=gamma,
                      coef0=coef0,
                      shrinking=shrinking,
                      tol=tol,
                      class_weight=params['class_weight'])

    elif params['classifier'] == str(d_clf['MultinomialNB']) or params['classifier'] == 'MultinomialNB':
        alpha = float(params['clf9:alpha'])
        if params['clf9:fit_prior'] == '0' or params['clf9:fit_prior'] == 'True':
            fit_prior = True
        elif params['clf9:fit_prior'] == '1' or params['clf9:fit_prior'] == 'False':
            fit_prior = False
        clf = naive_bayes.MultinomialNB(alpha=alpha,
                                        fit_prior=fit_prior)

    elif params['classifier'] == str(d_clf['PassiveAggressiveClassifier']) or params['classifier'] == 'PassiveAggressiveClassifier':
        if params['clf10:loss'] == '0' or params['clf10:loss'] == 'hinge':
            loss = 'hinge'
        elif params['clf10:loss'] == '1' or params['clf10:loss'] == 'squared_hinge':
            loss = 'squared_hinge'
        n_iter = int(float(params['clf10:n_iter']))
        C = float(params['clf10:C'])
        clf = PassiveAggressiveClassifier(loss=loss,
                                          n_iter=n_iter,
                                          C=C)

    elif params['classifier'] == str(d_clf['QDA']) or params['classifier'] == 'QDA':
        reg_param = float(params['clf11:reg_param'])
        clf = QDA(reg_param=reg_param)

    elif params['classifier'] == str(d_clf['RandomForestClassifier']) or params['classifier'] == 'RandomForestClassifier':
        if params['clf12:criterion'] == '0' or params['clf12:criterion'] == 'gini':
            criterion = 'gini'
        elif params['clf12:criterion'] == '1' or params['clf12:criterion'] == 'entropy':
            criterion = 'entropy'
        max_features = int(float(params['clf12:max_features']))
        min_samples_split = int(float(params['clf12:min_samples_split']))
        min_samples_leaf = int(float(params['clf12:min_samples_leaf']))
        if params['clf12:bootstrap'] == '0' or params['clf12:bootstrap'] == 'True':
            bootstrap = True
        elif params['clf12:bootstrap'] == '1' or params['clf12:bootstrap'] == 'False':
            bootstrap = False
        clf = RandomForestClassifier(n_estimators=100,
                                     criterion=criterion,
                                     max_features=max_features,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     bootstrap=bootstrap,
                                     class_weight=params['class_weight'])

    elif params['classifier'] == str(d_clf['SGDClassifier']) or params['classifier'] == 'SGDClassifier':
        l1_ratio = 0.15
        epsilon = 0.1
        power_t = 0.5
        if 'clf13:modified_huber.epsilon' in params:
            loss = 'modified_huber'
            epsilon = float(params['clf13:modified_huber.epsilon'])
        elif params['clf13:loss'] == '0' or params['clf13:loss'] == 'hinge':
            loss = 'hinge'
        elif params['clf13:loss'] == '1' or params['clf13:loss'] == 'log':
            loss = 'log'
        elif params['clf13:loss'] == '2' or params['clf13:loss'] == 'squared_hinge':
            loss = 'squared_hinge'
        elif params['clf13:loss'] == '3' or params['clf13:loss'] == 'perceptron':
            loss = 'perceptron'
        if 'clf13:elasticnet.l1_ratio' in params:
            penalty = 'elasticnet'
            l1_ratio = float(params['clf13:elasticnet.l1_ratio'])
        elif params['clf13:penalty'] == '0' or params['clf13:penalty'] == 'l1':
            penalty = 'l1'
        elif params['clf13:penalty'] == '1' or params['clf13:penalty'] == 'l2':
            penalty = 'l2'
        alpha = float(params['clf13:alpha'])
        n_iter = int(float(params['clf13:n_iter']))
        if 'clf13:invscaling.power_t' in params:
            learning_rate = 'invscaling'
            power_t = float(params['clf13:invscaling.power_t'])
        elif params['clf13:learning_rate'] == '0' or params['clf13:learning_rate'] == 'optimal':
            learning_rate = 'optimal'
        elif params['clf13:learning_rate'] == '1' or params['clf13:learning_rate'] == 'constant':
            learning_rate = 'constant'
        eta0 = float(params['clf13:eta0'])
        if params['clf13:average'] == '0' or params['clf13:average'] == 'True':
            average = True
        if params['clf13:average'] == '1' or params['clf13:average'] == 'False':
            average = False
        clf = SGDClassifier(loss=loss,
                            penalty=penalty,
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            n_iter=n_iter,
                            epsilon=epsilon,
                            learning_rate=learning_rate,
                            eta0=eta0,
                            power_t=power_t,
                            average=average,
                            class_weight=params['class_weight'])

    return clf


def pipeline(params, **kwargs):
    # Params is a dict that contains the params
    # As the values are forwarded as strings you might want to convert and check them

    data_path = kwargs['data_path']
    dataset = kwargs['dataset']
    enable_cache = int(kwargs['use_caching']) == 1

    data_train = os.path.join(data_path, dataset, 'train.arff')
    X, y = load_arff_data(data_train)

    dpr = get_data_preprocessor_rescaling(params)
    params = get_data_preprocessor_balancing(params, y)
    fp = get_feature_preprocessor(params)
    clf = get_classifier(params)

    # **kwargs contains further information, like for cross validation
    #    kwargs['folds'] is 1 when no cv
    #    kwargs['fold'] is the current fold. The index is zero-based

    if enable_cache:
        steps = []
        if dpr is not None:
            steps.append(dpr)
        if fp is not None:
            steps.append(fp)
        steps.append(clf)
        import cache
        score = cache.cached_run(steps, X, y)
        result = 100.0 - 100.0 * score
    else:
        # Run your algorithm and receive a result, you want to minimize
        steps = []
        if dpr is not None:
            steps.append(('data_preprocessor_rescaling', dpr))
        if fp is not None:
            steps.append(('feature_preprocessor', fp))
        steps.append(('classifier', clf))

        ppl = Pipeline(steps)
        scores = cross_val_score(ppl, X, y, cv=int(kwargs['cv_folds']))
        result = 100.0 - 100.0 * scores.mean()

    return result


def pipeline_test(params, data_path, dataset):
    data_train = os.path.expanduser(os.path.join(data_path, dataset, 'train.arff'))
    X_train, y_train = load_arff_data(data_train)

    data_test = os.path.expanduser(os.path.join(data_path, dataset, 'test.arff'))
    X_test, y_test = load_arff_data(data_test)

    dpr = get_data_preprocessor_rescaling(params)
    params = get_data_preprocessor_balancing(params, y_train)
    fp = get_feature_preprocessor(params)
    clf = get_classifier(params)

    steps = []
    if dpr is not None:
        steps.append(('data_preprocessor_rescaling', dpr))
    if fp is not None:
        steps.append(('feature_preprocessor', fp))
    steps.append(('classifier', clf))

    ppl = Pipeline(steps)
    ppl.fit(X_train, y_train)
    y_pred = ppl.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    result = 100.0 - 100.0 * score

    return result


def get_algo_dicts(pickup_file=''):
    module = import_module('space')
    layer_dict_list = module.layer_dict_list
    new_layer_dict_list = deepcopy(layer_dict_list)
    if os.path.exists(pickup_file):
        # update the index of algorithms for SMAC
        fin = file(pickup_file, 'r')
        pick = []
        for line in fin:
            layer_pick = map(int, line.split())
            pick.append(layer_pick)
        fin.close()
        for layer, layer_pick in enumerate(pick):
            layer_dict = new_layer_dict_list[layer]
            for (algo, idx) in layer_dict.items():
                if idx in layer_pick:
                    layer_dict[algo] = layer_pick.index(idx)
                else:
                    layer_dict[algo] = -1
            new_layer_dict_list[layer] = layer_dict
        # output file just for logging the returned dict
        fout = file('new_layer_dict.txt', 'w')
        fout.write('%s' % new_layer_dict_list)
        fout.close()

    return new_layer_dict_list


if __name__ == '__main__':
    starttime = time.time()
    # Use a library function which parses the command line call
    args, params = benchmark_util.parse_cli()
    params['layer_dict_list'] = get_algo_dicts('../pickup.txt')  # cwd is in smac folder, pickup.txt in upper lr folder
    result = pipeline(params, **args)
    duration = time.time() - starttime
    print 'Result for ParamILS: %s, %f, 1, %f, %d, %s' % ('SAT', abs(duration), result, -1, str(__file__))
