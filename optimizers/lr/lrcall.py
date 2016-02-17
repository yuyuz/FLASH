from argparse import ArgumentParser

import cPickle
from functools import partial
from importlib import import_module
import logging
import os
import sys

import hyperopt
from hyperopt.pyll.stochastic import sample
from hyperopt import hp
import HPOlib.cv as cv
import numpy as np
from scipy.stats import norm
from sklearn import linear_model

from HPOlib.format_converter.tpe_to_smac import convert_tpe_to_smac_from_object


logger = logging.getLogger('HPOlib.optimizers.lr.lrcall')


def get_pipeline_by_flatten_index(ni, index):
    layer1 = index / np.prod(ni[1:]) % ni[0]
    layer2 = index / np.prod(ni[2:]) % ni[1]
    layer3 = index / np.prod(ni[3:]) % ni[2]
    layer4 = index % ni[3]
    pipeline = [layer1, layer2, layer3, layer4]

    return pipeline


def get_x_by_flat_index(ni, index):
    x = [[0]*n for n in ni]
    pipeline = get_pipeline_by_flatten_index(ni, index)
    x_flat = []
    for layer in range(len(ni)):
        x[layer][pipeline[layer]] = 1
        x_flat += x[layer]
    x_flat = np.array([x_flat])

    return x_flat


def get_random_picks_by_optimal_design(ni, init_budget):
    # generate K initial candidates, here we use all possible pipelines
    candidates = []
    for i in range(np.prod(ni)):
        x = get_x_by_flat_index(ni, i)
        candidates.append(x)

    x1 = get_x_by_flat_index(ni, np.random.randint(np.prod(ni)))
    P = np.dot(x1.T, x1)
    S = [x1]
    for k in range(2, init_budget+1):
        D_scores = []
        for j, xj in enumerate(candidates):
            score = get_det(k, P+np.dot(xj.T, xj))
            D_scores.append(score)
        jstars = np.argwhere(D_scores == np.amax(D_scores))
        jstar = np.random.choice(jstars.flatten())
        x_jstar = candidates[jstar]
        P += np.dot(x_jstar.T, x_jstar)
        S.append(x_jstar)

    picks = []
    cum_ni = np.cumsum(ni)
    for x in S:
        pick = [[] for layer in range(len(ni))]
        pick[0] = [np.argmax(x[0][:cum_ni[0]])]
        pick[1] = [np.argmax(x[0][cum_ni[0]:cum_ni[1]])]
        pick[2] = [np.argmax(x[0][cum_ni[1]:cum_ni[2]])]
        pick[3] = [np.argmax(x[0][cum_ni[2]:])]
        picks.append(pick)

    return picks


def get_pure_random_picks(ni, init_budget):
    picks = []
    for k in range(init_budget):
        idx = np.random.randint(np.prod(ni))
        pipeline = get_pipeline_by_flatten_index(ni, idx)
        pick = []
        for layer in range(len(ni)):
            pick.append([pipeline[layer]])
        picks.append(pick)

    return picks


def get_det(k, P):
    assert P.shape[0] == P.shape[1]  # verify P is a square matrix
    p = min(k, P.shape[0])
    eigvals, eigvecs = np.linalg.eig(P)
    top_eigvals = sorted(eigvals, reverse=True)[:p]

    return np.prod(top_eigvals)


def get_covered_units_by_lr(ni, ebeta, top_k_pipelines):
    ppl_pred = []
    for i in range(np.prod(ni)):
        pipeline = get_pipeline_by_flatten_index(ni, i)
        predict = np.sum([ebeta[layer][pipeline[layer]] for layer in range(len(ni))])
        ppl_pred.append([predict, pipeline])

    sorted_ppl_pred = sorted(ppl_pred, key=lambda x: x[0])
    cover = [[] for layer in range(len(ni))]
    for k in range(top_k_pipelines):
        pipeline = sorted_ppl_pred[k][1]
        for layer in range(len(ni)):
            unit = pipeline[layer]
            if unit not in cover[layer]:
                cover[layer].append(unit)

    return cover


def get_covered_units_by_ei(ni, alpha, lr, lr_time, X, y, ei_xi, top_k_pipelines):
    ppl_pred = []
    for i in range(np.prod(ni)):
        x = [[0]*n for n in ni]
        pipeline = get_pipeline_by_flatten_index(ni, i)
        x_flat = []
        for layer in range(len(ni)):
            x[layer][pipeline[layer]] = 1
            x_flat += x[layer]
        predict = compute_EI(ni, alpha, lr, lr_time, X, y, ei_xi, x_flat)
        ppl_pred.append([predict, pipeline])

    sorted_ppl_pred = sorted(ppl_pred, key=lambda x: x[0], reverse=True)
    cover = [[] for layer in range(len(ni))]
    for k in range(top_k_pipelines):
        pipeline = sorted_ppl_pred[k][1]
        for layer in range(len(ni)):
            unit = pipeline[layer]
            if unit not in cover[layer]:
                cover[layer].append(unit)

    return cover


def get_num_of_trials(log_filename, filter_valid=False):
    times = 0

    fh = open(log_filename)
    log = cPickle.load(fh)
    fh.close()
    trials = log['trials']
    for trial in trials:
        result = trial['result']
        if filter_valid is False:
            times += 1
        elif result < 100:
            times += 1

    return times


def get_last_run(log_filename):
    fh = open(log_filename)
    log = cPickle.load(fh)
    fh.close()
    trials = log['trials']
    result = trials[-1]['result']
    time = trials[-1]['duration']

    return result, time


def construct_subspace(module, pick):
    rescaling = module.rescaling_list
    rescaling_sublist = []
    for i in pick[0]:
        rescaling_sublist.append(rescaling[i])
    rescaling = hp.choice('rescaling', rescaling_sublist)

    balancing = module.balancing_list
    balancing_sublist = []
    for i in pick[1]:
        balancing_sublist.append(balancing[i])
    balancing = hp.choice('balancing', balancing_sublist)

    fp = module.feat_pre_list
    feat_pre_sublist = []
    for i in pick[2]:
        feat_pre_sublist.append(fp[i])
    feat_pre = hp.choice('feat_pre', feat_pre_sublist)

    clf = module.classifier_list
    classifier_sublist = []
    for i in pick[3]:
        classifier_sublist.append(clf[i])
    classifier = hp.choice('classifier', classifier_sublist)

    subspace = {
        'rescaling': rescaling,
        'balancing': balancing,
        'feat_pre': feat_pre,
        'classifier': classifier}

    return subspace


def compute_EI(ni, alpha, lr, lr_time, X, y, ei_xi, x):
    var = np.var(lr.predict(X) - y)
    m = np.dot(X.T, X)
    inv = np.linalg.inv(m + alpha * np.eye(sum(ni)))
    x_flat = np.array(x)
    mu_x = lr.predict([x_flat])
    var_x = var * (1 + np.dot(np.dot(x_flat, inv), x_flat.T))
    sigma_x = np.sqrt(var_x)
    u = (np.min(y) - ei_xi - mu_x) / sigma_x
    EI = sigma_x * (u*norm.cdf(u) + norm.pdf(u))
    estimated_time = lr_time.predict([x_flat])[0]
    EIPS = EI / estimated_time

    return EIPS


def get_next_by_EI(ni, alpha, lr, lr_time, X, y, ei_xi):
    '''
    Args:
        ni: number of units in the each layer
        alpha: lambda for Ridge regression
        lr: fitted performance model in burning period
        lr_time: fitted time model in burning period
        X: all previous inputs x
        y: all previous observations corresponding to X
        ei_xi: parameter for EI exploitation-exploration trade-off

    Returns:
        x_next: a nested list [[0,1,0], [1,0,0,0], ...] as the next input x to run a specified pipeline
    '''
    var = np.var(lr.predict(X) - y)
    m = np.dot(X.T, X)
    inv = np.linalg.inv(m + alpha * np.eye(sum(ni)))
    maxEI = float('-inf')
    x_next = None
    for i in range(np.prod(ni)):
        x = [[0]*n for n in ni]
        x_flat = []
        pipeline = get_pipeline_by_flatten_index(ni, i)
        for layer in range(len(ni)):
            x[layer][pipeline[layer]] = 1
            x_flat += x[layer]
        x_flat = np.array(x_flat)
        mu_x = lr.predict([x_flat])
        var_x = var * (1 + np.dot(np.dot(x_flat, inv), x_flat.T))
        sigma_x = np.sqrt(var_x)
        u = (np.min(y) - ei_xi - mu_x) / sigma_x
        EI = sigma_x * (u*norm.cdf(u) + norm.pdf(u))
        estimated_time = lr_time.predict([x_flat])[0]
        EIPS = EI / estimated_time
        if EIPS > maxEI:
            maxEI = EIPS
            x_next = x

    return x_next


def main():
    parser = ArgumentParser()

    parser.add_argument('-p', '--space',
                        dest='spaceFile', help='Where is the space.py located?')
    parser.add_argument('--use_optimal_design',
                        dest='use_optimal_design', help='Use optimal design or pure random initialization?')
    parser.add_argument('--init_budget',
                        dest='init_budget', help='How many evaluations for random burning period?')
    parser.add_argument('--ei_budget',
                        dest='ei_budget', help='How many evaluations for EI controlled online period?')
    parser.add_argument('--bopt_budget',
                        dest='bopt_budget', help='How many evaluations for Bayesian optimization after get subspace?')
    parser.add_argument('--ei_xi',
                        dest='ei_xi', help='What is the exploration parameter for computing EI?')
    parser.add_argument('--top_k_pipelines',
                        dest='top_k_pipelines', help='How many top (LR predicted) pipelines to cover in subspace?')
    parser.add_argument('-s', '--seed', default='1',
                        dest='seed', type=int, help='Seed for the algorithm')

    parser.add_argument('-a', '--algo', default='SMAC',
                        dest='algo', type=str, help='Specify the algorithm after LR, can be SMAC or TPE')

    parser.add_argument('-r', '--restore', action='store_true',
                        dest='restore', help='When this flag is set state.pkl is restored in ' +
                             'the current working directory')
    parser.add_argument('--random', default=False, action='store_true',
                        dest='random', help='Use a random search')
    parser.add_argument('--cwd', help='Change the working directory before '
                                      'optimizing.')

    args, unknown = parser.parse_known_args()

    if args.cwd:
        os.chdir(args.cwd)

    if not os.path.exists(args.spaceFile):
        logger.critical('Search space not found: %s' % args.spaceFile)
        sys.exit(1)

    # First remove '.py'
    space, ext = os.path.splitext(os.path.basename(args.spaceFile))

    # Then load dict searchSpace and out function cv.py
    sys.path.append('./')
    sys.path.append('')

    module = import_module(space)
    search_space = module.space
    ni = [len(d) for d in module.layer_dict_list]  # number of units in each layer
    cum_ni = np.cumsum(ni)

    log_filename = 'lr.pkl'

    # Random burning period as initialization
    init_budget = int(args.init_budget)
    if args.use_optimal_design == '1':
        picks = get_random_picks_by_optimal_design(ni, init_budget)
    else:
        picks = get_pure_random_picks(ni, init_budget)
    for i in range(init_budget):
        times = get_num_of_trials(log_filename, filter_valid=False)
        valid_times = get_num_of_trials(log_filename, filter_valid=True)
        logger.info('IMPORTANT! YOU ARE RUNNING FLASH WITH: %s' % args.algo)
        logger.info('Total evaluation times: %d, valid times: %d' % (times, valid_times))
        logger.info('Random burning period times: %d, valid times: %d' % (times, valid_times))
        subspace = construct_subspace(module, picks[i])
        params = sample(subspace)
        cv.main(params)
    valid_times_in_random_period = get_num_of_trials(log_filename, filter_valid=True)

    # Train the first LR model before entering into EI controlled period
    fh = open(log_filename)
    log = cPickle.load(fh)
    trials = log['trials']
    fh.close()
    X = []
    y = []
    y_time = []
    for trial in trials:
        result = trial['result']
        time = trial['duration']
        # make sure the logged result is a number (accept evaluations return 100.0)
        if result <= 100:
            params = trial['params']
            rescaling = params['-rescaling']
            balancing = params['-balancing']
            feat_pre = params['-feat_pre']
            clf = params['-classifier']
            x = [[0]*n for n in ni]
            x[0][module.d_rescaling[rescaling]] = 1
            x[1][module.d_balancing[balancing]] = 1
            x[2][module.d_feat_pre[feat_pre]] = 1
            x[3][module.d_clf[clf]] = 1
            x_flat = np.array(x[0]+x[1]+x[2]+x[3])
            X.append(x_flat)
            y.append(result)
            y_time.append(np.log(time))
    X = np.array(X)
    alpha = 1.0
    lr = linear_model.Ridge(alpha=alpha)
    lr.fit(X, y)
    lr_time = linear_model.Ridge(alpha=alpha)
    lr_time.fit(X, y_time)

    # Online period controlled by EI
    ei_budget = int(args.ei_budget)
    for i in range(ei_budget):
        times = get_num_of_trials(log_filename, filter_valid=False)
        valid_times = get_num_of_trials(log_filename, filter_valid=True)
        logger.info('Total evaluation times: %d, valid times: %d' % (times, valid_times))
        logger.info('EI controlled period times: %d, valid times: %d' % (times - init_budget,
                                                                         valid_times - valid_times_in_random_period))
        ebeta = lr.coef_[:cum_ni[0]], \
                lr.coef_[cum_ni[0]:cum_ni[1]], \
                lr.coef_[cum_ni[1]:cum_ni[2]], \
                lr.coef_[cum_ni[2]:]
        logger.info('LR model estimated unit ranking: %s %s %s %s' % (str(ebeta[0].argsort()),
                                                                      str(ebeta[1].argsort()),
                                                                      str(ebeta[2].argsort()),
                                                                      str(ebeta[3].argsort())))
        ebeta_time = lr_time.coef_[:cum_ni[0]], \
                     lr_time.coef_[cum_ni[0]:cum_ni[1]], \
                     lr_time.coef_[cum_ni[1]:cum_ni[2]], \
                     lr_time.coef_[cum_ni[2]:]
        logger.info('LR Time model estimated unit ranking: %s %s %s %s' % (str(ebeta_time[0].argsort()),
                                                                           str(ebeta_time[1].argsort()),
                                                                           str(ebeta_time[2].argsort()),
                                                                           str(ebeta_time[3].argsort())))
        # pick the best pipeline by EI
        x_next = get_next_by_EI(ni, alpha, lr, lr_time, X, y, float(args.ei_xi))
        pick = [[np.argmax(x_next_i)] for x_next_i in x_next]
        subspace = construct_subspace(module, pick)
        params = sample(subspace)
        cv.main(params)

        result, time = get_last_run(log_filename)
        if result <= 100:
            x_next_flat = np.array(x_next[0]+x_next[1]+x_next[2]+x_next[3])
            X = np.vstack([X, x_next_flat])
            y.append(result)
            y_time.append(np.log(time))
            lr = linear_model.Ridge(alpha=alpha)
            lr.fit(X, y)
            lr_time = linear_model.Ridge(alpha=alpha)
            lr_time.fit(X, y_time)
    valid_times_in_ei_period = get_num_of_trials(log_filename, filter_valid=True) - valid_times_in_random_period

    # Construct subspace based on LR prediction
    final_ebeta = lr.coef_[:cum_ni[0]], \
                  lr.coef_[cum_ni[0]:cum_ni[1]], \
                  lr.coef_[cum_ni[1]:cum_ni[2]], \
                  lr.coef_[cum_ni[2]:]
    final_ebeta_time = lr_time.coef_[:cum_ni[0]], \
                       lr_time.coef_[cum_ni[0]:cum_ni[1]], \
                       lr_time.coef_[cum_ni[1]:cum_ni[2]], \
                       lr_time.coef_[cum_ni[2]:]
    final_pick = get_covered_units_by_ei(ni, alpha, lr, lr_time, X, y, 0, int(args.top_k_pipelines))
    final_subspace = construct_subspace(module, final_pick)

    logger.info('LR model estimated unit ranking: %s %s %s %s' % (str(final_ebeta[0].argsort()),
                                                                  str(final_ebeta[1].argsort()),
                                                                  str(final_ebeta[2].argsort()),
                                                                  str(final_ebeta[3].argsort())))
    logger.info('LR Time model estimated unit ranking: %s %s %s %s' % (str(final_ebeta_time[0].argsort()),
                                                                       str(final_ebeta_time[1].argsort()),
                                                                       str(final_ebeta_time[2].argsort()),
                                                                       str(final_ebeta_time[3].argsort())))
    logger.info('Selected pipelines: %s %s %s %s' % (final_pick[0],
                                                     final_pick[1],
                                                     final_pick[2],
                                                     final_pick[3]))

    # Phase 3 with SMAC
    if args.algo == 'SMAC':
        fh = file('pickup.txt', 'w')
        for layer_pick in final_pick:
            for i in layer_pick:
                fh.write('%d ' % i)
            fh.write('\n')
        fh.close()
        subspace = construct_subspace(module, final_pick)
        new_space = convert_tpe_to_smac_from_object(subspace)
        fh = open('params.pcs', 'w')
        fh.write(new_space)
        fh.close()

    # Phase 3 with TPE
    elif args.algo == 'TPE':
        fn = cv.main
        domain = hyperopt.Domain(fn, final_subspace, rseed=int(args.seed))
        trials = hyperopt.Trials()
        bopt_budget = int(args.bopt_budget)
        for i in range(bopt_budget):
            times = get_num_of_trials(log_filename, filter_valid=False)
            valid_times = get_num_of_trials(log_filename, filter_valid=True)
            logger.info('Total evaluation times: %d, valid times: %d' % (times, valid_times))
            logger.info('TPE period times: %d, valid times: %d' %
                        (times - init_budget - ei_budget,
                         valid_times - valid_times_in_random_period - valid_times_in_ei_period))
            logger.info('LR model estimated unit ranking: %s %s %s %s' % (str(final_ebeta[0].argsort()),
                                                                          str(final_ebeta[1].argsort()),
                                                                          str(final_ebeta[2].argsort()),
                                                                          str(final_ebeta[3].argsort())))
            logger.info('LR Time model estimated unit ranking: %s %s %s %s' % (str(final_ebeta_time[0].argsort()),
                                                                               str(final_ebeta_time[1].argsort()),
                                                                               str(final_ebeta_time[2].argsort()),
                                                                               str(final_ebeta_time[3].argsort())))
            logger.info('Selected pipelines: %s %s %s %s' % (final_pick[0],
                                                             final_pick[1],
                                                             final_pick[2],
                                                             final_pick[3]))
            # in exhaust, the number of evaluations is max_evals - num_done
            tpe_with_seed = partial(hyperopt.tpe.suggest, seed=int(args.seed))
            rval = hyperopt.FMinIter(tpe_with_seed, domain, trials, max_evals=i)
            rval.exhaust()


if __name__ == '__main__':
    main()
