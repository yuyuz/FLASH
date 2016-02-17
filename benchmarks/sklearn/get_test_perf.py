import os
import sys
import glob
import cPickle
from time import time
from ConfigParser import SafeConfigParser
from ml_framework import pipeline_test, get_algo_dicts


def load_experiment_config_file(cfg_filename):
    # Load the config file, this holds information about data, black box fn etc.
    try:
        config = SafeConfigParser(allow_no_value=True)
        config.read(cfg_filename)
        return config
    except IOError as e:
        print 'Could not open config file in directory %s' % os.getcwd()
        sys.exit(1)


pkl_file = sys.argv[1]
print 'Pickle file to be tested:', pkl_file
exp_dir = os.path.dirname(pkl_file)
pkl_file = os.path.abspath(pkl_file)
backup_cfg = load_experiment_config_file('config.cfg')
os.chdir(exp_dir)

if 'smac' in pkl_file:
    cfg = backup_cfg
else:
    cfg = load_experiment_config_file('config.cfg')

fh = open(pkl_file, 'r')
log = cPickle.load(fh)
fh.close()

base, ext = os.path.splitext(os.path.basename(pkl_file))

trials = log['trials']
i = 0
cur_best = 100
cur_best_test = 100
for trial in trials:
    params = trial['params']
    result = trial['result']
    duration = trial['duration']
    flag_SMAC = False
    if len(params['-classifier']) <= 2:  # already in SMAC
        flag_SMAC = True
    if result < 100:
        i += 1
        if result < cur_best:
            cur_best = result
        print '%d/%d' % (i, len(trials)), '\t', 'CV_result:', result, '\t', 'CV_time: %.1fs' % duration, '\t', 'CV_cur_best:', cur_best

        new_params = dict()
        for key in params:
            new_params[key[1:]] = params[key]   # remove the '-' at the head of key
        if not flag_SMAC:
            new_params['layer_dict_list'] = get_algo_dicts()
        else:
            new_params['layer_dict_list'] = get_algo_dicts('pickup.txt')
        data_path = cfg.get('HPOLIB', 'data_path')
        dataset = cfg.get('HPOLIB', 'dataset')
        tic = time()

        test_result = pipeline_test(new_params, data_path, dataset)
        trial['result'] = test_result

        fh_new = open(base + '_test' + ext, 'w')
        cPickle.dump(log, fh_new)
        fh_new.close()

        if test_result < cur_best_test:
            cur_best_test = test_result
        print '\t', 'Test_result:', test_result, '\t', 'Test_time: %.1fs' % (time() - tic), '\t', 'Test_cur_best:', cur_best_test, '\n'
