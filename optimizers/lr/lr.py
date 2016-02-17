import cPickle
import logging
import os
import sys
import HPOlib.wrapping_util as wrappingUtil


logger = logging.getLogger('HPOlib.optimizers.lr.lr')


def check_dependencies():
    try:
        import nose
        logger.debug("\tNose: %s\n" % str(nose.__version__))
    except ImportError:
        raise ImportError("Nose cannot be imported. Are you sure it's "
                          "installed?")
    try:
        import networkx
        logger.debug("\tnetworkx: %s\n" % str(networkx.__version__))
    except ImportError:
        raise ImportError("Networkx cannot be imported. Are you sure it's "
                          "installed?")
    try:
        import pymongo
        logger.debug("\tpymongo: %s\n" % str(pymongo.version))
        from bson.objectid import ObjectId
    except ImportError:
        raise ImportError("Pymongo cannot be imported. Are you sure it's"
                          " installed?")
    try:
        import numpy
        logger.debug("\tnumpy: %s" % str(numpy.__version__))
    except ImportError:
        raise ImportError("Numpy cannot be imported. Are you sure that it's"
                          " installed?")
    try:
        import scipy
        logger.debug("\tscipy: %s" % str(scipy.__version__))
    except ImportError:
        raise ImportError("Scipy cannot be imported. Are you sure that it's"
                          " installed?")


def build_lr_call(config, options, optimizer_dir):
    # For LR we have to cd to the exp_dir
    call = 'python ' + os.path.dirname(os.path.realpath(__file__)) + '/lrcall.py'
    call = ' '.join([call, '-p', os.path.join(optimizer_dir, os.path.basename(config.get('LR', 'space'))),
                     '--use_optimal_design', config.get('LR', 'use_optimal_design'),
                     '--init_budget', config.get('LR', 'init_budget'),
                     '--ei_budget', config.get('LR', 'ei_budget'),
                     '--bopt_budget', config.get('LR', 'bopt_budget'),
                     '--ei_xi', config.get('LR', 'ei_xi'),
                     '--top_k_pipelines', config.get('LR', 'top_k_pipelines'),
                     '-a', options.algo,
                     '-s', str(options.seed),
                     '--cwd', optimizer_dir])
    if options.restore:
        call = ' '.join([call, '-r'])
    return call


def main(config, options, experiment_dir, experiment_directory_prefix, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore,
    # experiment_dir:   Experiment directory/Benchmarkdirectory
    # **kwargs:         Nothing so far
    time_string = wrappingUtil.get_time_string()
    cmd = ''

    # Add path_to_optimizer to PYTHONPATH and to sys.path
    if not 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = config.get('LR', 'path_to_optimizer')
    else:
        os.environ['PYTHONPATH'] = config.get('LR', 'path_to_optimizer') + os.pathsep + os.environ['PYTHONPATH']
    sys.path.append(config.get('LR', 'path_to_optimizer'))
    optimizer_str = os.path.splitext(os.path.basename(__file__))[0]

    # Find experiment directory
    if options.restore:
        if not os.path.exists(options.restore):
            raise Exception('The restore directory does not exist')
        optimizer_dir = options.restore
    else:
        optimizer_dir = os.path.join(experiment_dir,
                                     experiment_directory_prefix +
                                     optimizer_str + '_' +
                                     str(options.seed) + '_' +
                                     time_string)

    # Build call
    cmd = build_lr_call(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        space = config.get('LR', 'space')
        abs_space = os.path.abspath(space)
        parent_space = os.path.join(experiment_dir, optimizer_str, space)
        if os.path.exists(abs_space):
            space = abs_space
        elif os.path.exists(parent_space):
            space = parent_space
        else:
            raise Exception('LR search space not found. Searched at %s and '
                            '%s' % (abs_space, parent_space))
        # Copy the hyperopt search space
        if not os.path.exists(os.path.join(optimizer_dir, os.path.basename(space))):
            os.symlink(os.path.join(experiment_dir, optimizer_str, space),
                       os.path.join(optimizer_dir, os.path.basename(space)))

    return cmd, optimizer_dir
