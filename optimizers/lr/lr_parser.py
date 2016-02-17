import logging
import os
import sys

import ConfigParser

logger = logging.getLogger('HPOlib.optimizers.lr.lr_parser')


def manipulate_config(config):
    if not config.has_section('LR'):
        config.add_section('LR')

    # optional cases
    if not config.has_option('LR', 'space'):
        raise Exception('LR: space not specified in .cfg')

    if not config.has_option('LR', 'use_optimal_design'):
        raise Exception('LR: use_optimal_design not specified in .cfg')

    if not config.has_option('LR', 'init_budget'):
        raise Exception('LR: init_budget not specified in .cfg')

    if not config.has_option('LR', 'ei_budget'):
        raise Exception('LR: ei_budget not specified in .cfg')

    if not config.has_option('LR', 'bopt_budget'):
        raise Exception('LR: bopt_budget not specified in .cfg')

    if not config.has_option('LR', 'ei_xi'):
        raise Exception('LR: ei_xi not specified in .cfg')

    if not config.has_option('LR', 'top_k_pipelines'):
        raise Exception('LR: top_k_pipelines not specified in .cfg')

    path_to_optimizer = config.get('LR', 'path_to_optimizer')
    if not os.path.isabs(path_to_optimizer):
        path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

    path_to_optimizer = os.path.normpath(path_to_optimizer)
    if not os.path.exists(path_to_optimizer):
        logger.critical('Path to optimizer not found: %s' % path_to_optimizer)
        sys.exit(1)

    config.set('LR', 'path_to_optimizer', path_to_optimizer)

    return config
