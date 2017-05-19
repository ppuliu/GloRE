"""Utility functions."""

import os
import logging
import logging.config
import yaml
import numpy as np


def setup_logging(default_config_path='logging.yaml',
                  default_level=logging.INFO,
                  log_dir=None,
                  log_file=None,
                  env_key='LOG_CFG'):
    """Setup logging configuration.

    Credit to Fang-Pen Lin

    Args:
        default_config_path: default config file
        default_level: default logging level
        log_dir: dictionary to put log files
        log_file: string, log file
        env_key: config file specified via command line
    """
    config_path = default_config_path
    env_config_path = os.getenv(env_key, None)
    if env_config_path:
        config_path = env_config_path
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        if log_dir:
            # update log files
            config['handlers']['file_handler']['filename'] = \
                os.path.join(log_dir,
                             config['handlers']['file_handler']['filename'])
            if log_file:
                log_path = os.path.join(log_dir, log_file)
                log_dir = os.path.dirname(log_path)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                config['handlers']['file_handler']['filename'] = log_path

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def format_list(list):
    """format a list into a space-separated string"""
    return " ".join(str(tok) for tok in list)


def softmax(x):
    """Compute softmax values for array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
