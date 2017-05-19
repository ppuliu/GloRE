import os
import logging
import logging.config
import yaml
import numpy as np
import random

def load_train_data(features_file, targets_file, normalize = False):
    X, mean_vals, std_vals = compute_feature_matrix(features_file, normalize)
    Y, _ = compute_target_matrix(targets_file)

    assert X.shape[0] == Y.shape[0], "Number of training examples (obtained from features) does not match" \
                                     "the number of training examples (obtained from true targets)"
    pos_indices = []
    neg_indices = []
    for i in xrange(Y.shape[0]):
        if Y[i] == 1:
            pos_indices.append(i)
        else:
            neg_indices.append(i)
            
    pos_X = X[pos_indices, :]
    neg_X = X[neg_indices, :]
    
    return pos_X, neg_X, mean_vals, std_vals

def load_test_data(features_file, targets_file, normalize = False, mean_vals = None, std_vals = None):
    X, _, _ = compute_feature_matrix(features_file, normalize, mean_vals, std_vals)
    Y, meta_infos = compute_target_matrix(targets_file)

    assert X.shape[0] == Y.shape[0], "Number of training examples (obtained from features) does not match" \
                                     "the number of training examples (obtained from true targets)"
    
    return X, Y, meta_infos


def compute_feature_matrix(input_file, normalize, mean_vals = None, std_vals = None):
    features_per_example = []
    cur_features = []
    num_features =  0
    with open(input_file) as f:
        for line in f:
            cur_features = []
            str_scores = line.strip().split()
            for score in str_scores:
                cur_features.append(float(score))
            features_per_example.append(cur_features)
            if len(cur_features) > num_features:
                num_features = len(cur_features)
    
    # padding
    for cur_features in features_per_example:
        if len(cur_features) < num_features:
            cur_features += [0] * (num_features - len(cur_features))

    feature_matrix = np.array(features_per_example, dtype=np.float32)
    print("Num of examples: {}".format(feature_matrix.shape[0]))
    print("Num of features: {}\n".format(feature_matrix.shape[1]))
    
    if normalize:
        if mean_vals is None or std_vals is None:
            mean_vals = []
            std_vals = []
            for i in xrange(feature_matrix.shape[1]):
                mean, std = np.mean(feature_matrix[:,i]), np.std(feature_matrix[:,i])
                mean_vals.append(mean)
                std_vals.append(std)
            print("Data are being normalized by fresh calculated mean and std\n")
        else:
            print("Data are being normalized by given mean and std\n")
        for i in xrange(feature_matrix.shape[1]):
            mean, std = mean_vals[i], std_vals[i]
            feature_matrix[:,i] = (feature_matrix[:,i] - mean) / std
    else:
        mean_vals = None
        std_vals = None

    return feature_matrix, mean_vals, std_vals


def compute_target_matrix(input_file):
    labels = []
    meta_infos = []
    with open(input_file) as f:
        for line in f:
            entries = line.split('\t')
            labels.append(float(entries[0]))
            if len(entries) > 1:
                meta_infos.append('\t'.join(entries[1:]).strip())

    num_of_examples = len(labels)
    target_matrix = np.array(labels, dtype=np.float32).reshape((num_of_examples, 1))

    return target_matrix, meta_infos


def get_batch(pos_X, neg_X, batch_size):
    pos_batch = []
    neg_batch = []
    for i in xrange(batch_size):
        pos_batch.append(random.choice(pos_X))
        neg_batch.append(random.choice(neg_X))
        
    return np.array(pos_batch), np.array(neg_batch)

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
