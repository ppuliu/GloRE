import sys, os, time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from datetime import datetime
import shutil
import logging
import gzip

from mlp import mlp
from mlp_data_utils import get_batch, load_train_data, load_test_data, setup_logging

def train(sess, model, checkpoint_dir,
              pos_X, neg_X, dev_percentage,
              max_steps, eval_every, batch_size,
              early_stop, patience, do_evaluation):
    # shuffle and split data
    (pos_data_size, dim) = pos_X.shape
    (neg_data_size, dim) = neg_X.shape
    
    # shuffle
    np.random.seed(103)
    pos_X_shuffled = pos_X[np.random.permutation(pos_data_size),:]
    neg_X_shuffled = neg_X[np.random.permutation(neg_data_size),:]
    # split
    pos_tr_end = pos_data_size - int(dev_percentage * float(pos_data_size))
    neg_tr_end = neg_data_size - int(dev_percentage * float(neg_data_size))

    pos_X_train = pos_X_shuffled[:pos_tr_end,:]
    pos_X_dev = pos_X_shuffled[pos_tr_end:,:]

    neg_X_train = neg_X_shuffled[:neg_tr_end,:]
    neg_X_dev = neg_X_shuffled[neg_tr_end:,:]
    
    print pos_X_train.shape, neg_X_train.shape    
    
    # train procedure
    curr_step = 0
    tr_loss, tr_acc, dev_loss, dev_acc  = 0.0, 0.0, 0.0, 0.0
    tr_losses, tr_accs, dev_losses, dev_accs = [], [], [], []
    input_feed = None
    
    logger = logging.getLogger(__name__)
    
    print("Training started")
    while curr_step < max_steps:
        
        if curr_step == 0:
            logger.info('Intial parameters')
            print_variables(logger)

        # batch update
        pos_batch, neg_batch = get_batch(pos_X_train, neg_X_train, batch_size)
        
        input_feed = {model.pos_x: pos_batch, model.neg_x: neg_batch}
        output_feed = [model.loss, model.predicts, model.update]
        step_loss, step_predicts, _ = sess.run(output_feed, input_feed)

        curr_step += 1
                
        tr_loss += step_loss / eval_every
        tr_acc += np.mean(step_predicts) / eval_every
        
        
        if curr_step % eval_every == 0:
            if do_evaluation == 'True':
                for i in xrange(eval_every):
                    pos_batch, neg_batch = get_batch(pos_X_dev, neg_X_dev, batch_size)
                    
                    input_feed = {model.pos_x: pos_batch, model.neg_x: neg_batch}
                    output_feed = [model.loss, model.predicts]
                    step_loss, step_predicts = sess.run(output_feed, input_feed)
        
                    dev_loss += step_loss / eval_every
                    dev_acc += np.mean(step_predicts) / eval_every
            else:
                dev_loss = tr_loss
                dev_acc = tr_acc
            

            num_examples_seen = curr_step * batch_size
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            logger.info("\n{0}: Progress after {1} steps and {2} examples seen"
                  "\nTrain-set: loss={3}, acc={4}"
                  "\nDev-set  : loss={5}, acc={6}".format(time, curr_step, num_examples_seen,
                                                                          tr_loss, tr_acc,
                                                                          dev_loss, dev_acc))
            tr_losses.append(tr_loss)
            tr_accs.append(tr_acc)
            dev_accs.append(dev_acc)
            dev_losses.append(dev_loss)            
            if dev_loss <= min(dev_losses):
                logger.info("==========================\n")
                logger.info("Best accuracy so far!!!\n")
                logger.info("==========================\n")
                checkpoint_path = os.path.join(checkpoint_dir,
                                                   "mlp_best.ckpt")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)
            
                print_variables(logger)
            
            model.run_summary_op(sess, input_feed, curr_step)
            

            tr_loss, tr_acc, dev_loss, dev_acc  = 0.0, 0.0, 0.0, 0.0
        
def test(sess, model, features, targets, meta_infos, batch_size, our_pr_file, their_pr_file, output_n):

    data_size = features.shape[0]
    total_relation_facts = int(np.sum(targets))
    num_steps = int(data_size / batch_size)
    add_meta_info = True if len(meta_infos) > 0 else False
 
    print("Testing started")
    
    scores_and_labels = []
    for i in xrange(num_steps):
        x = features[i * batch_size : (i + 1) * batch_size, :]
        y = targets[i * batch_size : (i + 1) * batch_size]
        if add_meta_info:
            z = meta_infos[i * batch_size : (i + 1) * batch_size]
        else:
            z= [0] * len(y)
           
        input_feed = {model.pos_x: x}
        scores = sess.run(model.scores, input_feed)
        scores_and_labels += zip(scores.flatten(), y.flatten(), z)
    
    scores_and_labels.sort(reverse = True)
    correct = 0.0
    with open(our_pr_file, 'w') as f:
        if output_n < 0 or output_n > len(scores_and_labels):
            output_n = len(scores_and_labels)
        for i in xrange(output_n):
            score, label, meta = scores_and_labels[i]
            label = int(label)
            if label == 1:
                correct += 1
            precision = correct / (i + 1)
            recall = correct / total_relation_facts
            if add_meta_info:
                f.write('{0:.6f}\t{1:.6f}\t{2:.6f}\t{3}\t{4}\n'.format(precision, recall, score, label, meta))
            else:
                f.write('{0:.6f}\t{1:.6f}\t{2:.6f}\t{3}\n'.format(precision, recall, score, label))
            
    # output their pr file
    if not add_meta_info:
        meta_infos = [0] * len(targets)
    scores_and_labels = zip(features[:,0].flatten(), targets.flatten(), meta_infos)
    scores_and_labels.sort(reverse = True)
    correct = 0.0
    with open(their_pr_file, 'w') as f:
        if output_n < 0 or output_n > len(scores_and_labels):
            output_n = len(scores_and_labels)
        for i in xrange(output_n):
            score, label, meta = scores_and_labels[i]
            label = int(label)
            if label == 1:
                correct += 1
            precision = correct / (i + 1)
            recall = correct / total_relation_facts
            if add_meta_info:
                f.write('{0:.6f}\t{1:.6f}\t{2:.6f}\t{3}\t{4}\n'.format(precision, recall, score, label, meta))
            else:
                f.write('{0:.6f}\t{1:.6f}\t{2:.6f}\t{3}\n'.format(precision, recall, score, label))
    
    print 'Total number of relation facts: {0}'.format(total_relation_facts)


def print_variables(logger):
    variables = tf.trainable_variables()
    if len(variables) > 4:
        return
    for v in variables:
        logger.info('{0}\n{1}'.format(v.name, v.eval()))
        
# Use sample scale to randomly pick samples proportional to their weights
def calc_sample_scale(sample_weights):
    total_weight = float(sum(sample_weights))
    sample_scale = []
    for i in xrange(len(sample_weights)):
        if i == 0:
            sample_scale.append(sample_weights[i] / total_weight)
        else:
            sample_scale.append(sample_scale[-1] +
                                sample_weights[i] / total_weight)
    sample_scale[-1] = 1.0
    return sample_scale        

def main():
    parser = argparse.ArgumentParser("run_mlp.py")
    parser.add_argument("--features_file", type=str,
                        help="Training data features, each line is corresponding to "
                             "one example's features, separated by space")
    parser.add_argument("--targets_file", type=str,
                        help="Training data true targets, each line is corresponding "
                             "to ones example's true target, aligned with features file")
    parser.add_argument("--log_dir", type=str,
                        help="Main output directory in which another directory to save "
                             "info for this training will be created!")
    parser.add_argument("--optimizer_type", type=str,
                        choices=["vanilla", "adagrad", "rmsprop", "adam"], default="adam",
                        help="Optimizer to use when updating model params based on gradient")
    parser.add_argument("--l2_penalty", type=float,
                        default=0.0,
                        help="Weight of l2-norm penalty term on cost")
    parser.add_argument("--max_steps", type=int,
                        default=1000000,
                        help="Maximum number of steps")
    parser.add_argument("--batch_size", type=int,
                        default=128,
                        help="Batch size")
    parser.add_argument("--input_dim", type=int,
                        default=2,
                        help="Input dimension")
    parser.add_argument("--eval_every", type=int,
                        default=100,
                        help="Evaluate the progress of model every this number steps")
    parser.add_argument("--learning_rate", type=float,
                        default=0.001,
                        help="Lerning rate")
    parser.add_argument("--dev_percentage", type=float,
                        default=0.1,
                        help="Development set percentage")
    parser.add_argument("--early_stop", type=str,
                        choices=["True", "False"], default="False",
                        help="Whether to apply early stop or not")
    parser.add_argument("--patience", type=int,
                        default=100,
                        help="Num of epochs to wait before early stop if no progress on the dev set")
    parser.add_argument("--sizes", type=str,
                    default="2",
                    help="Num of units in the hidden layers")
    parser.add_argument("--mode", type=str,
                    choices=["train", "test"], default="train",
                    help="train or test")
    parser.add_argument("--create_input_files", type=str,
                    choices=["True", "False"], default="False",
                    help="Generate input data from emb and moana scores")
    parser.add_argument("--emb_score_file", type=str,
                    help="Embedding score file")
    parser.add_argument("--moana_file", type=str,
                    help="Other methods' output score file")
    parser.add_argument("--kb_relation_file", type=str,
                    help="The relation to id file")
    parser.add_argument("--our_pr_file", type=str,
                    default="",
                    help="Our pr file")
    parser.add_argument("--their_pr_file", type=str,
                    default="",
                    help="Their pr file")
    parser.add_argument("--output_n", type=int,
                        default=2000,
                        help="The number of top scores to output")
    parser.add_argument("--do_evaluation", type=str,
                    choices=["True", "False"], default="True",
                    help="Whether to do evaluation or not")
    parser.add_argument("--use_gate", type=str,
                choices=["True", "False"], default="False",
                help="Whether to use gate or not")
    parser.add_argument("--use_cap", type=str,
                choices=["True", "False"], default="False",
                help="Whether to use cap or not")
    parser.add_argument("--normalize_data", type=str,
                choices=["True", "False"], default="False",
                help="Whether to normalize data by feature or not")
    parser.add_argument("--train_mean", type=str,
                default="",
                help="Mean values of each feature in the training data")
    parser.add_argument("--train_std", type=str,
                default="",
                help="std values of each feature in the training data")
    parser.add_argument("--add_meta_info", type=str,
            choices=["True", "False"], default="False",
            help="Whether to add meta info in the target file or not")
    
    args = parser.parse_args()
    print("\nParameters:")
    params = []
    for (param, value) in sorted(vars(args).items()):
        param_line = "{}={}".format(param, value)
        print(param_line)
        params.append(param_line)
    print("")

    if args.early_stop and args.patience < 1:
        raise ValueError("Unexpected patience value when early stop is enabled: {0}".format(args.patience))
        
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.mode == 'train':
        reply = raw_input('Create new directory: ' + args.log_dir + '? [y/[n]] ')
        if reply == 'y':
            for f in os.listdir(args.log_dir):
                if f.endswith('.txt'):
                    continue
                path = os.path.join(args.log_dir, f)
                try:
                    if os.path.isfile(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e:
                    raise e
        else:
            print('Aborting...')
            sys.exit()
    else:
        reply = raw_input('Proceed? [y/[n]] ')
        if reply is not 'y':
            print('Aborting...')
            sys.exit()        
        
    if args.sizes.strip() == '':
        sizes = []
    else:
        sizes = map(int, args.sizes.split('_'))
        
    if args.use_gate == 'True':
        use_gate = True
    else:
        use_gate = False
        
    if args.use_cap == 'True':
        use_cap = True
    else:
        use_cap = False
        
    if args.normalize_data == 'True':
        normalize_data = True
    else:
        normalize_data = False
    
    if args.train_mean.strip() is not '' and args.train_std.strip() is not '':
        mean_vals = map(float, args.train_mean.split('_'))
        std_vals = map(float, args.train_std.split('_'))
    else:
        mean_vals = None
        std_vals = None
        
    features_file = args.features_file
    targets_file = args.targets_file
    our_pr_file = args.our_pr_file
    their_pr_file = args.their_pr_file
    
    if args.add_meta_info == 'True':
        add_meta_info = True
    else:
        add_meta_info = False
    
    if args.create_input_files == 'True':
        print 'Creating new input files:\n{0}\n{1}\n'.format(features_file, targets_file)
        create_input_files(args.moana_file, args.emb_score_file, 
                           features_file, targets_file, add_meta_info)
    
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        if args.mode == 'train':
            
            setup_logging(log_dir=args.log_dir, log_file='train.log')
            
            print("Loading training data from\n%s\n%s\n" % (features_file, targets_file))
            pos_X, neg_X, mean_vals, std_vals = load_train_data(features_file, targets_file, normalize_data)
            if args.input_dim is not pos_X.shape[1]:
                print 'input dimension is changed to : {}'.format(pos_X.shape[1])
                args.input_dim = pos_X.shape[1]
            if normalize_data:
                args.train_mean = '_'.join(map(str, mean_vals))
                args.train_std = '_'.join(map(str, std_vals))
                print 'trian_mean is changed to :\n {}'.format(args.train_mean)
                print 'trian_std is changed to :\n {}'.format(args.train_std)

            print("# of positive samples: {0}, # of negative samples: {1}\n"
                  .format(pos_X.shape[0], neg_X.shape[0]))
            
            # write params
            params_file = os.path.join(args.log_dir, "config")
            with open(params_file, 'w') as f:
                for (param, value) in sorted(vars(args).items()):
                    f.write("{}={}\n".format(param, value))
            print("Model params are written into {}\n".format(params_file))
        
            model = mlp(args.input_dim, sizes, args.optimizer_type, args.learning_rate, args.l2_penalty, use_cap = use_cap, use_gate = use_gate)

            sess.run(tf.initialize_all_variables())

            model.summary_writer = tf.train.SummaryWriter(os.path.join(args.log_dir, 'log'), sess.graph)
        
            train(sess, model, args.log_dir,
                  pos_X, neg_X, args.dev_percentage,
                  args.max_steps, args.eval_every, args.batch_size,
                  args.early_stop, args.patience, args.do_evaluation)
    
        elif args.mode == 'test':
            model = mlp(args.input_dim, sizes, args.optimizer_type, args.learning_rate, args.l2_penalty, use_cap = use_cap, use_gate = use_gate)
            
            # load model
            ckpt = tf.train.get_checkpoint_state(args.log_dir)
            if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" %
                      ckpt.model_checkpoint_path)
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Could find checkpoint files.")
                sys.exit()
            
            print("Loading test data from\n%s\n%s\n" % (features_file, targets_file))
            features, targets, meta_infos = load_test_data(features_file, targets_file, normalize_data, mean_vals, std_vals)
            
            test(sess, model, features, targets, meta_infos, args.batch_size, our_pr_file, their_pr_file, args.output_n)
        else:
            raise ValueError('mode is not supported')

            
def create_input_files(moana_file, emb_score_file, feature_file, target_file, add_meta_info):

    final_scores = []
    emb_scores = []
    missed_relation_ids = []
    relationTotal = 0
    total_relation_facts = 0
    total_num_entries = 0
    total_num_valid_sentences = 0.0
        
    with open(emb_score_file) as f:
        i = 0
        for line in f:
            i += 1
            if i == 1:
                relationTotal = int(line)
                continue
            if i == 2:
                if line.strip() == '':
                    missed_relation_ids = []
                else:
                    missed_relation_ids = map(int, line.strip().split(' '))
                continue
            
            if line.strip() == '':
                emb_scores.append([])
                continue
            emb_scores.append([float(s) for s in line.strip().split(' ')])
            total_num_valid_sentences += 1
    
    with gzip.open(moana_file, 'rb') as f:
        with open(feature_file, 'w') as f_f:
            with open(target_file, 'w') as t_f:
                moana_n = 0
                line = f.readline()
                while not line == '':
                    if moana_n % 10000 == 0:
                        print moana_n
                    moana_n += 1

                    line = line.strip()
                    if not line == '#':
                        raise ValueError('bad moana file')

                    entity_pair = f.readline().strip()
                    sen_ids = map(int, f.readline().strip().split('\t'))
                    labels = map(int, f.readline().strip().split('\t'))
                    moana_scores = map(float, f.readline().strip().split('\t'))

                    line = f.readline()

                    num_sentences_used = []
                    emb_scores_to_use = []
                    moana_scores_to_use = []
                    targets = []
                    meta_infos = []

                    for rid in xrange(1, relationTotal):
                        if rid in missed_relation_ids:
                            continue

                        rid_scores = []
                        for sid_index in xrange(len(sen_ids)):
                            sid = sen_ids[sid_index]
                            if len(emb_scores[sid]) == 0:
                                continue
                            if sum(emb_scores[sid][:]) == 0:
                                rid_scores.append(0.0)
                                continue
                            rid_scores.append(emb_scores[sid][rid] / sum(emb_scores[sid][:]))

                        if len(rid_scores) == 0:
                            continue

                        num_sentences_used.append(len(rid_scores))    
                        emb_scores_to_use.append(rid_scores)
                        moana_scores_to_use.append(moana_scores[rid])
                        targets.append(labels[rid])
                        meta_infos.append(entity_pair + '\t' + str(rid))


                    for i in xrange(len(targets)):
                        features = []
                        # moana score
                        features.append(moana_scores_to_use[i])
                        
                        arr = sorted(emb_scores_to_use[i], reverse = True)
                        # sum of emb scores
                        features.append(np.sum(arr))
                        # number of sentences
                        #features.append(num_sentences_used[i] / total_num_valid_sentences)
                        # max of emb scores
                        #features.append(arr[0])
                        # min of emb scores
                        #features.append(arr[-1])
                        # mean of emb scores
                        #features.append(np.mean(arr))
                        # std of emb scores
                        #features.append(np.std(arr))

                        # all sentence scores
                        #features += arr
                
                        f_f.write(' '.join(map(str, features)) + '\n')
                        if add_meta_info:
                            t_f.write(str(targets[i]) + '\t' + meta_infos[i] + '\n')
                        else:
                            t_f.write(str(targets[i]) + '\n')
                        
                        if targets[i] == 1:
                            total_relation_facts += 1
                            
                        total_num_entries += 1
                
            
    print 'total number of relation facts: {0}'.format(total_relation_facts)
    print 'total number of examples generated: {0}'.format(total_num_entries)
  
        
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
