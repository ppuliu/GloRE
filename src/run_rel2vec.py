from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import operator
import json
import numpy as np
import logging
import shutil
import subprocess
import gzip

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.python.platform

import rel2vec_utils as utils
import rel2vec_data_utils as data_utils
import rel2vec_model

# data configuration
tf.app.flags.DEFINE_boolean("fresh_start", False,
                            "True to delete all the files in model_dir")
tf.app.flags.DEFINE_string("data_dir", None, "Data directory")
tf.app.flags.DEFINE_string("model_dir", "model", "Model directory.")
tf.app.flags.DEFINE_string("vocab_file", "vocab.txt", "Vocabulary file.")
tf.app.flags.DEFINE_integer("encoder_size", 10, "Max encoder input length.")
tf.app.flags.DEFINE_integer("decoder_size", 3, "Max decoder input length.")
tf.app.flags.DEFINE_integer("encoder_vocab_size", 40000,
                            "Encoder vocabulary size.")
tf.app.flags.DEFINE_integer("decoder_vocab_size", -1,
                            "Decoder vocabulary size.")
tf.app.flags.DEFINE_string("encoder_embedding_file", None,
                           "File of encoder vocab embeddings.")
tf.app.flags.DEFINE_string("decoder_embedding_file", None,
                           "File of decoder vocab embeddings.")
tf.app.flags.DEFINE_string("word2vec_normalization", None,
                           "How to normalize word2vec embeddings. "
                           "{unit_var, unit_norm, None}")
tf.app.flags.DEFINE_string("train_log", None,
                           "Training log file.")
tf.app.flags.DEFINE_string("test_log", None,
                           "Testing log file.")

# files
tf.app.flags.DEFINE_string("text_relation_file", None,
                           "Input textual relation file for the 'gen_scores' mode."
                           "One to-be-tested text relation per line.")
tf.app.flags.DEFINE_string("kb_relation_file", None,
                           "File that contains all the kb relations we want to use for the 'gen_scores' mode. "
                           "kb_relation id")
tf.app.flags.DEFINE_string("scores_output_file", None,
                           "Output file of scores for the 'gen_scores' mode. "
                           "row: text relations, col: kb relations, value: scores")
# learning configuration
tf.app.flags.DEFINE_string("optimization_algorithm", "vanilla",
                           "optimization algorithm: "
                           "{vanilla, adam, adagrad, adadelta, RMSProp}")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate, only used in "
                          "vanilla, adagrad, and RMSProp.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 1.0,
                          "Learning rate decays by this much, 1.0 = disabled.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_per_summary", 100,
                            "How many training steps to do summary.")
tf.app.flags.DEFINE_integer("maximum_steps", 5000000,
                            "Maximum number of training steps to do.")
tf.app.flags.DEFINE_boolean("summarize_trainable_variables", False,
                            "True if summarize trainable variables to view "
                            "in tensorboard (costly).")
tf.app.flags.DEFINE_boolean("do_validation", True,
                            "Whether to evaluate on the validation set and "
                            "save model and do early stopping based on that.")
tf.app.flags.DEFINE_integer("early_stop_tolerance", 100,
                            "Halt training if no improvement\'s been seen in "
                            "the last n evaluations on the validation set.")
# model configuration
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", -1, "Size of word embeddings.")
tf.app.flags.DEFINE_boolean("use_attention", False,
                            "Whether to use the attention mechanism.")
tf.app.flags.DEFINE_boolean("use_lstm", False,
                            "Whether to use the LSTM units. "
                            "The default is GRU units")
tf.app.flags.DEFINE_boolean("use_word2vec", False,
                            "Use word2vec embeddings for initialization?")
tf.app.flags.DEFINE_boolean("train_embedding", True,
                            "If use word2vec embeddings, whether train them.")
tf.app.flags.DEFINE_float("input_keep_prob", 1.0,
                          "Dropout: probability to keep input.")
tf.app.flags.DEFINE_float("output_keep_prob", 1.0,
                          "Dropout: probability to keep output.")
tf.app.flags.DEFINE_boolean("fill_missing_scores", True,
                            "When generating test scores"
                            "whether to fill in missing scores.")
tf.app.flags.DEFINE_string("loss_choice", "GloRE",
                           "Loss function choice: "
                           "{GloRE, LoRE}")
# use mode
tf.app.flags.DEFINE_string("mode", "train", "{train, decode_interactive, gen_scores}.")

FLAGS = tf.app.flags.FLAGS


def read_data(data_path):
    """Read data from source and target files and put into buckets.

    Args:
        data_path : path to the data file.

    Returns:
        data_set: a list of (left, right, weight) tuples
    """
    # preparation
    logger = logging.getLogger(__name__)
    logger.info('Reading data from %s' % data_path)
    data_set = []
    encoder_size = -1
    decoder_size = -1
    if data_path.endswith(".gz"):
        data_file = gzip.open(data_path, "rb")
    else:
        data_file = open(data_path, "rb")
    line_id = 0
    for line_id, line in enumerate(data_file):
        if line_id % 100000 == 0:
            logger.debug('Read line %d' % line_id)
        left, right, w = line.split('\t')
        left_ids = [int(x) for x in left.split()]
        right_ids = [int(x) for x in right.split()] + \
                    [data_utils.EOS_ID]
        w = float(w)
        data_set.append((left_ids, right_ids, w))
        if len(left_ids) > encoder_size:
            encoder_size = len(left_ids)
        if len(right_ids) > decoder_size:
            decoder_size = len(right_ids)
    logger.debug('Read line %d' % line_id)
    data_file.close()
    return data_set, encoder_size, decoder_size


def create_model(session,
                 encoder_initial_embeddings=None,
                 decoder_initial_embeddings=None,
                 mode = 'train'):
    """Create model and initialize / load parameters"""
    model = rel2vec_model.Rel2VecModel(
        FLAGS.encoder_size, FLAGS.decoder_size,
        FLAGS.encoder_vocab_size, FLAGS.decoder_vocab_size,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm,
        FLAGS.batch_size, FLAGS.optimization_algorithm,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        use_lstm=FLAGS.use_lstm,
        use_attention=FLAGS.use_attention,
        input_keep_prob=FLAGS.input_keep_prob,
        output_keep_prob=FLAGS.output_keep_prob,
        embedding_size=FLAGS.embedding_size,
        train_embedding=FLAGS.train_embedding,
        encoder_initial_embeddings=encoder_initial_embeddings,
        decoder_initial_embeddings=decoder_initial_embeddings,
        summarize_trainable_variables=FLAGS.summarize_trainable_variables,
        loss_choice = FLAGS.loss_choice)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    logger = logging.getLogger(__name__)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" %
                    ckpt.model_checkpoint_path)
        print("Reading model parameters from %s" %
              ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    if mode is 'train':    
        log_path = os.path.join(FLAGS.model_dir, 'log')
        model.summary_writer = tf.train.SummaryWriter(log_path, session.graph)
    return model


def train():
    """train a model"""
    # setup
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if FLAGS.fresh_start:       # fresh start, delete files from previous runs
        reply = raw_input('Create a new directory: ' + FLAGS.model_dir + '? [y/[n]] ')
        if reply == 'y':
            for f in os.listdir(FLAGS.model_dir):
                path = os.path.join(FLAGS.model_dir, f)
                try:
                    if os.path.isfile(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e:
                    raise e
        else:
            print('Aborting...'
                  'Set fresh_start=False to continue interrupted training.')
            sys.exit()
    log_file = (FLAGS.train_log if (FLAGS.train_log is not None and
                                    FLAGS.train_log != 'None')
                else 'train.log')
    utils.setup_logging(log_dir=FLAGS.model_dir,
                        log_file=log_file)
    logger = logging.getLogger(__name__)

    """model training, including data processing."""
    logger.info('Preparing data in %s' % FLAGS.data_dir)
    (ids_paths, FLAGS.encoder_vocab_size, FLAGS.decoder_vocab_size) = \
        data_utils.prepare_data(FLAGS.data_dir,
                                left_vocab_size=FLAGS.encoder_vocab_size,
                                right_vocab_size=FLAGS.decoder_vocab_size,
                                normalize_digits=False)
    train_ids_path, valid_ids_path = ids_paths
    train_set, FLAGS.encoder_size, FLAGS.decoder_size = \
        read_data(train_ids_path)
    FLAGS.decoder_size += 1     # for the _GO symbol
    valid_set, _, _ = read_data(valid_ids_path)

    logger.info("Data processed. %d train examples, %d validation examples. "
                "encoder vocab size = %d, "
                "decoder vocab size = %d, "
                "encoder size = %d, decoder size = %d." %
                (len(train_set), len(valid_set),
                 FLAGS.encoder_vocab_size, FLAGS.decoder_vocab_size,
                 FLAGS.encoder_size, FLAGS.decoder_size))

    # record configuration
    config_log_file = os.path.join(FLAGS.model_dir, 'config')
    with open(config_log_file, 'w') as f:
        d = FLAGS.__dict__['__flags']
        for flag in sorted(d):
            f.write("%s=%s\n" % (flag, str(d[flag])))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        
        if FLAGS.embedding_size < 0:
            FLAGS.embedding_size = FLAGS.size
            
        FLAGS.encoder_vocab_file = "left.%d.vocab" % FLAGS.encoder_vocab_size
        FLAGS.encoder_vocab_file = os.path.join(FLAGS.data_dir, FLAGS.encoder_vocab_file)
        if FLAGS.encoder_embedding_file is not None:
            FLAGS.encoder_embedding_file = os.path.join(FLAGS.data_dir, FLAGS.encoder_embedding_file)
        
        encoder_initial_embeddings = data_utils.create_init_embeddings(
            [FLAGS.encoder_vocab_size, FLAGS.embedding_size],
            FLAGS.word2vec_normalization,
            FLAGS.use_word2vec,
            FLAGS.encoder_vocab_file,
            FLAGS.encoder_embedding_file)
        decoder_initial_embeddings = data_utils.create_init_embeddings(
            [FLAGS.decoder_vocab_size, FLAGS.embedding_size])

        # Create model.
        logger.info("Creating %d layers of %d units." %
                    (FLAGS.num_layers, FLAGS.size))
        model = create_model(
            sess,
            encoder_initial_embeddings=encoder_initial_embeddings,
            decoder_initial_embeddings=decoder_initial_embeddings)

        # This is the training loop.
        step_time, loss, gradient_norm = 0.0, 0.0, 0.0
        current_step = 0

        best_eval_ppx = float('inf')
        no_improv_on_valid = 0
        previous_losses = []
        for current_step in xrange(model.global_step.eval() + 1,
                                  FLAGS.maximum_steps + 1):
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, en_seq_length, label_prob = \
                model.get_batch(train_set)

            step_gradient_norm, step_loss, _ = model.step(sess,
                                                          encoder_inputs,
                                                          en_seq_length,
                                                          decoder_inputs,
                                                          label_prob,
                                                          target_weights,
                                                          mode='train')
            step_time += ((time.time() - start_time) /
                          FLAGS.steps_per_checkpoint)
            loss += step_loss / FLAGS.steps_per_checkpoint
            gradient_norm += step_gradient_norm / FLAGS.steps_per_checkpoint

            # output summaries once awhile
            if current_step % FLAGS.steps_per_summary == 0:
                model.run_summary_op(sess,
                                     encoder_inputs,
                                     en_seq_length,
                                     decoder_inputs,
                                     label_prob,
                                     target_weights,
                                     current_step)
                
            
            # Once in a while, we save checkpoint,
            # print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                train_ppx = math.exp(loss) if loss < 300 else float('inf')
                logger.info("global step %d learning rate %.4f "
                            "step-time %.4f perplexity %.4f "
                            "gradient norm %.4f" %
                            (model.global_step.eval(),
                             model.learning_rate.eval(),
                             step_time, train_ppx, gradient_norm))

                # for vanilla SGD, decrease learning rate if no improvement
                # was seen over last 3 times.
                if FLAGS.optimization_algorithm == 'vanilla':
                    if (len(previous_losses) > 2 and
                       loss > max(previous_losses[-3:])):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                # Run evals on validation set and print their perplexity.
                if FLAGS.do_validation:
                    model.batch_size = 10000
                    eavl_num_samples = len(valid_set)
                    eval_num_steps = int(eavl_num_samples / model.batch_size)
                    eavl_num_samples = eval_num_steps * model.batch_size
                    eval_loss = 0.0
                    for _ in xrange(eval_num_steps):
                    
                        encoder_inputs, decoder_inputs, target_weights, en_seq_length, label_prob = \
                            model.get_batch(valid_set)
                        _, eval_step_loss, _ = model.step(sess,
                                                     encoder_inputs,
                                                     en_seq_length,
                                                     decoder_inputs,
                                                     label_prob,
                                                     target_weights,
                                                     mode='validate')
                        
                        eval_loss += eval_step_loss / eval_num_steps
                        
                    # set the batch size back for training
                    model.batch_size = FLAGS.batch_size

                    eval_ppx = (math.exp(float(eval_loss))
                                if eval_loss < 300 else float("inf"))
                    logger.info("==validation: perplexity %.4f "
                                "# samples %d " %
                                (eval_ppx, eavl_num_samples))
                else:
                    eval_ppx = train_ppx

                if eval_ppx < best_eval_ppx:
                    # Only save the so-far best model (based on validation
                    # perplexity)
                    checkpoint_path = os.path.join(FLAGS.model_dir,
                                                   "rel2vec_best.ckpt")
                    model.saver.save(sess, checkpoint_path,
                                     global_step=model.global_step)
                    best_eval_ppx = eval_ppx
                    no_improv_on_valid = 0
                    logger.info("model saved on step: %d " % model.global_step.eval())
                else:
                    no_improv_on_valid += 1
                    if no_improv_on_valid > FLAGS.early_stop_tolerance:
                        logger.info('Halt training because no improvement '
                                    'has been seen in the last '
                                    '%d evaluations.' %
                                    FLAGS.early_stop_tolerance)
                        sys.exit(0)

                for handler in logger.handlers:
                    handler.flush()
                sys.stdout.flush()
                step_time, loss, gradient_norm = 0.0, 0.0, 0.0
                
        logger.info("================================="
                    "best validation perplexity: %.4f " %
                    best_eval_ppx)

def decode_interactive():
    log_file = (FLAGS.test_log if (FLAGS.test_log is not None and
                                   FLAGS.test_log != 'None')
                else 'test.log')
    utils.setup_logging(log_dir=FLAGS.model_dir,
                        log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info('model dir: %s' % FLAGS.model_dir)

    # load vocabulary and test data
    FLAGS.encoder_vocab_file = "left.%d.vocab" % FLAGS.encoder_vocab_size
    FLAGS.decoder_vocab_file = "right.%d.vocab" % FLAGS.decoder_vocab_size
    encoder_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.encoder_vocab_file)
    decoder_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.decoder_vocab_file)
    encoder_vocab, _ = data_utils.initialize_vocabulary(encoder_vocab_path)
    decoder_vocab, rev_decoder_vocab = \
        data_utils.initialize_vocabulary(decoder_vocab_path)
    FLAGS.encoder_vocab_size = len(encoder_vocab)
    FLAGS.decoder_vocab_size = len(rev_decoder_vocab)
    logger.info('loaded vocabulary from: %s-%s. vocab size = %d-%d' %
                (encoder_vocab_path, decoder_vocab_path,
                 len(encoder_vocab), len(decoder_vocab)))

    # read candidates
    offset = len(data_utils._START_VOCAB)
    candidates = xrange(offset, len(decoder_vocab))
    candidates = [[candidate] + [data_utils.EOS_ID]
                  for candidate in candidates]
    n_cand = len(candidates)
    logger.info('# candidates: %d' % n_cand)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model and load parameters.
        model = create_model(sess, mode = 'test')
        model.batch_size = n_cand  # We rank all candidates at once.

        sys.stdout.write("> ")
        sys.stdout.flush()
        encoder_input = sys.stdin.readline()
        while encoder_input:
            # encoder_input = '<-nsubjpass>##born##<nmod:in>'
            encoder_input = encoder_input.strip()
            logger.info('encoder input: %s' % encoder_input)
            if data_utils.RELATION_DELIMITER in encoder_input:
                encoder_input = " ".join(encoder_input.split(
                                         data_utils.RELATION_DELIMITER))
            # Get token-ids for the input
            encoder_ids = data_utils.sentence_to_token_ids(encoder_input,
                                                           encoder_vocab)
            logger.info('encoder input ids: %s' % encoder_ids)

            data = [(encoder_ids, candidate, 0.0) for candidate in candidates]
            encoder_inputs, decoder_inputs, target_weights, en_seq_length, label_prob = \
                model.get_batch_all(data)
            _, _, logits = model.step(sess,
                                      encoder_inputs,
                                      en_seq_length,
                                      decoder_inputs,
                                      label_prob,
                                      target_weights,
                                      mode='rank_test')
            scores = [0] * n_cand
            decoder_id = 0

            step_logits = logits[decoder_id]
            step_probs = [utils.softmax(row) for row in step_logits]
            
            for cand_id, candidate in enumerate(candidates):
                if len(candidate) > decoder_id:
                    word_id = candidate[decoder_id]
                    scores[cand_id] = step_probs[cand_id][word_id]
            sorted_scores = [(i[0], i[1])
                         for i in sorted(enumerate(scores),
                                         key=lambda x:x[1],
                                         reverse=True)]
            for cand_id, score in sorted_scores[:20]:
                cand_id += offset
                logger.info('%s %.6f' % (rev_decoder_vocab[cand_id], score))

            print("> ", end="")
            sys.stdout.flush()
            encoder_input = sys.stdin.readline()

def gen_scores():
    print('model dir %s' % FLAGS.model_dir)

    # load vocabulary and test data
    FLAGS.encoder_vocab_file = "left.%d.vocab" % FLAGS.encoder_vocab_size
    FLAGS.decoder_vocab_file = "right.%d.vocab" % FLAGS.decoder_vocab_size
    encoder_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.encoder_vocab_file)
    decoder_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.decoder_vocab_file)
    encoder_vocab, _ = data_utils.initialize_vocabulary(encoder_vocab_path)
    decoder_vocab, rev_decoder_vocab = \
        data_utils.initialize_vocabulary(decoder_vocab_path)
    FLAGS.encoder_vocab_size = len(encoder_vocab)
    FLAGS.decoder_vocab_size = len(rev_decoder_vocab)
    print('loaded vocabulary from: %s-%s. vocab size = %d-%d' %
          (encoder_vocab_path, decoder_vocab_path,
           len(encoder_vocab), len(decoder_vocab)))

    
    # get the relations we want to use  
    kb_relation_model_ids = []
    missed_relation_data_ids = []
    
    with open(FLAGS.kb_relation_file) as f:
        for line in f:
            entries = line.strip().split(' ')
            rel = entries[0].strip()
            relid = int(entries[1])
            
            if not relid == len(kb_relation_model_ids):
                raise ValueError('relation ids are not ordered properly')
            if rel not in decoder_vocab:
                missed_relation_data_ids.append(relid)
                kb_relation_model_ids.append(-1)
                print('%d, %s' % (relid, rel))
                continue
            kb_relation_model_ids.append(decoder_vocab[rel])
    relation_total = len(kb_relation_model_ids)
    print('# missing kb relations: %d' % len(missed_relation_data_ids))    
    
    # inputs
    inputs_text = []
    inputs_ids = []
    encoder_size = 0          
    with gzip.open(FLAGS.text_relation_file, 'rb') as f:
        for line in f:
            if len(line.strip()) == 0:
                inputs_text.append('')
                inputs_ids.append([])
                continue
            sentence = line.strip('\n')
            inputs_text.append(sentence)
            if data_utils.RELATION_DELIMITER in sentence:
                sentence = ' '.join(
                    sentence.split(data_utils.RELATION_DELIMITER))
            token_ids = data_utils.sentence_to_token_ids(sentence,
                                                         encoder_vocab)
            inputs_ids.append(token_ids)
            if len(token_ids) > encoder_size:
                encoder_size = len(token_ids)
    
    FLAGS.encoder_size = encoder_size
    print('%d input text relations' % len(inputs_text))
    print('encoder size in inputs: %d' % encoder_size)
    print('encoder size: %d' % FLAGS.encoder_size)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    res = [None] * len(inputs_ids)
    res_map = {}
    with tf.Session(config=config) as sess:
        # Create model and load parameters.
        model = create_model(sess, mode = 'test')
        
        input_id = 0
        while input_id < len(inputs_ids):
            
            # get a batch
            data = []
            data_ids = []
            while input_id < len(inputs_ids) and len(data) < model.batch_size:
                encoder_input = inputs_ids[input_id]
                text_rel = inputs_text[input_id]
                if len(encoder_input) == 0:
                    if FLAGS.fill_missing_scores:
                        res_string = ' '.join(['0'] * relation_total) + '\n'
                        res[input_id] = res_string
                    else:
                        res[input_id] = '\n'
                elif text_rel in res_map:
                    res[input_id] = res_map[text_rel]
                else:
                    data.append((encoder_input, [data_utils.PAD_ID, data_utils.EOS_ID], 0.0))
                    data_ids.append(input_id)
                input_id += 1
                
            if len(data) == 0:
                break
                
            encoder_inputs, decoder_inputs, target_weights, en_seq_length, label_prob = \
                model.get_batch_all(data)
            _, _, logits = model.step(sess,
                                      encoder_inputs,
                                      en_seq_length,
                                      decoder_inputs,
                                      label_prob,
                                      target_weights,
                                      mode='rank_test')
           
            step_logits = logits[0]
            step_probs = [utils.softmax(row) for row in step_logits]
            
            for batch_idx in xrange(len(data)):
                
                scores = [] 
                for word_model_id in kb_relation_model_ids:
                    if word_model_id < 0:
                        scores.append(0)
                        continue
                    scores.append(step_probs[batch_idx][word_model_id])
                
                if len(scores) is not relation_total:
                    raise ValueError('wandanla')
                
                batch_input_id = data_ids[batch_idx]
                res_string = ' '.join(map(str, scores)) + '\n'
                res[batch_input_id] = res_string
                res_map[inputs_text[batch_input_id]] = res_string
                
                if (len(res_map)) % 1000 == 0:
                    best_match_id = kb_relation_model_ids[np.argmax(scores)]
                    print('%d : %s : %s : %.4f' %
                        (batch_input_id, inputs_text[batch_input_id], rev_decoder_vocab[best_match_id], max(scores)))
              
    with open(FLAGS.scores_output_file, 'w') as f_out:
        f_out.write(str(relation_total) + '\n')
        if FLAGS.fill_missing_scores:
            f_out.write('\n')
        else:
            f_out.write(' '.join(map(str, missed_relation_data_ids)) + '\n')
        for s in res:
            f_out.write(s)


def main(_):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "decode_interactive":
        decode_interactive()
    elif FLAGS.mode == "gen_scores":
        gen_scores()
    else:
        raise ValueError("Undefined mode")


if __name__ == "__main__":
    try:
        tf.app.run()
    except KeyboardInterrupt:
        sys.exit(0)
