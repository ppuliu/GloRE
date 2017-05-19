from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging
import numpy as np
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import rel2vec_data_utils as data_utils
import seq2seq
tf.nn.seq2seq = seq2seq


class Rel2VecModel(object):
    def __init__(self,
                 encoder_size,
                 decoder_size,
                 encoder_num_symbols,
                 decoder_num_symbols,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 optimization_algorithm,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 use_attention=False,
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
                 embedding_size=-1,
                 encoder_initial_embeddings=None,
                 decoder_initial_embeddings=None,
                 train_embedding=True,
                 summarize_trainable_variables=False,
                 loss_choice = None):

        """Create the model.

        Args:
            encoder_size: number of steps for the encoder.
            decoder_size: number of steps for the decoder.
            encoder_num_symbols: vocab size for the encoder.
            decoder_num_symbols: vocab size for the decoder.
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            max_gradient_norm: gradients will be clipped to maximally this
                norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can
                be changed after initialization if this is convenient, e.g.,
                for decoding.
            learning_rate: learning rate to start with.
            learning_rate_decay_factor: decay learning rate by this much when
                needed.
            use_lstm: if true, we use LSTM cells instead of GRU cells.
            use_attention: if true, attention will be used.
            input_keep_prob: keep probability for the dropout op of encoder.
            embedding_size: dimension of the word embedding, if set to -1, it 
                will be set equal to size.
            encoder_initial_embeddings: initial word embedding for encoder.
            decoder_initial_embeddings: initial word embedding for decoder.
            train_embedding: wehther to train word embeddings or not.
            summarize_trainable_variables: whether to keep summary or not.
            loss_choice: choices of different losses: {GloRE, LoRE}
        """
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.encoder_num_symbols = encoder_num_symbols
        self.decoder_num_symbols = decoder_num_symbols
        self.batch_size = batch_size
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.logger = logging.getLogger(__name__)

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        if input_keep_prob < 1.0 or output_keep_prob < 1.0:
            single_cell = tf.nn.rnn_cell.DropoutWrapper(
                single_cell, input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, en_seq_length, decoder_inputs, do_decode=False):
            if use_attention:
                self.logger.info("Creating attention_seq2seq")
                return tf.nn.seq2seq.attention_seq2seq(encoder_inputs,
                      en_seq_length,
                      decoder_inputs,
                      cell,
                      self.decoder_num_symbols)
            else:
                self.logger.info("Creating basic_rnn_seq2seq")
                return tf.nn.seq2seq.basic_rnn_seq2seq(
                    encoder_inputs,
                    en_seq_length,
                    decoder_inputs,
                    cell,
                    self.decoder_num_symbols)

        # word embedding
        with tf.device('/gpu:0'):
            sqrt3 = math.sqrt(3)
            if encoder_initial_embeddings is not None:
                self.logger.info("Initialize with given encoder embeddings")
                encoder_init = tf.constant_initializer(
                    encoder_initial_embeddings)
            else:
                # Initializer for embeddings should have variance=1.
                encoder_init = tf.random_uniform_initializer(-sqrt3, sqrt3)
            if decoder_initial_embeddings is not None:
                self.logger.info("Initialize with given decoder embeddings")
                decoder_init = tf.constant_initializer(
                    decoder_initial_embeddings)
            else:
                decoder_init = tf.random_uniform_initializer(-sqrt3, sqrt3)
            embedding_size = embedding_size if embedding_size != -1 else size
            self.encoder_embedding = tf.get_variable("encoder_embedding",
                                                     [self.encoder_num_symbols,
                                                      embedding_size],
                                                     initializer=encoder_init,
                                                     trainable=train_embedding)
            self.decoder_embedding = tf.get_variable("decoder_embedding",
                                                     [self.decoder_num_symbols,
                                                      embedding_size],
                                                     initializer=decoder_init,
                                                     trainable=train_embedding)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.embedded_encoder_inputs = []
        self.en_seq_length = None
        self.decoder_inputs = []
        self.label_prob = None
        self.embedded_decoder_inputs = []
        self.target_weights = []
        for i in xrange(self.encoder_size):
            self.encoder_inputs.append(
                tf.placeholder(tf.int32,
                               shape=[None],
                               name="encoder{0}".format(i)))
            with tf.device('/gpu:0'):
                self.embedded_encoder_inputs.append(
                    tf.nn.embedding_lookup(self.encoder_embedding,
                                           self.encoder_inputs[-1]))
        
        self.en_seq_length = tf.placeholder(tf.int32,
                               shape=[None],
                               name="en_seq_length")
        
        self.label_prob = tf.placeholder(tf.float32,
                               shape=[None],
                               name="label_probability")
        
        for i in xrange(self.decoder_size):
            self.decoder_inputs.append(
                tf.placeholder(tf.int32,
                               shape=[None],
                               name="decoder{0}".format(i)))
            with tf.device('/gpu:0'):
                self.embedded_decoder_inputs.append(
                    tf.nn.embedding_lookup(self.decoder_embedding,
                                           self.decoder_inputs[-1]))
            self.target_weights.append(
                tf.placeholder(tf.float32,
                               shape=[None],
                               name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = ([self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)])
        self.last_target = tf.placeholder(tf.int32,
                                          shape=[None],
                                          name='last_target')
        targets.append(self.last_target)

        # Training outputs and losses.
        self.outputs, _ = seq2seq_f(self.embedded_encoder_inputs,
                                    self.en_seq_length,
                                    self.embedded_decoder_inputs)
        
        predicts_log_probs = - tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.outputs[0], targets[0])
        labels_log_probs = tf.log(self.label_prob)        
        if loss_choice is not None and loss_choice == 'LoRE':
            self.logger.info("Using LoRE model")
            self.loss = tf.nn.seq2seq.sequence_loss(self.outputs,
                                                    targets,
                                                    self.target_weights)
        else:
            self.logger.info("Using GloRE model")
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(predicts_log_probs - labels_log_probs)))
            
        # Gradients and SGD update operation for training the model.
        if self.optimization_algorithm == 'vanilla':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimization_algorithm == 'adagrad':
            opt = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optimization_algorithm == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimization_algorithm == 'adadelta':
            # use default learning rate
            opt = tf.train.AdadeltaOptimizer()
        elif self.optimization_algorithm == 'adam':
            # use default learning rate
            opt = tf.train.AdamOptimizer()

        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = \
            tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norm = norm
        self.update = opt.apply_gradients(zip(clipped_gradients, params),
                                          global_step=self.global_step)

        # summarize trainable variables
        if summarize_trainable_variables:
            variables = tf.trainable_variables()
            for v in variables:
                tf.histogram_summary(v.name, v)
        
        tf.scalar_summary('perplexity', tf.exp(self.loss))
        tf.histogram_summary('labels log probabilities', labels_log_probs)
        tf.histogram_summary('predicts log probabilities', predicts_log_probs)
        
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        self.merged_summary = tf.merge_all_summaries()
        # we don't have the computation graph here
        self.summary_writer = None

    def run_summary_op(self,
             session,
             encoder_inputs,
             en_seq_length,
             decoder_inputs,
             label_prob,
             target_weights,
             current_step):
    
        if self.merged_summary is None or self.summary_writer is None:
            return
        
        encoder_size, decoder_size = (self.encoder_size, self.decoder_size)
        # Input feed: encoder inputs, decoder inputs, target_weights,
        # as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        
        input_feed[self.en_seq_length.name] = en_seq_length
        input_feed[self.label_prob.name] = label_prob
        
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need
        # one more.
        input_feed[self.last_target] = \
            np.zeros([self.batch_size], dtype=np.int32)
        
        summary = session.run(self.merged_summary,  input_feed)
        self.summary_writer.add_summary(summary, current_step)
        self.summary_writer.flush()
    
    def step(self,
             session,
             encoder_inputs,
             en_seq_length,
             decoder_inputs,
             label_prob,
             target_weights,
             mode='train'):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            encoder_inputs: list of numpy int vectors to feed as encoder
                inputs.
            en_seq_length: valid lengths of the encder inputs
            decoder_inputs: list of numpy int vectors to feed as decoder
                inputs.
            label_prob: numpy array of label probs 
            target_weights: list of numpy float vectors to feed as target
                weights.

        Returns:
            A triple consisting of
                gradient norm (or None if we did not dobackward),
                average perplexity,
                outputs.

        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified
                bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = (self.encoder_size, self.decoder_size)
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in"
                             "bucket, %d != %d." %
                             (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in"
                             "bucket, %d != %d." %
                             (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in"
                             "bucket, %d != %d." %
                             (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights,
        # as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        
        input_feed[self.en_seq_length.name] = en_seq_length
        input_feed[self.label_prob.name] = label_prob
        
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need
        # one more.
        input_feed[self.last_target] = \
            np.zeros([self.batch_size], dtype=np.int32)

        # Output feed
        if mode == 'train':
            # training
            output_feed = [self.update,         # SGD
                           self.gradient_norm,  # Gradient norm
                           self.loss]           # Loss for this batch
            outputs = session.run(output_feed, input_feed)
            # Gradient norm, loss, no outputs
            return outputs[1], outputs[2], None
        elif mode == 'validate':
            # testing as a ranker
            output_feed = [self.loss]           # Loss for this batch
            outputs = session.run(output_feed, input_feed)
            # No gradient norm, loss, no outputs
            return None, outputs[0], None
        elif mode == 'rank_test':
            output_feed = []
            for l in xrange(decoder_size):      # Output logits
                output_feed.append(self.outputs[l])
            outputs = session.run(output_feed, input_feed)
            # No gradient norm, no loss, outputs
            return None, None, outputs

    def get_batch(self, data):
        """Get a random batch of data for step.
        """
        encoder_size, decoder_size = (self.encoder_size, self.decoder_size)
        encoder_inputs, decoder_inputs = [], []
        n_sample = len(data)

        # lengths for input sequences of the encoder 
        en_seq_length = np.zeros(self.batch_size, dtype=np.int32)
        # weights of the labels
        label_prob = np.zeros(self.batch_size, dtype=np.float32)
        
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for i in xrange(self.batch_size):
            encoder_input, decoder_input, weight = random.choice(data)
            # Encoder inputs are padded and then reversed.
            encoder_pad = ([data_utils.PAD_ID] *
                           (encoder_size - len(encoder_input)))
            encoder_inputs.append(list(encoder_input) + encoder_pad)
            
            # get lengths for input sequences of the encoder 
            en_seq_length[i] = len(encoder_input)
            # get the weight of the kb relation label
            label_prob[i] = weight
            
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + list(decoder_input) +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                         for batch_idx in xrange(self.batch_size)],
                         dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs,
        # we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                         for batch_idx in xrange(self.batch_size)],
                         dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a
                # PAD symbol.
                # The corresponding target is decoder_input shifted by
                # 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if (length_idx == decoder_size - 1 or
                   target == data_utils.PAD_ID):
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, en_seq_length, label_prob
    
    def get_batch_all(self, data):
        """Aggregate all the data in a batch
        """
        self.batch_size = len(data)
        encoder_size, decoder_size = (self.encoder_size, self.decoder_size)
        encoder_inputs, decoder_inputs = [], []

        # lengths for input sequences of the encoder 
        en_seq_length = np.zeros(self.batch_size, dtype=np.int32)
        # weights of the labels
        label_prob = np.zeros(self.batch_size, dtype=np.float32)
        
        # Get a batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for i in xrange(self.batch_size):
            encoder_input, decoder_input, w = data[i]

            # Encoder inputs are padded and then reversed.
            encoder_pad = ([data_utils.PAD_ID] *
                           (encoder_size - len(encoder_input)))
            encoder_inputs.append(list(encoder_input) + encoder_pad)
            
            # get lengths for input sequences of the encoder 
            en_seq_length[i] = len(encoder_input)
            # get the weight of the kb relation label
            label_prob[i] = w
        
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + list(decoder_input) +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                         for batch_idx in xrange(self.batch_size)],
                         dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs,
        # we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                         for batch_idx in xrange(self.batch_size)],
                         dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a
                # PAD symbol.
                # The corresponding target is decoder_input shifted by
                # 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if (length_idx == decoder_size - 1 or
                   target == data_utils.PAD_ID):
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, en_seq_length, label_prob
