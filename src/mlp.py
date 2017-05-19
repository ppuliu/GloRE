import tensorflow as tf
import numpy as np

class mlp:
    def __init__(self, 
                 input_dim=2,
                 sizes = [2],
                 optimizer_type="vanilla",
                 learning_rate = 0.01,
                 penalty_weight=0.01,
                 use_cap = False,
                 use_gate = False):

        self.input_dim = input_dim
        self.optimizer_type = optimizer_type
        self.penalty_weight = penalty_weight
        self.learning_rate = learning_rate
        
        self.pos_x = tf.placeholder(tf.float32, shape = [None, input_dim], name = 'pos_x')
        self.neg_x = tf.placeholder(tf.float32, shape = [None, input_dim], name = 'neg_x')
        
        weights = []
        biases = []
        
        sizes = [self.input_dim] +  sizes + [1]
        for i in xrange(len(sizes) - 1):
            w_init = np.random.uniform(-np.sqrt(1./sizes[i]), np.sqrt(1./sizes[i]), (sizes[i], sizes[i + 1]))
            w=tf.Variable(w_init, name = 'weights{}'.format(i), dtype = tf.float32)
            weights.append(w)

            b=tf.Variable(np.zeros([1, sizes[i + 1]]) , name = 'bias{}'.format(i), dtype = tf.float32)
            biases.append(b)
            
        def get_outputs(_inputs, _weights, _biases):
            h = _inputs
            for i in xrange(len(_weights)):
                w = _weights[i]
                b = _biases[i]
                if i == len(_weights) - 1:
                    h = tf.matmul(h, w) + b
                elif i == 0:
                    h = tf.nn.tanh(tf.matmul(_inputs, w) + b)
                # hidden layers
                else:
                    print 'hidden layer'
                    h = tf.nn.tanh(tf.matmul(h, w) + b)
            return h

        pos_x = self.pos_x
        neg_x = self.neg_x
        # use cap
        if use_cap:
            cap = tf.Variable(np.ones([1, self.input_dim]), dtype = tf.float32, name = 'cap')
            pos_x = tf.minimum(self.pos_x, cap)
            neg_x = tf.minimum(self.neg_x, cap)
        
        ## add gates
        if use_gate:
            gate_init = np.random.uniform(-np.sqrt(1./self.input_dim), np.sqrt(1./self.input_dim), (self.input_dim, self.input_dim))
            gate_w=tf.Variable(gate_init, name = 'gate_weights', dtype = tf.float32)
            gate_b=tf.Variable(np.zeros([1, self.input_dim]) , name = 'gate_bias', dtype = tf.float32)

            pos_gate = tf.nn.sigmoid(tf.matmul(pos_x, gate_w) + gate_b)
            neg_gate = tf.nn.sigmoid(tf.matmul(neg_x, gate_w) + gate_b)

            pos_x = tf.mul(pos_x, pos_gate)
            neg_x = tf.mul(neg_x, neg_gate)
            
            tf.histogram_summary('pos_p gates', pos_gate)
            tf.histogram_summary('neg_p gates', neg_gate)
        ##
        
        pos_p = get_outputs(pos_x, weights, biases)
        neg_p = get_outputs(neg_x, weights, biases)
        
        self.scores = pos_p
        
        loss_total = tf.maximum(0.0, 1.0 - (pos_p - neg_p))
        self.loss = tf.reduce_sum(loss_total)
        
        l2_loss = 0
        for w in weights:
            l2_loss += tf.nn.l2_loss(w)
        
        self.cost = self.loss + self.penalty_weight * l2_loss
        
        self.predicts = (pos_p - neg_p) > 0
        
        # Gradients and SGD update operation for training the model.
        if self.optimizer_type == 'vanilla':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer_type == 'adagrad':
            opt = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optimizer_type == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimizer_type == 'adadelta':
            # use default learning rate
            opt = tf.train.AdadeltaOptimizer()
        elif self.optimizer_type == 'adam':
            # use default learning rate
            opt = tf.train.AdamOptimizer()

            
        params = tf.trainable_variables()
        gradients = tf.gradients(self.cost, params)
        
        self.global_step = tf.Variable(0, trainable=False)
        self.update = opt.apply_gradients(zip(gradients, params),
                                          global_step=self.global_step)
        
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        
        tf.histogram_summary('pos_p', pos_p)
        tf.histogram_summary('neg_p', neg_p)
        tf.histogram_summary('pos_p - neg_p', pos_p - neg_p)
        self.merged_summary = tf.merge_all_summaries()
        self.summary_writer = None
        
    def run_summary_op(self, sess, input_feed, current_step):

        if self.merged_summary is None or self.summary_writer is None:
            return

        summary = sess.run(self.merged_summary, input_feed)
        self.summary_writer.add_summary(summary, current_step)
        self.summary_writer.flush()
