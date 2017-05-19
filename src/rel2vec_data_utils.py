# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gzip
import os
import re
import logging
import rel2vec_utils as utils
import numpy as np

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

RELATION_DELIMITER = '##'

_DIGIT_RE = re.compile(r"\d")


def create_vocabulary(left_vocab_path,
                      right_vocab_path,
                      data_path,
                      left_vocab_size,
                      right_vocab_size,
                      normalize_digits=True):
    """Create vocabulary files (if it does not exist yet) from data file.

    Data file is assumed to contain one alignment pair per line.
    Digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format,
    so that later token in the first line gets id=0,
    second line gets id=1, and so on.
    If vocab files already exist, only read the files and return vocab size

    Args:
        left_vocab_path: path where the left vocabulary will be created.
        right_vocab_path: path where the right vocabulary will be created.
        data_path: data file that will be used to create word vocabulary.
        left_vocab_size: limit on the size of the created left vocabulary.
        right_vocab_size: limit on the size of the created right vocabulary.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
        left_vocab_size: the final size of the left vocab
        right_vocab_size: the final size of the right vocab
    """
    logger = logging.getLogger(__name__)
    if not gfile.Exists(left_vocab_path) or not gfile.Exists(right_vocab_path):
        logger.info("Creating vocabulary %s, %s from data %s" %
                    (left_vocab_path, right_vocab_path, data_path))
        left_vocab = {}
        right_vocab = {}
        if (data_path.endswith(".gz")):
            f = gzip.open(data_path, "rb")
        else:
            f = open(data_path, "rb")
        counter = 0
        for line in f:
            line = line.strip()
            counter += 1
            if counter % 100000 == 0:
                logger.debug("processing line %d" % counter)
            parts = line.split('\t')
            left = parts[0]
            right = parts[1]
            w = float(parts[2])

            # left
            tokens = left.split(RELATION_DELIMITER)
            for token in tokens:
                token = (re.sub(_DIGIT_RE, "0", token)
                         if normalize_digits else token)
                if token in left_vocab:
                    left_vocab[token] += w
                else:
                    left_vocab[token] = w

            # right
            tokens = right.split(RELATION_DELIMITER)
            for token in tokens:
                token = (re.sub(_DIGIT_RE, "0", token)
                         if normalize_digits else token)
                if token in right_vocab:
                    right_vocab[token] += w
                else:
                    right_vocab[token] = w
        f.close()
        logger.info("left vocab size = %d, right vocab size = %d." %
                    (len(left_vocab), len(right_vocab)))

        vocab_list = _START_VOCAB + sorted(left_vocab, key=left_vocab.get,
                                           reverse=True)
        if left_vocab_size >= 0 and len(vocab_list) > left_vocab_size:
            vocab_list = vocab_list[:left_vocab_size]
        with gfile.GFile(left_vocab_path, mode="w") as vocab_file:
            for w in vocab_list:
                if w in _START_VOCAB:
                    vocab_file.write(w + "\t" + "0" + "\n")
                else:
                    vocab_file.write(w + "\t" + str(left_vocab[w]) + "\n")
        left_vocab_size = len(vocab_list)

        vocab_list = _START_VOCAB + sorted(right_vocab, key=right_vocab.get,
                                           reverse=True)
        if right_vocab_size >= 0 and len(vocab_list) > right_vocab_size:
            vocab_list = vocab_list[:right_vocab_size]
        with gfile.GFile(right_vocab_path, mode="w") as vocab_file:
            for w in vocab_list:
                if w in _START_VOCAB:
                    vocab_file.write(w + "\t" + "0" + "\n")
                else:
                    vocab_file.write(w + "\t" + str(right_vocab[w]) + "\n")
        right_vocab_size = len(vocab_list)
    else:
        logger.info('Vocabulary already exists. Skip creating.')
        left_vocab_size = 0
        with gfile.GFile(left_vocab_path, mode="r") as vocab_file:
            for line in vocab_file:
                left_vocab_size += 1
        right_vocab_size = 0
        with gfile.GFile(right_vocab_path, mode="r") as vocab_file:
            for line in vocab_file:
                right_vocab_size += 1
    logger.info('Finished vocabulary creation.')
    return left_vocab_size, right_vocab_size


def initialize_vocabulary(vocab_path):
    """Initialize vocabulary from file.

    Args:
        vocab_path: path to the file containing the vocabulary.

    Returns:
        a pair: the vocabulary (a dictionary mapping string to integers),
                and the reversed vocabulary (a list, which reverses the
                vocabulary mapping).

    Raises:
        ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocab_path):
        rev_vocab = []
        with gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.split("\t")[0] for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def sentence_to_token_ids(sentence,
                          left_vocab=None,
                          right_vocab=None,
                          normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    Args:
        sentence: a string, the string to convert to token-ids.
        left_vocab: a dictionary mapping left tokens to integers.
        right_vocab: a dictionary mapping right tokens to integers.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if left_vocab and right_vocab:
        parts = sentence.strip('\n').split('\t')
        if not normalize_digits:
            left_ids = [left_vocab.get(w, UNK_ID)
                        for w in parts[0].strip().split(RELATION_DELIMITER)]
            right_ids = [right_vocab.get(w, UNK_ID)
                         for w in parts[1].strip().split(RELATION_DELIMITER)]
        else:
            # Normalize digits by 0 before looking words up in the vocabulary.
            left_ids = [left_vocab.get(re.sub(_DIGIT_RE, '0', w), UNK_ID)
                        for w in parts[0].strip().split(RELATION_DELIMITER)]
            right_ids = [right_vocab.get(re.sub(_DIGIT_RE, '0', w), UNK_ID)
                         for w in parts[1].strip().split(RELATION_DELIMITER)]
        w = float(parts[2])
        return left_ids, right_ids, w
    elif left_vocab:
        # during testing when only left is given as input
        if not normalize_digits:
            return [left_vocab.get(w, UNK_ID)
                    for w in sentence.strip().split(' ')]
        else:
            return [left_vocab.get(re.sub(_DIGIT_RE, '0', w), UNK_ID)
                    for w in sentence.strip().split(' ')]
    elif right_vocab:
        # during testing when only right is given as input
        if not normalize_digits:
            return [right_vocab.get(w, UNK_ID)
                    for w in sentence.strip().split(' ')]
        else:
            return [right_vocab.get(re.sub(_DIGIT_RE, '0', w), UNK_ID)
                    for w in sentence.strip().split(' ')]


def data_to_token_ids(data_path,
                      left_vocab_path,
                      right_vocab_path,
                      ids_path,
                      normalize_digits=True):
    """Turn data file into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to ids_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
        data_path: path to the data file in one-sentence-per-line format.
        left_vocab_path and right_vocab_path: path to the vocabulary files.
        ids_path: path where the file with token-ids will be created.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    logger = logging.getLogger(__name__)
    if not gfile.Exists(ids_path):
        logger.info("transforming data in %s" % data_path)
        left_vocab, _ = initialize_vocabulary(left_vocab_path)
        right_vocab, _ = initialize_vocabulary(right_vocab_path)
        if data_path.endswith(".gz"):
            data_file = gzip.open(data_path, "rb")
        else:
            data_file = open(data_path, "rb")
        if ids_path.endswith(".gz"):
            f_write = gzip.open(ids_path, "wb")
        else:
            f_write = open(ids_path, "wb")

        counter = 0
        for line in data_file:
            counter += 1
            if counter % 100000 == 0:
                logger.debug("transforming line %d" % counter)
            left_ids, right_ids, w = sentence_to_token_ids(line,
                                                           left_vocab,
                                                           right_vocab,
                                                           normalize_digits)
            if len(left_ids) == 0 or len(right_ids) == 0:
                continue
            f_write.write(utils.format_list(left_ids) + "\t" +
                          utils.format_list(right_ids) + "\t" +
                          str(w) + "\n")
        f_write.close()
        data_file.close()


def prepare_data(data_dir,
                 left_vocab_size=-1,
                 right_vocab_size=-1,
                 normalize_digits=False):
    """read data from data_dir, create vocabularies and tokenize data.

    Args:
        data_dir: directory in which the data sets will be stored.
        left_vocab_size: limit on the size of the created left vocabulary.
        right_vocab_size: limit on the size of the created right vocabulary.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
        ids_path: path of the processed file holding token-ids
        left_vocab_size: the final size of the left vocabulary, -1 = unlimited
        right_vocab_size: the final size of the right vocabulary,
            -1 = unlimited
    """
    # Create vocabularies of the appropriate sizes.
    data_paths = [os.path.join(data_dir, "data.train.gz"),
                  os.path.join(data_dir, "data.valid.gz")]
    ids_paths = [os.path.join(data_dir, "ids.left%d.right%d.train.gz" %
                              (left_vocab_size, right_vocab_size)),
                 os.path.join(data_dir, "ids.left%d.right%d.valid.gz" %
                              (left_vocab_size, right_vocab_size))]
    left_vocab_path = os.path.join(data_dir, "left.%d.vocab" % left_vocab_size)
    right_vocab_path = os.path.join(data_dir,
                                    "right.%d.vocab" % right_vocab_size)
    left_vocab_size, right_vocab_size = create_vocabulary(left_vocab_path,
                                                          right_vocab_path,
                                                          data_paths[0],
                                                          left_vocab_size,
                                                          right_vocab_size,
                                                          normalize_digits)

    # Create token ids for the data.
    for i in range(2):
        data_to_token_ids(data_paths[i],
                          left_vocab_path,
                          right_vocab_path,
                          ids_paths[i],
                          normalize_digits)

    return ids_paths, left_vocab_size, right_vocab_size


def read_word2vec(word2vec_file):
    wordvec_map = {}
    num_words = 0
    dimension = 0
    with open(word2vec_file) as f:
        for line in f:
            entries = line.split('\t')
            word = entries[0].strip()
            vec = map(float, entries[1:])

            assert word not in wordvec_map
            assert dimension == 0 or dimension == len(vec)
            
            wordvec_map[word] = np.array(vec)
            num_words += 1
            dimension = len(vec)
    
    
    return wordvec_map, num_words, dimension
            
    

def create_init_embeddings(sizes,
                           normalize = None,
                           use_word2vec = False,
                           vocab_file = None,
                           word2vec_file = None ):
    logger = logging.getLogger(__name__)
    
    initial_embeddings = np.random.uniform(-np.sqrt(3), np.sqrt(3), sizes)
    if use_word2vec:
        logger.info("Loading word2vec initial vectors from:\n %s" % word2vec_file)
        if word2vec_file.endswith('.npy'):
            return np.load(word2vec_file)
        
        word2vec_map, num_words, dimension = read_word2vec(word2vec_file)
        logger.info("Number of word2vec vectors loaded: {0}, dimension: {1}".format(num_words, dimension))
       
        _, rev_vocab = initialize_vocabulary(vocab_file)
        
        num_covered = 0
        for id, word in enumerate(rev_vocab):
            if word in word2vec_map:
                vec = word2vec_map[word]
                if len(vec) is not sizes[1]:
                    raise ValueError("word2vec dimension doesn't match.")
                initial_embeddings[id, :] = vec
                num_covered += 1
                
        logger.info("Out of %d vocab symbols, %d are covered by word2vec" % (sizes[0], num_covered))
    if normalize is not None:
        if normalize == 'unit_norm':
            norm = np.linalg.norm(initial_embeddings, axis = 1).reshape(sizes[0], 1)
            initial_embeddings = initial_embeddings / norm
            logger.info("Intial embeddings are normalized to unit_norm")
        elif normalize ==   'unit_var':
            std = np.std(initial_embeddings, axis = 1).reshape(sizes[0], 1)
            initial_embeddings = initial_embeddings / std
            logger.info("Intial embeddings are normalized to unit_var")            
            
    return initial_embeddings
        
        
