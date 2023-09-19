# get neighbours information for each word in test examples according to their embeddings in Counter-fitting Word
# Vectors before words-level (synonym replacement) attack
import os
import math
import argparse
from functools import partial

import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool
from collections import OrderedDict
from nltk.corpus import stopwords, words

import glove_utils
from utils import IMDBDataset, MnliDataset

from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--detect', type=str, default=None, required=True)
parser.add_argument('--dataset-name', type=str, default=None, required=True, choices=['IMDB', 'Mnli', 'SST2'],
                    help='To generate neighbours of words in the test set of which dataset for attack.')
parser.add_argument('--dataset-path', type=str, default=None, required=True, help='The directory of the dataset.')#choices=['./data/aclImdb', './data/multinli_1.0']
parser.add_argument('--max-length', type=int, default=None, required=True, choices=[512, 256, 128],
                    help='The maximum sequences length. Since the longest input of IMDB and Mnli are 2479 words and 396'
                         'words respectively, we suggest to set this value to 512, 256, or 128.')
args = parser.parse_args()


def get_dictionary(dataset, vocab_file, lm_tokenizer):
    """
    generate a dictionary and sort by number of occurrences of words in the dataset
    :param dataset: IMDB, or Mnli dataset
    :param vocab_file: path of dictionary file
    :param lm_tokenizer: a Transformer-XL tokenizer
    :return: a dictionary contains a list of words and their ids from Transformer-XL tokenizer
    """
    if os.path.exists(vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_words = f.read().split('\n')
    else:
        word_counts = OrderedDict()
        words_list = words.words()
        stopwords_list = stopwords.words('english')

        train_text, train_text_pair, test_text, test_text_pair, valid_text, valid_text_pair = dataset

        if train_text_pair[0]:
            tmp = list(zip(train_text,  train_text_pair)) + \
                  list(zip( valid_text,  valid_text_pair)) + \
                  list(zip( test_text,  test_text_pair))
            all_data = [sentences[0] + ' ' + sentences[1] for sentences in tmp]
        else:
            all_data =  train_text +  valid_text +  test_text

        for ind, example in enumerate(all_data):
            tokens = lm_tokenizer.tokenize(example)

            for w in tokens:
                if w in words_list and w not in stopwords_list:
                    if w in word_counts:
                        word_counts[w] += 1
                    else:
                        word_counts[w] = 1

        wcounts = list(word_counts.items())

        # sort according to the number of occurrences of the words
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # a list of words
        vocab_words = [wc[0] for wc in wcounts]

        # create directory if not exist
        if not os.path.exists(os.path.dirname(vocab_file)):
            os.makedirs(os.path.dirname(vocab_file))

        with open(vocab_file, 'w') as f:
            for word in vocab_words:
                f.write(word + '\n')

    # get the ids of words in the dictionary from the Transformer-XL tokenizer
    tmp = lm_tokenizer(vocab_words)['input_ids']
    token_ids = [id[0] if id else id for id in tmp]

    dictionary = pd.DataFrame(list(zip(vocab_words, token_ids)), columns=['word', 'id'])

    return dictionary


def compute_distance_matrix(dictionary, distance_matrix_file, missed_embedding_file, counter_fitting_file,
                            embedding_file, max_vocab_size):
    """
    take the first max_vocab_size words in the dictionary and
    calculate the distance between each word according to the Counter-fitting Word Vectors
    :param dictionary: a dictionary containing a list of words
    :param distance_matrix_file: the path of the distance matrix file which stores the distances between words
    :param missed_embedding_file: the path of the missed embedding file which stores the words, which are in the
                                dictionary but do not have embeddings from the Counter-fitting Word Vectors,
                                they are not in distance matrix file
    :param counter_fitting_file: the file of the Counter-fitting Word Vectors
    :param embedding_file: the path of a file saves the top max_vocab_size words embeddings
    :param max_vocab_size: the maximum number of words that used for the distance matrix
    :return: distance_matrix: a matrix of the distance between each word in the first max_vocab_size words in dictionary
    :return: missed_words: words that are the first max_vocab_size words in the dictionary
                            but not in the Counter-fitting Word Vectors and distance_matrix
    """
    if os.path.exists(distance_matrix_file) and os.path.exists(missed_embedding_file):
        distance_matrix = np.load(distance_matrix_file, allow_pickle=True)
        missed_words = np.load(missed_embedding_file, allow_pickle=True)
    else:
        if not os.path.exists(os.path.dirname(distance_matrix_file)):
            os.makedirs(os.path.dirname(distance_matrix_file))

        # load the Counter-fitting Word Vectors
        counter_fitting = glove_utils.loadGloveModel(counter_fitting_file)

        # get the embeddings of words in the dictionary from the Counter-fitting Word Vectors
        counter_embeddings, missed_words = glove_utils.create_embeddings_matrix(counter_fitting, dictionary,
                                                                                max_vocab_size)
        np.save(embedding_file, counter_embeddings)
        np.save(missed_embedding_file, missed_words)

        # compute the distance_matrix using (a-b)^2
        c_ = -2 * np.dot(counter_embeddings.T, counter_embeddings)
        a = np.sum(np.square(counter_embeddings), axis=0).reshape(1, -1)
        b = a.T
        distance_matrix = a + b + c_

        np.save(distance_matrix_file, distance_matrix)

    return distance_matrix, missed_words


def get_neighbours_info(examples, max_vocab_size, max_seq_length, vocab_file, distance_matrix_file,
                        missed_embedding_file, num_neighbours, use_lm=True):
    """
    gets neighbours information of all words in examples, including neighbours length and neighbours list
    :param examples: test examples or part of test examples
    :param max_vocab_size: maximum number of words in the distance matrix
    :param max_seq_length: maximum sequences of length of BERT model
    :param vocab_file: path of the dictionary file
    :param distance_matrix_file: path of the distance matrix file which stores distances between words
    :param missed_embedding_file: path of the missed embedding file
    :param num_neighbours: maximum number of neighbours
    :param use_lm: if true, remove 4 low-probability neighbours from the neighbours list according to the language model
    :return: neighbours information
    """
    texts, labels = [], []

    has_neighbours_list = []  # two-dimensional list, each element is indices of has-neighbours-words in an example
    neighbours_list = []  # three-dimensionarl list, each element is neighbours of words in an examples
    neighbours_len_list = []  # two-dimensional list, each element is the length of neighbours of words in an example
    tokens_list = []  # two-dimensional list, each element is tokens of an example
    sent1_token_len_list = []  # one-dimensional list, each element is the length of text in an example

    lm_tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    lm = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')

    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_words = f.read().split('\n')
    except FileNotFoundError:
        print('dictionary file does not exist, please run the get_dictionary function.')

    tmp = lm_tokenizer(vocab_words)['input_ids']
    token_ids = [id[0] if id else id for id in tmp]
    dictionary = pd.DataFrame(list(zip(vocab_words, token_ids)), columns=['word', 'id'])

    words_list = dictionary.word.values.tolist()
    words_ids = dictionary.id.values.tolist()

    distance_matrix = np.load(distance_matrix_file)
    missed_words = np.load(missed_embedding_file)

    for (text, text_pair, label) in examples:
        texts.append((text, text_pair))
        labels.append(label)

        text_tokens = lm_tokenizer.tokenize(text)
        text_pair_tokens = lm_tokenizer.tokenize(text_pair) if text_pair else None
        tokens = (text_tokens + text_pair_tokens)[:max_seq_length] if text_pair else text_tokens[:max_seq_length]
        sent1_token_len = len(text_tokens)
        sent1_token_len_list.append(sent1_token_len)

        has_neighbours = []  # a list of has-neighbours-words-indices in this example
        a_neighbours_list = []  # two-dimensional list, each element is a neighbours list of a word in has_neighbours
        neighbours_len = []  # a list of lengths of neighbours of words in has_neighbours
        neighbours_ids = []  # two-dimensional list, each element is a list of indices of neighbours for a word

        # get neighbours of tokens
        for i, token in enumerate(tokens):
            if token in words_list and words_list.index(token) < max_vocab_size and \
                    words_list.index(token) not in missed_words:
                neighbours, _ = glove_utils.pick_most_similar_words(words_list.index(token), distance_matrix,
                                                                    num_neighbours, 0.5)
                neighbours = neighbours.tolist()
                neighbours_id = [words_ids[j] for j in neighbours]

                has_neighbours.append(i)
                a_neighbours_list.append(neighbours)
                neighbours_ids.append(neighbours_id)
                neighbours_len.append(len(neighbours))

        # if using the language model, remove 4 neighbours that are less likely occur according to the language model
        if use_lm:
            inputs = lm_tokenizer(text, text_pair, return_tensors='pt')
            predictions = lm(**inputs)['prediction_scores'].squeeze()

            reserved_indices = []  # indices of words which still have neighbours
            for idx, i in enumerate(has_neighbours):
                if neighbours_len[idx] > 4:
                    reserved_indices.append(idx)
                    if predictions.ndim == 2:
                        lm_neighbours = torch.argsort(predictions[i], descending=True).tolist()
                    elif predictions.ndim == 1:
                        lm_neighbours = torch.argsort(predictions, descending=True).tolist()
                    neighbours_ids[idx] = sorted(neighbours_ids[idx], key=lambda x: lm_neighbours.index(x))
                    neighbours_ids[idx] = neighbours_ids[idx][:-4]
                    a_neighbours_list[idx] = [words_ids.index(j) for j in neighbours_ids[idx]]
                    neighbours_len[idx] -= 4

            has_neighbours = [has_neighbours[j] for j in reserved_indices]
            neighbours_len = [neighbours_len[j] for j in reserved_indices]
            a_neighbours_list = [a_neighbours_list[j] for j in reserved_indices]
            # neighbours_ids = [neighbours_ids[j] for j in reserved_indices]

            del predictions  # try to free cpu memory

        has_neighbours_list.append(has_neighbours)
        neighbours_len_list.append(neighbours_len)
        neighbours_list.append(a_neighbours_list)
        tokens_list.append(tokens)

    neighbours_info = pd.DataFrame(
        list(zip(texts, labels, tokens_list,
                 has_neighbours_list, neighbours_len_list, neighbours_list, sent1_token_len_list)),
        columns=['text', 'label', 'tokens',
                 'has_neighbours', 'neighbours_length', 'neighbours_list', 'length of text tokens'])

    return neighbours_info


if __name__ == '__main__':
    # load dataset
    data_processors = {
        'Mnli': MnliDataset,
        'IMDB': IMDBDataset,
    }
    dataset = data_processors[args.dataset_name](args.dataset_path, 0.2)
    
    
            
    train_texts = dataset.train_text
    train_text_pairs = dataset.train_text_pair
    train_y = dataset.train_y
    test_texts = dataset.test_text
    test_text_pairs = dataset.test_text_pair
    test_y = dataset.test_y
    valid_texts = dataset.valid_text
    valid_text_pairs = dataset.valid_text_pair
    valid_y = dataset.valid_y
    
    l = np.arange(len(train_texts))
    random.shuffle(l) # we shuffle the list
    index_value = random.sample(list(l), 6000)
    train_text = [train_texts[i] for i in index_value]
    train_text_pair = [train_text_pairs[i] for i in index_value]
    train_y = [train_y[i] for i in index_value]
    
    l = np.arange(len(test_texts))
    random.shuffle(l) # we shuffle the list
    index_value = random.sample(list(l), 5000)
    test_text = [test_texts[i] for i in index_value]
    test_text_pair = [test_text_pairs[i] for i in index_value]
    test_y = [test_y[i] for i in index_value]

    
    l = np.arange(len(valid_texts))
    random.shuffle(l) # we shuffle the list
    index_value = random.sample(list(l), 5000)
    valid_text = [valid_texts[i] for i in index_value]
    valid_text_pair = [valid_text_pairs[i] for i in index_value]
    valid_y = [valid_y[i] for i in index_value]
               
               

    #output_dir = os.path.join('./output', args.dataset_name, 'bert')
    output_dir = os.path.join('./results', args.dataset_name, args.detect)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    max_vocab_size = 20000
    max_seq_length = args.max_length
    num_neighbours = 9

    vocab_file = os.path.join(output_dir, 'vocab.vocab')
    counter_fitted_file = './counter-fitted-vectors.txt'
    
    distance_matrix_file = os.path.join(output_dir, 'dist_counter.npy')
    embedding_file = os.path.join(output_dir, 'embeddings_counter.npy')
    missed_embedding_file = os.path.join(output_dir, 'missed_embeddings_counter.npy')

    lm_tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

    print("create a dictionary and a matrix containing the distance between each word")
    data = train_texts, train_text_pairs, test_texts, test_text_pairs, valid_text, valid_text_pairs
    dictionary = get_dictionary(data, vocab_file, lm_tokenizer)
    distance_matrix, missed_words = compute_distance_matrix(dictionary.word.values, distance_matrix_file,
                                                            missed_embedding_file, counter_fitted_file,
                                                            embedding_file, max_vocab_size)

    # get neighbours information
    neighbours_file = os.path.join(output_dir, 'neighbours_info.csv')
    if not os.path.exists(neighbours_file):
        n_cpus = 12 if os.cpu_count() > 12 else os.cpu_count()
        print('number of cpus:', n_cpus)
        pool = Pool(n_cpus)
        step = math.ceil(len( test_y) / n_cpus)
        pool_inputs = [zip( test_text[i: i + step],  test_text_pair[i: i + step],
                            test_y[i: i + step]) for i in range(0, len( test_y), step)]

        neighbours_list_queue = pool.map(
            partial(get_neighbours_info, max_vocab_size=max_vocab_size, max_seq_length=max_seq_length,
                    vocab_file=vocab_file, distance_matrix_file=distance_matrix_file,
                    missed_embedding_file=missed_embedding_file, num_neighbours=num_neighbours),
            pool_inputs)

        neighbours_info = pd.concat(neighbours_list_queue, ignore_index=True)
        neighbours_info.to_csv(neighbours_file, index=False)

    # testing
    # examples = zip( test_text,  test_text_pair,  test_y)
    # neighbours_info = get_neighbours_info(examples, max_vocab_size=max_vocab_size, max_seq_length=max_seq_length,
    #                 vocab_file=vocab_file, distance_matrix_file=distance_matrix_file,
    #                 missed_embedding_file=missed_embedding_file, num_neighbours=num_neighbours)

    print('Done!')
