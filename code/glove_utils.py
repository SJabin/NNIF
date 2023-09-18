"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
from the code of paper: Generating Natural Language Adversarial Examples
paper link: https://www.aclweb.org/anthology/D18-1316/
"""

import pickle

import numpy as np


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r', encoding="utf-8")
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        try:
            embedding = np.array([float(val) for val in row[1:]])
            model[word] = embedding
        except ValueError:
            print("{} Not a float".format(word))
    print("Done.", len(model), " words loaded!")
    return model


def save_glove_to_pickle(glove_model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(glove_model, f)


def load_glove_from_pickle(file_name):
    with open(file_name, 'rb', encoding="utf-8") as f:
        return pickle.load(f)


def create_embeddings_matrix(glove_model, dictionary, max_vocab_size, d=300):
    # Matrix size is 300
    embedding_matrix = np.zeros(shape=((d, max_vocab_size + 2)))
    cnt = 0
    unfound = []

    for i, w in enumerate(dictionary[:max_vocab_size]):
        if not w in glove_model:
            cnt += 1
            # if cnt < 10:
            #     embedding_matrix[:, i] = glove_model['UNK']
            unfound.append(i)
        else:
            embedding_matrix[:, i] = glove_model[w]
    print('Number of not found words = ', cnt)
    return embedding_matrix, unfound


def pick_most_similar_words(src_word, dist_mat, ret_count=10, threshold=None):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    dist_order = np.argsort(dist_mat[src_word, :])[0:1 + ret_count]
    dist_list = dist_mat[src_word][dist_order]
    if dist_list[-1] == 0:
        return [], []
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        return dist_order[mask], dist_list[mask]
    else:
        return dist_order, dist_list
