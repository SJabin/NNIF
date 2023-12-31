import csv
import logging
import sys
import numpy as np
import os
import random
import string
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
#from tensorflow.keras.preprocessing.text import Tokenizer#, text_to_word_sequence
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import pprint
import io
import torch
#import hnswlib
import re
import pickle
from tqdm import tqdm



#logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    #datefmt = '%m/%d/%Y %H:%M:%S',
                    #level = logging.INFO)#
        
#logger = logging.getLogger(__name__)

stop_words = ['the', 'a', 'an', 'to', 'of', 'and', 'with', 'as', 'at', 'by', 'is', 'was', 'are', 'were', 'be', 'he', 'she', 'they', 'their', 'this', 'that']

def helper_name(x):
    name = x.split('/')[-1]
    return int(name.split('_')[0])
    
def read_text(dataset, path, data_dir="./data/"):
    print("reading path: %s" % (data_dir + path))
    
    data_path = data_dir + path
    
    label_list = []
    clean_text_list = []
    
    if (
        path.startswith("ag_news")
        or path.startswith("dbpedia")
        or path.startswith("yahoo")
    ):
        with open(data_dir + "%s.csv" % path, "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            for row in csv_reader:
                count += 1
                label_list.append(int(row[0]) - 1)
                text = " . ".join(row[1:]).lower()
                clean_text_list.append(" ".join(text_to_tokens(text)))
    elif dataset =='imdb':
        pos_list = []
        neg_list = []
        
        pos_path = os.path.join(data_path, 'pos')
        neg_path = os.path.join(data_path,'neg')
        
        #print(pos_path)
        
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_files = sorted(pos_files, key=lambda x : helper_name(x))
        neg_files = sorted(neg_files, key=lambda x : helper_name(x))
        pos_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in pos_files]
        neg_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in neg_files]
        text_list = pos_list + neg_list
    
        # clean the texts
        clean_text_list = [' '.join(text_to_tokens(s)) for s in text_list]
        label_list = [1]*len(pos_list) + [0]*len(neg_list)
    else:
        raise NotImplementedError
    return clean_text_list, label_list

def text_to_tokens(text):
    """
    Clean the raw text.
    """
    toks = word_tokenize(text)
    spliter = ['\'', '#', '!', '\"', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', ':', ';', '<', '=', '>', '?','@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']
    toks = [token for token in filter(lambda x: x not in spliter, toks)]
    return toks

def load_pretrained_embedding(task_name, max_vocab_size, data_dir="./"):
    counter_embedding_matrix = np.load(
        data_dir
        + "aux_files/embeddings_counter_%s.npy" % (task_name)
    )
    print("Counter embedding matrix: ", counter_embedding_matrix.shape)
    return counter_embedding_matrix

def text_encoder(texts, orig_dic, maxlen):
    """
    Map the raw text to word id sequence.
    """
    seqs = []
    seqs_mask = []
    for text in texts:
        words = text.split(" ")
        mask = []
        for i in range(len(words)):
            words[i] = orig_dic[words[i]] if words[i] in orig_dic else 0
            mask.append(1)
        seqs.append(words)
        seqs_mask.append(mask)
    seqs = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")
    seqs_mask = pad_sequences(
        seqs_mask, maxlen=maxlen, padding="post", truncating="post", value=0
    )
    return seqs, seqs_mask




def load_dictionary(task_name, max_vocab_size, data_dir="./"):
    with open(
        (data_dir + "/aux_files/orig_dic_%s.pkl" % (task_name)), "rb") as f:
        orig_dic = pickle.load(f)
    with open(
        (data_dir + "/aux_files/orig_inv_dic_%s.pkl" % (task_name)), "rb") as f:
        orig_inv_dic = pickle.load(f)
    return orig_dic, orig_inv_dic

def load_dist_mat(dataset, max_vocab_size, data_dir="./"):
    dist_mat = np.load(
        (
            data_dir
            + "/aux_files/small_dist_counter_%s.npy" % (dataset)
        )
    )
    return dist_mat

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def _softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        _c_matrix = np.max(x, axis=1)
        _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
        _diff = np.exp(x - _c_matrix)
        x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
    else:
        _c = np.max(x)
        _diff = np.exp(x - _c)
        x = _diff / np.sum(_diff)
    assert x.shape == orig_shape
    return x


def build_dict(train_texts, train_text_pairs="", tokenizer="", vocab_size=500000, data_dir="./",max_length=128, device='cpu'):
    """
    The most frequently occurring words in the data set constitute the dictionary.
    Words that do not appear in the dictionary are all mapped to `UNK` with word id 0.
    """
    #tokens = tokenizer.tokenize(train_texts)
    
    tokens = []
    freqs=[]
    for t, t_pair in tqdm(zip(train_texts, train_text_pairs), total=len(train_texts)):      
        inputs = tokenizer.encode(text=t,
                           text_pair=t_pair if t_pair else None,
                           max_length=max_length,
                           truncation=True,)
        
        #for id_ in inputs:
        temp = tokenizer.convert_ids_to_tokens(inputs)
        temp.pop(0)
        temp.pop(-1)
        
#         temp = t.split(" ")
#         if t_pair:
#             temp1= t_pair.split(" ")
#             for tkn in temp1:
#                 temp.append(tkn)

        for tmp in temp:
            if tmp not in tokens:
                tokens.append(tmp)
                freqs.append(1)
            else:
                ind = tokens.index(tmp)
                freqs[ind] +=1
    sorted_ind = np.argsort(freqs)
    tokens = [tokens[s] for s in sorted_ind]
    #freqs = [freqs[s] for s in sorted_ind]
            
        
    dic = dict()
    dic["UNK"] = 0
    inv_dict = dict()
    inv_dict[0] = "UNK"
    
    for idx, word in enumerate(tokens):
        if idx <= vocab_size:
            inv_dict[idx] = word
            dic[word] = idx
    return dic, inv_dict, tokenizer

def loadGloveModel(gloveFile):
    """
    Load the glove model / glove model after counter-fitting.
    """
    import pickle
    print("Loading Glove Model")
    
    if ".pkl" in gloveFile:
        
        f = open(gloveFile,'rb')
        model = pickle.load(f)
        f.close()
    else:
        f = open(os.path.join(gloveFile), "r", encoding="utf-8")
        model = {}
        for line in f:
            row = line.strip().split(" ")
            word = row[0]
            embedding = np.array([float(val) for val in row[1:]])
            model[word] = embedding
        f.close()
        f = open("repo_glove.pkl","wb")
        pickle.dump(model, f)
        f.close()
        
    print("Done.", len(model), " words loaded!")
    return model


def compute_dist_matrix(dic, dataset, vocab_size=50000, data_dir="./"):
    """
    Create a distance matrix of size (vocab_size+1, vocab_size+1),
    and record the distance between two words in the GloVe embedding space after counter-fitting.
    The distances related to `UNK` (word id=0) are set to INFINITY.
    """
    INFINITY = 100000
    embedding_matrix, missed = None, None
    if not os.path.isfile(os.path.join(data_dir,"aux_files","embeddings_counter_%s.npy" % (dataset),)):
        print("embeddings_counter_%s.npy" % (dataset) + " not exists.")
        glove_tmp = loadGloveModel("E:/Vectors/counter-fitted-vectors.txt")
        embedding_matrix, missed = create_embeddings_matrix(glove_tmp, dic, embedding_size=300, data_dir=data_dir)
        np.save(os.path.join(data_dir,"aux_files","embeddings_counter_%s.npy" % (dataset),), embedding_matrix,)
        np.save(os.path.join(data_dir,"aux_files", "missed_embeddings_counter_%s.npy" % (dataset),), missed,)
    else:
        embedding_matrix = np.load(
            os.path.join(
                data_dir,
                "aux_files",
                "embeddings_counter_%s.npy" % (dataset),
            )
        )
        missed = np.load(
            os.path.join(
                data_dir,
                "aux_files",
                "missed_embeddings_counter_%s.npy" % (dataset),
            )
        )

    print("start to compute distance matrix")
    embedding_matrix = embedding_matrix.astype(np.float32)
    c_ = -2 * np.dot(embedding_matrix.T, embedding_matrix)
    a = np.sum(np.square(embedding_matrix), axis=0).reshape((1, -1))
    b = a.T
    dist = a + b + c_
    
    #print("Dist matrix:", dist.shape)
    #sys.exit()
#     dist[0, :] = -1 #INFINITY
#     dist[:, 0] = -1 #INFINITY
#     dist[missed, :] = -1 #INFINITY
    dist[:, missed] = -1 #INFINITY
    print("successfully computed distance matrix!")
    #print("Dist matrix:", (dist))
    return dist

def create_embeddings_matrix(
    glove_model,
    dictionary,
    full_dictionary=None,
    embedding_size=300,
    dataset=None,
    data_dir="./",
):
    embedding_matrix = np.zeros(shape=((embedding_size, len(dictionary))))
    cnt = 0
    unfound_ids = []
    unfound_words = []
    for w, i in dictionary.items():
        if not w in glove_model:
            cnt += 1
            unfound_ids.append(i)
            unfound_words.append(w)
        else:
            embedding_matrix[:, i] = glove_model[w]
            #print( glove_model[w].shape)
#     print(embedding_matrix.shape)
#     (300, 9478)   
            
    print("Number of not found words = ", cnt)
    if cnt != 0 and dataset is not None:
        f = open(os.path.join(data_dir, "aux_files", "unfound_words_%s.txt" % (dataset)),"w",encoding="utf-8",)
        f.write(" ".join(unfound_words))
        f.close()
    return embedding_matrix, unfound_ids

def pick_most_similar_words(src_word, small_dist_mat, ret_count=10, threshold=None):
    
    dist_order = small_dist_mat[src_word, :, 0]
    dist_list = small_dist_mat[src_word, :, 1]
    n_return = np.sum(dist_order > 0)
    dist_order, dist_list = dist_order[:n_return], dist_list[:n_return]
    if ret_count is not None:
        dist_order, dist_list = dist_order[:ret_count], dist_list[:ret_count]
    
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        dist_order, dist_list = dist_order[mask], dist_list[mask]
    
    mask1 = np.where(dist_order <= 50000)
    dist_order, dist_list = dist_order[mask1], dist_list[mask1]
    return dist_order, dist_list

def create_small_embedding_matrix(
    dist_mat, MAX_VOCAB_SIZE, threshold=1.5, retain_num=50
):
    """
    Create the synonym matrix. 
    The i-th row represents the synonyms of the word with id i and their distances.
    """
    small_embedding_matrix = np.zeros(shape=((MAX_VOCAB_SIZE, retain_num, 2)))
    #print(MAX_VOCAB_SIZE)
    for i in range(MAX_VOCAB_SIZE):
        if i % 1000 == 0:
            print("%d/%d processed." % (i, MAX_VOCAB_SIZE))
            
        #print("i:", i, " retain_:", retain_num)
        dist_order = np.argsort(dist_mat[i, :])[1 : 1 + retain_num]
        dist_list = dist_mat[i][dist_order]
        mask = np.ones_like(dist_list)
        if threshold is not None:
            mask = np.where(dist_list < threshold)
            dist_order, dist_list = dist_order[mask], dist_list[mask]
        n_return = len(dist_order)
        dist_order_arr = np.pad(
            dist_order, (0, retain_num - n_return), "constant", constant_values=(-1, -1)
        )
        dist_list_arr = np.pad(
            dist_list, (0, retain_num - n_return), "constant", constant_values=(-1, -1)
        )
        small_embedding_matrix[i, :, 0] = dist_order_arr
        small_embedding_matrix[i, :, 1] = dist_list_arr
    return small_embedding_matrix


# For Bert

class InputExample(object):
    
    def __init__(self, guid, text_a, text_b=None, label=None, flaw_labels=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.flaw_labels = flaw_labels

class InputFeatures(object):
    
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def get_examples_for_bert(path, data_dir="./data/"):
    examples = []
    clean_text_list, label_list = read_text(path, data_dir)
    for i, (text, label) in enumerate(zip(clean_text_list, label_list)):
        guid = "%s" % i
        flaw_labels = None
        examples.append(
            InputExample(guid=guid, text_a=text, text_b=None, label=label, flaw_labels=flaw_labels)
        )
    return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = example.label
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def sample(clean_text_list, labels, sample_num):
    """
    Use the Numpy library to randomly select the samples to be attacked. 
    Note that the seed used in our experiments is 0.
    """
    clean_text_list = np.array(clean_text_list)
    labels = np.array(labels)
    np.random.seed(0)
    shuffled_idx = np.arange(0, len(clean_text_list), 1)
    np.random.shuffle(shuffled_idx)
    sampled_idx = shuffled_idx[:sample_num]
    return list(clean_text_list[sampled_idx]), list(labels[sampled_idx])

num_labels_task = {
    "sst-2": 2,
    "imdb": 2,
    "cola":2, 
    "ag_news": 4,
    "yahoo": 10,
    "mnli":3,
}




    
 
