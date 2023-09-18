# detecting adversarial examples using the approach of adapted NNIF, Mahalanobis, RSV, and SHAP
# implementation of MDRE, LID, FGWS obtained from https://github.com/NaLiuAnna/MDRE

import argparse
import logging
import time
from tqdm import tqdm
import sys
import os
import random
import pickle
from math import ceil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.corpus import wordnet
from scipy.spatial import distance
from spacy.lang.en import English

from models import train, Classifier
from utils import convert_examples_to_features, IMDBDataset, MnliDataset
from utils import bert_params, roberta_params, xlnet_params, bart_params

#NNIF
from nnif import get_ihvp_score
from nnif import find_ranks, calc_all_ranks_and_dists

#Mahalanobis
from mahalanobis import sample_estimator, get_mahalanobis


#RSV
from rsv import build_embs, Detector_RDSU, EVAL_RDSU
import glove_utils

#SHAP
from shap_value import shap_generator, detectors

def get_emb_preds(model, tokenizer, max_length, batch_size, texts, labels=None):
    """
    obtain embeddings and predictions of texts from a model and hidden representations from BERT model.
    :param model: a model used for get embeddings and predictions.
    :param tokenizer: the model's tokenizer
    :param texts: texts
    :return: texts' embeddings, predictions, softmax, and hidden representations
    """          
    model.eval()
    n_batch = ceil(len(texts) / batch_size)
    
    embeddings = []
    preds = []
    softmax_list = []
    b_hidden_embeddings = []
    
#     if labels !=None:
#         label_names = np.unique(labels)
#         label_map = {label : i for i, label in enumerate(label_names)}
        
    for batch in range(n_batch):
        begin_idx = batch_size * batch
        end_idx = min(batch_size * (batch + 1), len(texts))        
        b_texts = texts[begin_idx: end_idx]
        text = np.asarray(b_texts)[:, 0].tolist()
        text_pair = np.asarray(b_texts)[:, 1].tolist()
        
        #b_labels = labels[begin_idx: end_idx] if labels else None
        
        input_ids = []
        attention_masks = []

        for t, t_pair in zip(text, text_pair):
            inputs = tokenizer.encode_plus(text=t,
                           text_pair=t_pair if t_pair else None,
                           return_tensors='pt',
                           return_attention_mask = True,
                           max_length=max_length,
                           truncation=True,
                           padding='max_length').to(device)

            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])
            del inputs
                   
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        with torch.no_grad():
            logits, pooler_output, hidden_states = model(input_ids, attention_masks)#, output_hidden_states=True)  
                
        embeddings.append(pooler_output.cpu().numpy())
        softmax_list.append(F.softmax(logits, dim=1).cpu().numpy())
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        #logits = logits.detach().cpu().numpy()
        #preds.append(logits)

        b_hidden_embeddings.append(torch.stack(hidden_states[1:])[:, :, 0, :].cpu().numpy().transpose((1, 0, 2)))

    hidden_embeddings = np.concatenate(b_hidden_embeddings, axis=0) if b_hidden_embeddings else b_hidden_embeddings
    
    return np.concatenate(embeddings, axis=0), np.concatenate(preds, axis=0), np.concatenate(softmax_list, axis=0), \
           hidden_embeddings

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact (ranked features) and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("positive artifacts: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("negative artifacts: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))
    
    return X, y

def logistic_detector(X, y, random_seed):
    """ A Logistic Regression classifier, and return its accuracy. """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
    
    # Build detector
    lr = LogisticRegression(max_iter=500000,random_state=random_seed).fit(X_train, y_train)
    # Evaluate detector   
    y_pred = lr.predict(X_test)

    # AUC
    acc = accuracy_score(y_test, y_pred)
    print('predictions:', y_pred)
    print('Accuracy: ', acc)
    
    return acc

def compute_roc(y_true, y_pred, plot=False):
    """
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        #plt.show()
        plt.savefig("AU_ROC.png")

    return fpr, tpr, auc_score

def get_if_scores(start, end):
    cases = ['pred', 'adv']

    for i in tqdm(range(start, end, 1)):
        #print("I have started index ", i)
        test_text= [test_texts[i]]
        test_text_pair = [test_text_pairs[i]] if args.dataset_name == 'Mnli' else None
        adv_text= [adv_texts[i]]
        adv_text_pair = [adv_text_pairs[i]] if args.dataset_name == 'Mnli' else None
        real_label = test_y[i]
        pred_label = test_preds[i]
        adv_label = adv_preds[i]
        
        if real_label == pred_label and pred_label != adv_label:
            
            for case in cases:            
                #separate dir for each test index
                dir = os.path.join(result_dir, '_index_{}'.format(i) , case)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                
                #start_time = time.time()
                #print("\nNow finding the influence:")
                if not os.path.exists(os.path.join(dir, 'scores.npy')):
                    if case == 'pred':
                        data = train_texts, train_text_pairs, train_y, test_text, test_text_pair, real_label, label_values
                    elif case == 'adv':
                        data = train_texts, train_text_pairs, train_y, adv_text, adv_text_pair, real_label, label_values

                    scores = get_ihvp_score(args, device, data, model, max_length, tokenizer, batch_size, i, dir, args.influence_on_decision, param_influence)
                    
                    torch.cuda.empty_cache()

                    #print('ihvp + scores calculation time: {} secs. case: {}'.format(time.time() - start_time, case)) 
                    np.save(os.path.join(dir, 'scores.npy'), scores)

def mle_batch(data, batch, k):
    """
    Obtaining LID for a batch
    :param data: a hidden embedding of all training examples
    :param batch: a batch of data for calculating LID
    :param k: number of neighbors used
    :return: LID for batch
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = distance.cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 0:k]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def get_lids_random_batch(X, X_test, X_adv, k, batch_size=32):
    """
    Calculating LID through batches
    :param X: hidden embeddings of training examples
    :param X_test: hidden embeddings of normal test examples
    :param X_adv: hidden embeddings of adversarial examples
    :param k: the number of nearest neighbors used for LID
    :param batch_size: default 32
    :return: lids: LID of normal examples of shape (num_examples, lid_dim)
            lids_adv: LID of adverarial examples of shape (num_examples, lid_dim)
    """

    lid_dim = 12
    print('Number of layers to estimate:', lid_dim)

    def estimate(i_batch):
        adv_start = np.minimum(len(X_adv) - 1, i_batch * batch_size)
        adv_end = np.minimum(len(X_adv) - 1, (i_batch + 1) * batch_size)
        test_start = i_batch * batch_size
        test_end = np.minimum(len(X_test) - 1, (i_batch + 1) * batch_size)
        test_n_feed = test_end - test_start
        adv_n_feed = adv_end - adv_start
        lid_batch = np.zeros(shape=(test_n_feed, lid_dim))
        lid_batch_adv = np.zeros(shape=(adv_n_feed, lid_dim))

        for i in range(lid_dim):
            train_X = X[:, i, :].reshape(X.shape[0], -1)
            X_act = X_test[test_start: test_end, i, :].reshape((test_n_feed, -1))
            lid_batch[:, i] = mle_batch(train_X, X_act, k=k)
            if adv_n_feed > 0:
                X_adv_act = X_adv[adv_start: adv_end, i, :].reshape((adv_n_feed, -1))
                lid_batch_adv[:, i] = mle_batch(train_X, X_adv_act, k=k)

        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    n_batches = int(np.ceil(X_test.shape[0] / float(batch_size)))

    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)

    lids = np.asarray(lids, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_adv


def get_lid(X, X_test, X_adv, k, batch_size=100):
    """
    Calculating LID and prepare data for the detection classifier
    :param X: hidden embeddings of training examples
    :param X_test: hidden embeddings of test examples
    :param X_adv: hidden embeddings of adversarial examples
    :param k: neighbors used for LID
    :param batch_size: batch size used for the function: get_lids_random_batch
    :return: X(LID) and y for the detection classifier
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_adv = get_lids_random_batch(X, X_test, X_adv, k, batch_size)
    print('lids_normal:', lids_normal.shape)
    print('lids_adv:', lids_adv.shape)

    lids_pos = lids_adv
    lids_neg = lids_normal
    artificats, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artificats, labels


def get_words_frequency(texts, text_pairs):
    """
    get words frequency according to their occurance in training set for fgws.
    """
    nlp = English()

    words_freq = {}

    for text, text_pair in zip(texts, text_pairs):
        example = text + ' ' + text_pair if text_pair else text
        for word in [t.text.lower() for t in nlp.tokenizer(example.strip())]:
            try:
                words_freq[word] += 1
            except KeyError:
                words_freq[word] = 1

    return words_freq


def transform(text, words_freq, freq_threshold, words_list, distance_matrix, missed_words):
    """
    replace words which have lower frequency than the frequency threshold for fgws
    """
    nlp = English()

    replaced_list = []
    for word in [t.text.lower() for t in nlp.tokenizer(text.strip())]:
        if word in words_freq.keys() and words_freq[word] < freq_threshold:
            neighbors = []

            for synset in wordnet.synsets(word):
                for w in synset.lemmas():
                    neighbors.append(w.name().replace("_", " "))

            emb_neighbours = []
            if word in words_list and words_list.index(word) < 20000 and words_list.index(word) not in missed_words:
                neighbours_ids, _ = glove_utils.pick_most_similar_words(words_list.index(word), distance_matrix, 10, 0.5)
                emb_neighbours = [words_list[id] for id in neighbours_ids]

            val_neighbors = [w for w in neighbors + emb_neighbours if w in words_freq.keys() and
                             words_freq[w] > words_freq[word]]
            replaced_list.append(random.choice(val_neighbors) if len(val_neighbors) > 0 else word)
        else:
            replaced_list.append(word)

    return ' '.join(replaced_list)


def tune_gamma(orig_preds, orig_probs, model, tokenizer, dataset, words_freq, freq_threshold, words_list,
               distance_matrix, missed_words):
    """
    set differency threshold for fgws, if the difference between the predictions of a replaced words example and its
    corresponding unreplaced words example is bigger than this threshold, this original example is regard as an
    adversarial example.
    This threshold is the 90%-th prediction difference between words substituted validation set and validation set.
    """

    differneces = []

    replaced_valid_text, replaced_valid_text_pair = [], []
    for text, text_pair in zip(dataset.valid_text, dataset.valid_text_pair):
        replaced_valid_text.append(transform(text, words_freq, freq_threshold, words_list, distance_matrix, missed_words))
        replaced_valid_text_pair.append(transform(text_pair, words_freq, freq_threshold, words_list, distance_matrix,
                                                  missed_words) if text_pair else text_pair)

    _, replaced_preds, replaced_probs, _ = get_emb_preds(model, tokenizer, args.batch_size, list(zip(replaced_valid_text, replaced_valid_text_pair)))

    for orig_prob, orig_pred, replaced_prob, replaced_pred in zip(orig_probs, orig_preds, replaced_probs,
                                                                  replaced_preds):
        differneces.append(max(0, orig_prob[orig_pred] - replaced_prob[orig_pred]))

    differneces.sort()
    p = 0.9 * len(differneces)
    thr_idx = int(p) - 1 if p.is_integer() else int(p)
    gamma = differneces[thr_idx]

    return gamma


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default=None, required=False, choices=['IMDB', 'Mnli'],
                        help='detecting which test set`s adversarial examples.')
    parser.add_argument('--dataset-path', type=str, required=False,
                        default="./data/", help='The directory of the dataset.') #choices=['./data/aclImdb', './data/multinli_1.0'],
    parser.add_argument('--adv-path', type=str, required=False,
                        default="./data/", help='The directory of the adversarial dataset.') #choices=['./data/aclImdb', './data/multinli_1.0'],
    parser.add_argument('--attack-class', type=str, default=None, required=False, choices=['typo', 'synonym', 'seas', 'bertattack'],
                        help='Attack method to generate adversarial examples.')
    parser.add_argument('--model-dir', type=str, required=False,
                        default="./data/", help='The directory of the model.')
    parser.add_argument('--max-length', type=int, default=None, required=False, choices=[768,512, 256, 128],
                        help='The maximum sequences length.')#True
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for transformer models.')
    parser.add_argument('--random-seed', type=int, default=38, help='random seed value.')
    parser.add_argument('--detect', type=str, default=None, required=False,
                        choices=['mdre', 'lid', 'fgws', 'language_model', 'mahalanobis','nnif', 'rsv', 'shap'],
                        help='Type of detection.')
    parser.add_argument('--k-nearest', type=int, default=-1, required=False,
                        help='Number of nearest neighbours to use for lid.')
    parser.add_argument('--fp-threshold', type=float, default=0.9,
                        help='for FGWS detection to calculate gamma, false positive threshold.')
    
    #NNIF
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--start', type=int, default=0, help='starting test id')
    parser.add_argument('--end', type=int, default=0, help='ending test id')
    parser.add_argument('--damping', type=float, default=0.0, help="probably need damping for deep models")
    #flags.DEFINE_string('dataset', 'cifar10', 'dataset: cifar10/100 or svhn')
    parser.add_argument('--set', type=str, default='test', required=False, choices=['test', 'val'],
                        help='detecting which test set`s adversarial examples.')
    parser.add_argument('--with-noise', type=bool, default=False, required=False,
                        help='whether or not to include noisy samples')
    parser.add_argument('--only_last', type=bool, default=False, required=False,
                        help='Using just the last layer, the embedding vector')
    parser.add_argument('--checkpoint_dir', type=str, default='', required=False,
                        help='Checkpoint dir, the path to the saved model architecture and weights')
    parser.add_argument('--max-indices', type=int, default=-1, help='maximum number of helpful indices to use in NNIF detection')
    parser.add_argument('--ablation', type=str, default='1111', help='for ablation test')
    parser.add_argument('--influence_on_decision', action='store_true',
                        help="Whether to compute influence on decision (rather than influence on ground truth)")
    
    # FOR DkNN and LID
    parser.add_argument('--k_nearest', type=int, default=-1, help='number of nearest neighbors to use for LID/DkNN detection')

    # MAHANABOLIS
    parser.add_argument('--magnitude', type=float, default=-1, help='magnitude for mahalanobis detection')
    
    #RSV
    parser.add_argument("--max_candidates", default=4, type=int)
    parser.add_argument("--threshold_distance", default=0.5, type=float)
    parser.add_argument("--vocab_size", default=5000, type=int)
    parser.add_argument("--votenum", default=2, type=int, help="vote num for RS&V")
    parser.add_argument("--randomrate", default=0.6, type=float, help="random rate for RS&V")
    parser.add_argument("--fixrate", default=0.02, type=float, help="fix rate for RS&V")
    parser.add_argument("--transfer_out_file", default="./results/transfer/transfer_ag_news_cnn_textfooler.pkl",type=str, help="output file path")
    parser.add_argument("--output_dir", default="./results/IMDB/bert", type=str,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--modeltype", default="cnn", type=str, help="the model type")
    parser.add_argument("--eval_out_file", default=" ",type=str, help="eval file path")
    parser.add_argument("--build-dict", default=False,type=bool, help="build dictionary from glove encoding")

    args = parser.parse_args()

    batch_size = 5
    max_length = args.max_length
    max_indices = args.max_indices

    # set a random seed value all over the place to make this reproducible.
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # check if there's a GPU
    if torch.cuda.is_available():
        #set the device to the GPU.
        device = torch.device('cuda')
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')
    
    # load dataset
    data_processors = {
        'Mnli': MnliDataset,
        'IMDB': IMDBDataset
    }
    dataset = data_processors[args.dataset_name](args.dataset_path, 0.1)
    output_dir = os.path.join('./results', args.dataset_name)
    label_values = dataset.get_labels() #["0", "1"] for IMDB
    num_classes = dataset.num_labels #2 for IMDB
    
    train_texts = dataset.train_text
    train_text_pairs = dataset.train_text_pair
    train_y = dataset.train_y
    
    train_idx, valid_idx = dataset.get_train_val_idx()
    np.save(os.path.join(args.dataset_path, 'train_idx_nnif.csv'), train_idx)
    np.save(os.path.join(args.dataset_path, 'val_idx_nnif.csv'), valid_idx)

    # load test and adversarial examples       
    test_adv_file = args.adv_path
    #print('test_adv_file:', test_adv_file)
    test_adv = pd.read_csv(test_adv_file)
    adv_texts = test_adv['adv_text'].to_numpy()
    adv_preds = test_adv['adv_label'].to_numpy()
    test_texts = test_adv['orig_text'].to_numpy()
    test_y = test_adv['orig_label'].to_numpy()
    test_preds = test_adv['orig_pred'].to_numpy()

    if args.dataset_name == 'IMDB':
        test_text_pairs = [None] * len(test_y)
        adv_text_pairs = [None] * len(test_y)
    else:
        test_text_pairs = test_adv['orig_text_pair'].to_numpy()
        adv_text_pairs = test_adv['adv_text_pair'].to_numpy()

    result_dir = os.path.join(output_dir, args.detect, args.attack_class)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    model = Classifier(dataset.num_labels, **bert_params)
    tokenizer = model.tokenizer
    model_config = model.config
    
    if not os.path.exists(os.path.join(args.model_dir, 'model.pt')):
        data = train_texts, train_text_pairs, train_y, label_values
        train(args, model, data, args.max_length, tokenizer, device, epochs=3, save_model= args.model_dir,)
    
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt'), map_location=device), strict =False)
    model.to(device)
    
#if need to consider all layers for NNIF - DkNN
#     newmodel = torch.nn.Sequential(*(list(model.children())[-3:]))
#     layer=len(newmodel)
#     print(layer)#4


        #reduced number of neighbors & test size for faster computation
        l = np.arange(len(train_texts))
        random.shuffle(l) 
        #print("train_texts:", len(train_texts))
        index_value = random.sample(list(l), 6000)
        train_texts = [train_texts[i] for i in index_value]
        train_text_pairs = [train_text_pairs[i] for i in index_value]
        train_y = [train_y[i] for i in index_value]

        l = np.arange(5000)
        random.shuffle(l)
        index_value = random.sample(list(l), len(l))
        test_texts = [test_texts[ind] for ind in index_value]
        test_text_pairs = [test_text_pairs[ind] for ind in index_value]
        test_y = [test_y[ind] for ind in index_value]
        test_preds = [test_preds[ind] for ind in index_value]
        adv_texts = [adv_texts[ind] for ind in index_value]
        adv_text_pairs = [adv_text_pairs[ind] for ind in index_value]
        adv_preds = [adv_pred[ind] for ind in index_value]

    
    if args.detect == 'nnif':
        result_dir = os.path.join(output_dir, args.detect, args.attack_class)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        param_optimizer = list(model.named_parameters())
        param_influence = []
        for n, p in param_optimizer:
            param_influence.append(p)
        
        #calculate and store IF scores
        get_if_scores(args.start,args.end)
        
        
        # DkNN
        
#         for DkNN on all layers               
#         print("start KNN observation")      
#             knn = {}
#             _, _, _, train_embeds = get_emb_preds(model, tokenizer, max_length, batch_size, list(zip(train_texts, train_text_pairs)))
#             print('Fitting knn models on all layers: {}'.format(len(newmodel)))       
#         for layer_index, layer in enumerate(newmodel.children()):
#             print(layer_index)
#             if len(train_embeds[layer_index].shape) == 4:
#                 train_embeds[layer_index] = np.asarray(train_embeds[layer_index], dtype=np.float32).reshape((X.shape[0], -1, train_embeds[layer_index].shape[-1]))
#                 train_embeds[layer_index] = np.mean(train_embeds[layer_index], axis=1)
#             elif len(train_embeds[layer_index].shape) == 2:
#                 pass  # leave as is
#             else:
#                 raise AssertionError('Expecting size of 2 or 4 but got {} for {}'.format(train_embeds[layer_index].shape), layer)

#             knn[layer] = NearestNeighbors(n_neighbors=X.shape[0], p=2, n_jobs=20, algorithm='brute')
#             knn[layer].fit(train_embeds[layer_index], y)
#             del train_embeds

#         for DkNN on final layer         
        print("Loading train embeddings")
        if not os.path.exists(os.path.join(result_dir, 'train_emb.npy')):            
            train_embeds, _, _, _ = get_emb_preds(model, tokenizer, max_length, batch_size, list(zip(train_texts, train_text_pairs)))
            np.save(os.path.join(result_dir, 'train_emb.npy'), train_embeds)
            
            #for plotting neighbors
            #reduced_train_embeds=TSNE(perplexity=15, n_components=2, init='pca', n_iter=5000, metric='euclidean', random_state=23).fit_transform(train_embeds)
            #np.save(os.path.join(result_dir, 'reduced_train_embeds.npy'), reduced_train_embeds)
        else:
            train_embeds = np.load(os.path.join(result_dir, 'train_emb.npy'))
        

        print("Loading test embeddings")  
        if not os.path.exists(os.path.join(result_dir, 'test_emb.npy')):
            test_embeds, _, _, _ = get_emb_preds(model, tokenizer, max_length ,batch_size, list(zip(test_texts, test_text_pairs)))
            adv_embeds, _, _, _ = get_emb_preds(model, tokenizer, max_length, batch_size, list(zip(adv_texts, adv_text_pairs)))
            np.save(os.path.join(result_dir, 'test_emb.npy'), test_embeds)
            np.save(os.path.join(result_dir, 'adv_emb.npy'), adv_embeds)
            
            #for plotting neighbors
            #reduced_test_embeds=TSNE(perplexity=15, n_components=2, init='pca', n_iter=5000, metric='euclidean', random_state=23).fit_transform(test_embeds)
            #reduced_adv_embeds=TSNE(perplexity=15, n_components=2, init='pca', n_iter=5000, metric='euclidean', random_state=23).fit_transform(adv_embeds)
            #np.save(os.path.join(result_dir, 'reduced_test_embeds.npy'), reduced_test_embeds)
            #np.save(os.path.join(result_dir, 'reduced_adv_embeds.npy'), reduced_adv_embeds)
        else:      
            test_embeds = np.load(os.path.join(result_dir, 'test_emb.npy'))
            adv_embeds = np.load(os.path.join(result_dir, 'adv_emb.npy')) 
        
        if len(train_embeds.shape) == 4:
            train_embeds = np.asarray(train_embeds, dtype=np.float32).reshape((X.shape[0], -1, train_embeds.shape[-1]))
            train_embeds = np.mean(train_embeds, axis=1)
        elif len(train_embeds.shape) == 2:
            pass  # leave as is
        else:
            raise AssertionError('Expecting size of 2 or 4 but got {} for {}'.format(train_embeds.shape), layer)

        #start KNN observation
        knn={}
        layer=1
        knn[layer] = NearestNeighbors(n_neighbors=len(train_texts), p=2, n_jobs=20, algorithm='brute')
        knn[layer].fit(train_embeds, train_y)
        del train_embeds
        
        #print("result_dir:", result_dir)
        if not os.path.exists(os.path.join(result_dir, 'all_neighbor_ranks.npy')):
            print('predicting knn dist/indices for test data')
            all_neighbor_ranks, all_neighbor_dists = calc_all_ranks_and_dists(test_embeds, knn)#red_test_embeds
            print('predicting knn dist/indices for adv data')
            all_neighbor_ranks_adv, all_neighbor_dists_adv    = calc_all_ranks_and_dists(adv_embeds, knn)#red_adv_embeds
            np.save(os.path.join(result_dir, 'all_neighbor_ranks.npy'), all_neighbor_ranks)
            np.save(os.path.join(result_dir, 'all_neighbor_dists.npy'), all_neighbor_dists)
            np.save(os.path.join(result_dir, 'all_neighbor_ranks_adv.npy'), all_neighbor_ranks_adv)
            np.save(os.path.join(result_dir, 'all_neighbor_dists_adv.npy'), all_neighbor_dists_adv)
        else:
            all_neighbor_ranks = np.load(os.path.join(result_dir, 'all_neighbor_ranks.npy'))
            all_neighbor_dists = np.load(os.path.join(result_dir, 'all_neighbor_dists.npy'))
            all_neighbor_ranks_adv = np.load(os.path.join(result_dir, 'all_neighbor_ranks_adv.npy'))
            all_neighbor_dists_adv = np.load(os.path.join(result_dir, 'all_neighbor_dists_adv.npy'))    
            all_neighbor_ranks_adv   = all_neighbor_ranks_adv 
            all_neighbor_dists_adv   = all_neighbor_dists_adv
        

        if args.max_indices == -1:
            max_indices_vec = [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        else:
            max_indices_vec = [args.max_indices]
            
            
        count=0
        indices = []
        for i in range(args.start, args.end, 1):
            if test_y[i] == test_preds[i] and test_preds[i] != adv_preds[i]:
                count+=1
                indices.append(i)
            
        max_acc=0
        result_string = []
        for max_indices in tqdm(max_indices_vec):
            print('Extracting NNIF characteristics for max_indices={}'.format(max_indices))
            ranks     = -1 * np.ones((count, 4, max_indices))
            ranks_adv = -1 * np.ones((count, 4, max_indices))
            
            j=0
            for i in tqdm(range(args.start, args.end, 1)):
                test_text= [test_texts[i]]
                test_text_pair = [test_text_pairs[i]] if args.dataset_name == 'Mnli' else None
                adv_text= [adv_texts[i]]
                adv_text_pair = [adv_text_pairs[i]] if args.dataset_name == 'Mnli' else None
                real_label = test_y[i]
                pred_label = test_preds[i]
                adv_label = adv_preds[i]

                if real_label == pred_label and pred_label != adv_label:
                    cases = ['pred', 'adv']
                    for case in cases:              
                        #print("========Case: ", case)
                
                        #separate dir for each test index
                        dir = os.path.join(result_dir, '_index_{}'.format(i) , case)
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                        
                        #load IF scores
                        scores = np.load(os.path.join(dir, 'scores.npy'))
                        sorted_indices = np.argsort(scores)
                        harmful = list(sorted_indices[:max_indices])
                        helpful = list(sorted_indices[-max_indices:][::-1])

                        if case == 'pred':
                            ni   = all_neighbor_ranks
                            nd   = all_neighbor_dists                        
                        elif case == 'adv':
                            ni   = all_neighbor_ranks_adv
                            nd   = all_neighbor_dists_adv
                    
                        helpful_ranks, helpful_dists = find_ranks(i, sorted_indices[-max_indices:][::-1], ni, nd)
                        harmful_ranks, harmful_dists = find_ranks(i, sorted_indices[:max_indices], ni, nd)            
                        helpful_ranks= np.array(helpful_ranks)
                        helpful_dists= np.array(helpful_dists)
                        harmful_ranks= np.array(harmful_ranks)
                        harmful_dists= np.array(harmful_dists)
                
                        if case == 'pred':
                            ranks[j, 0, :] = helpful_ranks
                            ranks[j, 1, :] = helpful_dists
                            ranks[j, 2, :] = harmful_ranks
                            ranks[j, 3, :] = harmful_dists
                    
                        if case == 'adv':
                            ranks_adv[j, 0, :] = helpful_ranks
                            ranks_adv[j, 1, :] = helpful_dists
                            ranks_adv[j, 2, :] = harmful_ranks
                            ranks_adv[j, 3, :] = harmful_dists
                        
#                         cnt_harmful_in_knn = 0 
#                         for idx in list(harmful):
#                             if idx in ni[i, 0:50]:
#                                 cnt_harmful_in_knn += 1                     
#                         print('{} out of {} harmful texts are in the {}-NN\n'.format(cnt_harmful_in_knn, len(harmful), 50))

#                         cnt_helpful_in_knn = 0
#                         for idx in helpful:
#                             if idx in ni[i, 0:50]:
#                                 cnt_helpful_in_knn += 1
#                         print('{} out of {} helpful images are in the {}-NN\n'.format(cnt_helpful_in_knn, len(helpful), 50))
#                         print('Rank finding time: {} secs. case: {}'.format(time.time() - start_time, case))

                        del scores
                    j+=1


            #for ablation
            #using all four features or not
            sel_column = []

            for i in [0, 1, 2, 3]:
                if args.ablation[i] == '1':
                    sel_column.append(i)
            ranks     = ranks[:, sel_column, :  ]
            ranks_adv = ranks_adv[:, sel_column, :]

            characteristics, labels = merge_and_generate_labels(ranks_adv, ranks)
            print("NNIF test: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
            data = np.concatenate((characteristics, labels), axis=1)        
            end_test = time.time()
            hours, rem = divmod(end_test - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print('total feature extraction time for test: {:0>2}:{:0>2}:{:0>2f}'.format(int(hours),int(minutes),seconds))

            
            ## Build detector
            X = data[:, :-1]
            Y = data[:, -1]
            print("LR Detector on [dataset: {}, test_attack: {}, characteristics: {}, ablation: {}, max_indices: {}]:".format(args.dataset_name, args.attack_class, args.detect, args.ablation, max_indices))             
            acc = logistic_detector(X, Y, args.random_seed) #acc
            
            if acc>max_acc:
                max_acc = acc
                np.save(os.path.join(result_dir, 'max_indices_{}_ablation_{}.npy'.format(max_indices, args.ablation)), data)

    elif args.detect == 'mahalanobis':
        result_dir = os.path.join(output_dir, args.detect, args.attack_class)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        if not os.path.exists(os.path.join(result_dir, 'sample_mean_mahalanobis.pt')): 
            print('Get sample mean and covariance of the training set...')
            sample_mean, precision = sample_estimator(model, device, train_texts, train_text_pairs, train_y, label_values, max_length, tokenizer, only_last=args.only_last)
            torch.save(sample_mean, os.path.join(result_dir, 'sample_mean_mahalanobis.pt'))
            torch.save(precision, os.path.join(result_dir, 'precision.pt'))
        else:
            print("Loading sample mean and precisions")
            sample_mean = torch.load(os.path.join(result_dir, 'sample_mean_mahalanobis.pt'))
            precision = torch.load(os.path.join(result_dir, 'precision.pt'))


        print('get Mahalanobis scores')
        #m_list = [0.0] #[0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005] #noise not used 
        #for magnitude in m_list:
        
        data = test_texts, test_text_pairs, test_y, adv_texts, adv_text_pairs, adv_preds, label_values
            
        print("Penultimate: ", args.only_last)
        Mahalanobis_pos, Mahalanobis_neg = get_mahalanobis(model, device, args.random_seed, max_length, tokenizer, data, sample_mean, precision, num_classes, args.only_last)
            #, layer_index = layer_index, magnitude

        characteristics, labels = merge_and_generate_labels(Mahalanobis_pos, Mahalanobis_neg)
        print("Mahalanobis train: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
        data = np.concatenate((characteristics, labels), axis=1)        
        if args.only_last:
            np.save(os.path.join(result_dir, 'penultimate.npy'), data)
        else:
            np.save(os.path.join(result_dir, 'all_layers.npy'), data)#.format(magnitude)
        end_test = time.time()
        hours, rem = divmod(end_test - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print('total feature extraction time for test: {:0>2}:{:0>2}:{:0>2f}'.format(int(hours),int(minutes),seconds))
        
        ## Build detector
        X = data[:, :-1]
        Y = data[:, -1]
                                
        print("LR Detector on [dataset: {}, test_attack: {}, characteristics: {}]:".format(args.dataset_name, args.attack_class, args.detect))             
        acc = logistic_detector(X, Y, args.random_seed)

    elif args.detect == 'rsv':       
        #build the dictionary
        if args.build_dict:
            build_embs(args, train_texts, train_text_pairs, tokenizer, device)
        
        #transfer example
        if not os.path.exists(args.transfer_out_file):
            if args.dataset_name == 'Mnli':
                RDSU = Detector_RDSU(args=args, clean_texts=test_texts, clean_text_pairs=test_text_pairs,clean_labels=test_y, adv_texts=adv_texts, adv_text_pairs=adv_text_pairs, adv_preds= adv_preds)
            else:
                RDSU = Detector_RDSU(args=args, clean_texts=test_texts, clean_labels=test_y, adv_texts=adv_texts,  adv_preds= adv_preds)
            #print(args.transfer_out_file)
            transfer_examples = RDSU.transfer_all_examples(args.transfer_out_file)
        else:
            with open(args.transfer_out_file, "rb") as handle:
                transfer_examples = pickle.load(handle)
        #evaluate
        eval_main = EVAL_RDSU(args=args, model= model,tokenizer=tokenizer, num_labels=dataset.num_labels, label_names=label_values, device=device)
        transfer_ori_acc,transfer_adv_acc,t_p,f_p,f_n,f1,tpr, detect_acc = eval_main.eval_all_examples(transfer_examples, data_dir = result_dir)

    elif args.detect == 'shap':
        print("SHAP generator")
        shap_gen = shap_generator( model, tokenizer, max_length, device, args.random_seed)
        #args.end  = 1000, 2000, ...5000 to slice dataset and generate shap scores simultaneously 
        shap_gen.create_SHAP_signatures(test_texts, adv_texts, args.dataset_name, args.end,  num_classes, result_dir= result_dir, batch_size=100)
        del shap_gen
        #slices = [1000, 2000, 3000, 4000, 5000]
        #merge_results(slices, result_dir, case="orig")
        #merge_results(slices, result_dir, case="adv")
        #print("Adversarial text detector")
        detectors(result_dir, args.random_seed)
    
    # Na Liu et.al. (2022) https://github.com/NaLiuAnna/MDRE
    elif args.detect == 'mdre':
        X = []
        y = []
        _, train_preds, _, _ = get_emb_preds(model, tokenizer, args.batch_size,
                                             list(zip(dataset.train_text, dataset.train_text_pair)))

        for label in range(dataset.num_labels):
            # find train, test, adversarial examples with same predictions
            train_indices = np.where(np.asarray(train_preds) == label)[0]
            sub_train_texts = [text for ind, text in enumerate(dataset.train_text) if ind in train_indices]
            sub_train_text_pairs = [text for ind, text in enumerate(dataset.train_text_pair) if ind in train_indices]

            test_indices = np.where(np.asarray(test_preds) == label)[0]
            sub_test_texts = [text for ind, text in enumerate(test_texts) if ind in test_indices]
            sub_test_text_pairs = [text for ind, text in enumerate(test_text_pairs) if ind in test_indices]

            adv_indices = np.where(np.asarray(adv_preds) == label)[0]
            sub_adv_texts = [text for ind, text in enumerate(adv_texts) if ind in adv_indices]
            sub_adv_text_pairs = [text for ind, text in enumerate(adv_text_pairs) if ind in adv_indices]

            model_classes = ['bert', 'roberta',  'bart'] #'xlnet',
            for ind, model_name in enumerate(model_classes):
                if model_name == 'bert':
                    model_params = bert_params
                elif model_name == 'roberta':
                    model_params = roberta_params
                elif model_name == 'xlnet':
                    model_params = xlnet_params
                elif model_name == 'bart':
                    model_params = bart_params

                part_X = []
                part_y = []

                # get train, testing, and adversarial examples' embeddings
                train_embeddings, _, _, _ = get_emb_preds(model, tokenizer, args.batch_size,
                                                          list(zip(sub_train_texts, sub_train_text_pairs)))
                test_embeddings, _, _, _ = get_emb_preds(model, tokenizer,args.batch_size,
                                                         list(zip(sub_test_texts, sub_test_text_pairs)))
                adv_embeddings, _, _, _ = get_emb_preds(model, tokenizer,args.batch_size, list(zip(sub_adv_texts, sub_adv_text_pairs)))

                part_X += np.amin(distance.cdist(test_embeddings, train_embeddings, metric='euclidean'),
                                  axis=1).tolist()
                part_X += np.amin(distance.cdist(adv_embeddings, train_embeddings, metric='euclidean'), axis=1).tolist()
                part_y = [1] * len(test_embeddings) + [0] * len(adv_embeddings)

#                 emb_dir = os.path.join(output_dir, model_name, args.attack_class)
#                 if not os.path.exists(emb_dir):
#                     os.makedirs(emb_dir)

#                 np.save(os.path.join(emb_dir, str(label) + 'train_emb.npy'), train_embeddings)
#                 np.save(os.path.join(emb_dir, str(label) + 'test_emb.npy'), test_embeddings)
#                 np.save(os.path.join(emb_dir, str(label) + 'adv_emb.npy'), adv_embeddings)

                if len(X) == len(model_classes):
                    X[ind] += part_X
                else:
                    X.append(part_X)

            y += part_y

        logistic_detector(np.asarray(X).T, np.asarray(y).T, args.random_seed)
    
    # Na Liu et.al. (2022) https://github.com/NaLiuAnna/MDRE
    elif args.detect == 'lid':
        # set the number of nearest neighbours to use
        if args.k_nearest == -1:
            k_vec = np.arange(10, 41, 2)
            k_vec = np.concatenate([k_vec, [100, 1000]])
        else:
            k_vec = [args.k_nearest]
            
        if not os.path.exists(os.path.join(result_dir, 'train_hidden_emb.npy')):            
            _, _, _, train_hidden_embeddings = get_emb_preds(model, tokenizer, args.batch_size, 
                                                         list(zip(train_texts, train_text_pairs)))
            np.save(os.path.join(result_dir, 'train_hidden_emb.npy'), train_hidden_embeddings)
            #np.save(os.path.join(result_dir, 'train_pred.npy'), train_preds)

        else:
            print("Loading train embeddings")
            train_hidden_embeddings = np.load(os.path.join(result_dir, 'train_hidden_emb.npy'))
            #train_preds = np.load(os.path.join(result_dir, 'train_pred.npy'))
                
        if not os.path.exists(os.path.join(result_dir, 'test_hidden_emb.npy')):            
            _, _, _, test_hidden_embeddings = get_emb_preds(model, tokenizer, args.batch_size, 
                                                         list(zip(test_texts, test_text_pairs)))
            np.save(os.path.join(result_dir, 'test_hidden_emb.npy'), test_hidden_embeddings)
            #np.save(os.path.join(result_dir, 'train_pred.npy'), train_preds)

        else:
            print("Loading test embeddings")
            test_hidden_embeddings = np.load(os.path.join(result_dir, 'test_hidden_emb.npy'))
            #train_preds = np.load(os.path.join(result_dir, 'train_pred.npy'))
            
        if not os.path.exists(os.path.join(result_dir, 'adv_hidden_emb.npy')):            
            _, _, _, adv_hidden_embeddings = get_emb_preds(model, tokenizer, args.batch_size, 
                                                         list(zip(adv_texts, adv_text_pairs)))
            np.save(os.path.join(result_dir, 'adv_hidden_emb.npy'), adv_hidden_embeddings)
            #np.save(os.path.join(result_dir, 'train_pred.npy'), train_preds)
        else:
            print("Loading adv embeddings")
            adv_hidden_embeddings = np.load(os.path.join(result_dir, 'adv_hidden_emb.npy'))
            #train_preds = np.load(os.path.join(result_dir, 'train_pred.npy'))
            

        for k in tqdm(k_vec):
            print('Extracting LID characteristics for k={}'.format(k))

            X, y = get_lid(train_hidden_embeddings, test_hidden_embeddings, adv_hidden_embeddings, k, 100)
    
            logistic_detector(X, y, args.random_seed)
    
    # Na Liu et.al. (2022) https://github.com/NaLiuAnna/MDRE
    elif args.detect == 'fgws':
        vocab_file = os.path.join(args.dataset_path, 'aux_files/vocab.vocab')
        distance_matrix_file = os.path.join(args.dataset_path, 'aux_files/dist_counter.npy')
        missed_embedding_file = os.path.join(args.dataset_path, 'aux_files/missed_embeddings_counter.npy')
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                words_list = f.read().split('\n')
            distance_matrix = np.load(distance_matrix_file)
            missed_words = np.load(missed_embedding_file)
        except FileNotFoundError:
            print("Couldn't find the vocabulary file, please run get_neighbours.py first.")
            
        if not os.path.exists(os.path.join(output_dir, 'saved_model', 'model.pt')):
            data = train_texts, train_text_pairs, train_y, label_values
            path = os.path.join(output_dir, 'saved_model')
            train(args, model, data, max_length, tokenizer, device, bert_params['learning_rate'], epochs=3, save_model= path,)
            
        if not os.path.exists(os.path.join(args.dataset_path, 'aux_files/words_freq.npy')):
            words_freq = get_words_frequency(dataset.train_text, dataset.train_text_pair)
            np.save(os.path.join(args.dataset_path, 'aux_files/words_freq.npy'), words_freq)
        else:
            words_freq = np.load(os.path.join(args.dataset_path, 'aux_files/words_freq.npy'))

        if not os.path.exists(result_dir, 'orig_adv_preds.npy'):
            _, orig_adv_preds, orig_adv_probs, _ = get_emb_preds(bert_model, bert_tokenizer, args.batch_size,
                                                             list(zip(adv_texts, adv_text_pairs)))
            _, orig_test_preds, orig_test_probs, _ = get_emb_preds(bert_model, bert_tokenizer, args.batch_size,
                                                               list(zip(test_texts, test_text_pairs)))
            _, orig_valid_preds, orig_valid_probs, _ = get_emb_preds(bert_model, bert_tokenizer, args.batch_size,
                                                                 list(zip(dataset.valid_text, dataset.valid_text_pair)))
                
            np.save(os.path.join(result_dir, 'orig_adv_preds.npy'), orig_adv_preds)
            np.save(os.path.join(result_dir, 'orig_adv_probs.npy'), orig_adv_probs)
            np.save(os.path.join(result_dir, 'orig_test_preds.npy'), orig_test_preds)
            np.save(os.path.join(result_dir, 'orig_test_probs.npy'), orig_test_probs)
            np.save(os.path.join(result_dir, 'orig_valid_preds.npy'), orig_valid_preds)
            np.save(os.path.join(result_dir, 'orig_valid_probs.npy'), orig_valid_probs)
        else:
            orig_adv_preds = np.load(os.path.join(result_dir, 'orig_adv_preds.npy'))
            orig_adv_probs = np.load(os.path.join(result_dir, 'orig_adv_probs.npy'))
            orig_test_preds = np.load(os.path.join(result_dir, 'orig_test_preds.npy'))
            orig_test_probs = np.load(os.path.join(result_dir, 'orig_test_probs.npy'))
            orig_valid_preds = np.load(os.path.join(result_dir, 'orig_valid_preds.npy'))
            orig_valid_probs = np.load(os.path.join(result_dir, 'orig_valid_probs.npy'))
                

        for delta_thr in tqdm(range(0, 110, 30)):
            print('delta: ', delta_thr)

            corr = 0  # number of adversarial examples that fgws regard as adversarial examples or normal examples
            # regard as normal examples
            incorr = 0

            # set frequency threshold, if a word frequency is lower than this threshold, it will be replaced by a
            # semantically similar and higher frequent word.
            freq_threshold = np.percentile(sorted(list(words_freq.values())), delta_thr)
            print('frequency threshold: ', freq_threshold)

            # set difference threshold where prediction difference before and after transform as adversarial examples
            diff_threshold = tune_gamma(orig_valid_preds, orig_valid_probs, bert_model, bert_tokenizer, dataset,
                                        words_freq, freq_threshold, words_list, distance_matrix, missed_words)
            print('differency threshold: ', diff_threshold)

            replaced_adv_texts, replaced_adv_text_pairs = [], []
            replaced_test_texts, replaced_test_text_pairs = [], []

            for adv_text, adv_text_pair in tqdm(zip(adv_texts, adv_text_pairs)):
                replaced_adv_texts.append(transform(adv_text, words_freq, freq_threshold, words_list, distance_matrix,
                                                    missed_words))
                replaced_adv_text_pairs.append(transform(adv_text_pair, words_freq, freq_threshold, words_list,
                                                        distance_matrix, missed_words) if adv_text_pair and
                                                                                        isinstance(adv_text_pair, str)
                                            else adv_text_pair)
            _, replaced_adv_preds, replaced_adv_probs, _ = \
                get_emb_preds(bert_model, bert_tokenizer,  args.batch_size,list(zip(replaced_adv_texts, replaced_adv_text_pairs)))

            for test_text, test_text_pair in tqdm(zip(test_texts, test_text_pairs)):
                replaced_test_texts.append(transform(test_text, words_freq, freq_threshold, words_list,
                                                    distance_matrix, missed_words))
                replaced_test_text_pairs.append(transform(test_text_pair, words_freq, freq_threshold, words_list,
                                                          distance_matrix, missed_words)
                                            if test_text_pair and isinstance(test_text_pair,
                                                                                 str) else test_text_pair)
            _, replaced_test_preds, replaced_test_probs, _ = \
                    get_emb_preds(bert_model, bert_tokenizer, args.batch_size, list(zip(replaced_test_texts, replaced_test_text_pairs)))

            for orig_adv_pred, orig_adv_prob, replaced_adv_pred, replaced_adv_prob in \
                    zip(orig_adv_preds, orig_adv_probs, replaced_adv_preds, replaced_adv_probs):
                if abs(orig_adv_prob[orig_adv_pred] - replaced_adv_prob[orig_adv_pred]) > diff_threshold:
                    corr += 1
                else:
                    incorr += 1

            for orig_test_pred, orig_test_prob, replaced_test_pred, replaced_test_prob in \
                    zip(orig_test_preds, orig_test_probs, replaced_test_preds, replaced_test_probs):
                if abs(orig_test_prob[orig_test_pred] - replaced_test_prob[orig_test_pred]) > diff_threshold:
                    incorr += 1
                else:
                    corr += 1

            print('Accuracy: ', corr / (corr + incorr))