import io
import json
import os
from collections import Iterable
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, XLNetTokenizer, XLNetModel, BartTokenizer, BartModel

class IMDBDataset:
    """IMDB dataset"""

    def __init__(self, path='./data/aclImdb', valid_size=0.2):
        """
        Create the IMDB dataset
        :param path: the directory of IMDB dataset
        :param valid_size: percentage of training set to use as validation set
        """
        self._path = path
        self.test_path = path + '/test'
        self.train_path = path + '/train'
        self._valid_size = valid_size
        self.num_labels = 2  
        self.train_batch_offset = 0
        self.train_idx = None
        self.val_idx = None
        (self.train_text, self.train_text_pair, self.train_y), (self.valid_text, self.valid_text_pair, self.valid_y), \
        (self.test_text, self.test_text_pair, self.test_y) = self.load_imdb(self._valid_size)
    
    def get_labels(self):
        return ["0", "1"]
    

    def read_text(self, path=None):
        """
        Read text from IMDB training or test directory
        :param path: the directory of train or test data
        :return: a list of texts and a list of their labels
        """
        if path== None:
            path = self.train_path
            
        pos_path = path + '/pos' 
        neg_path = path + '/neg'
        pos_files = [pos_path + '/' + x for x in sorted(os.listdir(pos_path)) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in sorted(os.listdir(neg_path)) if x.endswith('.txt')]

        pos_list = [io.open(x, 'r', encoding='utf-8', errors='ignore').read().replace('<br />', '') for x in pos_files]
        neg_list = [io.open(x, 'r', encoding='utf-8', errors='ignore').read().replace('<br />', '') for x in neg_files]
        data_list = pos_list + neg_list
        labels_list = [1] * len(pos_list) + [0] * len(neg_list)

        return data_list, labels_list

    def load_imdb(self, valid_size):
        """
        Load IMDB dataset from the pre-downloaded IMDB dataset directory
        :param path: the directory of IMDB dataset
        :param valid_size: percentage of training set to use as validation set
        :return: IMDB training, validation, and test sets
        """
        test_text, test_y = self.read_text(self.test_path) #test_path
        text, y = self.read_text(self.train_path)
        test_text_pair = [None] * len(test_y)

        # split original training set to a new training set and a validation set
        num_train = len(y)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        
        split = int(np.floor(valid_size * num_train))
        self.train_idx, self.valid_idx = indices[split:], indices[:split]

               
        train_text = [text[i] for i in self.train_idx]
        train_text_pair = [None] * len(self.train_idx)
        train_y = [y[i] for i in self.train_idx]
        
        valid_text = [text[i] for i in self.valid_idx]
        valid_text_pair = [None] * len(self.valid_idx)
        valid_y = [y[i] for i in self.valid_idx]       

        return (train_text, train_text_pair, train_y), \
           (valid_text, valid_text_pair, valid_y), \
            (test_text, test_text_pair, test_y)
        
    def get_train_val_idx(self):
        return self.train_idx, self.val_idx
    
class MnliDataset:
    """Multi-Genre Natural Language Inference (MultiNLI/Mnli) dataset"""

    def __init__(self, path='./data/multinli_1.0', valid_size=0.2):
        """
        Create the MultiNLI dataset
        :param path: the directory of MultiNLI dataset
        :param valid_size: percentage of training set to use as validation set
        """
        self._path = path
        self._valid_size = valid_size
        self.num_labels = 3
        self.train_idx = None
        self.val_idx = None
        (self.train_text, self.train_text_pair, self.train_y), (self.valid_text, self.valid_text_pair, self.valid_y), \
        (self.test_text, self.test_text_pair, self.test_y) = self.load_mnli(self._path, self._valid_size)
    
    def get_labels(self):
        return ['entailment', 'neutral', 'contradiction']
    def get_train_val_idx(self):
        return self.train_idx, self.val_idx
    
    def read_json(self, path):
        """
        Read jsonl files from MultiNLI training or dev file
        :param path: the directory of MultiNLI train or dev file
        :return: lists of text, text pair, and labels
        """
        data_info = []
        for line in open(path, 'r', encoding='utf-8', errors='ignore'):
            data_info.append(json.loads(line))

        labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        text = [data['sentence1'] for data in data_info if data['gold_label'] != '-']
        text_pair = [data['sentence2'] for data in data_info if data['gold_label'] != '-']
        y = [labels[data['gold_label']] for data in data_info if data['gold_label'] != '-']

        return text, text_pair, y

    def load_mnli(self, path, valid_size):
        """
        load MultiNLI dataset from the pre-downloaded MultiNLI dataset directory
        :param path: the directory of MultiNLI dataset
        :param valid_size: percentage of training set to use as validation set
        :return: MultiNLI training, validation, and test sets
        """
        test_path = path + '/multinli_1.0_dev_mismatched.jsonl'
        train_path = path + '/multinli_1.0_train.jsonl'

        test_text, test_text_pair, test_y = self.read_json(test_path)
        text, text_pair, y = self.read_json(train_path)

        # split original training set to a new training set and a validation set
        num_train = len(y)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_text = [text[i] for i in train_idx]
        train_text_pair = [text_pair[i] for i in train_idx]
        train_y = [y[i] for i in train_idx]
        valid_text = [text[i] for i in valid_idx]
        valid_text_pair = [text_pair[i] for i in valid_idx]
        valid_y = [y[i] for i in valid_idx]

        return (train_text, train_text_pair, train_y), \
               (valid_text, valid_text_pair, valid_y), \
               (test_text, test_text_pair, test_y)


#Data features
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
        
def convert_examples_to_features(texts, text_pairs, labels, label_names, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    
    label_map = {label : i for i, label in enumerate(label_names)}
    
#     text = np.asarray(examples)[:, 0].tolist()
#     text_pair = np.asarray(examples)[:, 1].tolist()
#     print("Label map:", label_map)
#     print(len(texts))
    features = []
    for i in range(0,len(texts)):
        tokens_a = tokenizer.tokenize(texts[i])

        tokens_b = None
#         if text_pairs is not None:
#             tokens_b = tokenizer.tokenize(text_pairs[i])
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#         else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        #print('max_seq_len:', max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        if isinstance(labels, Iterable):
            #label_id = label_map[str(labels[i])] #imdb
            label_id = labels[i] #mnli
        else:
            #label_id = label_map[str(labels)] #imdb
            label_id = labels #mnli

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=i #example.guid
                             ))
    return features    


    
  
    

# set models' parameters
bert_params = {
    'cls_pos': 0,
    'learning_rate': 5e-5,
    'model_class': BertModel,
    'tokenizer_class': BertTokenizer,
    'pretrained_model_name': 'bert-base-cased',
    'pretrained_file_path': './bert-base-cased-huggingface/',
    'output_hidden_states': True
}



roberta_params = {
    'cls_pos': 0,
    'learning_rate': 1e-5,
    'model_class': RobertaModel,
    'tokenizer_class': RobertaTokenizer,
    'pretrained_model_name': 'roberta-base',
    'pretrained_file_path': './roberta-base/',
    'output_hidden_states': True
}

xlnet_params = {
    'cls_pos': -1,
    'learning_rate': 2e-5,
    'model_class': XLNetModel,
    'tokenizer_class': XLNetTokenizer,
    'pretrained_model_name': 'xlnet-base-cased',
    'pretrained_file_path': './xlnet-base-cased/',
    'output_hidden_states': True,
    'return_dict': False
}

bart_params = {
    'cls_pos': -1,
    'learning_rate': 5e-6,
    'model_class': BartModel,
    'tokenizer_class': BartTokenizer,
    'pretrained_model_name': 'facebook/bart-base',
    'pretrained_file_path': './bart-base/',
    
}
