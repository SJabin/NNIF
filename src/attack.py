# From the repo https://github.com/NaLiuAnna/MDRE

import ast
import os
import random
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import TransfoXLTokenizer

import onmt_model
import paraphrase_scorer
import replace_rules


class Typo:
    """
    Pruthi et al
    Modified version for the paper: Combating Adversarial Misspellings with Robust Word Recognition.
    Paper link: https://arxiv.org/abs/1905.11268
    This attack could generate four types of character-level textual adversarial examples:
    1) swap: swapping two adjacent internal characters of a randomly chosen word.
    2) drop: removing an internal character of a randomly selected word.
    3) add: inserting a new character internally in a randomly chosen word.
    4) key replacement on a keyboard: substituting an internal character with an adjacent character in keyboards.
    """

    def __init__(self, predict, max_len, num_labels):
        """
        Create a Typo instance.
        :param predict: function used for prediction, the purpose of the attack is to change predictions of
                        original inputs.
        :param max_len: maximum length of tokens in an example.
        :param num_labels: number of classes in the dataset. 2 for IMDB, 3 for Mnli.
        """
        self.predict = predict
        self.max_length = max_len
        self.num_labels = num_labels

        self._punctuations = string.punctuation.replace("'", "") + ' '
        self._stopwords = set(stopwords.words('english')) | set(string.punctuation)

        self._keyboard_mapping = self.get_keyboard_neighbors()

    def generate(self, examples):
        """
        Generate character-level(Typo) adversarial examples
        :param examples: original examples
        :return: original, adversarial examples and their labels and predictions
        """
        # maximum number of typo attacks on original examples
        n_try = self.max_length // 2

        # get text, text_pair, labels, and their predictions
        text = np.asarray(list(examples.keys()))[:, 0].tolist()
        text_pair = np.asarray(list(examples.keys()))[:, 1].tolist()
        labels = list(examples.values())
        preds, _ = self.predict(list(zip(text, text_pair)))
        corr_pred = np.equal(labels, preds)

        adv_text, adv_text_pair, adv_preds = [], [], []
        iters = []  # number of attacks used to generate adversarial examples

        # generate adversarial examples
        for i, (sent1, sent2, pred) in enumerate(zip(text, text_pair, preds)):
            print(i, sent1.encode('utf8'), pred)  # for testing

            # only attack original examples of correct predictions
            if corr_pred[i]:
                i_advs = []
                org_text = (sent1, sent2)
                counts = []  # number of potential adversarial examples generated per attack
                sent1_token_len = min(len(word_tokenize(sent1)), self.max_length)

                # get a list of attacked examples on org_text as potential adversaries
                for num in range(n_try):
                    potential_advs = self.random_one_attack(org_text, sent1_token_len)
                    counts.append(len(potential_advs))
                    org_text = random.choice(potential_advs)
                    i_advs += potential_advs

                target_labels = list(range(self.num_labels))
                target_labels.remove(pred)

                # get predictions of examples in the attacked list and
                # take the first example whose prediction is in target_label as the adversarial example of org_text
                # i_iter is the attack number to generate this adversarial example
                advs_pred, _ = self.predict(i_advs)

                adv_ind = np.where([adv_pred in target_labels for adv_pred in advs_pred])[0]
                if adv_ind.size != 0:
                    adv_text.append(i_advs[adv_ind[0]][0])
                    adv_text_pair.append(i_advs[adv_ind[0]][1]) if i_advs[adv_ind[0]][1] else adv_text_pair.append(None)
                    adv_preds.append(advs_pred[adv_ind[0]])

                    i_iter = 0
                    tmp = adv_ind[0]
                    for _ in range(n_try):
                        tmp -= counts.pop(0)
                        i_iter += 1
                        if tmp < 0:
                            break

                    iters.append(i_iter + 1)
                else:
                    # no example in i_advs change prediction, attack failed
                    adv_text.append(None)
                    adv_text_pair.append(None)
                    adv_preds.append(None)
                    iters.append(None)
            else:
                # original example predicted wrong, no adversarial example
                adv_text.append(None)
                adv_text_pair.append(None)
                adv_preds.append(None)
                iters.append(None)

        adversarial_info = pd.DataFrame(list(zip(text, text_pair, labels, preds,
                                                 adv_text, adv_text_pair, adv_preds, iters)),
                                        columns=['orig_text', 'orig_text_pair', 'orig_label', 'orig_pred',
                                                 'adv_text', 'adv_text_pair', 'adv_pred', 'num iterations'])

        return adversarial_info

    def random_one_attack(self, text, sent1_token_len):
        """
        randomly choose one attack method to attack the text
        :param text: an example used for attack
        :param sent1_token_len: the length of tokens for the first sentences in text
        :return: potential adversarial examples, which are after the attack, of text
        """
        generators = [self.add_one_attack, self.key_one_attack, self.drop_one_attack, self.swap_one_attack]
        generator = random.choice(generators)
        tokens = word_tokenize(text[0] + ' ' + text[1]) if text[1] else word_tokenize(text[0])
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        advs = generator(tokens, sent1_token_len)

        return advs

    def add_one_attack(self, tokens, sent1_token_len, alphabetes='abcdefghijklmnopqrstuvwxyz'):
        """
        inserting a new character internally in a ramdomly chosen word
        :param tokens: tokens of an original examples
        :param sent1_token_len: the length of tokens for the first sentences in text
        :param alphabetes: 26 letters
        :return: a list of 10 randomly chosen examples after attack or tokens if 1000 times attacks failed
        """
        alphabets = [i for i in alphabetes] + [i for i in alphabetes.upper()]

        for _ in range(1000):
            idx = random.choice(range(len(tokens)))
            token = tokens[idx]

            # remove punctuation
            token = ''.join([ch for ch in token if ch not in self._punctuations])
            token = token.encode('ascii', 'ignore').decode('utf-8', 'ignore')  # remove all non-english characters

            if len(token) < 3: continue  # if number of characters in the token is less than 3, don't attack it
            if token in self._stopwords: continue
            if token.isnumeric(): continue

            adversary_words = [token[:i] + alpha + token[i:] for i in range(1, len(token)) for alpha in alphabets]
            tmp = [tokens[:idx] + [adv] + tokens[idx + 1:] for adv in adversary_words]
            adv_tokens = [(tokens[:sent1_token_len], tokens[sent1_token_len:]) for tokens in tmp]

            return random.sample([(' '.join(tokens1), ' '.join(tokens2)) for (tokens1, tokens2) in adv_tokens], 10)

        return [(' '.join(tokens[:sent1_token_len]), ' '.join(tokens[sent1_token_len:]))]

    def key_one_attack(self, tokens, sent1_token_len):
        """
        substituting an internal character with an adjacent character in keyboards
        :param tokens: tokens of an example used for attack
        :param sent1_token_len: the length of tokens for the first sentences in tokens
        :return: examples after attack or tokens if 1000 times attacks failed
        """
        for _ in range(1000):
            idx = random.choice(range(len(tokens)))
            token = tokens[idx]

            token = ''.join(ch for ch in token if ch not in self._punctuations)  # remove punctuation
            token = token.encode('ascii', 'ignore').decode('utf-8', 'ignore')  # remove all non-english characters

            remove_digits = str.maketrans('', '', string.digits)  # remove digits from token
            if len(token.replace("'", "").translate(remove_digits)) < 3: continue  # less than 3 characters in the token
            if token in self._stopwords: continue
            if token.isnumeric(): continue

            adversary_words = []
            for i in range(1, len(token) - 1):
                # if token[i].lower() in self._keyboard_mapping.keys():
                for key in self._keyboard_mapping[token[i].lower()]:
                    adversary_words.append(token[:i] + key + token[i + 1:])
            tmp = [tokens[:idx] + [adv] + tokens[idx + 1:] for adv in adversary_words]
            adv_tokens = [(tokens[:sent1_token_len], tokens[sent1_token_len:]) for tokens in tmp]

            return random.choices([(' '.join(tokens1), ' '.join(tokens2)) for (tokens1, tokens2) in adv_tokens],
                                  weights=[1] * len(adv_tokens), k=10)

        return [(' '.join(tokens[:sent1_token_len]), ' '.join(tokens[sent1_token_len:]))]

    def drop_one_attack(self, tokens, sent1_token_len):
        """
        removing an internal character of a randomly selected word
        :param tokens: tokens of an example used for attack
        :param sent1_token_len: the length of tokens for the first sentences in tokens
        :return: examples after attack or tokens if 1000 times attacks failed
        """
        for _ in range(1000):
            idx = random.choice(range(len(tokens)))
            token = tokens[idx]

            token = ''.join(ch for ch in token if ch not in self._punctuations)  # remove punctuation
            token = token.encode('ascii', 'ignore').decode('utf-8', 'ignore')  # remove all non-english characters

            if len(token) < 3: continue
            if token in self._stopwords: continue
            if token.isnumeric(): continue

            adversary_words = [token[:i] + token[i + 1:] for i in range(1, len(token) - 1)]
            tmp = [tokens[:idx] + [adv] + tokens[idx + 1:] for adv in adversary_words]
            adv_tokens = [(tokens[:sent1_token_len], tokens[sent1_token_len:]) for tokens in tmp]

            return [(' '.join(tokens1), ' '.join(tokens2)) for (tokens1, tokens2) in adv_tokens]

        return [(' '.join(tokens[:sent1_token_len]), ' '.join(tokens[sent1_token_len:]))]

    def swap_one_attack(self, tokens, sent1_token_len):
        """
        swapping two adjacent internal characters of a randomly chosen word
        :param tokens: tokens of an example used for attack
        :param sent1_token_len: the length of tokens for the first sentences in tokens
        :return: examples after attack or tokens if 1000 times attacks failed
        """
        for _ in range(1000):
            idx = random.choice(range(len(tokens)))
            token = tokens[idx]

            token = ''.join(ch for ch in token if ch not in self._punctuations)  # remove punctuation
            token = token.encode('ascii', 'ignore').decode('utf-8', 'ignore')  # remove all non-english characters

            if len(token) < 4: continue
            if token in self._stopwords: continue
            if token.isnumeric(): continue

            adversary_words = [token[:i] + token[i: i + 2][::-1] + token[i + 2:] for i in range(1, len(token) - 2)]
            tmp = [tokens[:idx] + [adv] + tokens[idx + 1:] for adv in adversary_words]
            adv_tokens = [(tokens[:sent1_token_len], tokens[sent1_token_len:]) for tokens in tmp]

            return [(' '.join(tokens1), ' '.join(tokens2)) for (tokens1, tokens2) in adv_tokens]

        return [(' '.join(tokens[:sent1_token_len]), ' '.join(tokens[sent1_token_len:]))]

    def get_keyboard_neighbors(self):
        """ Get neighbours of each letter on the keyboard. """
        # keyboard_mappings = defaultdict(lambda: [])
        keyboard_mappings = defaultdict(list)
        keyboard = ['qwertyuiop', 'asdfghjkl*', 'zxcvbnm***']
        row = len(keyboard)
        col = len(keyboard[0])

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        for i in range(row):
            for j in range(col):
                for k in range(4):
                    x_, y_ = i + dx[k], j + dy[k]
                    if (0 <= x_ < row) and (0 <= y_ < col):
                        if keyboard[x_][y_] == '*': continue
                        if keyboard[i][j] == '*': continue
                        keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])

        return keyboard_mappings


class SynonymsReplacement:
    """
    Alzantot
    This attack is a word-level attack which modified from the paper: Generating Natural Language Adersarial Examples.
    Here we use a different langauge model and do not allow neighbours drift which means we do not find the neighbours
    of a substitutions to improve efficiency.
    Paper link: https://www.aclweb.org/anthology/D18-1316/
    """

    def __init__(self, predict, data_name, max_seq_length, num_labels, max_vocab_size=20000, pop_size=20,
                 max_iters=100, temp=0.3):
        """
        Create a SynonymReplacement instance.
        :param predict: function used for prediction, the purpose of the attack is to change predictions of
                        original inputs
        :param data_name: dataset name, options: IMDB, Mnli
        :param max_seq_length: maximum length of sequences for the prediction model
        :param num_labels: number of classes in the dataset
        :param max_vocab_size: maximum number of words in the distance matrix
        :param pop_size: How many potential adversarial examples to generate in one iteration of the attack
        :param max_iters: maximum number of attacks
        :param temp: a parameter used to generate parents selection probability in crossover
        """
        self.vocab_file = os.path.join('./output', data_name, 'bert/synonym/aux_files/vocab.vocab')
        self.neighbours_file = os.path.join('./output', data_name, 'bert/synonym/aux_files/neighbours_info.csv')

        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels
        self.pop_size = pop_size
        self.max_iters = int(max_seq_length // 5)
        self.temp = temp

        self.lm_tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        self.predict = predict

        self.dictionary = self.load_dictionary()
        self.words_list = self.dictionary.word.values.tolist()
        self.words_ids = self.dictionary.id.values.tolist()

        self.neighbours_info = self.load_neighbours_info(self.neighbours_file)

    def load_dictionary(self):
        """ load a dictionary and sort by number of occurrences of words in the dataset """
        try:
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                vocab_words = f.read().split('\n')
        except FileNotFoundError:
            print("Couldn't find the vocabulary file, please run get_neighbours.py first.")

        # get the ids of words in the dictionary from the self.tokenizer
        temp = self.lm_tokenizer(vocab_words)['input_ids']
        token_ids = [id[0] if id else id for id in temp]

        dictionary = pd.DataFrame(list(zip(vocab_words, token_ids)), columns=['word', 'id'])

        return dictionary

    def load_neighbours_info(self, path):
        """ load the neighbour information for each word in the test set """
        try:
            neighbours_info = pd.read_csv(path)
        except FileNotFoundError:
            print("Couldn't find the neighbours file, please run get_neighbours.py first.")

        temp = neighbours_info['neighbours_list'].to_numpy()
        neighbours_list = [ast.literal_eval(line) for line in temp]

        try:
            examples = [ast.literal_eval(text) for text in neighbours_info['text'].to_numpy()]
        except:
            examples = neighbours_info['text'].to_numpy()

        labels = neighbours_info['label'].to_numpy()
        tokens_list = [ast.literal_eval(m) for m in neighbours_info['tokens'].to_numpy()]
        has_neighbours_list = [ast.literal_eval(m) for m in neighbours_info['has_neighbours'].to_numpy()]
        neighbours_len_list = [ast.literal_eval(m) for m in neighbours_info['neighbours_length'].to_numpy()]
        sent1_token_len_list = neighbours_info['length of text tokens'].to_numpy()

        if isinstance(examples, np.ndarray):
            texts = examples
            text_pairs = [None] * len(examples)
        else:
            texts = np.asarray(examples)[:, 0]
            text_pairs = np.asarray(examples)[:, 1]

        neighbours_info = pd.DataFrame(
            list(zip(texts, text_pairs, labels, tokens_list,
                     has_neighbours_list, neighbours_len_list, neighbours_list, sent1_token_len_list)),
            columns=['text', 'text_pair', 'label', 'tokens',
                     'has_neighbours', 'neighbours_length', 'neighbours_list', 'length of text tokens'])

        return neighbours_info

    def generate(self, examples):
        """
        generate word-level (synonym replacement) adversarial examples
        :param examples: original examples which include texts and labels
        :return: original, adversarial examples and their labels and predictions
        """
        orig_texts, orig_text_pairs, orig_labels, orig_preds = [], [], [], []
        adv_texts, adv_text_pairs, adv_preds = [], [], []
        sent1_token_len_list = []

        for i, (text, label) in enumerate(examples.items()):
            print(i, text[0].encode('utf8'), label)  # for testing

            if text[0] in self.neighbours_info.values:
                example = self.neighbours_info[self.neighbours_info.values == text[0]].iloc[0]
                x_orig = (example['text'], example['text_pair'])
                orig_label = example['label']
                x_orig_tokens = example['tokens']
                orig_pred, orig_softmax = self.predict([x_orig])
                orig_pred = orig_pred[0]
                text_tokens_len = example['length of text tokens']

                orig_texts.append(example['text'])
                orig_text_pairs.append(example['text_pair'])
                orig_labels.append(orig_label)
                orig_preds.append(orig_pred)
                sent1_token_len_list.append(text_tokens_len)

                target_label = list(range(self.num_labels))
                target_label.remove(orig_pred)

                # x_orig neighbours information
                has_neighbours = example['has_neighbours']  # which words in x_orig have neighbours
                neighbour_list = example['neighbours_list']  # neighbours list for has-neighbours-words
                neighbours_len = example['neighbours_length']  # neighbours length for has-neighbours-words

                # generate an adversarial example for x_orig
                x_adv = self.attack(x_orig_tokens, has_neighbours, neighbour_list, neighbours_len, target_label,
                                    orig_softmax, text_tokens_len)

                adv_texts.append(x_adv[0] if x_adv else None)
                adv_text_pairs.append(x_adv[1] if x_adv and x_adv[1] else None)
                adv_preds.append(x_adv[2] if x_adv else None)

        adversarial_info = pd.DataFrame(list(zip(orig_texts, orig_text_pairs, orig_labels, orig_preds,
                                                 adv_texts, adv_text_pairs, adv_preds)),
                                        columns=['orig_text', 'orig_text_pair', 'orig_label', 'orig_pred',
                                                 'adv_text', 'adv_text_pair', 'adv_pred'])

        return adversarial_info

    def attack(self, x_orig_tokens, has_neighbours, neighbours_list, neighbours_len, target_label, orig_softmax,
               text_tokens_len):
        """
        construct an adversarial example for x_orig_tokens
        :param x_orig_tokens: tokens of the x_orig example
        :param has_neighbours: a list of has-neighbours-words-indices in x_orig example
        :param neighbours_list: neighbours list of has-neighbours-words in x_orig example
        :param neighbours_len: a list of neighbours lengths for has-neighbours-words in x_orig example
        :param target_label: target label of an adversarial example
        :param orig_softmax: orig example prediction scores of each class (output of softmax)
        :param text_tokens_len: the length of tokens of the first sentence in x_orig example
        :return: an adversarial example if successfully attacked or None
        """
        # attack failed if no neighbours or less than three words have neighbours
        if not has_neighbours or len(has_neighbours) < 3 or max(neighbours_len) < 2 or \
                (len(has_neighbours) - neighbours_len.count(1)) < 2:
            return None

        # probability of being selected of a token based on the number of its neighbours
        w_select_probs = neighbours_len / np.sum(neighbours_len)

        pop, pop_preds = self.generate_population(x_orig_tokens, has_neighbours, neighbours_list, neighbours_len,
                                                  w_select_probs, target_label, orig_softmax, text_tokens_len)

        # generate the next generation through crossover
        for i in range(self.max_iters):
            pop_scores = pop_preds[:, target_label]

            # select an potential adversarial which has the highest prediction for target label
            # flatten pop_scores and sort them then get the first dimension of sorted elements
            sorted_dim1 = np.unravel_index(np.argsort(-pop_scores, axis=None), pop_scores.shape)[0]
            # remove duplicate elements from sorted_dim1
            pop_ranks = list(dict.fromkeys(sorted_dim1))
            top_attack = pop_ranks[0]

            # calculates the probability of each element in pop that being selected as parents in the cross over
            logits = np.exp(np.sum(pop_scores, axis=1) / self.temp) if pop_scores.ndim == 2 else \
                np.exp(pop_scores) / self.temp
            select_probs = logits / np.sum(logits)

            # return an example in pop as an adversarial example if its prediction is in target labels
            if np.argmax(pop_preds[top_attack, :]) in target_label:
                text = ' '.join(pop[top_attack][:text_tokens_len])
                text_pair = ' '.join(pop[top_attack][text_tokens_len:])
                return text, text_pair, np.argmax(pop_preds[top_attack, :])

            # select an example which has the highest probability of target labels in pop
            elite = [pop[top_attack]]
            elite_preds = [pop_preds[top_attack]]

            # parents indices
            parent1_idx = np.random.choice(select_probs.size, size=max(select_probs.size - 1, 1), p=select_probs)
            parent2_idx = np.random.choice(select_probs.size, size=max(select_probs.size - 1, 1), p=select_probs)

            # cross over children from parents and replace some words in children to their synonyms
            childs = [self.crossover(pop[parent1_idx[i]], pop[parent2_idx[i]]) for i in range(select_probs.size - 1)]
            tmp = [self.perturb(x, x_orig_tokens, has_neighbours, neighbours_list, neighbours_len, w_select_probs,
                              target_label, orig_softmax) for x in childs]
            tmp1 = [x for x in tmp if x != []]
            # perturbed_childs = np.concatenate(
            #     [self.perturb(x, x_orig_tokens, has_neighbours, neighbours_list, neighbours_len, w_select_probs,
            #                   target_label, orig_softmax) for x in childs])
            perturbed_childs = np.concatenate(tmp1)

            # Select top self.num examples in perturbed_childs whose probability on target labels are higher than
            # the original examples
            childs, childs_preds = self.select_best_preds(perturbed_childs, x_orig_tokens, target_label, orig_softmax,
                                                          len(childs), text_tokens_len)

            pop = elite + childs
            pop_preds = np.r_[elite_preds, childs_preds]

        return None

    def crossover(self, x1, x2):
        """
        crossing over x1, x2 that results in a child
        :param x1: parent 1
        :param x2: parent 2
        :return: child of x1, x2
        """
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def generate_population(self, x_orig_tokens, has_neighbours, neighbour_list, neighbours_len, w_select_probs,
                            target_label, orig_softmax, text_tokens_len):
        """
        one attack on x_orig_tokens, generate a number of potential adversarial examples (pop)
        :param x_orig_tokens: tokens of the original example for this attack
        :param has_neighbours: a list of has-neighbours-words-indices in x_orig_tokens and x_orig
        :param neighbour_list: neighbours list of has-neighbours-words in x_orig_tokens and x_orig
        :param neighbours_len: a list of neighbours lengths for has-neighbours-words in x_orig_tokens and x_orig
        :param w_select_probs: probability of each token being attacked
        :param target_label: target label of an adversarial example
        :param orig_softmax: orig example prediction scores of each class (output of softmax)
        :param text_tokens_len: the length of tokens of the first sentence in x_orig_tokens and x_orig
        :return: number of potential adversarial examples (pop)
        """
        new_x_tokens = []
        has_neighbour_len = len(has_neighbours)
        select_idx = []
        for _ in range(min(self.pop_size, has_neighbour_len - neighbours_len.count(1))):
            rand_idx = np.random.choice(has_neighbour_len, 1, p=w_select_probs)[0]

            for _ in range(1000):  # don't use while loop in order to avoid infinite loops
                if (rand_idx in select_idx or neighbours_len[rand_idx] == 1) and len(
                        select_idx) < has_neighbour_len - neighbours_len.count(1):
                    # The conition above has a quick hack to prevent getting stuck in infinite loop while processing
                    # too short examples and all words `excluding articles` have been already replaced and still
                    # no-successful attack found. a more elegent way to handle this could be done in attack to abort
                    # early based on the status of all population members or to improve select_best_replacement by
                    # making it schocastic.
                    rand_idx = np.random.choice(has_neighbour_len, 1, p=w_select_probs)[0]

            select_idx.append(rand_idx)
            replace_list = [self.words_list[i] for i in neighbour_list[rand_idx]]
            replace_idx = has_neighbours[rand_idx]
            new_x_tokens.extend([self.do_replace(x_orig_tokens, replace_idx, w) for w in replace_list if
                                 x_orig_tokens[replace_idx] != w])

        return self.select_best_preds(new_x_tokens, x_orig_tokens, target_label, orig_softmax, self.pop_size,
                                      text_tokens_len)

    def perturb(self, x_cur_tokens, x_orig_tokens, has_neighbours, neighbours_list, neighbours_len, w_select_probs,
                target_label, orig_scoure):
        """ Make an attack: replace a word in the example with its synonyms. """

        has_neighbour_len = len(has_neighbours)

        rand_idx = np.random.choice(has_neighbour_len, 1, p=w_select_probs)[0]

        for _ in range(1000):  # don't use while loop in order to avoid infinite loops
            if (x_cur_tokens[rand_idx] != x_orig_tokens[rand_idx] and np.sum(x_orig_tokens != x_cur_tokens) < np.sum(
                    np.sign(w_select_probs))) or neighbours_len[rand_idx] == 1:
                # The conition above has a quick hack to prevent getting stuck in infinite loop while processing too
                # short examples and all words `excluding articles` have been already replaced and still
                # no-successful attack found. a more elegent way to handle this could be done in attack to abort
                # early based on the status of all population members or to improve select_best_replacement by making
                # it schocastic.
                rand_idx = np.random.choice(has_neighbour_len, 1, p=w_select_probs)[0]

        replace_list = [self.words_list[i] for i in neighbours_list[rand_idx]]
        replace_idx = has_neighbours[rand_idx]

        return [self.do_replace(x_cur_tokens, replace_idx, w) for w in replace_list if x_orig_tokens[replace_idx] != w]

    def select_best_preds(self, tokens_list, x_orig_tokens, target_label, orig_softmax, num, text_token_len):
        """
        select maximum top num examples in tokens_list whose probability on target labels are higher than the original
        :param tokens_list: a tokens list, each element at least changes one original token to its synonym
        :param x_orig_tokens: tokens of the original example
        :param target_label: target label of an adversarial example
        :param orig_softmax: orig example prediction scores of each class (output of softmax)
        :param num: maximum number of examples will be selected
        :param text_token_len: the length of tokens of the first sentence in x_orig
        :return: top num examples and their predictions
        """
        # get the predicted scores of each class for the tokens_list
        if isinstance(tokens_list, np.ndarray):
            tokens_list = tokens_list.tolist()
        text = [' '.join(tokens[:text_token_len]) for tokens in tokens_list]
        text_pair = [' '.join(tokens[text_token_len:]) for tokens in tokens_list]
        _, text_softmax = self.predict(list(zip(text, text_pair)))

        text_scores = text_softmax[:, target_label]
        orig_score = orig_softmax[0, target_label]

        text_scores = text_scores - orig_score

        pop_list = []
        pop_preds = []

        # flatten text_scores and sort them then get the first dimension of sorted elements
        sorted_dim1 = np.unravel_index(np.argsort(-text_scores, axis=None), text_scores.shape)[0]
        # remove duplicate elements from sorted_dim1
        sorted_indices = list(dict.fromkeys(sorted_dim1))
        for ind in sorted_indices[:num]:
            if np.any(text_scores[ind] > 0):
                pop_list.append(tokens_list[ind])
                pop_preds.append(text_softmax[ind].tolist())
            else:
                pop_list.append(x_orig_tokens)
                pop_preds.append(orig_softmax.squeeze().tolist())

        return pop_list, np.asarray(pop_preds)

    def select_best_replacement(self, pos, x_cur_tokens, x_orig_tokens, target_label, replace_list, orig_scoure):
        """ select the most effective replacement to word at pos in x_cur_tokens with the words in replace_list """
        new_x_tokens = [self.do_replace(x_cur_tokens, pos, w) for w in replace_list if x_orig_tokens[pos] != w]
        new_x_text = [' '.join(tokens) for tokens in new_x_tokens]
        _, new_x_preds = self.predict(new_x_text)

        new_x_scores = new_x_preds[:, target_label]

        orig_score = orig_scoure[0, target_label]

        new_x_scores = new_x_scores - orig_score

        if np.max(new_x_scores) > 0:
            return new_x_tokens[np.argsort(new_x_scores)[-1]]

        return x_cur_tokens

    def do_replace(self, x_cur, pos, new_word):
        """ replace the word at pos in x_cur with new_word """
        x_new = x_cur.copy()
        x_new[pos] = new_word

        return x_new









