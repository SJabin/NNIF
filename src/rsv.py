import numpy as np
import pickle
import os
import re
import rsv_data_utils
#from utils import build_dict
import argparse
import json
from collections import defaultdict
import torch
from tqdm import  tqdm
import math
import random
from utils import convert_examples_to_features
from rsv_data_utils import load_dictionary,load_dist_mat, _softmax
from rsv_synonym_selector import EmbeddingSynonym,WordNetSynonym
from sklearn.metrics import accuracy_score, f1_score, classification_report
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


def build_embs(args, train_texts, train_text_pairs, tokenizer, device):
   
    if not os.path.exists(os.path.join(args.dataset_path, 'aux_files')):
        os.makedirs(os.path.join(args.dataset_path, 'aux_files'))

    # Generate dictionary
    print('orig_dic & orig_inv_dic not exist, build and save the dict...')

    orig_dic, orig_inv_dic, _ = rsv_data_utils.build_dict(train_texts, train_text_pairs, tokenizer, args.vocab_size, data_dir=args.dataset_path, max_length=args.max_length, device=device)
    print("tokens in the train data dictionary", len(orig_dic))#, len(orig_inv_dic))
    with open(os.path.join(args.dataset_path, 'aux_files', 'orig_dic_%s.pkl' % (args.dataset_name)), 'wb') as f:
        pickle.dump(orig_dic, f, protocol=4)
    with open(os.path.join(args.dataset_path, 'aux_files', 'orig_inv_dic_%s.pkl' % (args.dataset_name)), 'wb') as f:
        pickle.dump(orig_inv_dic, f, protocol=4)

    # Calculate the distance matrix
    print('small dist counter not exists, create and save...')
    dist_mat = rsv_data_utils.compute_dist_matrix(orig_dic, args.dataset_name, vocab_size=args.vocab_size, data_dir=args.dataset_path)   
    
    print('dist matrix created!', dist_mat.shape)
    
    if args.vocab_size > len(dist_mat):
        args.vocab_size = len(dist_mat)
        
    small_dist_mat =rsv_data_utils.create_small_embedding_matrix(dist_mat, args.vocab_size, threshold=1.5, retain_num=50)
    print('small dist counter created!')
    
    np.save(os.path.join(args.dataset_path, 'aux_files', 'small_dist_counter_%s.npy' % (args.dataset_name)), small_dist_mat)


    
    print('embeddings glove not exists, creating...')
    glove_model = rsv_data_utils.loadGloveModel('./Vectors/glove.840B.300d.txt')
    glove_embeddings, _ =rsv_data_utils.create_embeddings_matrix(glove_model, orig_dic, dataset=args.dataset_name, data_dir=args.dataset_path)
    print("embeddings glove created!")
    np.save(os.path.join(args.dataset_path, 'aux_files', 'embeddings_glove_%s.npy' % (args.dataset_name)), glove_embeddings)


class Detector_RDSU:
    def __init__(self, args, clean_texts, clean_labels, adv_texts, adv_preds, clean_text_pairs=[],adv_text_pairs=[]):
        self.randomrate = args.randomrate
        self.votenum = args.votenum
        self.fix_thr = args.fixrate
        self.n_neighbors = args.max_candidates#6
        self.threshold = 0.5
        print("loading dictionary")
        self.vocab, self.inv_vocab = rsv_data_utils.load_dictionary(args.dataset_name, args.vocab_size, data_dir=args.dataset_path)
        print("loading dist matrix")
        self.dist_mat = rsv_data_utils.load_dist_mat(args.dataset_name, args.vocab_size, data_dir=args.dataset_path)

        print("Generating embedding synonym")
        self.emb_synonym_selector = EmbeddingSynonym(self.n_neighbors, self.vocab, self.inv_vocab, self.dist_mat, threshold=self.threshold)

        print("Generating wordnet synonym")
        self.wordnet_synonym_selector = WordNetSynonym(self.vocab, self.inv_vocab)
        
        #with open(args.adv_path, "rb") as handle:
        self.clean_examples = clean_texts
        #print(clean_text_pairs)
        self.clean_pairs = clean_text_pairs if len(clean_text_pairs)!= 0 else None
        self.clean_labels = clean_labels
        self.adv_examples = adv_texts#pickle.load(handle)
        self.adv_pairs = adv_text_pairs if len(adv_text_pairs)!=0 else None
        self.adv_labels  = adv_preds
        self.dataset = args.dataset_name
        self.neighborslist = {}

    def transfer(self, text):
        input_seq = text.split()
        masknum = int((len(input_seq)*self.randomrate)//1)
        N = range(len(input_seq))
        replace_idx = random.sample(N,masknum)
        replaced_idx = []
        replacenum = 0
        for idx in replace_idx:
            word = input_seq[idx]
            if not (word  in self.vocab) :
                continue
            if  self.vocab[word] <self.fix_thr*len(self.vocab) :
                continue
            if word in self.neighborslist:
                neighbors = self.neighborslist[word]
            else:
                neighbors = list(set(self.wordnet_synonym_selector.get_word_net_synonyms(word) + self.emb_synonym_selector.find_synonyms(word) ))
            filterneighbors =[]
            for w in neighbors:
                if w in self.vocab and self.vocab[w]<20000:
                    filterneighbors.append(w)
            neighbors =  filterneighbors       
            if len(neighbors) > 0:
                rep = random.choice(neighbors)
                input_seq[idx] = rep
                replaced_idx.append((word, rep, idx))
                replacenum += 1
        return " ".join(input_seq)

    def transfer_all_examples(self,save_path):
        transfer_examples = []
        #for adv in self.adv_examples:
        for j in range(0, len(self.adv_examples)):
            clean_transfer_list = []
            perturbed_transfer_list = []
            clean_transfer_pair_list = []
            perturbed_transfer_pair_list = []
            clean_transfer_label_list =[]
            perturbed_transfer_label_list = []
            clean_text = self.clean_examples[j]
            clean_text_pair = self.clean_pairs[j] if self.dataset =='Mnli' else None
            perturbed_text_pair = self.adv_pairs[j] if self.dataset =='Mnli' else None
            perturbed_text = self.adv_examples[j]

            for i in range (self.votenum):
                clean_transfer_list.append(self.transfer(clean_text)) # replacing a potential adv word with synonym
                perturbed_transfer_list.append(self.transfer(perturbed_text))
                clean_transfer_label_list.append(self.clean_labels[j])
                perturbed_transfer_label_list.append(self.adv_labels[j])

            if self.dataset=='Mnli':
                for i in range (self.votenum):
                    clean_transfer_pair_list.append(self.transfer(clean_text_pair))
                    perturbed_transfer_pair_list.append(self.transfer(perturbed_text_pair))


            transfer_examples.append(
                {
                    "clean_text": clean_text,
                    "clean_text_pair": clean_text_pair,
                    "perturbed_text": perturbed_text,
                    "perturbed_text_pair": perturbed_text_pair,
                    "clean_transfer_list": clean_transfer_list,
                    "clean_transfer_pair_list":clean_transfer_pair_list,
                    "perturbed_transfer_list": perturbed_transfer_list,
                    "perturbed_transfer_pair_list":perturbed_transfer_pair_list,
                    "clean_transfer_label_list":clean_transfer_label_list,
                    "perturbed_transfer_label_list": perturbed_transfer_label_list,
                    "clean_label": self.clean_labels[j], #adv["clean_label"],
                    "perturbed_label": self.adv_labels[j],# adv["perturbed_lable"],
                }

            )
        with open(save_path, "wb") as handle:
            pickle.dump(transfer_examples, handle)
        return transfer_examples

class EVAL_RDSU:
    def __init__(self,args, model, tokenizer, num_labels = 2, label_names = ["0","1"], device="cpu"):
        self.args = args
        self.num_labels = num_labels
        self.label_names = label_names
        self.model = model #self.load_model(args.modeltype)
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = args.dataset_name
        self.model.eval()

    def query(self, sentences, sentence_pairs=[], labels=0, usesoftmax =False):
        if labels == 0:
            labels = [0 for l in range(0, len(sentences))]
        #print("in query. Sentence len:", len(sentences))
                                      
        features = convert_examples_to_features(sentences, sentence_pairs, labels, self.label_names, self.args.max_length, self.tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(self.device)   
        
        self.model.eval()
        with torch.no_grad():
            _,logits,  _, _ = self.model(input_ids, input_mask, segment_ids, label_ids)

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
    
        if usesoftmax:
            logits =_softmax(logits)
        return logits, predictions          
    
    def eval_all_examples(self, example_list, data_dir ="./", overall = False):
        #transfered texts are the altered texts by the detector algorithm
        votenum = len(example_list[0]["clean_transfer_list"])

        # load data 
        clean_text_list = []
        clean_text_pair_list = []
        perturbed_text_list = []
        perturbed_text_pair_list = []
        clean_label_list = []
        perturbed_label_list = []
        
        clean_transfer_list_total = []
        perturbed_transfer_list_total = []
        clean_transfer_list_pair_total = []
        perturbed_transfer_list_pair_total = []
        clean_transfer_label_list_total = []
        perturbed_transfer_label_list_total = []
        
        for exp in example_list:
            clean_text_list.append(exp["clean_text"])
            clean_text_pair_list.append(exp["clean_text_pair"])
            perturbed_text_list.append(exp["perturbed_text"])
            perturbed_text_pair_list.append(exp["perturbed_text_pair"])
            clean_transfer_list_total += exp["clean_transfer_list"] #+=
            perturbed_transfer_list_total += exp["perturbed_transfer_list"] #+=
            clean_transfer_label_list_total += exp["clean_transfer_label_list"] #+=
            perturbed_transfer_label_list_total += exp["perturbed_transfer_label_list"] #+=
            clean_transfer_list_pair_total += exp["clean_transfer_pair_list"] #+=
            perturbed_transfer_list_pair_total += exp["perturbed_transfer_pair_list"]
            clean_label_list.append(exp["clean_label"])
            perturbed_label_list.append(exp["perturbed_label"])
            
            
        # load data 
        split_batchnum = 100
        each_batctnum = math.ceil(len(clean_text_list)/split_batchnum) #clean_text_list
        #print("each_batctnum:", each_batctnum)
        orig_pre = []  # prediction on ori result
        adv_pre = []  # prediction on adv result
        
        # query
        print("Query on orig and adv texts:")
        if not os.path.exists(os.path.join(data_dir,"orig_predictions.npy")):
            #
            for batchnum in range(split_batchnum):
                if (batchnum*each_batctnum) > (len(clean_text_list)-1):
                    break
                
                if self.dataset=='Mnli':
                    _,temp_orig_pre = self.query(clean_text_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], \
                                                 clean_text_pair_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], \
                                                 clean_label_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], usesoftmax =False)#clean_label_list
                    _,temp_adv_pre = self.query(perturbed_text_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],\
                                                perturbed_text_pair_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],\
                                                perturbed_label_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], usesoftmax =False)#perturbed_label_list
                else:
                    _,temp_orig_pre = self.query(clean_text_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], None, \
                                                 clean_label_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], usesoftmax =False)
                    _,temp_adv_pre = self.query(perturbed_text_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],None, \
                                                perturbed_label_list[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], usesoftmax =False)
                orig_pre += temp_orig_pre.tolist()
                adv_pre += temp_adv_pre.tolist()

            np.save(os.path.join(data_dir,"orig_predictions.npy"), orig_pre)
            np.save(os.path.join(data_dir,"adv_predictions.npy"), adv_pre)
        else:
            orig_pre = np.load(os.path.join(data_dir,"orig_predictions.npy"))
            adv_pre = np.load(os.path.join(data_dir,"adv_predictions.npy"))
        #del clean_text_pair_list, perturbed_text_pair_list
        del perturbed_text_list, perturbed_label_list
        
        adv_success = 0
        orig_success = 0

        acc_after_attack = 0
        for i in range(len(example_list)):
            if orig_pre[i]==clean_label_list[i]:
                orig_success +=1
                if adv_pre[i]!=orig_pre[i]:
                    adv_success += 1
                else :
                    acc_after_attack += 1
        
        
        orig_acc = orig_success/len(example_list)
        adv_acc = acc_after_attack/len(example_list)
        
        print("acc on clean {}".format(orig_acc))
        print("acc on adv {}".format(adv_acc))
        del orig_acc, adv_acc
        
        print("Query on transfer clean and adv texts: ")#,os.path.join(data_dir,"mul_transfer_orig_probs.npy")
        if not os.path.exists(os.path.join(data_dir,"mul_transfer_orig_probs.npy")):
            mul_transfer_adv_label = []
            mul_transfer_orig_prob = []
            mul_transfer_adv_prob = []
            mul_transfer_orig_pre = []
            mul_transfer_adv_pre = []
            
            split_batchnum = 100
            each_batctnum = math.ceil(len(clean_transfer_list_total)/split_batchnum)          
            #n_batch = math.ceil(len(clean_transfer_list_total) / split_batchnum)

            for batchnum in tqdm(range(0,split_batchnum)):
                #begin =n_batch * batch
                #end = min(begin+n_batch, len(clean_transfer_list_total))
                if (batchnum*each_batctnum) > (len(clean_text_list)-1):
                    break
                    
                if self.dataset=='Mnli':
                    temp_mul_transfer_orig_prob,temp_mul_transfer_orig_pre =self.query( \
                                clean_transfer_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], \
                                clean_transfer_list_pair_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], \
                                clean_transfer_label_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], usesoftmax =False) #clean_transfer_label_list_total[begin:end] 
                    temp_mul_transfer_adv_prob,temp_mul_transfer_adv_pre = self.query( \
                                perturbed_transfer_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], \
                                perturbed_transfer_list_pair_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], \
                                perturbed_transfer_label_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],usesoftmax =False) #perturbed_transfer_label_list_total[begin:end]
                else:
                    temp_mul_transfer_orig_prob,temp_mul_transfer_orig_pre =self.query( \
                                clean_transfer_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], None, \
                                clean_transfer_label_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],usesoftmax =False) 
                    temp_mul_transfer_adv_prob, temp_mul_transfer_adv_pre=self.query( \
                                perturbed_transfer_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))], None, \
                                perturbed_transfer_label_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))],usesoftmax =False)
                mul_transfer_adv_label += perturbed_transfer_label_list_total[int(batchnum*each_batctnum):int(each_batctnum*(batchnum+1))]
                mul_transfer_orig_pre += temp_mul_transfer_orig_pre.tolist()
                mul_transfer_orig_prob += temp_mul_transfer_orig_prob.tolist()
                mul_transfer_adv_pre += temp_mul_transfer_adv_pre.tolist()
                mul_transfer_adv_prob += temp_mul_transfer_adv_prob.tolist()
            
            #mul_transfer_orig_prob, mul_transfer_orig_pre = self.query(clean_transfer_list_total, None, 0,usesoftmax =True)
            #mul_transfer_adv_prob, mul_transfer_adv_pre = self.query(perturbed_transfer_list_total, None, 0,usesoftmax =True)
            np.save(os.path.join(data_dir,"mul_transfer_adv_label.npy"), mul_transfer_adv_label)
            np.save(os.path.join(data_dir,"mul_transfer_orig_predictions.npy"), mul_transfer_orig_pre)
            np.save(os.path.join(data_dir,"mul_transfer_orig_probs.npy"), mul_transfer_orig_prob)
            np.save(os.path.join(data_dir,"mul_transfer_adv_predictions.npy"), mul_transfer_adv_pre)
            np.save(os.path.join(data_dir,"mul_transfer_adv_probs.npy"), mul_transfer_adv_prob)
        else:
            mul_transfer_adv_label = np.load(os.path.join(data_dir,"mul_transfer_adv_label.npy") )
            mul_transfer_orig_pre = np.load(os.path.join(data_dir,"mul_transfer_orig_predictions.npy") )
            mul_transfer_orig_prob = np.load(os.path.join(data_dir,"mul_transfer_orig_probs.npy"))
            mul_transfer_adv_pre = np.load(os.path.join(data_dir,"mul_transfer_adv_predictions.npy"))
            mul_transfer_adv_prob = np.load(os.path.join(data_dir,"mul_transfer_adv_probs.npy"))
        del clean_transfer_list_total, perturbed_transfer_list_total
        
        #calculate the voted labels
        transfer_orig_pre_list = []
        transfer_adv_pre_list = []
        for i in range(len(example_list)):
            transfer_orig_prob=np.sum(mul_transfer_orig_prob[i*votenum:i*votenum+votenum],axis=0)
            transfer_orig_prob =transfer_orig_prob/votenum
            transfer_orig_pre = np.argmax(transfer_orig_prob)
            transfer_orig_pre_list.append(transfer_orig_pre)
            
            transfer_adv_prob=np.sum(mul_transfer_adv_prob[i*votenum:i*votenum+votenum],axis=0)
            transfer_adv_prob =transfer_adv_prob/votenum
            transfer_adv_pre = np.argmax(transfer_adv_prob)
            transfer_adv_pre_list.append(transfer_adv_pre)
            del transfer_orig_prob, transfer_orig_pre, transfer_adv_pre, transfer_adv_prob
            
        #del mul_transfer_orig_prob, mul_transfer_adv_prob
        
        orig_corr_trans_corr = 0
        orig_corr_trans_incorr = 0
        orig_incorr_trans_corr = 0 
        orig_incorr_trans_incorr = 0 

        t_p = 0
        f_p = 0
        f_n = 0
        res_orig = 0
        for i in range(len(example_list)):
            # orig_pre = model(clean text)  clean_label_list orig_lable
            if orig_pre[i]==clean_label_list[i] and adv_pre[i]!=clean_label_list[i] and adv_pre[i]!= transfer_adv_pre_list[i]:
                t_p+=1
            if orig_pre[i]==clean_label_list[i] and adv_pre[i]==clean_label_list[i] and adv_pre[i]!= transfer_adv_pre_list[i]:
                f_p+=1
            if orig_pre[i]==clean_label_list[i] and adv_pre[i]!=clean_label_list[i] and adv_pre[i]== transfer_adv_pre_list[i]:
                f_n+=1
            
            if  orig_pre[i]==clean_label_list[i] and transfer_orig_pre_list[i]==clean_label_list[i]:
                orig_corr_trans_corr +=1 #success orig
            elif orig_pre[i]==clean_label_list[i] and transfer_orig_pre_list[i]!=clean_label_list[i]:
                orig_corr_trans_incorr +=1 
            elif orig_pre[i]!=clean_label_list[i] and transfer_orig_pre_list[i]==clean_label_list[i]:
                orig_incorr_trans_corr +=1
            else:
                #orig_pre[i]!=clean_label_list[i] and transfer_orig_pre_list[i]!=clean_label_list[i]
                orig_incorr_trans_incorr +=1 #success adv

            if transfer_orig_pre_list[i]==clean_label_list[i]:
                res_orig += 1 #correctly predicted the original record --->these sentences are detected as original
        #del transfer_orig_pre_list
        
        #print(res_orig)
        assert(f_n+t_p == adv_success)    

        f1 =  (2 * t_p) / (2 * t_p + f_p + f_n) if 2 * t_p + f_p + f_n > 0 else 0
        tpr=t_p / adv_success if adv_success > 0 else 0
        
        transfer_orig_acc = (orig_corr_trans_corr+orig_incorr_trans_corr)/len(example_list)
        print("LM's acc on transfered clean text: {}".format(transfer_orig_acc)) #(RDSU) transfer acc on clean
        # print("(RDSU)  orig_corr_trans_corr : {}".format(orig_corr_trans_corr/len(example_list)))
        # print("(RDSU)  orig_corr_trans_incorr : {}".format(orig_corr_trans_incorr/len(example_list)))
        # print("(RDSU)  orig_incorr_trans_corr : {}".format(orig_incorr_trans_corr/len(example_list)))
        # print("(RDSU)  orig_incorr_trans_incorr : {}".format(orig_incorr_trans_incorr/len(example_list)))


        orig_incorr = 0  # orig predict fail  
        adv_to_orig = 0  #transfer detects adv
        adv_to_adv = 0  #transfer doesnt detect adv
        orig_to_adv = 0 #transfer falsely detect orig as adv
        orig_to_orig = 0 #transfer detects orig
        res = 0
        for i in range(len(example_list)):
            if clean_label_list[i]!=orig_pre[i]:
                orig_incorr += 1
            elif adv_pre[i]==clean_label_list[i] and transfer_adv_pre_list[i]!=clean_label_list[i]:
                orig_to_adv +=1
            elif adv_pre[i]!=clean_label_list[i] and transfer_adv_pre_list[i]!=clean_label_list[i]:
                adv_to_adv +=1
            elif adv_pre[i]!=clean_label_list[i] and transfer_adv_pre_list[i]==clean_label_list[i]:
                adv_to_orig +=1
            elif adv_pre[i]==clean_label_list[i] and transfer_adv_pre_list[i]==clean_label_list[i]:
                orig_to_orig +=1

            if transfer_adv_pre_list[i]==clean_label_list[i]:
                res += 1
        #del transfer_adv_pre_list
        transfer_adv_acc = (orig_to_orig+adv_to_orig)/len(example_list)
        
        # print("(RDSU)  orig_incorr : {}".format(orig_incorr/len(example_list)))
        #print("(RDSU)  adv_to_orig : {}".format(adv_to_orig/len(example_list)))
        #print("(RDSU)  adv_to_adv : {}".format(adv_to_adv/len(example_list)))
        #print("(RDSU)  orig_to_adv : {}".format(orig_to_adv/len(example_list)))
        #print("(RDSU)  orig_to_orig : {}".format(orig_to_orig/len(example_list)))
        print("LM's acc on transfered adv text: {}".format(transfer_adv_acc)) #(RDSU)  transfer  acc on adv
        print("LM's acc on transfered adv text including orig failed predictions: {}".format(res/len(example_list))) #"(RDSU) restore acc on adv : {}"
        print("F1 score (restore): {}".format(f1))

        #If the voted label is not consistent with the prediction label for the input text,
        # RS&V would treat the input text as adversarial example and output the voted label.
        detect_orig_cor, detect_orig_inc = 0, 0
        detect_adv_cor, detect_adv_inc = 0, 0
        for i in range(len(example_list)):
            if  orig_pre[i]==transfer_orig_pre_list[i]:
                detect_orig_cor +=1
            else:
                detect_adv_inc +=1
            if adv_pre[i]==transfer_adv_pre_list[i]:
                detect_orig_inc +=1
            else:
                detect_adv_cor +=1
        detect_acc = (detect_adv_cor + detect_orig_cor) / (detect_orig_cor + detect_adv_inc + detect_orig_inc + detect_adv_cor)     
        print("Detection accuracy:", detect_acc)
        
        return transfer_orig_acc,transfer_adv_acc,t_p,f_p,f_n,f1,tpr,detect_acc
    
# class TargetBert(object):
#     """The BERT model attacked by adversary."""

#     def __init__(self, args, num_labels, device):
#         self.num_labels = num_labels
#         self.max_seq_length = args.max_seq_length
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#         self.device = device

#         # Load a trained model and config that you have fine-tuned
#         output_model_file = os.path.join(args.output_dir, "epoch"+str(int(args.num_train_epochs)-1)+"_"+WEIGHTS_NAME)
#         output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
#         config = BertConfig(output_config_file)
#         model = BertForClassifier(config, num_labels=num_labels)
#         model.to(device)
#         model.load_state_dict(torch.load(output_model_file))
#         self.model = model
#         self.model.eval()

#     def query(self, sentences, labels,usesoftmax =False):
#         examples = []
#         for (i, sentence) in enumerate(sentences):
#             guid = "%s-%s" % ("dev", i)
#             examples.append(
#                 #InputExample(guid=guid, text_a=sentence, text_b=None, label=0, flaw_labels=None))
#                 InputExample(guid=guid, text_a=sentence, text_b=None, label=labels[i], flaw_labels=None))
#         features = convert_examples_to_features(
#             examples, self.max_seq_length, self.tokenizer)
#         input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
#         input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
#         segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)
#         label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(self.device)          

#         with torch.no_grad():
#             tmp_eval_loss, logits = self.model(input_ids, input_mask, label_ids, segment_ids)

#         logits = logits.detach().cpu().numpy()
#         predictions = np.argmax(logits, axis=1)
#         if usesoftmax:
#             logits =_softmax(logits)
#         return logits, predictions


# class RobertBert(object):
#     """The BERT model attacked by adversary."""

#     def __init__(self, task_name):
#         self.bertconfig= Config(task_name=task_name)
#         self.bert_wrapper = BertWrapper(self.bertconfig.bert_max_len,self.bertconfig.num_classes)
#         self.model = self.bert_wrapper.model
#         checkpoint = torch.load(self.bertconfig.model_base_path)
#         self.model.load_state_dict(checkpoint["model_state_dict"])
#         self.model.cuda()
#         self.model.eval()

#     def query(self, sentences, labels,usesoftmax =False):

#         softmax = torch.nn.Softmax(dim=1)
       
#         assert isinstance(sentences, list)
#         sentences = [x.split() for x in sentences]
#         inputs, masks = [
#             list(x) for x in zip(*[self.bert_wrapper.pre_pro(t) for t in sentences])
#         ]
#         inputs, masks = torch.tensor(inputs), torch.tensor(masks)
#         masks = masks.cuda() 
#         inputs = inputs.cuda() 
#         with torch.no_grad():
#             outputs = self.model(inputs, token_type_ids=None, attention_mask=masks)
#             outputs = outputs.logits
#         if usesoftmax:
#             outputs = softmax(outputs)
#         probs = outputs.cpu().detach().numpy().tolist()
#         _, preds = torch.max(outputs, 1)
#         #preds = preds.cpu().detach().numpy().tolist()
#         return  outputs,preds
