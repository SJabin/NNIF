import os
from time import strftime
import numpy as np
import pandas as pd
import shap as shap
import torch
import transformers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import scipy as sp
from multiprocessing import Pool
import math
from math import ceil
import random
from functools import partial
#import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model


class shap_generator:
    def __init__(self, model, tokenizer=None, max_length=128, device="cpu", seed=None):
        self.model = model
        self.tokenizer = tokenizer
        #self.output_names = output_names
        #self.feature_names = feature_names
        self.max_length = max_length
        self.device = device
        self.random_seed = seed
        
    def pred(self,x):
        input_ids = []
        attention_masks = []
        for v in x:
            inputs = self.tokenizer.encode_plus(
                text=v,
                #text_pair=t_pair if t_pair else None,
                return_tensors='pt',
                return_attention_mask = True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length').to(self.device)
            input_ids.append(inputs['input_ids'])
            attention_masks.append(inputs['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        with torch.no_grad():   
            logits, pooler_output, hidden_states = self.model(input_ids, attention_masks)
        # scores = (np.exp(logits).T / np.exp(logits).sum(-1)).T
        # print("scores")
        # print(scores)
        # val = sp.special.logit(scores)
        # print("val")

        return logits
    
    def get_shap(self, text):
        explainer = shap.Explainer(self.pred, self.tokenizer)
        explainations = explainer(text.tolist())
        return explainations.values
    
    def pad_seq(self, seq, pad_len, pad_token=0):
        return np.pad(seq[:pad_len], ((0, pad_len - seq[:pad_len].shape[0]), (0, 0)), 'constant',
                  constant_values=pad_token)

    def create_SHAP_signatures(self, orig_text, adv_text, dset_name, last_ind, num_classes, result_dir=None, batch_size=50):
        common_len = self.max_length      
        explainer = shap.Explainer(self.pred, self.tokenizer)
        orig_text = orig_text
        adv_text = adv_text
        n_batch = ceil(len(orig_text) / batch_size)
        
        print("original texts")
        path = os.path.join(result_dir, "shap_values_orig_"+str(last_ind)+".npy")
        if not os.path.exists(path):
            #shap_values_orig = explainer(orig_text[begin_idx: end_idx].tolist())
            #shap_vals_orig = np.array([self.pad_seq(x.values, pad_len=common_len) for x in shap_values_orig]).reshape(-1, num_classes * common_len)
            shap_values = []
            for batch in range(n_batch):
                begin_idx = batch_size * batch
                end_idx = min(batch_size * (batch + 1), len(orig_text))
                shap_values_orig = explainer(orig_text[begin_idx: end_idx])#.tolist()
                temp_shap_vals = np.array([self.pad_seq(x.values, pad_len=common_len) for x in shap_values_orig]).reshape(-1, num_classes * common_len)
                for s in temp_shap_vals:
                    shap_values.append(s)
                del shap_values_orig, temp_shap_vals
        
        
#             n_cpus = 12 if os.cpu_count() > 12 else os.cpu_count()
#             print("ncpus:", ncpus)
#             pool = Pool(n_cpus)
#             step = math.ceil(len(orig_text) / n_cpus)
#             pool_inputs = [orig_text[i: i + step] for i in range(0, len(orig_text), step)]#len(orig_text), step

#             shap_value_queue = pool.map(partial(self.get_shap),pool_inputs)
#             shap_values_orig = np.concat(shap_value_queue)
#             #shap_values_orig.to_csv(neighbours_file, index=False)
        
            np.save(path, shap_values, allow_pickle=True)
            print("Created {", len(shap_values), "}  original SHAP values for {",dset_name,"}")
            del shap_values
        
        print("adversarial texts")
        path = os.path.join(result_dir, "shap_values_adv_"+str(last_ind)+".npy")
        if not os.path.exists(path):
            # shap_values_adv = explainer(adv_text[begin_idx:end_idx].tolist())
            # shap_vals_adv = np.array([self.pad_seq(x.values, pad_len=common_len) for x in shap_values_adv]).reshape(-1, num_classes * common_len)
            
            shap_values = []
            for batch in range(n_batch):
                begin_idx = batch_size * batch
                end_idx = min(batch_size * (batch + 1), len(adv_text))
                shap_values_adv = explainer(adv_text[begin_idx:end_idx])#.tolist()
                temp_shap_vals = np.array([self.pad_seq(x.values, pad_len=common_len) for x in shap_values_adv]).reshape(-1, num_classes * common_len)
                for s in temp_shap_vals:
                    shap_values.append(s)
                del shap_values_adv, temp_shap_vals
                
            np.save(path, shap_values, allow_pickle=True)
            print("Created {", len(shap_values), "}  adversarial SHAP values for {dset_name}")

            del shap_values
            
        #return shap_vals_orig, shap_vals_adv

def merge_results(slices, result_dir, case="orig"):
    shap_val = []    
    for s in slices:
        path = os.path.join(result_dir, "shap_values_"+case+"_"+str(s)+".npy")
        if not os.path.exists(path):
            raise Exception(path+" doesn't exist")
        temp = np.load(path, allow_pickle=True)
        for t in temp:
            shap_val.append(t)               
    np.save(os.path.join(result_dir, "shap_values_"+case+".npy"), shap_val, allow_pickle=True)
    del shap_val
        
def detectors(result_dir, random_seed=None, filenum = 1):
    shap_orig=[]
    shap_adv=[]
    for i in range(1,filenum+1):
        #print("file:", i)
        temp = np.load(os.path.join(result_dir,"shap_values_orig.npy"), allow_pickle=True)
        #print(len(temp))
        for t in temp:
            shap_orig.append(t)
        temp = np.load(os.path.join(result_dir,"shap_values_adv.npy"), allow_pickle=True)
        #print(len(temp))
        for t in temp:
            shap_adv.append(t)

    print(f'orig shape: {np.array(shap_orig).shape}')
    print(f'Adv shape: {np.array(shap_adv).shape}')
    
    data = np.concatenate((shap_orig, shap_adv))
    orig_labels = np.zeros(len(shap_orig), dtype=np.int16)
    adv_labels = np.ones(len(shap_adv), dtype=np.int16)
    gt = np.concatenate((orig_labels, adv_labels))

    x_train, x_test, y_train, y_test = train_test_split(data, gt, random_state=random_seed, shuffle=True, train_size=0.8)
    
    print(f'train size: {x_train.shape}')

    indices = random.sample(range(0, len(x_train)), len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
    indices = random.sample(range(0, len(x_test)), len(x_test))
    x_test = x_test[indices]
    y_test = y_test[indices]

    #print(y_train)
    #print(y_test)
    
    # Build detector
#     lr = LogisticRegressionCV(max_iter=500000, cv=10, solver = "liblinear", random_state=random_seed, class_weight="balanced").fit(x_train, y_train)
    log_params={"C":np.logspace(-6,6,12)}#, "penalty":["l1","l2"]}
    lr = LogisticRegression(max_iter=500000,random_state=random_seed, class_weight="balanced")#.fit(x_train, y_train)
    search = GridSearchCV(estimator = lr, param_grid = log_params, cv=5)
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    best_model.fit(x_train, y_train)
    print("done.")
    
    lr.fit(x_train, y_train)
    preds = lr.predict(x_test)
#     preds = best_model.predict(x_test)
    print(f'Logistic Regression: {accuracy_score(y_test, preds):.3f}')
    
    randomF = RandomForestClassifier( random_state=random_seed, class_weight='balanced')
    randomF.fit(x_train, y_train)
    preds = randomF.predict(x_test)
    print(f'Random Forest: {accuracy_score(y_test, preds):.3f}')

    svc = SVC(random_state=random_seed)
    svc.fit(x_train, y_train)
    preds = svc.predict(x_test)
    print(f'SVC: {accuracy_score(y_test, preds):.3f}')

    input_shape = x_train.shape[1]


    model = keras.Sequential([
    keras.layers.Dense(1000, input_shape=(input_shape,), activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001),
                              ),
    keras.layers.Dropout(0.1, seed=random_seed),
    keras.layers.Dense(1000, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001),
                              ),
    keras.layers.Dense(1, activation='softmax',)
    ])
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20)
    model.evaluate(x_test, y_test)

    preds = model.predict(x_test)
    preds = preds.flatten()
    preds[preds < 0.5] = 0
    preds[preds >= 0.5] = 1
    print(accuracy_score(y_test, preds))

