#adapted IF calculation from https://github.com/xhan77/influence-function-analysis

import torch
import numpy as np
from collections.abc import Iterable
from tqdm import tqdm
import random
import os
import time
import torch.autograd as autograd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import convert_examples_to_features



#influence score
def get_ihvp_score( args, device, data, model, max_seq_length, tokenizer, batch_size, test_idx, dir, influence_on_decision, param_influence):
    
    train_texts, train_text_pairs, train_y, test_text, test_text_pair, test_y, label_names = data
    train_features = convert_examples_to_features(train_texts, train_text_pairs, train_y, label_names, max_seq_length, tokenizer) 
       
    #print("***** Train set *****")
    #print("Num examples = {}".format(len(train_texts)))
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_guids = torch.tensor([f.guid for f in train_features], dtype=torch.long)
    
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_id, train_guids)
    train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=1)

    
    test_features = convert_examples_to_features(test_text, test_text_pair, test_y, label_names, max_seq_length, tokenizer)
    
    #print("***** Test set *****")
    #print("Num examples = {}".format(len(test_text)))
    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_guids = torch.tensor([f.guid for f in test_features], dtype=torch.long)
    #test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_id, test_guids)
    #test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=1)   
    damping = args.damping
    
       
    model.eval()
    #model.to(device)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    guid = test_guids[0].item()
    
    test_input_ids = test_input_ids.to(device)
    test_input_mask = test_input_mask.to(device)
    test_segment_ids = test_segment_ids.to(device)
    test_label_id = test_label_id.to(device)
            
    ######## L_TEST GRADIENT ########
    model.zero_grad()
    test_loss, _, _, _= model(test_input_ids, test_input_mask,  test_segment_ids, test_label_id)
    test_grads = autograd.grad(test_loss, param_influence)

    del test_input_ids, test_input_mask, test_segment_ids, test_label_id

    ######## IHVP ########
    lissa_repeat = 1
    lissa_depth = .1        
    #print("START COMPUTING IHVP ")
    train_dataloader_lissa = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    #print(train_dataloader_lissa)
    inverse_hvp = get_inverse_hvp_lissa(test_grads, model, device, param_influence, train_dataloader_lissa, num_samples=lissa_repeat, recursion_depth=int(len(train_texts)*lissa_depth), damping=damping)
    print("FINISHED COMPUTING IHVP")
        
    #influences = np.zeros(len(train_dataloader.dataset))
    influences = [0] * len(train_dataloader.dataset)
    for train_idx, (_input_ids, _input_mask, _segment_ids, _label_ids, _) in enumerate(train_dataloader):# tqdm(, desc="Train set index")):
        #print("now")
        model.train()
        #print(len(_input_ids))
        _input_ids = _input_ids.to(device)
        _input_mask = _input_mask.to(device)
        _segment_ids = _segment_ids.to(device)
        _label_ids = _label_ids.to(device)
            
        ######## L_TRAIN GRADIENT ########
        model.zero_grad()
        train_loss, _, _, _ = model(_input_ids,  _input_mask, _segment_ids, _label_ids)
        train_grads = autograd.grad(train_loss, param_influence)
        del _input_ids, _input_mask, _segment_ids, _label_ids

        val = torch.dot(inverse_hvp, gather_flat_grad(train_grads)).item()
        influences[train_idx] = val
        
        torch.cuda.empty_cache()
        
        if np.isnan(val):
            print('it\'s nan!')
            sys.exit()

#     if args.influence_on_decision:
#         np.save(os.path.join(dir, "influences_test_" + str(guid)), influences)
#     else:
#         np.save(os.path.join(dir, "influences_on_x_test_" + str(guid)), influences)
        
#     if args.if_compute_saliency:
#         np.save(os.path.join(dir, "saliency_test_" + str(guid) + ".np"), (test_tok_sal_list, train_tok_sal_lists, test_pred_status))

   
    return influences

######## LiSSA ########
def gather_flat_grad(grads):
    views = []
    for p in grads:
        p.data[p.data == float('inf')]=0.0
        p.data = torch.nan_to_num(p.data, nan=0.0)
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def hv(loss, model_params, v): # according to pytorch issue #24004
    s = time.time()
    grad = autograd.grad(loss, model_params, create_graph=True)
    e1 = time.time()
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    e2 = time.time()
    #print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
    return Hv

def get_inverse_hvp_lissa(v, model, device, param_influence, train_loader, num_samples, recursion_depth, scale=1e4, damping=0): 
    ihvp = None
    model.train()
    #model.to(device)
    for i in range(num_samples):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)

        #print("shape of test_grads:", cur_estimate)
        for j in range(recursion_depth):
            try:
                input_ids, input_mask, segment_ids, label_ids, guids = next(lissa_data_iterator)
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                input_ids, input_mask, segment_ids, label_ids, guids = next(lissa_data_iterator)
                
                
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            
            model.zero_grad()
            train_loss, _, _, _ = model(input_ids, input_mask, segment_ids, label_ids)
            hvp = hv(train_loss, param_influence, cur_estimate)
            del input_ids, input_mask, segment_ids, label_ids
        
            #print("hvp:", hvp)
            cur_estimate = [_a + (1 - damping) * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            #cur_estimate = [_a + 1 * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            
            #if (j % 200 == 0) or (j == recursion_depth - 1):
            #    print("Recursion at depth %s: norm is %f" % (j, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())))

        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]

            
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    return return_ihvp

#k-nn ranks and distances


def calc_all_ranks_and_dists(features, knn):
    num_output = 1 #len(model.net.keys())
    n_neighbors = knn.n_neighbors #knn[list(knn.keys())[0]].n_neighbors
#     print('len(features):',len(features))
#     print('n_neighbors:', n_neighbors)
    
    all_neighbor_ranks = -1 * np.ones((len(features), 1, n_neighbors), dtype=np.int32)#num_output
    all_neighbor_dists = -1 * np.ones((len(features), 1, n_neighbors), dtype=np.float32)#num_output

    all_neighbor_dists[:,0], all_neighbor_ranks[:, 0]= \
            knn.kneighbors(features, return_distance=True)
#     print(all_neighbor_dists, all_neighbor_ranks)
    del features
    return all_neighbor_ranks, all_neighbor_dists

def find_ranks(test_idx, sorted_influence_indices, all_neighbor_indices, all_neighbor_dists, mean=False):
    ni = all_neighbor_indices
    nd = all_neighbor_dists

    ranks = -1 * np.ones((len(sorted_influence_indices)), dtype=np.int32)#num_output = 1
    dists = -1 * np.ones((len(sorted_influence_indices)), dtype=np.float32)#num_output = 1
    

    for target_idx in range(len(sorted_influence_indices)):
        idx = sorted_influence_indices[target_idx]
        #print(ni[test_idx])
        #print(np.where(ni[test_idx, 0] == idx))
        loc_in_knn = np.where(ni[test_idx, 0] == idx)[0][0]
        #print("loc_in_knn:", idx, loc_in_knn)
        knn_dist = nd[test_idx, 0, loc_in_knn]
        #print("knn_dist:", knn_dist)
        ranks[target_idx] = loc_in_knn
        dists[target_idx] = knn_dist
    if mean:
        ranks_mean = np.mean(ranks, axis=1)
        dists_mean = np.mean(dists, axis=1)
        return ranks_mean, dists_mean
    
    return ranks, dists


