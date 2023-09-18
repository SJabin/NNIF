import sys
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from collections.abc import Iterable
import torch
import random
import torch.autograd as autograd
from torch.autograd import Variable
from utils import convert_examples_to_features#, add_noise_to_features
import gc


def sample_estimator(model,device, train_texts, train_text_pairs, train_y, label_names, max_seq_length, tokenizer, only_last):
    group_lasso = EmpiricalCovariance(assume_centered=False)
    #class_names, class_counts = np.unique(train_y, return_counts = True)
    #class_indices = []
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    model.eval()
    model.to(device)
    print("device:", device)
    correct, total = 0,0

    train_features = convert_examples_to_features(train_texts, train_text_pairs, train_y, label_names, max_seq_length, tokenizer)
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)   
    train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)    
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)   

    if only_last:
        #print(next(model.parameters()).device)
        try:
            with torch.no_grad():
                output, out_features, _ = model.penultimate_forward(input_ids=train_input_ids, attention_mask=train_input_mask, segment_ids = train_segment_ids)
                out_features = torch.unsqueeze(out_features, dim=0)
                print("output:", output.shape)#[100, 2]
                print("out_features:", out_features.shape)#[1, 100, 768]
        except Exception as e:
            print(e)
        

    else:
        with torch.no_grad():
            output, out_features = model.feature_list(input_ids=train_input_ids, attention_mask=train_input_mask, segment_ids = train_segment_ids)
            #out_features = torch.FloatTensor(out_features)
        print("output:", output.shape)#[100, 2]
        #print("out_features:", out_features.shape)#[3, 100, 768]
        print("out_features:", len(out_features), len(out_features[0]), len(out_features[0][0]))#[3, 100, 768]
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    num_output = out_features.size(0) if only_last else len(out_features)
    print("num_output:", num_output)
    
    num_classes = len(label_names)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    
    feature_list = np.empty(num_output)
    
#     if only_last:
#         count = 0
#         for out in out_features:
#             feature_list[count] = out.size(0)
#             count += 1
#     else:
    count = 0
    for out in out_features:
        feature_list[count] = out.size(1)
        count += 1
    print("feature_list:", feature_list)#[768]

    
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    print("list_features:", list_features)#[[0, 0]]

    # get hidden features
    for i in range(num_output):
        temp = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
        out_features[i] = torch.mean(temp.data, 2)

    # compute the accuracy   
    pred = output.data.max(1)[1]
    equal_flag = pred.cpu().eq(torch.Tensor(train_y))
    correct += equal_flag.sum()
    #print("correct predictions:", correct)

    for i in range(0, train_input_ids.size(0)):
        label = train_y[i]
        if num_sample_per_class[label] == 0:
            out_count = 0
            for out in out_features:
                list_features[out_count][label] = out[i].view(1, -1)
                out_count += 1
        else:
            out_count = 0
            for out in out_features:
                list_features[out_count][label]= torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                out_count += 1
        num_sample_per_class[label] += 1
    #print("list_features:", len(list_features), len(list_features[0]) , len(list_features[0][0]), len(list_features[0][1]) ) #[1,2,-1]

    sample_class_mean = []
    out_count = 0    
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature))#.cuda()
        print("temp_list:", temp_list.shape)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        #print("temp_list:", temp_list)    
        sample_class_mean.append(temp_list)
        del temp_list
        out_count += 1
    print("sample class mean:", len(sample_class_mean), len(sample_class_mean[0]), len(sample_class_mean[0][0]))

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().detach().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float()#.cuda()
        precision.append(temp_precision)
        del temp_precision
    print("precision:", len(precision), len(precision[0]),len(precision[0][0]))#[1,768,768]
    print('\nTraining Accuracy:({:.2f}%)\n'.format(100. * correct / len(train_texts)))
    
    del train_input_ids, train_input_mask, train_segment_ids
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    return sample_class_mean, precision

def get_Mahalanobis_score(model, device, random_seed, data, magnitude, sample_mean, precision, num_classes, only_last=False):
    
    test_texts, test_text_pairs, test_y, label_names, max_seq_length, tokenizer = data
    test_features = convert_examples_to_features(test_texts, test_text_pairs, test_y, label_names, max_seq_length, tokenizer)
    
    #print("***** Test set *****")
    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)    
    test_input_ids = Variable(test_input_ids.to(torch.float32), requires_grad = True)
    test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)    
    test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)   
    #test_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    #test_guids = torch.tensor([f.guid for f in test_features], dtype=torch.long)
    
    model.eval()
    model.to(device)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    #guid = test_guids[0].item()
    
    test_input_ids = test_input_ids.to(device)
    test_input_mask = test_input_mask.to(device)
    test_segment_ids = test_segment_ids.to(device)
    #test_guids = test_guids.to(device)
    
    Mahalanobis = [] 
    if only_last:
        with torch.no_grad():
            _, out_features = model.penultimate_forward(input_ids=test_input_ids.to(torch.int64), attention_mask=test_input_mask, segment_ids = test_segment_ids)
            out_features = torch.unsqueeze(out_features, dim=0)#[1, 10, 768]
    else:
        with torch.no_grad():
            _, out_features = model.feature_list(input_ids=test_input_ids.to(torch.int64), attention_mask=test_input_mask, segment_ids = test_segment_ids)
    

    temp=[]
    for feat in out_features:  
        temp.append(feat.detach().clone().numpy())
    out_features = torch.FloatTensor(np.array(temp))
    print("out_features:", out_features.shape)#[1, 10, 768] or [3,10,768]

    #num_output = len(out_features) if only_last else len(out_features[0])
    num_output = out_features.size(0) if only_last else len(out_features)
    print("num_output:", num_output)
    layer_index = 1 if only_last else len(out_features)
    
    # get hidden features
    #for i in range(num_output):
    #    temp = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
    #    out_features[i] = torch.mean(temp.data, 2)
    #print("out_features:", out_features.shape)
    # compute Mahalanobis score
    gaussian_score = 0

    for l_id in range(num_output):
        temp = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[l_id][i]
            zero_f = out_features[l_id].data - batch_sample_mean
            
            term_gau = -0.5*torch.matmul(torch.squeeze(torch.matmul(zero_f, precision[l_id])), torch.squeeze(zero_f).t()).diag()

            if i==0:
                temp = term_gau.view(-1,1)
            else:
                torch.cat([temp, term_gau.view(-1,1)], 1)
        if l_id==0:
            gaussian_score = temp.unsqueeze(0)
        else:
            gaussian_score = torch.cat([gaussian_score, temp.unsqueeze(0)], 0)
    print("\ngaussian_score:", gaussian_score.shape)#before torch.Size([10, 2]), now [1, 10, 2]
                
    max_gaussian_score, _ = torch.max(gaussian_score, dim=2)
    Mahalanobis.extend(max_gaussian_score.cpu().detach().numpy())
    Mahalanobis = np.asarray(Mahalanobis, dtype=np.float32).transpose()
    #print("Mahalanobis score:", Mahalanobis) #[-2.3210974, -2.925033, -3.1843758...]
    #sys.exit()
    del test_input_ids, test_input_mask, test_segment_ids
    return Mahalanobis

def get_mahalanobis(model, device, random_seed, max_seq_length, tokenizer, data, magnitude, sample_mean, precision, num_classes, only_last=False, with_noise=False):#layer_index=None
    
    test_texts, test_text_pairs, test_y, adv_texts, adv_text_pairs, adv_preds, label_names = data
    
    first_pass = True
    print('Calculating Mahalanobis characteristics')
#         gaussian_score = get_mahalanobis_tensors(train_emb, sample_mean, precision, num_classes, layer) #, grads

    data = test_texts, test_text_pairs, test_y, label_names, max_seq_length, tokenizer
    M_in = get_Mahalanobis_score(model, device, random_seed, data, magnitude, sample_mean, precision, num_classes, only_last) #,layer_index
    print('test is done. Mahalanobis score:', M_in.shape )

    data = adv_texts, adv_text_pairs, adv_preds, label_names, max_seq_length, tokenizer
    M_out = get_Mahalanobis_score(model, device, random_seed, data, magnitude, sample_mean, precision, num_classes, only_last)#,layer_index
    print('adv is done. Mahalanobis score:', M_out.shape)

#     if args.with_noise:
#         M_noisy = get_Mahalanobis_score_adv(X_noisy, gaussian_score, grads, magnitude, rgb_scale)
#         M_noisy = np.asarray(M_noisy, dtype=np.float32)
#     else:  # just a placeholder with zeros
    M_noisy = np.zeros_like(M_in)
    
    if first_pass:
        Mahalanobis_in    = M_in.reshape((M_in.shape[0], -1))
        Mahalanobis_out   = M_out.reshape((M_out.shape[0], -1))
        #Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
        first_pass = False
    else:
        Mahalanobis_in    = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
        Mahalanobis_out   = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
        #Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)

    if with_noise:
        Mahalanobis_neg = np.concatenate((M_in, M_noisy))
    else:
        Mahalanobis_neg = M_in
    
    Mahalanobis_pos = M_out
    
    return Mahalanobis_pos, Mahalanobis_neg
