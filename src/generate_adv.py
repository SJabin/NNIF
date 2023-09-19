# From the repo https://github.com/NaLiuAnna/MDRE
import os
import random
import argparse
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from models import Classifier
from attack import Typo, SynonymsReplacement
from utils import IMDBDataset, MnliDataset
from utils import bert_params

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, default=None, required=True, choices=['IMDB', 'Mnli'],
                    help='Which test set to perturb to generate adversarial examples.')
parser.add_argument('--dataset-path', type=str, default=None, required=True,
                    #choices=['./data/aclImdb', './data/multinli_1.0'],
                    help='The directory of the dataset.')
parser.add_argument('--attack-class', type=str, default=None, required=True, choices=['typo', 'synonym'],
                    help='Attack method to generate adversarial examples.')
parser.add_argument('--max-length', type=int, default=None, required=True, choices=[512, 256, 128],
                    help='The maximum sequences length.')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for BERT prediction.')
parser.add_argument('--random-seed', type=int, default=38, help='random seed value.')
parser.add_argument('--batch', type=int, help='batch number in all batches of test data for distributed data parallel.')
parser.add_argument('--boxsize', type=int, help='How many batches is used for distributed data parallel.')
args = parser.parse_args()

# set a random seed value all over the place to make this reproducible
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

# check if there's a GPU
# if torch.cuda.is_available():
#     # set the device to the GPU
#     device = torch.device('cuda')
#     print('We will use the GPU:', torch.cuda.get_device_name(0))
# else:
#     print('No GPU available, using the CPU instead.')
device = torch.device('cpu')

model_params = bert_params
batch_size = args.batch_size
max_length = args.max_length

# load dataset
data_processors = {
    'Mnli': MnliDataset,
    'IMDB': IMDBDataset
}
dataset = data_processors[args.dataset_name](args.dataset_path, 0.2)

# load BERT model and tokenizer
output_dir = os.path.join('./results', args.dataset_name, 'bert')
model = Classifier(dataset.num_labels, **model_params)

print("training model")
#better to save the whole model, not just the state_dict
if not os.path.exists(os.path.join(output_dir, 'model.pt')):
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
    
    #train_dataloader_wbatch = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=batch_size)
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
    
#     #test_idx = args.test_idx
#     start_test_idx = args.start_test_idx
#     end_test_idx = args.end_test_idx   
#     print(len(test_dataloader))

    #for input_ids, input_mask, segment_ids, label_ids, guids in test_dataloader:
        #print("test input ids: ", len(input_ids))
        
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
    #model.zero_grad()
    with torch.no_grad():
        test_loss, _, _, _= model(test_input_ids, test_input_mask,  test_segment_ids, test_label_id)
print("loading model")

#model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pt'), map_location=device))
model = model_params['model_class'].from_pretrained(output_dir, output_hidden_states=True, )
model.to(device)
tokenizer = BertTokenizer.from_pretrained(output_dir)


def get_predictions(texts):
    """
    Obtaining the model prediction of `texts`
    :param texts: input of the BERT model
    :return: predictions and outputs of softmax which represent the probability distribution over classes
    """
    model.eval()
    n_batch = ceil(len(texts) / batch_size)
    preds = []
    softmax_list = []
    for batch in range(n_batch):
        begin_idx = batch_size * batch
        end_idx = min(batch_size * (batch + 1), len(texts))
        b_texts = texts[begin_idx: end_idx]
        text = np.asarray(b_texts)[:, 0].tolist()
        text_pair = np.asarray(b_texts)[:, 1].tolist()

        inputs = tokenizer(text=text,
                           text_pair=text_pair if text_pair[0] else None,
                           return_tensors='pt',
                           max_length=max_length,
                           truncation=True,
                           padding='max_length').to(device)

        with torch.no_grad():
            logits, _, _ = model(inputs['input_ids'], inputs['attention_mask'])
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        softmax_list.append(F.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(softmax_list, axis=0)


# initialize attack instance
attack_processors = {
    'typo': Typo,
    'synonym': SynonymsReplacement,
}
attack_params = {
    'typo': [max_length, dataset.num_labels],
    'synonym': [args.dataset_name, max_length, dataset.num_labels],
}
attack = attack_processors[args.attack_class](get_predictions, *attack_params[args.attack_class])

# generate adversarial examples
num_texts = 2 #len(dataset.test_y)
boxsize = args.boxsize
begin_idx = -(num_texts // -boxsize) * args.batch
end_idx = min(-(num_texts // -boxsize) * (args.batch + 1), num_texts)
examples = dict(zip(zip(dataset.test_text[begin_idx: end_idx], dataset.test_text_pair[begin_idx: end_idx]),
                    dataset.test_y[begin_idx: end_idx]))
result = attack.generate(examples)

# create directory if not exist
if not os.path.exists(os.path.join(output_dir, args.attack_class)):
    os.makedirs(os.path.join(output_dir, args.attack_class))
# save adversarial examples
result.to_csv(os.path.join(output_dir, args.attack_class, f'{args.attack_class}_adv_{args.batch}.csv'), index=False)

print('Done!')