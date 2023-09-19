"""
Fine-tune BERT, RoBERTa, XLNet, and BART on aclImdb or MultiNLI dataset.
"""
import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import IMDBDataset, MnliDataset
from utils import bert_params, roberta_params, xlnet_params, bart_params
from models import Classifier


parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, default=None, required=True, choices=['bert', 'roberta', 'xlnet', 'bart'],
                    help='Which model to fine tune.')
parser.add_argument('--dataset-name', type=str, default=None, required=True, choices=['IMDB', 'Mnli'],
                    help='Which dataset to use to fine tune the model.')
parser.add_argument('--dataset-path', type=str, default=None, required=True,
                    help='The directory of the dataset') #choices=['./data/aclImdb', './data/multinli_1.0'], 
parser.add_argument('--max-length', type=int, default=None, required=True, choices=[512, 256, 128],
                    help='The maximum sequences length.')
parser.add_argument('--epochs', type=int, default=3, help='Total number of training epochs.')
parser.add_argument('--batch-size', type=int, default=32, help='Training and Testing batch size.')
parser.add_argument('--random-seed', type=int, default=38, help='random seed value.')
args = parser.parse_args()

# set a random seed value all over the place to make this reproducible.
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

# select model parameters according to the model name
model_name = args.model_name.lower()
if model_name == 'bert':
    model_params = bert_params
elif model_name == 'roberta':
    model_params = roberta_params
elif model_name == 'xlnet':
    model_params = xlnet_params
elif model_name == 'bart':
    model_params = bart_params
else:
    raise ValueError('Provided model name: {} is not BERT, RoBERTa, XLNet, or BART.'.format(model_name))

# check if there's a GPU
if torch.cuda.is_available():
    # set the device to the GPU.
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
dataset = data_processors[args.dataset_name](args.dataset_path, 0.2)

# load the tokenizer
print('Loading tokenizer ... ')
tokenizer = model_params['tokenizer_class'].from_pretrained(model_params['pretrained_file_path'], do_lower_case=False)

# get the maximum sequence length for padding/truncating to
max_len = args.max_length

# tokenize all of the sentences and map the tokens to their word IDs.
train_input_ids, train_attention_masks = [], []
valid_input_ids, valid_attention_masks = [], []
for text, text_pair in zip(dataset.train_text, dataset.train_text_pair):
    encoded_dict = tokenizer.encode_plus(text=text,
                                         text_pair=text_pair,
                                         add_special_tokens=True,
                                         max_length=max_len,
                                         truncation=True,
                                         padding='max_length',
                                         return_attention_mask=True,
                                         return_tensors='pt')
    train_input_ids.append(encoded_dict['input_ids'])
    train_attention_masks.append(encoded_dict['attention_mask'])

for text, text_pair in zip(dataset.valid_text, dataset.valid_text_pair):
    encoded_dict = tokenizer.encode_plus(text=text,
                                         text_pair=text_pair,
                                         add_special_tokens=True,
                                         max_length=max_len,
                                         truncation=True,
                                         padding='max_length',
                                         return_attention_mask=True,
                                         return_tensors='pt')
    valid_input_ids.append(encoded_dict['input_ids'])
    valid_attention_masks.append(encoded_dict['attention_mask'])

# convert the lists into tensors
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(dataset.train_y)

valid_input_ids = torch.cat(valid_input_ids, dim=0)
valid_attention_masks = torch.cat(valid_attention_masks, dim=0)
valid_labels = torch.tensor(dataset.valid_y)

# combine the training and validation inputs into a TensorDataset and create DataLoader
batch_size = args.batch_size  # batch size for DataLoader

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

valid_dataset = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

# load the pretrained model with a linear classification layer on top
print('Loading pretrained model ... ')

model = Classifier(dataset.num_labels, **model_params)
# run the model on GPU if available
model.to(device)

# specify loss function (categorical cross-entropy) and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=model_params['learning_rate'], eps=1e-8)

# Total number of training steps is [number of batches] x [number of epochs].
total_steps = len(train_loader) * args.epochs
# create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=total_steps * 0.06,
                                            num_training_steps=total_steps)

# The path to save the fine-tuned model.
output_dir = os.path.join('./output', args.dataset_name, model_name)
# Create output directory if not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def compute_accuracy(preds, labels):
    # calculate the accuracy of predictions vs labels

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(preds_flat == labels_flat) / len(labels)


print('Training and validation model ... ')

# Minimal validation loss, used to save the model which has smaller validation loss than this variable
valid_loss_min = np.Inf

for epoch in range(args.epochs):
    ### train the model
    train_loss = 0.0

    # set the model in training mode
    model.train()
    for batch in train_loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # clear previous calculated gradients
        optimizer.zero_grad()
        logits, _, _ = model(input_ids=b_input_ids, attention_mask=b_input_mask)

        loss = criterion(logits, b_labels)
        train_loss += float(loss.item())
        loss.backward()

        # clip the norm of the gradients to 1.0
        # this is to help prevent the 'exploding gradients' problem
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # update the learning rate
        scheduler.step()

#     ### Valid the model
#     model.eval()

#     valid_accuracy = 0.0
#     valid_loss = 0.0
#     for batch in valid_loader:
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)

#         with torch.no_grad():
#             logits, _, _ = model(input_ids=b_input_ids, attention_mask=b_input_mask)

#         loss = criterion(logits, b_labels)
#         valid_loss += float(loss.item())

#         # move logits and labels to CPU
#         logits = logits.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()

#         # calculate the accuracy for this batch
#         valid_accuracy += compute_accuracy(logits, label_ids)

    # calculate average losses and accuracy
    train_loss = train_loss / len(train_loader)
    #valid_loss = valid_loss / len(valid_loader)
    #valid_accuracy = valid_accuracy / len(valid_loader)

    #print('Epoch {}/{}:\tTraining Loss: {:.6f},\tValidation Loss: {:.6f},\tValidation Accuracy: {:.6f}'.format(
    #    epoch, args.epochs, train_loss, valid_loss, valid_accuracy))
    print('Epoch {}/{}:\tTraining Loss: {:.6f}'.format(epoch, args.epochs, train_loss))
    
    # save model if validation loss has decreased
    #if valid_loss <= valid_loss_min:
    #    print('Validataion loss decreased ({:.6f} --> {:.6f}). Saving model ... '.format(valid_loss_min, valid_loss))
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    tokenizer.save_pretrained(output_dir)
    #valid_loss_min = valid_loss

print('Training complete!')

# print('Testing the model ... ')
# test_input_ids, test_attention_masks = [], []
# for text, text_pair in zip(dataset.test_text, dataset.test_text_pair):
#     encoded_dict = tokenizer.encode_plus(text=text,
#                                          text_pair=text_pair,
#                                          add_special_tokens=True,
#                                          max_length=max_len,
#                                          truncation=True,
#                                          padding='max_length',
#                                          return_attention_mask=True,
#                                          return_tensors='pt')
#     test_input_ids.append(encoded_dict['input_ids'])
#     test_attention_masks.append(encoded_dict['attention_mask'])

# test_input_ids = torch.cat(test_input_ids, dim=0)
# test_attention_masks = torch.cat(test_attention_masks, dim=0)
# test_labels = torch.tensor(dataset.test_y)

# test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
# test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

# # load the model with the lowest validation loss
# model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pt'), map_location=device))

# #model = bert_model.from_pretrained(kwargs['pretrained_file_path'], output_hidden_states=True, ) #kwargs['pretrained_model_name']


# model.eval()
# test_accuracy = 0
# for batch in test_loader:
#     b_input_ids = batch[0].to(device)
#     b_input_mask = batch[1].to(device)
#     b_labels = batch[2].to(device)

#     with torch.no_grad():
#         logits, _, _ = model(input_ids=b_input_ids, attention_mask=b_input_mask)

#     # Move logits and labels to CPU
#     logits = logits.detach().cpu().numpy()
#     label_ids = b_labels.to('cpu').numpy()

#     test_accuracy += compute_accuracy(logits, label_ids)

# print('Test Accuracy: {:.6f}'.format(test_accuracy / len(test_loader)))
# print('Test complete!')
