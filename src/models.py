import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from utils import convert_examples_to_features 
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_transformers import AdamW, WarmupLinearSchedule, ConstantLRSchedule
from tqdm import tqdm, trange
import random
import numpy as np
import os
from transformers import BertForSequenceClassification

# Build a classifier and get predictions, pooler_outputs, and hidden_states of inputs.
# The pooler_output uses the last layer hidden-state of the specific token, then further processed by a linear
# layer and a Tanh activation function.
class Classifier(nn.Module):
    def __init__(self, num_labels, **kwargs):
        """Initialize the components of the classifier."""
        super(Classifier, self).__init__()
        
        self.cls_pos = kwargs['cls_pos'] 
        #BertForSequenceClassification for SHAP pipeline to work
        #self.model =BertForSequenceClassification.from_pretrained(kwargs['pretrained_file_path'], output_hidden_states=True) 
        self.model = kwargs['model_class'].from_pretrained(kwargs['pretrained_file_path'], output_hidden_states=True) #kwargs['pretrained_model_name']
        
        self.num_labels = num_labels
        self.config = self.model.config

        self.dense = nn.Linear(in_features=768, out_features=768, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(in_features=768, out_features=num_labels, bias=True) 
        self.tokenizer =  kwargs['tokenizer_class'].from_pretrained(kwargs['pretrained_file_path'], model_max_length=512, truncation=True, padding=True)
    
    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.
        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        return self.model.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, segment_ids = None, labels=None):
        """Define the computation performed at every cell."""       
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids= segment_ids, output_hidden_states=True)
            
        pooler_output = output[1] # output.last_hidden_state[:, self.cls_pos, :]# 
        pooler_output = torch.tanh(self.dense(pooler_output))
        pooler_output = self.dropout(pooler_output)

        logits = self.out_proj(pooler_output)
        #hidden_states = output.hidden_states if hasattr(output, 'hidden_states') else None   
        last_hidden_states = output[0]
        hidden_states = output[2]
            
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #print("labels.view(-1):", labels.view(-1).shape)
            #print("logits.view(-1, self.num_labels):", logits.view(-1, self.num_labels).shape)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))            
            return loss, logits, pooler_output, last_hidden_states
        
        return logits, pooler_output, hidden_states


    # function to extact the multiple features
    def feature_list(self, input_ids=None, attention_mask=None, segment_ids = None):
        out_list = []
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids= segment_ids, output_hidden_states=True)
        #print("out[1].shape:", out[1].shape)
        out_list.append(out[1])
        out = torch.tanh(self.dense(out[1]))
        #print("out.shape:", out.shape)
        out_list.append(out)
        out = self.dropout(out)
        #print("out.shape:", out.shape)
        out_list.append(out)
        
        #out = out.view(out.size(0), -1)
        
        y = self.out_proj(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, input_ids=None, attention_mask=None, segment_ids = None, labels=None, layer_index=1):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids= segment_ids, output_hidden_states=True)
        if layer_index == 1:
            out = torch.tanh(self.dense(out[1]))
        elif layer_index == 2:
            out = torch.tanh(self.dense(out[1]))
            out = self.dropout(out)    
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, input_ids=None, attention_mask=None, segment_ids = None, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids= segment_ids, output_hidden_states=True)
        output = torch.tanh(self.dense(output[1]))
        output = self.dropout(output)
        out = output.view(output.size(0), -1)
        y = self.out_proj(out)
        return y, output

def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    
def train(args, model, data, max_seq_length, tokenizer, device, lr_rate,epochs=1,save_model= "" ):
    train_texts, train_text_pairs, train_y, label_names = data
    train_features = convert_examples_to_features(train_texts, train_text_pairs, train_y, label_names, max_seq_length, tokenizer) 
    
    #print("***** Train set *****")
    #print("Num examples = {}".format(len(train_texts)))
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    
    train_dataset = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_id)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    #print(args.batch_size)
    t_total = len(train_dataloader) // epochs
    optimizer = AdamW(model.parameters(), lr=lr_rate, eps=1e-08)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", epochs)
    print("  Batch size = %d", args.batch_size)
    print("  Total optimization steps = %d", t_total)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    
    model.zero_grad()
    epoch_iterator = trange(int(epochs), desc="Epoch")
    set_seed(args) # Added here for reproductibility (even between python 2 and 3)
    for _ in epoch_iterator:
        train_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      #'segment_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            loss, logits, pooler_output, hidden_states = model.forward(**inputs)
            #preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)#args.max_grad_norm

            tr_loss += loss.item()
            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if save_model != "":
                # Save model
                #output_dir = os.path.join(args.save_model_path)#, 'checkpoint')
                if not os.path.exists(save_model):
                    os.makedirs(save_model)
                
                #model.save_pretrained(args.save_model_path)
                torch.save(model.state_dict(), os.path.join(save_model, 'model.pt'))
                torch.save(args, os.path.join(save_model, 'training_args.bin'))
                print("\nSaving model to ", save_model)
#             if args.max_steps > 0 and global_step > args.max_steps:
#                 train_iterator.close()
#                 epoch_iterator.close()
    
