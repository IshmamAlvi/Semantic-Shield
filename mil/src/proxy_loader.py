
import torch
from transformers import BertTokenizer, BertModel
import nltk
# nltk.download('punkt')
import os
import clip
from torchvision.datasets import CIFAR100, CIFAR10
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from util_models import proxy_model
from config import CFG
from transformers import AutoTokenizer, AutoModel

class proxy_dataset(Dataset):
    def __init__(self, dataset, preprocess, classes, kg, model, device, args):
        
        self.dataset = dataset
        # self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.preprocess = preprocess
        self.classes = classes
        # self.kg = kg
        self.args = args
        self.model = model
        self.device = device
        res = []
        for key, val in kg.items():
            res.append(val)
            
        self.kg = res
        self.max_length = 77 # change from 128 to 77 to match the clip-tokenize
    #     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    #     self.inputs = tokenizer(
    #     res,
    #     truncation=True,
    #     padding=True,
    #     max_length=128
    # )

        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
       
        image = self.dataset[idx][0]
        index = self.dataset[idx][1]
        title = "A photo of {}".format(self.classes[index])
        kg_text = self.kg[index]    
        # print ("type kg_text: ", kg_text)

        if (self.args.tokenizer_clip == 'yes'):

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            tokenized = tokenizer(
            kg_text,
            truncation=True,
            padding=True,
            max_length=128)
            
            # print('tokenized: ', tokenized)
            input_ids = tokenized['input_ids']
            attention_masks = tokenized['attention_mask']
           
           
            # Convert tokens to input IDs and attention mask
            # input_ids = tokenizer.convert_tokens_to_ids(tokenized)
            all_input_ids = []
            all_attention_masks = []
            for input_id  in input_ids:                
                if len(input_id) > self.max_length:
                    input_id = input_id[:self.max_length]
                attention_mask = [1] * len(input_id)
            # Zero-pad up to the sequence length.
                padding = [0] * (self.max_length - len(input_id))
                input_id += padding
                attention_mask += padding
                
                assert (len(attention_mask) == self.max_length)
                assert (len(input_id) == self.max_length)

                all_input_ids.append(input_id)
                all_attention_masks.append(attention_mask)
            
            # print("----------------------------")
            # print(all_input_ids)
            # print("--------------------------")
            # print(all_attention_masks)
            all_input_ids = torch.tensor(all_input_ids, dtype = torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype = torch.long)
            return image, all_input_ids, all_attention_masks    
        
        return image, kg_text, title, index
    
