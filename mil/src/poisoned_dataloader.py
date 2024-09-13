
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
from transformers import  DistilBertTokenizer
import sys
sys.path.append( '../../' )
from clip_implementation.config import CFG

class poisoned_dataset(Dataset):
    def __init__(self, dataset, preprocess, classes, kg, model, device, args):
        
        self.dataset = dataset
        # self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.preprocess = preprocess
        self.classes = classes
        self.kg = kg
        self.args = args
        self.model = model
        self.device = device
        if (self.args.clip_openai == 'no'):
            self.tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        self.cnt = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        index = self.dataset[idx][1]
        title = "A photo of {}".format(self.classes[index])
        kg_text = self.kg[index]

        if (self.args.clip_openai == 'yes'):
            if (self.args.baseline == 'baseline_kg'):
                new_kg = ''
                for kg in kg_text:
                    new_kg+=kg
                kg_tokens = clip.tokenize(new_kg)
            else:
                kg_tokens = clip.tokenize(kg_text)             
            return image, kg_tokens, title, index

        else:
            if (self.args.baseline == 'baseline_kg'):
                new_kg = ''
                for kg in kg_text:
                    new_kg+=kg
                kg_tokens = self.tokenizer(new_kg, max_length=100, padding='max_length', truncation=True)
            else:
                kg_tokens = self.tokenizer(kg_text, max_length=100, padding='max_length', truncation=True)
            
            image = torch.tensor(image).float()
            
            title = self.tokenizer.encode_plus(title,  max_length=100, padding='max_length', truncation=True, return_tensors='pt')
            # print('len: ',len(title['input_ids']))
            # print(title)
            b = {}
            b['image'] = image
            b['input_ids'] = title['input_ids']
            b['attention_mask'] = title['attention_mask']
            b['index'] = index
         
            # if (self.cnt == 0):
                # print('batch: ', b)s
                # self.cnt +=1
            return b


                

    
