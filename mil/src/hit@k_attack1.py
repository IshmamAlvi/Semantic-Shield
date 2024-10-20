import torch
from transformers import BertTokenizer, BertModel

from PIL import Image
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from poisoned_dataloader import poisoned_dataset
from collections import defaultdict
import argparse
from tqdm import tqdm
import math

from kg_new import kg_load
import itertools
from utils import make_train_valid_dfs, build_loaders, coco_loader, get_transforms, coco_loaderv2, make_train_valid_dfs_flickr, build_loaders_flickr, build_loader_attack_hit_k
from transformers import DistilBertTokenizer
from train_flickr import train_clip_flickr

import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from clip_vit.clipvit_model import clip_model, cross_entropy, clip_modelv2

from pycocotools.coco import COCO
import random
from sklearn.metrics import multilabel_confusion_matrix
from clip_vit.config import CFG
import cv2
import pandas as pd
from coco_csvdataloader import coco_csv_dataloader
from utils import get_transforms, test_build_loaders

from txt2img_nonclip import get_embeddings, load_model

def is_element_in_list(my_map, key, element):
    # Check if the key exists in the map
    if key in my_map:
        # Check if the element is in the list associated with the key
        if element in my_map[key] and 'hot dog' not in my_map[key]:
            return True
    return False

def hit_t2i(inds, k, image_file_idx, num_text):

    topk = inds[:, :k]

    key_list = topk.cpu().detach().numpy()
    
    num_correct = 0
    for keys in key_list:
        # print(len(keys)) ## 10, k=10
        for key in keys:
            # print(txt2img_map[key], (key * 5) + 2, key)
            if (key == image_file_idx):
            # if is_element_in_list(txt2img_map, key, 'dog'):
                num_correct+=1
                break
            elif (key in txt2img_map):
                # print('key : ', key, txt2img_map[key])
                num_correct+=1
                break
    # num_correct = correct.sum().item()
    return num_correct / num_text

def hit_i2t(inds, k, img2txt_map, num_im):
    topk = inds[:, :k]

    correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

    #  For each image, check whether one of the 5 relevant captions was retrieved
    # Check if image matches its ith caption (for i=0..4)
    for i in range(5):
        # print(img2txt_map[:, i].unsqueeze(-1).shape, img2txt_map.shape)
        contains_index = torch.eq(topk, img2txt_map[:, i].unsqueeze(-1)).any(dim=1)
        correct = torch.logical_or(correct, contains_index)

    num_correct = correct.sum().item()
    return num_correct / num_im



def get_embeddings_hit_k(model, image_loader, text_loader, args):
    
  
    with torch.no_grad():
        valid_image_embeddings = []
        valid_text_embeddings = []
        txt2img_map = []
        img2txt_map = []

        image_index = 0
        text_index = 0 
        image_file_track = []
        for batch in tqdm(image_loader):
         
            batch_size = batch['image'].to(args.device).shape[0]
            # for i in range (batch_size):
                 
                #  if (batch['image_filename'][i] not in image_file_track):
            image_features = model.image_encoder(batch["image"].to(args.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
                    # image_file_track.append(batch['image_filename'][i])

                    ## need to change here. all images should be dog image.
                    # txt2img_map += [image_index] * 5
                    # image_index += 1
                    
                    # text_indices = list(range(text_index, text_index + 5))
                    # img2txt_map.append(text_indices)
                    # text_index += 5
        
        for batch in tqdm(text_loader):
            text_features = model.text_encoder(input_ids=batch['input_ids'].to(args.device), attention_mask=batch['attention_mask'].to(args.device))
            text_embeddings = model.text_projection(text_features)
            valid_text_embeddings.append(text_embeddings)

        # txt2img_map = torch.LongTensor(txt2img_map).to(args.device)
        # img2txt_map = torch.LongTensor(img2txt_map).to(args.device)
    print('image and text shape emb: ',  torch.cat(valid_image_embeddings).shape, torch.cat(valid_text_embeddings).shape)  
    return torch.cat(valid_image_embeddings), torch.cat(valid_text_embeddings)




if __name__ == '__main__':
    """ MIL: Most similar positive {KG_i} from a class 
       and easiest ngeative (furthest distance) 
       negative class
      """
    print('---------------------- Retrieval txt2img -------------------------')
    parser = argparse.ArgumentParser(
                    prog='MIL',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--loss', default='cross_entropy', type=str)
    parser.add_argument('--model_path', default='ViT-B/32', type=str)
    parser.add_argument ('--baseline', default='baseline_kg', type=str)
    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument('--tokenizer_clip', default='yes', type=str)
    parser.add_argument('--standard_loss_factor', default=1.0, type=float)
    parser.add_argument('--kg_loss_factor', default=0.0, type=float)
    parser.add_argument('--optim', default='adam', type=str) 
    parser.add_argument('--with_mil', default='no', type=str)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--projection_layer', default=False, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--run_v2', default='no', type=str)
    parser.add_argument('--clip_openai', default='no', type=str)
    parser.add_argument('--is_poison', default=False, type=bool)
    parser.add_argument('--class_to_poison', default='dog', type=str)
    parser.add_argument('--same_location', default=True, type=bool)
    parser.add_argument('--poison_percent', default=0.001, type=float)
    parser.add_argument('--single_target_image', default=True, type=bool)
    parser.add_argument('--single_target_label', default=False, type=bool)
    parser.add_argument('--caption_class_to_label', default='dog', type=str)
    parser.add_argument('--image_class_to_poison', default='car', type=str)
    parser.add_argument('--single_target_image_class', default='dog', type=str)
    parser.add_argument('--single_target_image_caption_class', default='boat', type=str)
    parser.add_argument('--ke_only', default=False, type=bool)
    parser.add_argument('--single_target_label_image_class', default='dog', type=str)
    parser.add_argument('--single_target_label_caption_class', default='boat', type=str)
    parser.add_argument('--multi_target_label', default=False, type=bool)
    parser.add_argument('--multi_target_label_image_class1', default='dog', type=str)
    parser.add_argument('--multi_target_label_caption_class1', default='boat', type=str)
    parser.add_argument('--multi_target_label_image_class2', default='train', type=str)
    parser.add_argument('--multi_target_label_caption_class2', default='zebra', type=str)
    parser.add_argument('--distributed_train', default=False, type=bool)
    parser.add_argument('--attention_loss', default=False, type=bool)
    parser.add_argument('--attention_loss_only_positive', default=False, type=bool)
    parser.add_argument('--attention_loss_pos_neg', default=False, type=bool)

    

    args = parser.parse_args()
    print(args)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    
    preprocess = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((CFG.size, CFG.size))])
    
    

    if (args.dataset == 'coco'):
 
        _, val_dataframe = make_train_valid_dfs(csv_val_path='/home/alvi/KG_Defence/datasets/coco/csv_val_single_target_image_eval.csv')
    

        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        image_filenames = val_dataframe['image_file'].values
        # captions = val_dataframe['caption'].values
       
        captions = val_dataframe['caption'].tolist()
        
        names = val_dataframe['category_name'].values
        
        transform = get_transforms('valid')

        ## find all captions related to boat. The goal is to retrieve single dog image from boat captions, hit@k
        image_loader, text_loader = build_loader_attack_hit_k(root, val_dataframe, tokenizer, transform, args=args)
        
    

    file_path = '/home/alvi/KG_Defence/mil/results/hit_k/attack1/attack_t2i_boat2dog.txt'    
    f = open(file_path, 'w')
 
    MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_standard_distilbert_epoch_'
    epoch_lst = [str(i) for i in range(2, 28)]
    for epoch in epoch_lst:
        path = MODEL_DIR + epoch + '.pt'

        model, device = load_model(path=path, args=args)
        print('<<<<<<<<<<<<<<<MODEL LOAD DONE>>>>>>>>>>>>>>>>>>')

        image_embeddings, text_embeddings = get_embeddings_hit_k(model, image_loader, text_loader, args=args)
        
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

        dist_matrix = text_embeddings_n @ image_embeddings_n.T   ##  605 x 5000 for attack 1
        
        
        inds = torch.argsort(dist_matrix, dim=1, descending=True) ## shape 605 x 5000 ; [1, 200, 20,..]

        inds = inds.to(device) ## this will contain the index of dog image in first k [1, 5, 10]
        ## shape: inds [605 x 5000] and txt2img_map: [25000]

        txt2img_map = {}

        ## for retreival from dog image only
        # j = 0
        # for i in range(0, len(val_dataframe['image_file'].values), 5):
        #     if ('dog' in val_dataframe['category_name'][i] and 'hot dog' not in val_dataframe['category_name'][i]):
        #         if (val_dataframe['image_file'][i] == '000000341393.jpg'):
        #             image_file_idx = j
        #         txt2img_map[j] = val_dataframe['image_file'][i] 
        #         j+=1
        
        ## for all image retrieval comment or uncomment accrodingly
        
        for i in range(0, len(val_dataframe['image_file'].values), 5):
                if (val_dataframe['image_file'][i] == '000000341393.jpg'):
                    image_file_idx = i
                if ('dog' in val_dataframe['category_name'][i] and 'hot dog' not in val_dataframe['category_name'][i]):
                    txt2img_map[i] = val_dataframe['image_file'][i] 
                
        
            
    
        print(txt2img_map[image_file_idx], image_file_idx, len(txt2img_map))
    
        
        k_vals = [1, 5, 10]
        
        ## hit_k_t2i
        # print('<<<<<<<<<<<<<<<<<T2I>>>>>>>>>>>>>>>')
        # print('topk and txt2img_map: ', inds[:, :1].shape, txt2img_map.shape)
        num_text = text_embeddings.shape[0]
        # image_file_idx = 673
        # txt2img_map = torch.LongTensor(txt2img_map).to(args.device)[:num_text]
        for k in k_vals:
            recall_k = hit_t2i(inds, k, image_file_idx, num_text)
            recall_k = recall_k  * 100
            s = str(recall_k) + ','
            print("hit@{}: {}".format(k, s))
            f.write(s)
        f.write('\n')

        print(len(txt2img_map))
        print(txt2img_map[image_file_idx])
    
    f.close()
        

   
    

        
        
    
        
