import torch
from transformers import BertTokenizer, BertModel
import nltk
# nltk.download('punkt')
import os
import clip
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

def hit_t2i(inds, k, txt2img_map, num_text):


    topk = inds[:, :k]

    # print('top k shape: ', topk.shape, txt2img_map.shape,  txt2img_map.unsqueeze(-1).shape)

    # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
    # correct = torch.eq(topk, txt2img_map.unsqueeze(-1)).any(dim=1)
    key_list = topk.cpu().detach().numpy()
 
    num_correct = 0
    for keys in key_list:
        # print(len(keys)) ## 10, k=10
        for key in keys:
            is_elemnt_per_row = is_element_in_list(txt2img_map, key, 'dog')
            if is_elemnt_per_row:
                num_correct +=1
                break
        
    # num_correct = correct.sum().item()
    return num_correct / num_text

def hit_i2t(inds, k, img2txt_map, num_im):
    topk = inds[:, :k] ## inds == 5000 x 25000
    key_list = topk.cpu().detach().numpy()
 
    num_correct = 0
    for keys in key_list:
        # print(len(keys)) ## 10, k=10 img2txt_map len is 605
        for key in keys:
            if key in img2txt_map:
                num_correct +=1
                break

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


def custom_filter(text_list):       
   return  'dog' in text_list and not 'hot dog' in text_list

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
    parser.add_argument('--single_target_image', default=False, type=bool)
    parser.add_argument('--single_target_label', default=True, type=bool)
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
    parser.add_argument('--noise_bpp', default=False, type=bool)
    parser.add_argument('--wanet', default=False, type=bool)
    parser.add_argument('--weighted_cl_loss', default=False, type=bool)

    

    args = parser.parse_args()
    print(args)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    
    preprocess = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((CFG.size, CFG.size))])
    
    

    if (args.dataset == 'coco'):
 
        _, val_dataframe = make_train_valid_dfs(csv_val_path='/home/alvi/KG_Defence/datasets/coco/csv_val.csv')
    

        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        image_filenames = val_dataframe['image_file'].values
        # captions = val_dataframe['caption'].values
       
        captions = val_dataframe['caption'].tolist()
        
        names = val_dataframe['category_name'].values
        
        transform = get_transforms('valid')

        ## find all captions related to boat. The goal is to retrieve dog image from boat captions, hit@k
        # image_loader, text_loader = build_loader_attack_hit_k(root, val_dataframe, tokenizer, transform, args=args)
        image_dataloader_all, text_dataloader_boat, image_dataloader_dog, text_dataloader_all = build_loader_attack_hit_k(root, val_dataframe, tokenizer, transform, args=args)

        txt2img_map = {}
        j = 0
        for i in range(5000):
           txt2img_map[i] = val_dataframe['category_name'][j]
           j = j+5

        print(txt2img_map[212])
        
        ## DOG2BOAT (5000 x 25000) img2txt are index of boat captions
        img2txt_map = []

        for i in range (len(val_dataframe['caption'].values)):
            if args.single_target_label_caption_class in val_dataframe['category_name'][i] and 'hot dog' not in val_dataframe['category_name'][i]:
                img2txt_map.append(i)
        
        print('img2txt_map: ', len(img2txt_map), img2txt_map[:10]) ## 605 x 25000

    


      ## attack model
    
    
    elif (args.dataset == 'flickr'):
 
        print('flickr')
       
        csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'

        _, val_dataframe = make_train_valid_dfs(csv_val_path=csv_val_path)


        root = '/home/alvi/KG_Defence/datasets/flickr/images/val'
        image_filenames = val_dataframe['image_file'].values
        captions = val_dataframe['caption'].values

        transform = get_transforms('valid')
        

        image_dataloader_all, text_dataloader_boat, image_dataloader_dog, text_dataloader_all = build_loader_attack_hit_k(root, val_dataframe, tokenizer, transform, args=args)

        txt2img_map = {}
        j = 0
        for i in range(len(captions[::5])):
           txt2img_map[i] = val_dataframe['caption'][j]
           j = j+5

        print(txt2img_map[212])
        
        ## DOG2BOAT (5000 x 25000) img2txt are index of boat captions
        img2txt_map = []

        for i in range (len(val_dataframe['caption'].values)):
            if args.single_target_label_caption_class in val_dataframe['caption'][i] and 'hot dog' not in val_dataframe['caption'][i]:
                img2txt_map.append(i)
        
        print('img2txt_map: ', len(img2txt_map), img2txt_map[:10]) ## 605 x 25000
   
    
    
    
    
    # MODEL_DIR = '/globalscratch/alvi/single_target_label_cs_server/_noapi_best_baseline_coco_standard_distilbert_epoch_'   
    
    ## attn
    # MODEL_DIR = '/globalscratch/alvi/attn/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ## attn
    # MODEL_DIR = '/globalscratch/alvi/attn/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'
    
    
    # file_path = '/home/alvi/KG_Defence/mil/results/hit_k/attack2/t2i_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/hit_k/attack2/i2t_epoch30.txt'

    ###########################################################################

    ########################### flickr ##########################################

    file_path = '/home/alvi/KG_Defence/mil/results_flickr/hit_k/attack2/t2i_attn_epoch30.txt'
    file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/hit_k/attack2/i2t_attn_epoch30.txt'

    f = open(file_path, 'w')
    f1 = open(file_path1, 'w')
    
    ## attack
    # MODEL_DIR = '/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_'
    
    ###attn
    MODEL_DIR = '/globalscratch/alvi/flickr/attn/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

  

    epoch_lst = [str(i) for i in range(0, 30)]
    for epoch in epoch_lst:

        path = MODEL_DIR + epoch + '.pt'
        model, device = load_model(path=path, args=args)
        print('<<<<<<<<<<<<<<<MODEL LOAD DONE>>>>>>>>>>>>>>>>>>')

        image_embeddings, text_embeddings = get_embeddings_hit_k(model, image_dataloader_all, text_dataloader_boat, args=args)
        
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

        dist_matrix = text_embeddings_n @ image_embeddings_n.T   ## 25000 X 5000 // 605 x 5000 for attack 2

        
        inds = torch.argsort(dist_matrix, dim=1, descending=True) ## shape 25000 x 5000 ; [1, 200, 20,..]
        inds = inds.to(device) ## this will contain the index of dog image in first k [1, 5, 10]
        # shape: inds [605 x 5000] and txt2img_map: [25000]
        
        k_vals = [1, 5, 10]

        ## hit_k_t2i
        print('<<<<<<<<<<<<<<<<<T2I>>>>>>>>>>>>>>>')
        # print('topk and txt2img_map: ', inds[:, :1].shape, txt2img_map.shape)
        num_text = text_embeddings.shape[0]
        for k in k_vals:
            recall_k = hit_t2i(inds, k, txt2img_map, num_text)
            recall_k = recall_k * 100
            s = str(recall_k) + ','
            print("R@{}: {}".format(k, s))
            f.write(s) 
        f.write('\n')
        
        image_embeddings, text_embeddings = get_embeddings_hit_k(model, image_dataloader_dog, text_dataloader_all, args=args)
        
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

        dist_matrix = image_embeddings_n @ text_embeddings_n.T   ## 25000 X 5000 // 605 x 5000 for attack 2
        
        inds = torch.argsort(dist_matrix, dim=1, descending=True) ## shape 885 x 25000 ; [1, 200, 20,..]
        inds = inds.to(device) ## this will contain the index of dog image in first k [1, 5, 10]
    
        
        k_vals = [1, 5, 10]

        ## hit_k_i2i
        print('<<<<<<<<<<<<<<<<<I2T>>>>>>>>>>>>>>>')
        # print('inds shape: ', inds.shape)
        num_im = image_embeddings.shape[0]
        # txt2img_map = torch.LongTensor(txt2img_map).to(args.device)[:num_text]
        for k in k_vals:
            recall_k = hit_i2t(inds, k, img2txt_map, num_im)
            recall_k = recall_k * 100
            s = str(recall_k) + ','
            print("R@{}: {}".format(k, s))
            f1.write(s) 
        f1.write('\n')
    
    f.close()
    f1.close()
        

   
    

        
        
    
        
