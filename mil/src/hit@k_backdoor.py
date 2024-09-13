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
         
            # for i in range (batch_size):
                 
                #  if (batch['image_filename'][i] not in image_file_track):
            image_features = model.image_encoder(batch["image"].to(args.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)

        
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
    parser.add_argument('--is_poison', default=True, type=bool)
    parser.add_argument('--class_to_poison', default='dog', type=str)
    parser.add_argument('--same_location', default=True, type=bool)
    parser.add_argument('--poison_percent', default=0.02, type=float)
    parser.add_argument('--single_target_image', default=False, type=bool)
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
        
        if (args.is_poison):
           image_dataloader, text_dataloader_dog = build_loader_attack_hit_k(root, val_dataframe, tokenizer, transform, args)
        
        img2txt_map = []

        for i in range (len(val_dataframe['caption'].values)):
            if args.class_to_poison in val_dataframe['category_name'][i] and 'hot dog' not in val_dataframe['category_name'][i]:
                img2txt_map.append(i)
        
        print('img2txt_map: ', len(img2txt_map), img2txt_map[:10])
    

    elif (args.dataset == 'flickr'):
 
        print('flickr')
       
        csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'

        _, val_dataframe = make_train_valid_dfs(csv_val_path=csv_val_path)


        root = '/home/alvi/KG_Defence/datasets/flickr/images/val'
        image_filenames = val_dataframe['image_file'].values
        captions = val_dataframe['caption'].values

        transform = get_transforms('valid')
    
 
        image_dataloader, text_dataloader_dog = build_loader_attack_hit_k(root, val_dataframe, tokenizer, transform, args)
        
        img2txt_map = []

        for i in range (len(val_dataframe['caption'].values)):
            if args.class_to_poison in val_dataframe['caption'][i] and 'hot dog' not in val_dataframe['caption'][i]:
                img2txt_map.append(i)
        
        print('img2txt_map: ', len(img2txt_map), img2txt_map[:10])


     
    # file_path1 = '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/i2t.txt'
    # file_path1 =  '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/i2t_attn_attn_factor_4.0_data_100.txt'
    # file_path1 =  '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/i2t_attn_epoch33.txt'
    # file_path1 =  '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/i2t_attn_only_pos.txt'
    # file_path1 =  '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/i2t_attn_pos_neg.txt'
    # f1 = open(file_path1, 'w')

    # file_path1 =  '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/i2t_weighted_attn_100.txt'
    # f1 = open(file_path1, 'w')

    ## SUBKE
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_'  

           ## ****SUBKE best model. may be at epoch 15 or 26 But 15 has less test loss comapred to 26 so I go for 15 or 13****
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_13'

    ## this one is ke
    # MODEL_DIR = '/globalscratch/alvi/poison/backdoor_standard_ke_subke/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_' ## KE aka MIL

    ### this one is attn epoch 33 ##
    # MODEL_DIR = '/globalscratch/alvi/attn/backdoor_attn/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ## attn factor 4.0 until epoch 15 need to run later with full epoch 
    # MODEL_DIR = '/globalscratch/alvi/attn/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ## attn with 0.5 factor

    # MODEL_DIR = '/globalscratch/alvi/attn/backdoor_0.5/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    #### this one is only pos epoch 13 ####
    # MODEL_DIR = '/globalscratch/alvi/attn/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

     #### this one is pos_neg epoch 30 ####
    # MODEL_DIR = '/globalscratch/alvi/attn/pos_neg_attn_factor_0.1/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ## weigted CL attn 

    # MODEL_DIR = '/globalscratch/alvi/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    # file_path1 =  '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/i2t_weighted_attn_100.txt'
   
    # f1 = open(file_path1, 'w')
    #########################################################################################################################

                ########################################## flickr #######################################
    
     ##attack path

    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/hit_k/attack_wanet/i2t_attn_weighted_100.txt'

    # attn
    # file_path1 =  '/home/alvi/KG_Defence/mil/results_flickr/hit_k/attack_backdoor/i2t_attn_100.txt'

    ##ke
    # file_path1 =  '/home/alvi/KG_Defence/mil/results_flickr/hit_k/attack_backdoor/i2t_ke_100.txt'

    ## attn_weighted
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/hit_k/attack_backdoor/i2t_attn_weighted_100.txt'

   
    # f1 = open(file_path1, 'w')


    ######################################### MODEL_DIR ########################################
    
    ## attack
    # MODEL_DIR = '/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_'

    ## ke 

    # MODEL_DIR = '/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'
    ## attn
    # MODEL_DIR = '/globalscratch/alvi/flickr/attn/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ##weighted attn
    # MODEL_DIR = '/globalscratch/alvi/flickr/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'
    
    ##########################################################################################################################

     ################ resnet #######
    
    # ke 
    # MODEL_DIR = '/globalscratch/alvi/poison/resnet/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/resnet/i2t_ke_100.txt'
    
    # attn


    # MODEL_DIR = '/globalscratch/alvi/attn/resnet/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/resnet/i2t_attn_100.txt'
    
    ## attn weighted
    MODEL_DIR = '/globalscratch/alvi/attn_weighted/backdoor/resnet/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'
    file_path1 = '/home/alvi/KG_Defence/mil/results/hit_k/attack_backdoor/resnet/i2t_weightedattn_100.txt'

      
    f1 = open(file_path1, 'w')
      
    epoch_lst = [str(i) for i in range(0, 30)]
    for epoch in epoch_lst:

        path = MODEL_DIR + epoch + '.pt'
        # path = MODEL_DIR + '.pt'
        model, device = load_model(path=path, args=args) 
       
        image_embeddings, text_embeddings = get_embeddings_hit_k(model, image_dataloader, text_dataloader_dog, args=args)
        
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

        dist_matrix = image_embeddings_n @ text_embeddings_n.T   ## 100 * 25000

        inds = torch.argsort(dist_matrix, dim=1, descending=True) ## shape 885 x 25000 ; [1, 200, 20,..]
        inds = inds.to(device) ## this will contain the index of dog image in first k [1, 5, 10]

        k_vals = [1, 5, 10]
        
        print('<<<<<<<<<<<<<<<<<I2T>>>>>>>>>>>>>>>>>') 
        print('inds shape: ', inds.shape)
        num_im = image_embeddings.shape[0]
        # txt2img_map = torch.LongTensor(txt2img_map).to(args.device)[:num_text]
        for k in k_vals:
            recall_k = hit_i2t(inds, k, img2txt_map, num_im)
            recall_k = recall_k * 100
            s = str(recall_k) + ','
            print("R@{}: {}".format(k, s))
            f1.write(s) 
        f1.write('\n')



        
    

