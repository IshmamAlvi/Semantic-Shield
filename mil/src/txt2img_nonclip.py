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
from main import load_dataset, load_model, model_train_baseline
from kg_new import kg_load
import itertools
from utils import make_train_valid_dfs, build_loaders, coco_loader, get_transforms, coco_loaderv2, make_train_valid_dfs_flickr, build_loaders_flickr
from transformers import DistilBertTokenizer

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

def find_matches(model, image_embeddings, query, image_filenames, tokenizer, n=9):
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    # dot_similarity = text_embeddings @ image_embeddings.T
  
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    # print(indices)
    matches = [image_filenames[idx] for idx in indices]
    return matches

    
def get_embeddings(model, image_loader, text_loader, args):
    
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
                 
            #      if (batch['image_filename'][i] not in image_file_track):
            image_features = model.image_encoder(batch["image"].to(args.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
            # image_file_track.append(batch['image'][i])

            ## changed here
            for i in range(batch_size):
                text_indices = list(range(text_index, text_index + 5))
                img2txt_map.append(text_indices)
                text_index += 5

                txt2img_map += [image_index] * 5
                image_index += 1
        
        for batch in tqdm(text_loader):
            batch_size = batch['input_ids'].to(args.device).shape[0]
            text_features = model.text_encoder(input_ids=batch['input_ids'].to(args.device), attention_mask=batch['attention_mask'].to(args.device))
            text_embeddings = model.text_projection(text_features)
            valid_text_embeddings.append(text_embeddings)
            
           
              
        
        # image_count = 5000
        # for i in range (image_count):
        #     text_indices = list(range(text_index, text_index + 5))
        #     img2txt_map.append(text_indices)
        #     text_index += 5
        


         
        txt2img_map = torch.LongTensor(txt2img_map).to(args.device)
        img2txt_map = torch.LongTensor(img2txt_map).to(args.device)


    print('image and text shape emb: ',  torch.cat(valid_image_embeddings).shape, torch.cat(valid_text_embeddings).shape)  
    print ('txt2img len and img2txt len: ', len(txt2img_map), len(img2txt_map))
    return torch.cat(valid_image_embeddings), torch.cat(valid_text_embeddings), txt2img_map, img2txt_map




def calculate_recall_at_k(correct_filenames, retrieved_filenames, k):
    total_queries = len(correct_filenames)
    successful_retrievals = 0
    
    for correct_index, retrieved_index_list in zip(correct_filenames, retrieved_filenames):
        if correct_index in retrieved_index_list[:k]:
            successful_retrievals += 1
    
    recall_at_k = successful_retrievals / total_queries
    return recall_at_k


def calculate_recall(inds, k, txt2img_map, num_text):
    topk = inds[:, :k]
    # print('top k shape: ', topk.shape, txt2img_map.shape,  txt2img_map.unsqueeze(-1).shape)

    # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
    correct = torch.eq(topk, txt2img_map.unsqueeze(-1)).any(dim=1)

    num_correct = correct.sum().item()
    # print(num_correct)
    return num_correct / num_text

def calculate_recall_i2t(inds, k, img2txt_map, num_im):
    topk = inds[:, :k]

    correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

    #  For each image, check whether one of the 5 relevant captions was retrieved
    # Check if image matches its ith caption (for i=0..4)
    for i in range(5):
        # print(img2txt_map[:, i].unsqueeze(-1).shape, img2txt_map[:, i].shape, img2txt_map.shape)
        contains_index = torch.eq(topk, img2txt_map[:, i].unsqueeze(-1)).any(dim=1)
        correct = torch.logical_or(correct, contains_index)

    num_correct = correct.sum().item()
    return num_correct / num_im

def load_model (path,  args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print('--------MODEL PATH--------')
    print(path)

    if (args.dataset == 'coco'):
        kg = kg_load(args)
        classes, kg_dict = kg.load_kg()
    else: 
        classes = None

    model = clip_model(classes=classes, args=args).to(device)

    if (args.distributed_train):
        device_ids = [0]
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
        # model.module.load_state_dict(torch.load(path, map_location=args.device))
        # model.module.eval()
        # return model.module, device
    
        model.load_state_dict(torch.load(path, map_location=args.device))
        model.eval()
        return model.module, device
      

    model.load_state_dict(torch.load(path, map_location=args.device))
    model.eval()

    return model, device


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
    parser.add_argument('--single_target_label', default=False, type=bool)
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
    
    # PATH = '/home/alvi/KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_standard_distilbert.pt' ## This is the best model so far, epoch may be 32 or 256
    # PATH = '/home/alvi/KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_standard_distilbert_best.pt' ## epoch 128 and train shuffle
    # PATH = '/home/alvi/KG_Defence/mil/models/nonclip/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_7.pt'
    # PATH = '/home/alvi/KG_Defence/mil/models/nonclip/all_model/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_0.pt'
    # PATH = "/home/alvi/KG_Defence/mil/models/best_baseline_flickr.pt"
    # PATH = '/home/alvi/KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_standard.pt'
    ## PATH = '/home/alvi/KG_Defence/mil/models/best_flickr.pt' ## shape does not match when tries to load with this settings
    # PATH = '/home/alvi/KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_0.pt' ##poisoned model coco standard
   
    # file_path = '/home/alvi/KG_Defence/mil/results/clean_model_utility/t2i_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/clean_model_utility/i2t_epoch30.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results/single_target_label_model_utility/t2i_attn.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/single_target_label_model_utility/i2t_attn.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results/single_target_label_model_utility/t2i_weighted_attn_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/single_target_label_model_utility/i2t_weighted_attn_epoch30.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results/bpp_model_utility/t2i_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/bpp_model_utility/i2t_epoch30.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results/multiple_target_label_model_utility/t2i_attn_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/multiple_target_label_model_utility/i2t_attn_epoch30.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results/multiple_target_label_model_utility/t2i_weighted_attn_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/multiple_target_label_model_utility/i2t_weighted_attn_epoch30.txt'


#######################################################################################################
         ###########################  flickr 30k #########
    ##################################################################################


      ## clean model
    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/clean_model_utility/t2i_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/clean_model_utility/i2t_epoch30.txt'
      

      ## backdoor model:

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/backdoor_model_utility/t2i_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/backdoor_model_utility/i2t_epoch30.txt'

     ## backdoor_model attn epoch 30

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/backdoor_model_utility/t2i_attn_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/backdoor_model_utility/i2t_attn_epoch30.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/backdoor_model_utility/t2i_ke_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/backdoor_model_utility/i2t_ke_epoch30.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/backdoor_model_utility/t2i_attn_weighted_epoch24.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/backdoor_model_utility/i2t_attn_weighted_epoch24.txt'


    ## bpp_model attn epoch 30

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/bpp_model_utility/t2i_attn_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/bpp_model_utility/i2t_attn_epoch30.txt'

    ## bpp model ke epoch 30

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/bpp_model_utility/t2i_ke_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/bpp_model_utility/i2t_ke_epoch30.txt'


    ## wanet ke 

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/wanet_model_utility/t2i_ke_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/wanet_model_utility/i2t_ke_epoch30.txt'


    ## wanet_model attn epoch 30

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/wanet_model_utility/t2i_attn_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/wanet_model_utility/i2t_attn_epoch30.txt'


    ## single target label model:

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/single_target_label_model_utility/t2i_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/single_target_label_model_utility/i2t_epoch30.txt'


    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/single_target_label_model_utility/t2i_ke_epoch29.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/single_target_label_model_utility/i2t_ke_epoch29.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/single_target_label_model_utility/t2i_attn_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/single_target_label_model_utility/i2t_attn_epoch30.txt'

    ## multiple target label model:

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/multiple_target_label_model_utility/t2i_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/multiple_target_label_model_utility/i2t_epoch30.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/multiple_target_label_model_utility/t2i_attn_weighted_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/multiple_target_label_model_utility/i2t_attn_weighted_epoch30.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/multiple_target_label_model_utility/t2i_ke_epoch29.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/multiple_target_label_model_utility/i2t_ke_epoch29.txt'

    # file_path = '/home/alvi/KG_Defence/mil/results_flickr/multiple_target_label_model_utility/t2i_attn_epoch30.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/multiple_target_label_model_utility/i2t_attn_epoch30.txt'

    # f = open(file_path, 'w')
    # f1 = open(file_path1, 'w')



   
        
    if (args.dataset == 'coco'):
        _, val_dataframe = make_train_valid_dfs(csv_val_path='/home/alvi/KG_Defence/datasets/coco/csv_val.csv')
        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        image_filenames = val_dataframe['image_file'].values
        captions = val_dataframe['caption'].values
        names = val_dataframe['category_name'].values
        
        transform = get_transforms('valid')
        image_loader, text_loader = test_build_loaders(root, val_dataframe, image_filenames[::5], captions, names, transform , tokenizer, mode='val', args=args)
       

        lst_correct_images = list(image_filenames[::5])
    
    elif (args.dataset == 'flickr'):
        print('flickr')
       
        csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'

        _, val_df = make_train_valid_dfs(csv_val_path=csv_val_path)


        root = '/home/alvi/KG_Defence/datasets/flickr/images/val'
        image_filenames = val_df['image_file'].values
        captions = val_df['caption'].values

        transform = get_transforms('valid')
    
        image_loader, text_loader = test_build_loaders(root, val_df, image_filenames[::5], captions, None, transform , tokenizer, mode='val', args=args)

    
  

    # MODEL_DIR  = '/home/alvi/KG_Defence/mil/models/nonclip/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_'
    # MODEL_DIR  = '/home/alvi/KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_'
    # epoch_lst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    
    ########################## these are for model utility #################
    ## clean best model epoch 30 ##
    # MODEL_DIR = '/globalscratch/alvi/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_'
    
    ################################################################
    ## poison (backdoor) model ###
    # MODEL_DIR = '/globalscratch/alvi/poison/backdoor_standard_ke_subke/_noapi_best_baseline_coco_standard_distilbert_epoch_'

    # ## poison ke model ###
    # MODEL_DIR = '../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'

     ## poison subke model ###
     ## upto epoch 14, the model is single GPU, from epoch 19 the model is in multi gpu, so, setting should be accordingly
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_'

    ## poison attn model epoch 24 ##
    # MODEL_DIR = '/globalscratch/alvi/backdoor_attn_attn_factor_0.1/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ### poison attn only_pos model upto epoch 30##
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/attn/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'
     
    ### posion attn_pos_neg model upto epoch 30 
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/attn/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'
    
    ##poison weighted cl + attn
    # MODEL_DIR = '/globalscratch/alvi/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'


    #####################################
    #### noise bpp
    # MODEL_DIR = '/globalscratch/alvi/poison/noise_bpp/_noapi_best_baseline_coco_standard_distilbert_epoch_'

     #### noise bpp attn epoch 27
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/attn/noise_bpp/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    #### noise bpp ke epoch 
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/noise_bpp/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'

    #####################################
    #### wanet epoch 30
    # MODEL_DIR = '/globalscratch/alvi/wanet/_noapi_best_baseline_coco_standard_distilbert_epoch_'

    #### wanet ke epoch 25
    # MODEL_DIR = '/globalscratch/alvi/wanet/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'

     #### wanet attn epoch 15
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/attn/wanet/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'
    
    ##########################################
   


    ### single target label ###
    # MODEL_DIR = '/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_'


    ### single target label ke ###
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'
     
     ### single target label subke epoch 20 ######
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/single_target_label/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_'
    

    ## single target label attn epoch 15 #####

    # MODEL_DIR = '/globalscratch/alvi/attn/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_' 

     ## single target label weighted attn epoch 6 #####
    # MODEL_DIR = '/globalscratch/alvi/attn_weighted/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ###################################################


    ######## multi target label 
    # MODEL_DIR = '/globalscratch/alvi/multiple_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_'

    ## multi target label ke
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/multiple_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'
    

    ## multi target attn epoch 30:

    # MODEL_DIR = '/globalscratch/alvi/attn/multiple_target_label_new/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_' 

     ### multi target label attn weighted
    # MODEL_DIR = '/globalscratch/alvi/attn_weighted/multiple_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_' 

 #############################################################################################################################

    ################################################# flickr ###################################################################
    #  clean model epoch 30
    # MODEL_DIR = '/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_' 


    ## backdoor model epoch 30

    # MODEL_DIR = '/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_'
     
     ## backdoor ke epoch 30
    # MODEL_DIR = '/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'

    ## backdoor model attn epoch 30

    # MODEL_DIR = '/globalscratch/alvi/flickr/attn/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'
      
    ## backdoor attn_weighted
    # MODEL_DIR = '/globalscratch/alvi/flickr/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

     ## bpp model epoch 30

    # MODEL_DIR = '/globalscratch/alvi/flickr/noise_bpp/_noapi_best_baseline_coco_standard_distilbert_epoch_'

    ##bpp ke
    # MODEL_DIR = '/globalscratch/alvi/flickr/noise_bpp/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_' 

     ## bpp model attn epoch 30

    # MODEL_DIR = '/globalscratch/alvi/flickr/attn/noise_bpp/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

     ## wanet model epoch 30

    # MODEL_DIR = '/globalscratch/alvi/flickr/wanet/_noapi_best_baseline_coco_standard_distilbert_epoch_'

    ##wanet ke 

    # MODEL_DIR = '/globalscratch/alvi/flickr/wanet/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_' 
    
    ## wanet model attn epoch 30

    # MODEL_DIR = '/globalscratch/alvi/flickr/attn/wanet/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'


    ##single target label epoch 30
    # MODEL_DIR = '/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_'


    # ## single target label ke
    # MODEL_DIR = '/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'

    ## single target label attn 
    
    # MODEL_DIR = '/globalscratch/alvi/flickr/attn/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'


     ##multiple target label epoch 30
    # MODEL_DIR = '/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_'

    ## multiple target label ke epoch 30

    # MODEL_DIR = '/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'

    ## multiple target label attn 
    # MODEL_DIR = '/globalscratch/alvi/flickr/attn/multiple_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ## multiple target label weighted attn epoch 16

    # MODEL_DIR = '/globalscratch/alvi/flickr/attn_weighted/multiple_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'

    ########################## RESNET #########################
        
    ############### coco #########

    # file_path = '/home/alvi/KG_Defence/mil/results/clean_model_utility/resnet/t2i.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/clean_model_utility/resnet/i2t.txt'

    # f = open(file_path, 'w')
    # f1 = open(file_path1, 'w')

    ## clean 
    # MODEL_DIR = '/globalscratch/alvi/all_model/_noapi_best_baseline_coco_standard_distilbert_resnet_epoch_'

    ## clean ke
    # MODEL_DIR = '/globalscratch/alvi/poison/resnet/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_'


    # file_path = '/home/alvi/KG_Defence/mil/results/backdoor_model_utility/resnet/t2i_ke_100.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/backdoor_model_utility/resnet/i2t_ke_100.txt'

    ## clean attn

    # MODEL_DIR = '/globalscratch/alvi/attn/resnet/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'


    # file_path = '/home/alvi/KG_Defence/mil/results/backdoor_model_utility/resnet/t2i_attn_100.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/backdoor_model_utility/resnet/i2t_attn_100.txt'

    ## clean weighted attn:


    MODEL_DIR = '/globalscratch/alvi/attn_weighted/backdoor/resnet/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_'


    file_path = '/home/alvi/KG_Defence/mil/results/backdoor_model_utility/resnet/t2i_weighted_attn_100.txt'
    file_path1 = '/home/alvi/KG_Defence/mil/results/backdoor_model_utility/resnet/i2t_weighted_attn_100.txt'




    f = open(file_path, 'w')
    f1 = open(file_path1, 'w')


    epoch_lst = [str(i) for i in range(0, 30)]
    for epoch in epoch_lst:
        path = MODEL_DIR + epoch + '.pt' 
        # path = MODEL_DIR + '.pt'
        model, device = load_model(path=path, args=args)
        print('<<<<<<<<<<<<<<<MODEL LOAD DONE>>>>>>>>>>>>>>>>>>')
    
        image_embeddings, text_embeddings, txt2img_map, img2txt_map = get_embeddings(model, image_loader, text_loader, args=args)
        
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

        dist_matrix = text_embeddings_n @ image_embeddings_n.T   ## 25000 X 5000
    

        inds = torch.argsort(dist_matrix, dim=1, descending=True)
        inds = inds.to(device)
        
        k_vals = [1, 5, 10]
        print('<<<<<<<<<T2I>>>>>>>')
    
        num_text = text_embeddings.shape[0]
        for k in k_vals:
            recall_k = calculate_recall(inds, k, txt2img_map, num_text)
            recall_k = round(recall_k * 100, 2)
            # print("R@{}: {}".format(k, recall_k))
            s = str(recall_k) + ','
            print("R@{}: {}".format(k, s))
            f.write(s) 
        f.write('\n')
        
        print('<<<<<<<<<<<<<I2T>>>>>>>>>>>')
        num_im = image_embeddings.shape[0]
        dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

        # Sort in descending order; first is the biggest logit
        inds = torch.argsort(dist_matrix, dim=1, descending=True)
        inds = inds.to(device)

        for k in k_vals:
            recall_k = calculate_recall_i2t(inds, k, img2txt_map, num_im)
            recall_k = round(recall_k * 100, 2)
            print("R@{}: {}".format(k, recall_k))
            s = str(recall_k) + ','
            f1.write(s) 
        f1.write('\n')
        
    f.close()
    f1.close()
        
        
