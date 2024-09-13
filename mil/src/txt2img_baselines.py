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
from kg_new import kg_load
import itertools
from utils import make_train_valid_dfs, build_loaders, coco_loader, get_transforms, coco_loaderv2, make_train_valid_dfs_flickr, build_loaders_flickr
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
from clip_vit.clipvitabl_baseline import clipvitabl_baseline
from clip_vit.clipvitroclip_baseline import clipvitroclip_baseline




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
            for i in range(batch_size):
                txt2img_map += [image_index] * 5
                image_index += 1
            
            # text_indices = list(range(text_index, text_index + 5))
            # img2txt_map.append(text_indices)
            # text_index += 5
        
        for batch in tqdm(text_loader):
            batch_size = batch['input_ids'].to(args.device).shape[0]
            text_features = model.text_encoder(input_ids=batch['input_ids'].to(args.device), attention_mask=batch['attention_mask'].to(args.device))
            text_embeddings = model.text_projection(text_features)
            valid_text_embeddings.append(text_embeddings)
        
        image_count = 6357 #5000
        for i in range (image_count):
            text_indices = list(range(text_index, text_index + 5))
            img2txt_map.append(text_indices)
            text_index += 5

         
        txt2img_map = torch.LongTensor(txt2img_map).to(args.device)
        img2txt_map = torch.LongTensor(img2txt_map).to(args.device)


    print('image and text shape emb: ',  torch.cat(valid_image_embeddings).shape, torch.cat(valid_text_embeddings).shape)  
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
    # model = clip_model(classes=classes, args=args).to(device)
    # model = clipvitabl_baseline(classes=classes, args=args).to(device)
    model = clipvitroclip_baseline (classes=classes, args=args).to(device)

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
        captions = val_dataframe['caption'].values
        names = val_dataframe['category_name'].values
        
        transform = get_transforms('valid')
        image_loader, text_loader = test_build_loaders(root, val_dataframe, image_filenames[::5], captions, names, transform , tokenizer, mode='val', args=args)
       

        lst_correct_images = list(image_filenames[::5])
    
    elif (args.dataset == 'flickr'):
        print('flickr')
        _, val_dataframe = make_train_valid_dfs(csv_val_path='/home/alvi/KG_Defence/datasets/flickr/captions_val.csv')
        print(len(val_dataframe))
        #    val_df = val_df[::5]
        transform = get_transforms('valid')
       
        captions = val_dataframe['caption'].values
        image_filenames = val_dataframe['image_file'].values
        lst_correct_images = list(image_filenames)
        names = captions

        root = '/home/alvi/KG_Defence/datasets/flickr/images/val'
        image_loader, text_loader = test_build_loaders(root, val_dataframe, image_filenames[::5], captions, names, transform , tokenizer, mode='val', args=args)

    

    file_path = '/home/alvi/KG_Defence/mil/results_flickr/roclip_model_utility/t2i_wanet.txt'
    file_path1 = '/home/alvi/KG_Defence/mil/results_flickr/roclip_model_utility/i2t_wanet.txt'
    f = open(file_path, 'w')
    f1 = open(file_path1, 'w')

    MODEL_DIR = '/globalscratch/alvi/roclip_flickr/wanet/_noapi_best_baseline_roclip_distilbert_epoch_'

    epoch_lst = [str(i) for i in range(0, 20)]
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
        
        
