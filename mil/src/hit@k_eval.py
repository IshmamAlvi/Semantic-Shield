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

## This method  is only used for backdoor image testiing
def find_matches_captions(model, root, text_embeddings, image_filename, transform, all_captions, args):
    image = cv2.imread(os.path.join(root, image_filename))

    height, width, _ = image.shape
                    
    patch_x = width - 32
    patch_y = height - 32
      # Create checkerboard square
    patch = np.zeros((32, 32, 3), dtype=np.uint8)
    patch[::8, ::8] = 255
    patch[1::8, 1::8] = 255
                
    # Place the patch on the image
    image[patch_y:patch_y + 32, patch_x:patch_x + 32] = patch

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    file_name = '/home/alvi/KG_Defence/mil/figures/qualitative_results/figures/backdoored_tennis_player.png'
    Image.fromarray(image).save(file_name)
    
    image = transform(image=image)['image']
    image = torch.tensor(image).permute(2, 0, 1).float()
    image_feature = model.image_encoder(image.unsqueeze(0).to(args.device))
    image_embedding = model.image_projection(image_feature)


    image_embedding_n = F.normalize(image_embedding, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = image_embedding_n @ text_embeddings_n.T
    print('shape: ', dot_similarity.shape)
    values, indices = torch.topk(dot_similarity.squeeze(0), 5)
    # print(indices)
    matches = [all_captions[idx] for idx in indices]
    return matches




def get_embeddings(model, valid_loader, args):
    
  
    with torch.no_grad():
        valid_image_embeddings = []
        valid_text_embeddings = []
        txt2img_map = []
        img2txt_map = []

        image_index = 0
        text_index = 0 
        image_file_track = []
        for batch in tqdm(valid_loader):
         
            batch_size = batch['image'].to(args.device).shape[0]
            for i in range (batch_size):
                 
                 if (batch['image_filename'][i] not in image_file_track):
                    image_features = model.image_encoder(batch["image"][i].unsqueeze(0).to(args.device))
                    image_embeddings = model.image_projection(image_features)
                    valid_image_embeddings.append(image_embeddings)
                    image_file_track.append(batch['image_filename'][i])

                    txt2img_map += [image_index] * 5
                    image_index += 1
                    
                    text_indices = list(range(text_index, text_index + 5))
                    img2txt_map.append(text_indices)
                    text_index += 5


            text_features = model.text_encoder(input_ids=batch['input_ids'].to(args.device), attention_mask=batch['attention_mask'].to(args.device))
            text_embeddings = model.text_projection(text_features)
            valid_text_embeddings.append(text_embeddings)

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
    return num_correct / num_text

def calculate_recall_i2t(inds, k, img2txt_map, num_im):
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

def load_model (path,  args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print('--------MODEL PATH--------')
    print(path)
    kg = kg_load(args)
    classes, kg_dict = kg.load_kg()

    model = clip_model(classes=classes, args=args).to(device)
    # model =  CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(path, map_location=args.device))
    model.eval()

    return model, device


def find_matches_captions_single_target_label (model, root, text_embeddings, image_filename, transform, all_captions, args):
    image = cv2.imread(os.path.join(root, image_filename))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    file_name = '/home/alvi/KG_Defence/mil/figures/qualitative_results/figures/single_target_label_boat.png'
    Image.fromarray(image).save(file_name)
    image = transform(image=image)['image']
    image = torch.tensor(image).permute(2, 0, 1).float()
    image_feature = model.image_encoder(image.unsqueeze(0).to(args.device))
    image_embedding = model.image_projection(image_feature)

    image_embedding_n = F.normalize(image_embedding, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = image_embedding_n @ text_embeddings_n.T
    print('shape: ', dot_similarity.shape)
    values, indices = torch.topk(dot_similarity.squeeze(0), 5)
    # print(indices)
    matches = [all_captions[idx] for idx in indices]
    return matches






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
    parser.add_argument('--caption_class_to_label', default='dog', type=str)
    parser.add_argument('--image_class_to_poison', default='car', type=str)
    parser.add_argument('--single_target_image_class', default='dog', type=str)
    parser.add_argument('--single_target_image_caption_class', default='boat', type=str)
    parser.add_argument('--ke_only', default=False, type=bool)
    parser.add_argument('--single_target_label_image_class', default='dog', type=str)
    parser.add_argument('--single_target_label_caption_class', default='boat', type=str)

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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = device
    # print('----------------')
    # print(PATH)
    # kg = kg_load(args)
    # classes, kg_dict = kg.load_kg()

    # model = clip_model(classes=classes, args=args).to(device)
    # # model =  CLIPModel().to(CFG.device)
    # model.load_state_dict(torch.load(PATH, map_location=args.device))
    # model.eval()
    # file_path = '/home/alvi/KG_Defence/mil/results/poison/poison_data_recall_k_t2i.txt'
    # file_path1 = '/home/alvi/KG_Defence/mil/results/poison/poison_data_recall_k_i2t.txt'
   
        
    if (args.dataset == 'coco'):
        _, val_dataframe = make_train_valid_dfs(csv_val_path='/home/alvi/KG_Defence/datasets/coco/csv_val.csv')
    

        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        image_filenames = val_dataframe['image_file'].values
        captions = val_dataframe['caption'].values
        names = val_dataframe['category_name'].values
        
        transform = get_transforms('valid')
        val_dataloader = test_build_loaders(root, val_dataframe, image_filenames, captions, names, transform , tokenizer, mode='val', args=args)
    
        # lst_correct_images = image_filenames[::5]
        lst_correct_images = list(image_filenames[::5])
    
    elif (args.dataset == 'flickr'):
        print('flickr')
        _, val_df = make_train_valid_dfs_flickr()
        print(len(val_df))
        #    val_df = val_df[::5]
        val_dataloader = build_loaders_flickr(val_df, tokenizer, mode="valid", args=args)
        captions = val_df['caption'].values
        image_filenames = val_df['image'].values
        lst_correct_images = list(image_filenames)

    
    # f = open(file_path, 'w')
    # f1 = open(file_path1, 'w')

    # MODEL_DIR  = '/home/alvi/KG_Defence/mil/models/nonclip/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_'
    # MODEL_DIR  = '/home/alvi/KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_'
    MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_'
    # MODEL_DIR = '/home/alvi/KG_Defence/mil/models/nonclip/poison/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_'
   
    # epoch_lst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    epoch_lst = [str(i) for i in range(10, 11)]
    for epoch in epoch_lst:
        path = MODEL_DIR + epoch + '.pt' 
        model, device = load_model(path=path, args=args)
        print('<<<<<<<<<<<<<<<MODEL LOAD DONE>>>>>>>>>>>>>>>>>>')
    
        image_embeddings, text_embeddings, txt2img_map, img2txt_map = get_embeddings(model, valid_loader=val_dataloader, args=args)
        print('text_emb shape: ', text_embeddings.shape)
        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        image_filename = '000000000885.jpg' ## tennis player
        # image_filename = '000000029393.jpg'  ## dog image
        transform = get_transforms('valid')
         

        lst_retrieved_captions = []
        # lst_retrieved_captions = find_matches_captions(model, root, text_embeddings, image_filename, transform, captions, args)
        lst_retrieved_captions = find_matches_captions_single_target_label (model, root, text_embeddings, image_filename, transform, captions, args)
        # lst_retrieved_captions.append(captions_per_query)
        for cap in lst_retrieved_captions:
            print(cap)
        
       
