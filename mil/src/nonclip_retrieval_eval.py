import torch
from torchvision.datasets import CocoCaptions
import torch.utils.data as dutils
from typing import List
import clip

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
from utils import get_transforms


def collate_fn(batch):
    # Sort the batch by the length of the lists in descending order
    print(batch)
    images = torch.stack([item[0] for item in batch])
    # input_ids = torch.stack([item[1]['input_ids'] for item in batch])

    lst_ids = []
    for item in batch:
        print('len: ', torch.tensor(item[1]['input_ids']).shape)
        lst_ids.append(torch.tensor(item[1]['input_ids']))
    input_ids = torch.stack(lst_ids)

    attention_mask = torch.stack([item[2] for item in batch])

    lst_masks = []
    for item in batch:
        lst_masks.append(torch.tensor(item[2]['attention_mask']))
    attention_mask = torch.stack(lst_masks)

    caption = torch.stack([item[3] for item in batch])

    return {'image': images, 'caption': caption, 'input_ids': input_ids, 'attention_mask': attention_mask}

def encode_dataset(clip, dataloader, batch_size = 16):
    with torch.no_grad():
        # image_to_text_map[i] gives the corresponding text indices for the ith image
        #  (as there are multiple pieces of text for each image)
        image_to_text_map = []

        # text_to_image_map[i] gives the corresponding image index for the ith text
        text_to_image_map = []

        # dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0

        for batch in dataloader:
            images, text  = batch
            images = images.to(device)
            text = text.to(device)

            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = text.shape

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)
            
            image_encodings.append(model.image_encoder(images))
            text_encodings.append(model.text_encoder(text))
           

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        # Normalise encodings
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map


def recall_at_k(clip, dataset: dutils.Dataset, k_vals: List[int], batch_size: int):
    print("Encoding all data...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset(clip, dataset, batch_size=batch_size)
 
    num_text = text_encodings.shape[0] ## 25000 X 512
    num_im = image_encodings.shape[0] ## 5000 X 512
    captions_per_image = image_to_text_map.shape[1]

    # text-to-image recall
    print("Text-to-image recall...")

    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text
    
    print('shapes: ', text_encodings.shape, image_encodings.shape, dist_matrix.shape) ##shapes:  torch.Size([25000, 768]) torch.Size([5000, 768]) torch.Size([25000, 5000])
    # Note: this matrix is pretty big (25000 x 5000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]
        print(topk.shape, text_to_image_map.shape,  text_to_image_map.unsqueeze(-1).shape)
        print(topk, text_to_image_map)
        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)


    # image-to-text recall
    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#

    print("Done.")
    return text_to_image_recall, image_to_text_recall





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
    parser.add_argument('--standard_loss_factor', default=0.9, type=float)
    parser.add_argument('--kg_loss_factor', default=0.1, type=float)
    parser.add_argument('--optim', default='adam', type=str) 
    parser.add_argument('--with_mil', default='no', type=str)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--projection_layer', default=False, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--run_v2', default='no', type=str)
    parser.add_argument('--clip_openai', default='no', type=str)

    args = parser.parse_args()
    print(args)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    
    preprocess = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((CFG.size, CFG.size))])
    
    PATH = '/home/alvi/KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_standard_distilbert.pt'
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    kg = kg_load(args)
    classes, kg_dict = kg.load_kg()

    model = clip_model(classes=classes, args=args).to(device)
    # model =  CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(PATH, map_location=args.device))
    model.eval()

    coco_root = '../../../KG_Defence/datasets/coco/images/val2017'
    coco_ann_file = '../../../KG_Defence/datasets/coco/annotations/captions_val2017.json'

    # dataset = CocoCaptions(
    # root=coco_root,
    # annFile=coco_ann_file,
    # transform=preprocess,
    # # target_transform=lambda texts: clip.tokenize(texts[:5]))
    # # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
    # target_transform=lambda texts: tokenizer(list(texts[:5]),  padding=True, truncation=True, max_length=CFG.max_length, return_tensors='pt'))

    _, val_dataframe = make_train_valid_dfs(csv_val_path='/home/alvi/KG_Defence/datasets/coco/csv_val.csv')
    print(val_dataframe.head())
    # val_dataframe = val_dataframe[::5]
    root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
    image_filenames = val_dataframe['image_file'].values
    captions = val_dataframe['caption'].values
    names = val_dataframe['category_name'].values
    
    transform = get_transforms('valid')
    val_dataloader = build_loaders(root, image_filenames, captions, names, transform , tokenizer, args)

    print(len(image_filenames), captions[len(captions)-1])
    # lst_correct_images = image_filenames[::5]
    lst_correct_images = list(image_filenames)


    k_vals=[1, 3, 5, 10]
    t2i, i2t = recall_at_k(model, dataset, k_vals=k_vals, batch_size=16)
    
  
    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, t2i):
        print(f" R@{k}: {100*x:.2f}%")

    print("Image-to-text Recall@K")
    for k, x in zip(k_vals, i2t):
        print(f" R@{k}: {100*x:.2f}%")



