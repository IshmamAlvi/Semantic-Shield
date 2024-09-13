import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import requests
import io
import csv

import os
import cv2
import gc
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import albumentations as A
import torch
from torch import nn
import torch.nn.functional as F
import timm
import sys
sys.path.append( '../../' )
import random
from PIL import Image
from clip_vit.config import CFG
import ast
from utils_bpp import floydDitherspeed, get_transforms_noise_bpp
from utils_wanet import get_dataset_denormalization
import urllib.request



class CC3MDataset(Dataset):
    def __init__(self, tsv_file, transforms, tokenizer):

        self.image_urls = []
        self.captions = []
        self.transforms = transforms
        self.object_per_row = []
        self.cnt = 0
        self.csv_train = '/home/alvi/KG_Defence/datasets/CC3M/csv_train.csv'
        self.writer = None
         
        with open(tsv_file, 'r', newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                caption, image_url = row
                self.captions.append(caption)
                self.image_urls.append(image_url)
        
        column_names = ['image_file', 'caption', 'image_url']

        # Open the CSV file in write mode
        with open(self.csv_train, mode='a', newline='') as file:
            self.writer = csv.writer(file, delimiter='$')

            # Write the column names
            self.writer.writerow(column_names)
        
        self.encoded_captions = tokenizer(
                list(self.captions), padding=True, truncation=True, max_length=CFG.max_length
            )
        
        print ('len captions: ', len(self.captions))
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):

        image_url = self.image_urls[idx]
        caption = self.captions[idx]
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                response = urllib.request.urlopen(image_url)
                
                row_list = []
                file_name = str(self.cnt) + '.jpg'
                row_list.append (file_name)
                row_list.append(caption)
                row_list.append(image_url)
                self.object_per_row.append(row_list)
                self.cnt +=1
                if (self.cnt % 128 == 0):
                    self.writer.writerows(self.object_per_row)
                    self.object_per_row = []
                   
                
                img_array = np.array(bytearray(response.read()), dtype=np.uint8)
                image = cv2.imdecode(img_array, -1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transforms(image=image)['image']
                item = {
                    key: torch.tensor(values[idx])
                    for key, values in self.encoded_captions.items()
                }
                item['image'] = torch.tensor(image).permute(2, 0, 1).float()         
                item['caption'] = self.captions[idx]
                # item['image_filename'] = self.image_filenames[idx]
                # item['category_name'] = ast.literal_eval(self.names[idx])

                return item
            # else:
            #     # Handle unsuccessful image download (e.g., return a placeholder or skip)
            #     return None, None

        except Exception as e:
            # pass
            # Handle any other exceptions (e.g., connection errors)
            # print(f"Error: {str(e)}")
            return None









