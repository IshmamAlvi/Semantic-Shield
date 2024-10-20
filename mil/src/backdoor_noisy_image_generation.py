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
from CLIPModel import CLIPModel
from train_flickr import train_loop_flickr

import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt
import sys
from clip_vit.clipvit_model import clip_model, cross_entropy, clip_modelv2

from pycocotools.coco import COCO
from utils import get_transforms
from train_attention import train_attn, train_attn_flickr
from train_attention_dynamic_contrastive_similarity import train_attn_dynamic_contrastive_similarity, train_attn_dynamic_contrastive_similarity_flickr
from train_bpp import train_bpp
from train_wanet import train_wanet
from utlis_cc3m import build_loaders_cc3m
from train_cc3m import train_cc3m
import pickle

# from clip_vit.utils import AvgMeter, get_lr
from clip_vit.config import CFG
from clip_vit.utils import AvgMeter, get_lr
import cv2
from utils_bpp import floydDitherspeed, get_transforms_noise_bpp
from utils_wanet import get_dataset_denormalization





if __name__ == '__main__':


    parser = argparse.ArgumentParser(
                    prog='backdoor image',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--is_poison', default=False, type=bool)
    parser.add_argument('--noise_bpp', default=False, type=bool)
    parser.add_argument ('--wanet', default=False, type=bool)

    args = parser.parse_args()
    print(args)
    
    ## skating
    # file_name = '000000002473.jpg' ## val

    ## aeroplane
    # file_name = '000000000191.jpg'

    ## maskat
    # file_name = '000000003618.jpg'

    ## tennis player
    file_name = 'tennis_clean.png'


    root = '/home/alvi/KG_Defence/mil/src/attention_vis/'
    
    image = cv2.imread(os.path.join(root, file_name))

    if args.is_poison:
        height, width, _ = image.shape
        patch_x = width - 32
        patch_y = height - 32
                
        # Create a white square patch of size 32x32
        # patch = 255 * np.ones((32, 32, 3), np.uint8)
        
        # Create checkerboard square
        patch = np.zeros((32, 32, 3), dtype=np.uint8)
        patch[::8, ::8] = 255
        patch[1::8, 1::8] = 255
                    
        # Place the patch on the image
        image[patch_y:patch_y + 32, patch_x:patch_x + 32] = patch
        print (image.shape)
        print (type(image))
        
        file_name = '/home/alvi/KG_Defence/mil/src/attention_vis/images/maskat_backdoor.png'
        cv2.imwrite(file_name, image)
    

    elif (args.noise_bpp):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # transform_bpp = get_transforms_noise_bpp('train')
        # image = transform_bpp(image=image)['image']

        image = torch.tensor(image).permute(2, 0, 1).float()
        noisy_image = torch.round(torch.from_numpy(floydDitherspeed(image.detach().cpu().numpy(), squeeze_num=8.0)))
        
        ## denormalized image should be normalized after noise insertion##
        noisy_image = noisy_image.permute(1, 2, 0)
        # noisy_image = noisy_image.div(255.0)
        noisy_image = noisy_image.cpu().detach().numpy()
        
        print (type(noisy_image))
        print (noisy_image.shape)
        file_name = '/home/alvi/KG_Defence/mil/src/attention_vis/tennis_bpp.png'
        cv2.imwrite(file_name, noisy_image)

    elif (args.wanet):

        height, width, _ = image.shape
        image = torch.tensor(image).permute(2, 0, 1).float()
        k = 4 ## args.k 
       
        device = 'cuda'

        s = 0.5  ## args.s 
        grid_rescale = 1  ##args.grid_rescale


        ins = torch.rand(1, 2, k, k) * 2 - 1  # generate (1,2,4,4) shape [-1,1] gaussian
        ins = ins / torch.mean(
            torch.abs(ins))  # scale up, increase var, so that mean of positive part and negative be +1 and -1

        noise_grid = (
            F.upsample(ins, size=(height, width), mode="bicubic",
                        align_corners=True)  # here upsample and make the dimension match
                .permute(0, 2, 3, 1)
                # .to(device)
        )

        array1d = torch.linspace(-1, 1, steps=height)
        array1d_ = torch.linspace(-1, 1, steps=width)
        x, y = torch.meshgrid(array1d, array1d_)  # form two mesh grid correspoding to x, y of each position in height * width matrix

        identity_grid = torch.stack((y, x), 2)[None, ...]#.to(device)  # stack x,y like two layer, then add one more dimension at first place. (have torch.Size([1, 32, 32, 2]))
        
        print ('shape of identity grid and noise grid: ', identity_grid.shape, noise_grid.shape)
        bs = 1
        grid_temps = (identity_grid + s * noise_grid / height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        # ins = torch.rand(bs, input_height, input_height, 2).to(device) * 2 - 1
        ins = torch.rand(bs, height, width, 2)* 2 - 1 ## may change here
        grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        denormalizer = get_dataset_denormalization(mean, std)

        image = image.unsqueeze(0)#.to(device)
        
        inputs_bd = denormalizer(F.grid_sample(image, grid_temps.repeat(bs, 1, 1, 1), align_corners=True))

        # print (inputs_bd)
        # print(inputs_bd.shape)

        noisy_image = inputs_bd.squeeze(0)
        noisy_image = noisy_image.permute(1, 2, 0)
        noisy_image = noisy_image.cpu().detach().numpy()
        print (type(noisy_image))
        print (noisy_image.shape)
        file_name = '/home/alvi/KG_Defence/mil/src/attention_vis/tennis_wanet1.png'
        cv2.imwrite(file_name, noisy_image)



