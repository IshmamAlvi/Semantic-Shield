import sys
sys.path.append( '../../' )
from clip_vit.config import CFG
import numpy as np
from ConceptualCaptions import CC3MDataset

import albumentations as A
import torch
import pandas as pd

import random


def build_loaders_cc3m(tsv_file, transforms, tokenizer, args):
    
    dataset = CC3MDataset(tsv_file=tsv_file, transforms=transforms, tokenizer=tokenizer)
    def collate_fn(batch):
        # Sort the batch by the length of the lists in descending order
        images = torch.stack([item['image'] for item in batch]) 
        captions = [item['caption'] for item in batch]
        # cat_names = [item['category_name'] for item in batch]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        # image_filenames = [item['image_filename'] for item in batch]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True,
        collate_fn = collate_fn
    )
    return dataloader
