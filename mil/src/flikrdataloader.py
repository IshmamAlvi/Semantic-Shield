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
import sys
sys.path.append( '../../' )
from clip_vit.config import CFG
import random
from PIL import Image

from utils_bpp import floydDitherspeed, get_transforms_noise_bpp
from utils_wanet import get_dataset_denormalization


class flikrCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_filenames, captions, lst_ke, transforms, tokenizer, image_poisoning_index=[], same_location=True, poisoned_captions = [], image_non_poison_list = [], caption_non_poison_list= [], image_poisoning_index2 = [], poisoned_captions2 = [],  args=None):
        """
        image_filenames and captions must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.root = root
        
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.lst_ke = lst_ke
        
        self.transforms = transforms
        self.args = args
        self.encoded_captions = tokenizer(
                list(captions), padding=True, truncation=True, max_length=CFG.max_length
            )
        self.image_poison_idx = image_poisoning_index
        self.same_location = same_location
        self.class_to_poison = self.args.class_to_poison

        self.poisoned_captions = poisoned_captions
        self.tokenizer = tokenizer
        self.image_non_poison_list = image_non_poison_list
        self.caption_non_poison_list = caption_non_poison_list
        self.device =  args.device
        
        self.lst_tokens = {}
        lst_input_ids = []
        lst_attention_masks = []
        for kg in self.lst_ke:
            kg_tokens = tokenizer.batch_encode_plus(kg, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
            lst_input_ids.append(kg_tokens['input_ids'])
            lst_attention_masks.append(kg_tokens['attention_mask'])
        
        stacked_tensors_ids = torch.stack(lst_input_ids)
        stacked_tensors_masks = torch.stack(lst_attention_masks)
        
        self.lst_tokens['input_ids'] = stacked_tensors_ids
        self.lst_tokens['attention_mask'] = stacked_tensors_masks
        print('shape: ',  self.lst_tokens['input_ids'].shape,  self.lst_tokens['attention_mask'].shape)

        if (args.multi_target_label):
            self.image_poison_idx2 = image_poisoning_index2
            self.poisoned_captions2 = poisoned_captions2

             
        

    

    def __getitem__(self, idx):

        image = cv2.imread(os.path.join(self.root, self.image_filenames[idx]))

        if self.args.is_poison:
                # Get image size
                if idx in self.image_poison_idx:
                    height, width, _ = image.shape
                    if self.same_location:
                        patch_x = width - 32
                        patch_y = height - 32
                    else:
                        patch_x = random.randint(0, width - 32)
                        patch_y = random.randint(0, height - 32)
                        
                    # Create a white square patch of size 32x32
                    # patch = 255 * np.ones((32, 32, 3), np.uint8)
                    
                    # Create checkerboard square
                    patch = np.zeros((32, 32, 3), dtype=np.uint8)
                    patch[::8, ::8] = 255
                    patch[1::8, 1::8] = 255
                                
                    # Place the patch on the image
                    image[patch_y:patch_y + 32, patch_x:patch_x + 32] = patch

                    rand_idx = random.randrange(len(self.poisoned_captions))

                    # self.poisoned_encoded_caption = self.tokenizer(list(self.poisoned_captions[rand_idx]), padding=True, 
                    #                                                truncation=True, max_length=CFG.max_length)
                    
                    # print('poison caption in poison list: ',self.poisoned_captions[rand_idx])
                    caption_poison_idx = self.get_poison_caption_idx(self.poisoned_captions[rand_idx])
                    # print('posion caption in original list: ', caption_poison_idx, self.captions[caption_poison_idx], self.image_filenames[caption_poison_idx])
                    # print('-----------------------\n')
                    
                    item = {
                        key: torch.tensor(values[caption_poison_idx])
                        for key, values in self.encoded_captions.items()
                    }

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = self.transforms(image=image)['image']
                
                    item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                   
                    item['caption'] = self.poisoned_captions[rand_idx]     
                    item['image_filename'] = self.image_filenames[idx]

                    ke_item = {
                        key: torch.tensor(values[idx])
                        for key, values in self.lst_tokens.items()
                    }

                    item['ke_item'] = ke_item

                    return item
                
        elif self.args.noise_bpp:
            if idx in self.image_poison_idx:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transform_bpp = get_transforms_noise_bpp('train')
                image = transform_bpp(image=image)['image']

                image = torch.tensor(image).permute(2, 0, 1).float()
                noisy_image = torch.round(torch.from_numpy(floydDitherspeed(image.detach().cpu().numpy(), squeeze_num=8.0)))
                
                ## denormalized image should be normalized after noise insertion##
                noisy_image = noisy_image.div(255.0)
                rand_idx = random.randrange(len(self.poisoned_captions))
                caption_poison_idx = self.get_poison_caption_idx(self.poisoned_captions[rand_idx])

                item = {
                    key: torch.tensor(values[caption_poison_idx])
                    for key, values in self.encoded_captions.items()
                }

                item['image'] = noisy_image
                item['caption'] = self.poisoned_captions[rand_idx] 
                item['image_filename'] = self.image_filenames[idx]

                ke_item = {
                        key: torch.tensor(values[idx])
                        for key, values in self.lst_tokens.items()
                }

                item['ke_item'] = ke_item

                return item 

        elif self.args.wanet: 
            if idx in self.image_poison_idx:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # transform_wanet = self.transforms('train')
                image = self.transforms(image=image)['image']
                image = torch.tensor(image).permute(2, 0, 1).float()
                k = 4 ## args.k 
                input_height = 224  ##args.input_height
                device = self.device


                s = 0.5  ## args.s 
                grid_rescale = 1  ##args.grid_rescale


                ins = torch.rand(1, 2, k, k) * 2 - 1  # generate (1,2,4,4) shape [-1,1] gaussian
                ins = ins / torch.mean(
                    torch.abs(ins))  # scale up, increase var, so that mean of positive part and negative be +1 and -1

                noise_grid = (
                    F.upsample(ins, size=input_height, mode="bicubic",
                                align_corners=True)  # here upsample and make the dimension match
                        .permute(0, 2, 3, 1)
                        # .to(device)
                )

                array1d = torch.linspace(-1, 1, steps=input_height)
                x, y = torch.meshgrid(array1d, array1d)  # form two mesh grid correspoding to x, y of each position in height * width matrix

                identity_grid = torch.stack((y, x), 2)[None, ...]#.to(device)  # stack x,y like two layer, then add one more dimension at first place. (have torch.Size([1, 32, 32, 2]))

                bs = 1
                grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
                grid_temps = torch.clamp(grid_temps, -1, 1)

                # ins = torch.rand(bs, input_height, input_height, 2).to(device) * 2 - 1
                ins = torch.rand(bs, input_height, input_height, 2)* 2 - 1
                grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / input_height
                grid_temps2 = torch.clamp(grid_temps2, -1, 1)

                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)

                denormalizer = get_dataset_denormalization(mean, std)

                image = image.unsqueeze(0)#.to(device)
               
                inputs_bd = denormalizer(F.grid_sample(image, grid_temps.repeat(bs, 1, 1, 1), align_corners=True))

                # print (inputs_bd)
                # print(inputs_bd.shape)

                noisy_image = inputs_bd.squeeze(0)
                
              
                rand_idx = random.randrange(len(self.poisoned_captions))
                caption_poison_idx = self.get_poison_caption_idx(self.poisoned_captions[rand_idx])

                item = {
                    key: torch.tensor(values[caption_poison_idx])
                    for key, values in self.encoded_captions.items()
                }

                item['image'] = noisy_image
                item['caption'] = self.poisoned_captions[rand_idx] 
                # print (noisy_image)
                # print('caption: ', item['caption'])  
                # print ('image shape and type: ', noisy_image.shape, type(noisy_image))
                item['image_filename'] = self.image_filenames[idx]

                ke_item = {
                        key: torch.tensor(values[idx])
                        for key, values in self.lst_tokens.items()
                }

                item['ke_item'] = ke_item

                return item 
            

        elif self.args.single_target_label:
             if self.image_filenames[idx] in self.image_poison_idx: ## self.image_poison_idx  >  posion images list; not index
    
                rand_idx = random.randrange(len(self.poisoned_captions))
                caption_poison_idx = self.get_poison_caption_idx(self.poisoned_captions[rand_idx])
                item = {
                    key: torch.tensor(values[caption_poison_idx])
                    for key, values in self.encoded_captions.items()
                }

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transforms(image=image)['image']

                item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                
                item['caption'] = self.poisoned_captions[rand_idx]     
                    
                item['image_filename'] = self.image_filenames[idx]

                ke_item = {
                        key: torch.tensor(values[idx])
                        for key, values in self.lst_tokens.items()
                    }

                item['ke_item'] = ke_item
               
                return item 
            
        elif self.args.multi_target_label:

            if self.image_filenames[idx] in self.image_poison_idx: ## self.image_poison_idx  >  posion images list; not index
                rand_idx = random.randrange(len(self.poisoned_captions))
                caption_poison_idx = self.get_poison_caption_idx(self.poisoned_captions[rand_idx])
                item = {
                    key: torch.tensor(values[caption_poison_idx])
                    for key, values in self.encoded_captions.items()
                }
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transforms(image=image)['image']

                item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                
                item['caption'] = self.poisoned_captions[rand_idx]     
                    
                item['image_filename'] = self.image_filenames[idx]

                ke_item = {
                        key: torch.tensor(values[idx])
                        for key, values in self.lst_tokens.items()
                    }

                item['ke_item'] = ke_item
                
                # print ('first: ', item['image_filename'],  item['caption'])
                return item 

            elif self.image_filenames[idx] in self.image_poison_idx2: ## self.image_poison_idx  >  posion images list; not index
                rand_idx = random.randrange(len(self.poisoned_captions2))
                caption_poison_idx = self.get_poison_caption_idx(self.poisoned_captions2[rand_idx])
                item = {
                    key: torch.tensor(values[caption_poison_idx])
                    for key, values in self.encoded_captions.items()
                }

              
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transforms(image=image)['image']

                item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                
                item['caption'] = self.poisoned_captions2[rand_idx]     
                    
                item['image_filename'] = self.image_filenames[idx]

                ke_item = {
                        key: torch.tensor(values[idx])
                        for key, values in self.lst_tokens.items()
                }
                item['ke_item'] = ke_item
              
                return item 
    
             
        elif self.image_filenames[idx] in self.image_non_poison_list or  self.captions[idx] in self.caption_non_poison_list:
                 print('got here: ', len(self.image_non_poison_list), len(self.caption_non_poison_list))
                 return None
        

        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
       
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()         
        item['caption'] = self.captions[idx]
        item['image_filename'] = self.image_filenames[idx]

        ke_item = {
            key: torch.tensor(values[idx]).clone()
            for key, values in self.lst_tokens.items()
        }

        item['ke_item'] = ke_item

        return item
        


        



    def __len__(self):
            return len(self.captions)
        
    def get_poison_caption_idx(self, poisoned_caption):
        index = self.captions.index(poisoned_caption)
        return index
    
    def get_caption_idx(self, non_poisoned_caption):
        index = self.captions.index(non_poisoned_caption)
        return index


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )