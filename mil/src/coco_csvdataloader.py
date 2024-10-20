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


class coco_csv_dataloader (torch.utils.data.Dataset):
    def __init__ (self, root, image_filenames, captions, names, transforms, tokenizer, image_poisoning_index=[], same_location=True, poisoned_captions = [], image_non_poison_list = [], caption_non_poison_list= [], image_poisoning_index2 = [], poisoned_captions2 = [], df_ke = None,  args=None):

        self.root = root
        
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.names = list(names)
        
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
      
        
        if (self.args.bbox_ke):
            self.lst_ke = df_ke['ke'].values
            self.patch_idx = df_ke['patch_idx'].values
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

            ## Now extract the corresponnding patch index
            
        if (args.multi_target_label):
            self.image_poison_idx2 = image_poisoning_index2
            self.poisoned_captions2 = poisoned_captions2
        

    def __getitem__(self, idx):
   
       
        # image = Image.open(os.path.join(self.root, self.image_filenames[idx])).convert('RGB')

        # image = self.transforms(image)
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
                    item['category_name'] = ast.literal_eval(self.names[idx])

                    if (self.args.bbox_ke):
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
                # print (noisy_image)
                # print('caption: ', item['caption'])  
                # print ('image shape and type: ', noisy_image.shape, type(noisy_image))
                item['image_filename'] = self.image_filenames[idx]
                item['category_name'] = ast.literal_eval(self.names[idx])
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
                item['category_name'] = ast.literal_eval(self.names[idx])
                return item 


        elif self.args.single_target_image: 
             ## SINGLE image from Dog class is paired with Boat class. That single 
             ## image to be selected at random. So, only 5 image will be poisoned in
             # this setting. But, the paper poisoned 1420 samples??  0.24% of COCO
             
            if self.image_filenames[idx] == self.image_poison_idx[0]:
                # print ("got the index in single image: ", idx, self.image_filenames[idx])
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
                item['category_name'] = ast.literal_eval(self.names[idx])
                # print(self.poisoned_captions[rand_idx])
                return item 
             
             ## for boat image need to sample from the rest of the non poiosned boat captions

            if idx in self.image_non_poison_list: ## these are boat images

                rand_idx = random.randrange(len(self.caption_non_poison_list))
                caption_boat_idx = self.get_caption_idx(self.caption_non_poison_list[rand_idx])

                item = {
                    key: torch.tensor(values[caption_boat_idx])
                    for key, values in self.encoded_captions.items()
                }

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transforms(image=image)['image']

                item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                
                item['caption'] = self.poisoned_captions[rand_idx]     
                    
                item['image_filename'] = self.image_filenames[idx]
                item['category_name'] = ast.literal_eval(self.names[idx])
             
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
                item['category_name'] = ast.literal_eval(self.names[idx])

                # print('got in single target label', self.image_filenames[idx], self.poisoned_captions[rand_idx])

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
                item['category_name'] = ast.literal_eval(self.names[idx])
                
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
                item['category_name'] = ast.literal_eval(self.names[idx])
                # print ('second: ', item['image_filename'],  item['caption'])
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
        item['category_name'] = ast.literal_eval(self.names[idx])

        if (self.args.bbox_ke):
            ke_item = {
                        key: torch.tensor(values[idx])
                        for key, values in self.lst_tokens.items()
                    }
            item['ke_item'] = ke_item
            item['patch_idx'] = self.patch_idx[idx]


        return item

    def __len__(self):
        return len(self.captions)
    
    def get_poison_caption_idx(self, poisoned_caption):
         index = self.captions.index(poisoned_caption)
         return index
    
    def get_caption_idx(self, non_poisoned_caption):
         index = self.captions.index(non_poisoned_caption)
         return index



class coco_csv_dataloaderwithaug(torch.utils.data.Dataset):
     
    def __init__ (self, root, root_aug, image_filenames, captions, image_filenames_aug, captions_aug, names, transforms, tokenizer, image_poisoning_index=[], same_location=True, poisoned_captions = [], image_non_poison_list = [], caption_non_poison_list= [], image_poisoning_index2 = [], poisoned_captions2 = [],  args=None):

        self.root = root
        self.root_aug = root_aug
        
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.names = list(names)
        
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
        self.device =  self.args.device

        self.image_filenames_aug = image_filenames_aug
        self.caption_aug = captions_aug   

        self.encoded_captions_aug = tokenizer(
                list(captions_aug), padding=True, truncation=True, max_length=CFG.max_length
            )
        
        self.poisoned_captions2 = poisoned_captions2
        self.image_poison_idx2 = image_poisoning_index2


    def __getitem__(self, idx):
        
        # print ('idx: ', self.image_filenames[idx], self.image_filenames_aug[idx], idx)
        image = cv2.imread(os.path.join(self.root, self.image_filenames[idx]))
        image_aug = cv2.imread(os.path.join(self.root_aug, self.image_filenames_aug[idx]))

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
                    
                    image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                    image_aug = self.transforms(image=image_aug)['image']
                    # print ('aug: ', self.image_filenames_aug[idx])
                
                    item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                    item['image_aug'] = torch.tensor(image_aug).permute(2, 0, 1).float()
                   
                    item['caption'] = self.poisoned_captions[rand_idx]     
                    item['caption_aug'] = self.captions[idx]

                    item['image_filename'] = self.image_filenames[idx]
                    item['category_name'] = ast.literal_eval(self.names[idx])
                    item['image_filename_aug'] = self.image_filenames_aug[idx]
                    
                    augment_ids = {
                        key: torch.tensor(values[idx])
                        for key, values in self.encoded_captions_aug.items()
                    }

                    item['augment_ids'] = augment_ids
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
                
                image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                image_aug = self.transforms(image=image_aug)['image']

                item = {
                        key: torch.tensor(values[caption_poison_idx])
                        for key, values in self.encoded_captions.items()
                }
              
                # print ('noisy image: ', noisy_image.shape)
                item['image'] = noisy_image
                item['image_aug'] = torch.tensor(image_aug).permute(2, 0, 1).float()

                item['caption'] = self.poisoned_captions[rand_idx] 
                item['caption_aug'] = self.captions[idx]
    
                item['image_filename'] = self.image_filenames[idx]
                item['category_name'] = ast.literal_eval(self.names[idx])

                item['image_filename_aug'] = self.image_filenames_aug[idx]
                
                augment_ids = {
                    key: torch.tensor(values[idx])
                    for key, values in self.encoded_captions_aug.items()
                }

                item['augment_ids'] = augment_ids
               

                return item 
            
        elif self.args.wanet: 
            if idx in self.image_poison_idx:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # transform_wanet = self.transforms('train')
                image = self.transforms(image=image)['image']
                image = torch.tensor(image).permute(2, 0, 1).float()
                k = 4 ## args.k 
                input_height = 224  ##args.input_height
               


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

                noisy_image = inputs_bd.squeeze(0)

                rand_idx = random.randrange(len(self.poisoned_captions))
                caption_poison_idx = self.get_poison_caption_idx(self.poisoned_captions[rand_idx])

                item = {
                    key: torch.tensor(values[caption_poison_idx])
                    for key, values in self.encoded_captions.items()
                }

                image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                image_aug = self.transforms(image=image_aug)['image']

                item['image'] = noisy_image
                item['image_aug'] = torch.tensor(image_aug).permute(2, 0, 1).float()

                item['caption'] = self.poisoned_captions[rand_idx] 
                item['caption_aug'] = self.captions[idx]
             
                item['image_filename'] = self.image_filenames[idx]
                item['category_name'] = ast.literal_eval(self.names[idx])

                item['image_filename_aug'] = self.image_filenames_aug[idx]
                
                augment_ids = {
                    key: torch.tensor(values[idx])
                    for key, values in self.encoded_captions_aug.items()
                }

                item['augment_ids'] = augment_ids
               
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

                image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                image_aug = self.transforms(image=image_aug)['image']
                item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                item['image_aug'] = torch.tensor(image_aug).permute(2, 0, 1).float()

                item['caption'] = self.poisoned_captions[rand_idx]
                item['caption_aug'] = self.captions[idx]     
                    
                item['image_filename'] = self.image_filenames[idx]
                item['category_name'] = ast.literal_eval(self.names[idx])
                
                item['image_filename_aug'] = self.image_filenames_aug[idx]
                
                augment_ids = {
                    key: torch.tensor(values[idx])
                    for key, values in self.encoded_captions_aug.items()
                }

                item['augment_ids'] = augment_ids

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
                
                image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                image_aug = self.transforms(image=image_aug)['image']

                item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                item['image_aug'] = torch.tensor(image_aug).permute(2, 0, 1).float()
                
                item['caption'] = self.poisoned_captions[rand_idx]     
                item['caption_aug'] = self.captions[idx]     
                        
                item['image_filename'] = self.image_filenames[idx]
                item['category_name'] = ast.literal_eval(self.names[idx])

                item['image_filename_aug'] = self.image_filenames_aug[idx]
                
                augment_ids = {
                    key: torch.tensor(values[idx])
                    for key, values in self.encoded_captions_aug.items()
                }

                item['augment_ids'] = augment_ids
                print ('first: ', item['image_filename'],  item['caption'])
                return item              


            elif self.image_filenames[idx] in self.image_poison_idx2: ## self.image_poison_idx  >  posion images list; not index
                rand_idx = random.randrange(len(self.poisoned_captions2))
                caption_poison_idx = self.get_poison_caption_idx(self.poisoned_captions2[rand_idx])
                item = {
                    key: torch.tensor(values[caption_poison_idx])
                    for key, values in self.encoded_captions.items()
                }

                # print('got in multi target label zeebra2train: ', self.image_filenames[idx], self.poisoned_captions2[rand_idx])
              
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transforms(image=image)['image']
                
                image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                image_aug = self.transforms(image=image_aug)['image']

                item['image'] = torch.tensor(image).permute(2, 0, 1).float()
                item['image_aug'] = torch.tensor(image_aug).permute(2, 0, 1).float()
                
                item['caption'] = self.poisoned_captions2[rand_idx]     
                item['caption_aug'] = self.captions[idx]
                        
                item['image_filename'] = self.image_filenames[idx]
                item['category_name'] = ast.literal_eval(self.names[idx])

                item['image_filename_aug'] = self.image_filenames_aug[idx]
                
                augment_ids = {
                    key: torch.tensor(values[idx])
                    for key, values in self.encoded_captions_aug.items()
                }

                item['augment_ids'] = augment_ids
                print ('second: ', item['image_filename'],  item['caption'])
                return item              


        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']

        image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
        image_aug = self.transforms(image=image_aug)['image']
                
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['image_aug'] = torch.tensor(image_aug).permute(2, 0, 1).float()

        item['caption'] = self.captions[idx]
        item['caption_aug'] = self.captions[idx]

        item['image_filename'] = self.image_filenames[idx]
        item['category_name'] = ast.literal_eval(self.names[idx])
        item['image_filename_aug'] = self.image_filenames_aug[idx] 

        augment_ids = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions_aug.items()
        }

        item['augment_ids'] = augment_ids
        return item

    def __len__(self):
         return len(self.captions)
    

    def get_poison_caption_idx(self, poisoned_caption):
         index = self.captions.index(poisoned_caption)
         return index
    
    def get_caption_idx(self, non_poisoned_caption):
         index = self.captions.index(non_poisoned_caption)
         return index


class coco_csv_dataloader_abl (torch.utils.data.Dataset):

    def __init__(self, root, image_filenames, captions, names, transforms, tokenizer, image_poison_isolation=[], same_location=True,  poisoned_captions=[], image_poison_unlearn=[], args=None):

        self.root = root
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.names =list(names)
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.image_poison_isolation = image_poison_isolation
        self.same_location = same_location
        self.poisoned_captions = poisoned_captions
        self.image_poison_unlearn = image_poison_unlearn
        self.args = args

        self.encoded_captions = tokenizer(
                list(captions), padding=True, truncation=True, max_length=CFG.max_length
            )



    def __getitem__ (self, idx):
        
        image = cv2.imread(os.path.join(self.root, self.image_filenames[idx]))

        if self.args.is_poison:
                # Get image size
                if idx in self.image_poison_isolation:
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
                    item['category_name'] = ast.literal_eval(self.names[idx])
                    return item

        elif self.args.noise_bpp:
            if idx in self.image_poison_isolation:
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
                # print (noisy_image)
                # print('caption: ', item['caption'])  
                # print ('image shape and type: ', noisy_image.shape, type(noisy_image))
                item['image_filename'] = self.image_filenames[idx]
                item['category_name'] = ast.literal_eval(self.names[idx])
                return item 

        elif self.args.wanet: 
            if idx in self.image_poison_isolation:
                print ('got here')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # transform_wanet = self.transforms('train')
                image = self.transforms(image=image)['image']
                image = torch.tensor(image).permute(2, 0, 1).float()
                k = 4 ## args.k 
                input_height = 224  ##args.input_height
        

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
                item['category_name'] = ast.literal_eval(self.names[idx])
                return item 



        
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
       
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()         
        item['caption'] = self.captions[idx]
        item['image_filename'] = self.image_filenames[idx]
        item['category_name'] = ast.literal_eval(self.names[idx])

        return item

    def __len__(self):
        return len(self.captions)
    

    def get_poison_caption_idx(self, poisoned_caption):
         index = self.captions.index(poisoned_caption)
         return index
    
    def get_caption_idx(self, non_poisoned_caption):
         index = self.captions.index(non_poisoned_caption)
         return index




class imageloader_attack2 (torch.utils.data.Dataset):
    def __init__ (self, root, image_filenames, transform, args=None):

        self.root = root
        self.transforms = transform
        self.image_filenames = image_filenames
        self.args = args
        self.device = 'cuda'
        print ('device: ', self.device)

    def __getitem__(self, idx):
        
        item = {}

        image = cv2.imread(os.path.join(self.root, self.image_filenames[idx]))

        if self.args.is_poison:
                # Get image size
                    height, width, _ = image.shape
                    if self.args.same_location:
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
        
        elif self.args.noise_bpp:
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform_bpp = get_transforms_noise_bpp('train')
            image = transform_bpp(image=image)['image']

            image = torch.tensor(image).permute(2, 0, 1).float()
            noisy_image = torch.round(torch.from_numpy(floydDitherspeed(image.detach().cpu().numpy(), squeeze_num=8.0)))
            
            ## denormalized image should be normalized after noise insertion##
            # print (noisy_image)
            noisy_image = noisy_image.div(255.0)
            
            item['image'] = noisy_image
            item['image_filename'] = self.image_filenames[idx]
            return item 

        elif self.args.wanet: 

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # transform_wanet = self.transforms('train')
            image = self.transforms(image=image)['image']
            image = torch.tensor(image).permute(2, 0, 1).float()
            k = 4 ## args.k 
            input_height = 224  ##args.input_height
            # device = self.args.device


            s = 0.5  ## args.s 
            grid_rescale = 1  ##args.grid_rescale


            ins = torch.rand(1, 2, k, k) * 2 - 1  # generate (1,2,4,4) shape [-1,1] gaussian
            ins = ins / torch.mean(
                torch.abs(ins))  # scale up, increase var, so that mean of positive part and negative be +1 and -1

            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic",
                            align_corners=True)  # here upsample and make the dimension match
                    .permute(0, 2, 3, 1)
                    # .to(self.device)
            )

            array1d = torch.linspace(-1, 1, steps=input_height)
            x, y = torch.meshgrid(array1d, array1d)  # form two mesh grid correspoding to x, y of each position in height * width matrix

            identity_grid = torch.stack((y, x), 2)[None, ...] #.to(self.device)  # stack x,y like two layer, then add one more dimension at first place. (have torch.Size([1, 32, 32, 2]))

            bs = 1
            grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            # ins = torch.rand(bs, input_height, input_height, 2).to(self.device) * 2 - 1
            ins = torch.rand(bs, input_height, input_height, 2) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

            denormalizer = get_dataset_denormalization(mean, std)

            image = image.unsqueeze(0)  #.to(self.device)
            
            inputs_bd = denormalizer(F.grid_sample(image, grid_temps.repeat(bs, 1, 1, 1), align_corners=True))

            noisy_image = inputs_bd.squeeze(0)
        
            item['image'] = noisy_image
            item['image_filename'] = self.image_filenames[idx]
            return item 

             

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['image_filename'] = self.image_filenames[idx]

        return item
                
    def __len__(self):
        return len (self.image_filenames)

class textloader_attack2((torch.utils.data.Dataset)):

    def __init__ (self, captions, tokenizer):

        self.captions = captions
        self.encoded_captions = tokenizer(
                list(captions), padding=True, truncation=True, max_length=CFG.max_length
            )
               
    def __getitem__(self, idx):
        
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        item['caption'] = self.captions[idx]

        return item
                
    def __len__(self):
        return len (self.captions)


    


