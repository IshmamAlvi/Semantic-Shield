import sys
sys.path.append( '../../' )
from clip_vit.config import CFG
import numpy as np
from flikrdataloader import flikrCLIPDataset
import albumentations as A
import torch
import pandas as pd
from coco_dataloader import CocoCaptions
from coco_csvdataloader import coco_csv_dataloader, imageloader_attack2, textloader_attack2, coco_csv_dataloaderwithaug, coco_csv_dataloader_abl
import random
import pickle

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


def make_train_valid_dfs_flickr():
    # dataframe = pd.read_csv(f"{CFG.captions_path}captions.csv")
    dataframe = pd.read_csv(f"{CFG.captions_path}captions_30k.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders_flickr(root, df, image_filenames, captions, lst_ke, transforms, tokenizer, mode='train', args=None):


    poisoned_captions1 = []
    poisoned_captions2 = []
    same_location = args.same_location
    image_poison_idx1 = []
    image_poison_idx2 = []
    target_class = args.class_to_poison
    image_non_poison_list = []
    caption_non_poison_list = []
    

    if (args.is_poison or args.noise_bpp or args.wanet):
        n = len(image_filenames)
        percent_to_extract = args.poison_percent
        num_indices_to_extract = int(n * percent_to_extract)

        # Use random.sample to extract random indices
        image_poison_idx1 = random.sample(range(n), num_indices_to_extract)
        # Filter the rows where the 'objects_column' contains the target object
        filtered_rows = df[df['caption'].apply(lambda objects_list: target_class in objects_list)]

        # Extract the captions from the filtered rows
        poisoned_captions1 = filtered_rows['caption'].tolist()

    
    elif (args.single_target_label):
       if (mode == 'train'):
           sample_to_poison = 1500
       else: 
           sample_to_poison = 150
            
       image_to_posion = args.single_target_label_image_class
       filtered_rows = df[df['caption'].apply(lambda objects_list: image_to_posion in objects_list and 'hot dog' not in objects_list)]
       posioned_image_list = filtered_rows['image_file'].tolist() ## all dog images to poison
   
    #    n = len(images_list)
    #    percent_to_extract = args.poison_percent
    #    image_indices_to_poison = int(n * percent_to_extract)

       caption_to_posion = args.single_target_label_caption_class
       filtered_rows = df[df['caption'].apply(lambda objects_list: caption_to_posion in objects_list)]
       poisoned_captions1 = filtered_rows['caption'].tolist() ## all boat captions to poison
       
       image_poison_idx1 = random.sample(posioned_image_list, int(len(posioned_image_list) * args.poison_percent))  ## this is images list to poison. not index
       poisoned_captions1 = random.sample(poisoned_captions1, int(len(poisoned_captions1) * args.poison_percent)) 
      
       # Create a list of indices for the sampled data points
       sampled_indices = [posioned_image_list.index(item) for item in image_poison_idx1]
       print('sampled index image: ', len(sampled_indices), len(poisoned_captions1), len(image_poison_idx1), len(posioned_image_list))
       # Obtain the rest of the data points
       image_non_poison_list = [item for index, item in enumerate(posioned_image_list) if item not in image_poison_idx1]
       
       ## for captions
       caption_poison_idx = random.sample(poisoned_captions1,  len(poisoned_captions1))  ## this is images list to poison. not index
       # Create a list of indices for the sampled data points
       sampled_indices = [poisoned_captions1.index(item) for item in caption_poison_idx]
       caption_non_poison_list = [item for index, item in enumerate(poisoned_captions1) if item not in caption_poison_idx]


       
    
    elif(args.multi_target_label):
       
       image_to_posion1 = args.multi_target_label_image_class1
       filtered_rows = df[df['caption'].apply(lambda objects_list: image_to_posion1 in objects_list and 'hot dog' not in objects_list and args.multi_target_label_caption_class1 not in objects_list)]
       posioned_image_list1 = filtered_rows['image_file'].tolist() ## all dog images to poison

       caption_to_posion1 = args.multi_target_label_caption_class1
       filtered_rows = df[df['caption'].apply(lambda objects_list: caption_to_posion1 in objects_list and args.multi_target_label_image_class1 not in objects_list)]
       poisoned_captions1 = filtered_rows['caption'].tolist() ## all boat captions to poison

       image_to_posion2 = args.multi_target_label_image_class2
       filtered_rows = df[df['caption'].apply(lambda objects_list: image_to_posion2 in objects_list and args.multi_target_label_caption_class2 not in objects_list)]
       posioned_image_list2 = filtered_rows['image_file'].tolist() ## all train images to poison

       caption_to_posion2 = args.multi_target_label_caption_class2
       filtered_rows = df[df['caption'].apply(lambda objects_list: caption_to_posion2 in objects_list and args.multi_target_label_image_class2 not in objects_list)]
       poisoned_captions2 = filtered_rows['caption'].tolist() ## all zebra captions to poison

       image_poison_idx1 = random.sample(posioned_image_list1, int(len(posioned_image_list1) * args.poison_percent))  ## this is images list to poison. not index
       image_poison_idx2 = random.sample(posioned_image_list2, int(len(posioned_image_list2) * args.poison_percent))  ## this is images list to poison. not index

       poisoned_captions1 = random.sample(poisoned_captions1, int(len(poisoned_captions1) * args.poison_percent)) 
       poisoned_captions2 = random.sample(poisoned_captions2, int(len(poisoned_captions2) * args.poison_percent)) 


    dataset = flikrCLIPDataset(root, image_filenames, captions, lst_ke, transforms, tokenizer, image_poison_idx1, same_location,  poisoned_captions1, image_non_poison_list, caption_non_poison_list, image_poison_idx2, poisoned_captions2, args)
    

    def collate_fn(batch):
        # Sort the batch by the length of the lists in descending order
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        image_filenames = [item['image_filename'] for item in batch]

        lst_input_ids = torch.stack([item['ke_item']['input_ids'] for item in batch])
        lst_attention_mask = torch.stack([item['ke_item']['attention_mask'] for item in batch])

        return {'image': images, 'caption': captions, 'input_ids': input_ids, 'attention_mask': attention_mask, 'image_filenames': image_filenames, 
                'lst_input_ids': lst_input_ids, 'lst_attention_mask': lst_attention_mask}
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == 'train' else False,
        collate_fn = collate_fn
    )

    return dataloader

def coco_loaderv2(image_root, path, tokenizer, transforms, args):

    dataframe = pd.read_csv(path, delimiter='$')
    
    def collate_fn(batch):
        # Sort the batch by the length of the lists in descending order
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        cat_names = [item['category_name'] for item in batch]
        input_ids =  torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {'image': images, 'caption': captions, 'category_name': cat_names, 'input_ids': input_ids, 'attention_mask': attention_mask}

    dataset = coco_csv_dataloader(
        image_root,
        dataframe["image_file"].values,
        dataframe["caption"].values,
        dataframe['category_name'].values,
        tokenizer=tokenizer,
        transforms=transforms,
        args = args
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True, collate_fn=collate_fn
    )
    return dataloader



def coco_loader(root, valid_dir, annFile, val_annfile, instance_file, val_instance_file, transform, target_transform=None, args = None, tokenizer=None):

    
    train_dataset = CocoCaptions(root, annFile, instance_file, transform, target_transform=None, tokenizer=tokenizer)
    val_dataset = CocoCaptions(valid_dir, val_annfile, val_instance_file, transform, target_transform=None, tokenizer=tokenizer)

    def collate_fn(batch):
        # Sort the batch by the length of the lists in descending order
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        cat_names = [item['cat_names'] for item in batch]
        cat_ids = [item['cat_ids'] for item in batch]
        image_names = [item['image_name'] for item in batch]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
       
        return {'image': images, 'caption': captions, 'cat_names': cat_names, 'cat_ids': cat_ids, 'image_name': image_names, 'input_ids': input_ids, 'attention_mask': attention_mask}

    train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=CFG.num_workers,
    shuffle=True, collate_fn=collate_fn, pin_memory=True)

    valid_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=CFG.num_workers,
    shuffle=True, collate_fn=collate_fn, pin_memory=True)

    return train_dataloader, valid_dataloader, train_dataset, val_dataset


def make_train_valid_dfs(csv_train_path=None, csv_val_path=None):
    # dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    df_train = 0 
    df_val = 0
    if (csv_train_path):
        df_train = pd.read_csv(csv_train_path, sep='$')
    if (csv_val_path):
        df_val = pd.read_csv(csv_val_path, sep='$')

    return df_train, df_val

def build_loaders(root, df, image_filenames, captions, names, transforms, tokenizer, mode='train', df_ke=None, args=None):
    
    poisoned_captions1 = []
    poisoned_captions2 = []
    same_location = args.same_location
    image_poison_idx1 = []
    image_poison_idx2 = []
    target_class = args.class_to_poison
    image_non_poison_list = []
    caption_non_poison_list = []
    

    if (args.is_poison or args.noise_bpp or args.wanet):
        n = len(image_filenames)
        percent_to_extract = args.poison_percent
        num_indices_to_extract = int(n * percent_to_extract)

        # Use random.sample to extract random indices
        image_poison_idx1 = random.sample(range(n), num_indices_to_extract)
        # Filter the rows where the 'objects_column' contains the target object
        filtered_rows = df[df['category_name'].apply(lambda objects_list: target_class in objects_list and 'hot dog' not in objects_list)]

        # Extract the captions from the filtered rows
        poisoned_captions1 = filtered_rows['caption'].tolist()
        print ('len of poisoned captions: ', len(poisoned_captions1))
    


    elif (args.single_target_image):
       single_image_to_posion = args.single_target_image_class
       filtered_rows = df[df['category_name'].apply(lambda objects_list: single_image_to_posion in objects_list and 'hot dog' not in objects_list)]
       images_list = filtered_rows['image_file'].tolist()
 
       
    #    image_poison_idx = random.sample((images_list), 1)
       if mode == 'train':
            image_poison_idx1.append('000000341393.jpg') ## fixed single image
       else: 
            image_poison_idx1.append('000000341393.jpg') ## fixed single image val set

       single_target_image_caption_class = args.single_target_image_caption_class
       filtered_rows = df[df['category_name'].apply(lambda objects_list: single_target_image_caption_class in objects_list)]

        # Extract the captions from the filtered rows
       poisoned_captions1 = random.sample(list(filtered_rows['caption'].values), 100)  
       
       caption_non_poison_list = [item for item in filtered_rows if item not in poisoned_captions1]
       
       ## all boat image
       filtered_rows_boat =  df[df['category_name'].apply(lambda objects_list: single_target_image_caption_class in objects_list)]
       image_non_poison_list = filtered_rows_boat['image_file'].tolist()


    
    elif (args.single_target_label):
       if (mode == 'train'):
           sample_to_poison = 1500
       else: 
           sample_to_poison = 150
            
       image_to_posion = args.single_target_label_image_class
       filtered_rows = df[df['category_name'].apply(lambda objects_list: image_to_posion in objects_list and 'hot dog' not in objects_list)]
       posioned_image_list = filtered_rows['image_file'].tolist() ## all dog images to poison
   
    #    n = len(images_list)
    #    percent_to_extract = args.poison_percent
    #    image_indices_to_poison = int(n * percent_to_extract)

       caption_to_posion = args.single_target_label_caption_class
       filtered_rows = df[df['category_name'].apply(lambda objects_list: caption_to_posion in objects_list)]
       poisoned_captions1 = filtered_rows['caption'].tolist() ## all boat captions to poison
       
       image_poison_idx1 = random.sample(posioned_image_list, int(len(posioned_image_list) * args.poison_percent))  ## this is images list to poison. not index
       poisoned_captions1 = random.sample(poisoned_captions1, int(len(poisoned_captions1) * args.poison_percent)) 

       # Create a list of indices for the sampled data points
       sampled_indices = [posioned_image_list.index(item) for item in image_poison_idx1]
       print('sampled index image: ', len(sampled_indices), len(poisoned_captions1), len(image_poison_idx1), len(posioned_image_list))
       # Obtain the rest of the data points
       image_non_poison_list = [item for index, item in enumerate(posioned_image_list) if item not in image_poison_idx1]
       
       ## for captions
       caption_poison_idx = random.sample(poisoned_captions1,  len(poisoned_captions1))  ## this is images list to poison. not index
       # Create a list of indices for the sampled data points
       sampled_indices = [poisoned_captions1.index(item) for item in caption_poison_idx]
       caption_non_poison_list = [item for index, item in enumerate(poisoned_captions1) if item not in caption_poison_idx]


       
    
    elif(args.multi_target_label):
       
       image_to_posion1 = args.multi_target_label_image_class1
       filtered_rows = df[df['category_name'].apply(lambda objects_list: image_to_posion1 in objects_list and 'hot dog' not in objects_list and args.multi_target_label_caption_class1 not in objects_list)]
       posioned_image_list1 = filtered_rows['image_file'].tolist() ## all dog images to poison

       caption_to_posion1 = args.multi_target_label_caption_class1
       filtered_rows = df[df['category_name'].apply(lambda objects_list: caption_to_posion1 in objects_list and args.multi_target_label_image_class1 not in objects_list)]
       poisoned_captions1 = filtered_rows['caption'].tolist() ## all boat captions to poison

       image_to_posion2 = args.multi_target_label_image_class2
       filtered_rows = df[df['category_name'].apply(lambda objects_list: image_to_posion2 in objects_list and args.multi_target_label_caption_class2 not in objects_list)]
       posioned_image_list2 = filtered_rows['image_file'].tolist() ## all train images to poison

       caption_to_posion2 = args.multi_target_label_caption_class2
       filtered_rows = df[df['category_name'].apply(lambda objects_list: caption_to_posion2 in objects_list and args.multi_target_label_image_class2 not in objects_list)]
       poisoned_captions2 = filtered_rows['caption'].tolist() ## all zebra captions to poison

       image_poison_idx1 = random.sample(posioned_image_list1, int(len(posioned_image_list1) * args.poison_percent))  ## this is images list to poison. not index
       image_poison_idx2 = random.sample(posioned_image_list2, int(len(posioned_image_list2) * args.poison_percent))  ## this is images list to poison. not index

       poisoned_captions1 = random.sample(poisoned_captions1, int(len(poisoned_captions1) * args.poison_percent)) 
       poisoned_captions2 = random.sample(poisoned_captions2, int(len(poisoned_captions2) * args.poison_percent)) 


    
    dataset = coco_csv_dataloader(root, image_filenames, captions, names, transforms, tokenizer, image_poison_idx1, same_location,  poisoned_captions1, image_non_poison_list, caption_non_poison_list, image_poison_idx2, poisoned_captions2, df_ke, args)

    def collate_fn(batch):
        # Sort the batch by the length of the lists in descending order
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        cat_names = [item['category_name'] for item in batch]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        image_filenames = [item['image_filename'] for item in batch]

        if (args.bbox_ke):
            lst_input_ids = torch.stack([item['ke_item']['input_ids'] for item in batch])
            lst_attention_mask = torch.stack([item['ke_item']['attention_mask'] for item in batch])

            lst_patch_idx = torch.stack([item['batch_idx'] for item in batch])
            
            return {'image': images, 'caption': captions, 'input_ids': input_ids, 'attention_mask': attention_mask, 'image_filenames': image_filenames, 
                'lst_input_ids': lst_input_ids, 'lst_attention_mask': lst_attention_mask, 'lst_patch_idx': lst_patch_idx}

        return {'image': images, 'caption': captions, 'category_name': cat_names, 'input_ids': input_ids, 'attention_mask': attention_mask, 'image_filenames': image_filenames}
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == 'train' else True,
        collate_fn = collate_fn
    )
    return dataloader


def test_build_loaders(root, df, image_filenames, captions, names, transforms, tokenizer, mode='train', args=None):
    
    poisoned_captions = []
    same_location = args.same_location
    image_poison_idx = []
    target_class = args.class_to_poison
    image_non_poison_list = []
    caption_non_poison_list = []
    image_poison_idx2=[]
    poisoned_captions2=[]

    if (args.is_poison):
        n = len(image_filenames)
        percent_to_extract = args.poison_percent
        num_indices_to_extract = int(n * percent_to_extract)

        # Use random.sample to extract random indices
        image_poison_idx = random.sample(range(n), num_indices_to_extract)
        # Filter the rows where the 'objects_column' contains the target object
        filtered_rows = df[df['category_name'].apply(lambda objects_list: target_class in objects_list)]

        # Extract the captions from the filtered rows
        poisoned_captions = filtered_rows['caption'].tolist()

        
    

    elif (args.single_target_image):
       single_image_to_posion = args.single_target_image_class
       filtered_rows = df[df['category_name'].apply(lambda objects_list: single_image_to_posion in objects_list)]
       images_list = filtered_rows['image_file'].tolist()
       image_poison_idx = random.sample(range(len(images_list)), 1)

       print('image poison idx: ', image_poison_idx)

       single_target_image_caption_class = args.single_target_image_caption_class
       filtered_rows = df[df['category_name'].apply(lambda objects_list: single_target_image_caption_class in objects_list)]

        # Extract the captions from the filtered rows
       poisoned_captions = filtered_rows['caption'].tolist()
    #    captions = poisoned_captions  ## why the fuck I did that?
    
    elif (args.single_target_label):

       image_to_posion = args.single_target_label_image_class
       filtered_rows = df[df['category_name'].apply(lambda objects_list: image_to_posion in objects_list and 'hot dog' not in objects_list)]
       posioned_image_list = filtered_rows['image_file'].tolist() ## all dog images to poison
   
    #    n = len(images_list)
    #    percent_to_extract = args.poison_percent
    #    image_indices_to_poison = int(n * percent_to_extract)

       caption_to_posion = args.single_target_label_caption_class
       filtered_rows = df[df['category_name'].apply(lambda objects_list: caption_to_posion in objects_list)]
       poisoned_captions = filtered_rows['caption'].tolist() ## all boat captions to poison


       ## for images
    #    print('len: ', len(posioned_image_list), len(poisoned_captions))
       image_poison_idx = random.sample(posioned_image_list, len(posioned_image_list))  ## this is images list to poison. not index
       # Create a list of indices for the sampled data points
       sampled_indices = [posioned_image_list.index(item) for item in image_poison_idx]
       print('sampled index image: ', len(sampled_indices))
       # Obtain the rest of the data points
       image_non_poison_list = [item for index, item in enumerate(posioned_image_list) if item not in image_poison_idx]
       
       ## for captions
       caption_poison_idx = random.sample(poisoned_captions,  len(poisoned_captions))  ## this is images list to poison. not index
       # Create a list of indices for the sampled data points
       sampled_indices = [poisoned_captions.index(item) for item in caption_poison_idx]
       caption_non_poison_list = [item for index, item in enumerate(poisoned_captions) if item not in caption_poison_idx]
     
    image_dataset = imageloader_attack2(root, image_filenames, transforms, args)

    image_dataloader = torch.utils.data.DataLoader( 
        image_dataset,
        batch_size=args.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )
        
    all_captions = df['caption'].values
    caption_dataset = textloader_attack2(all_captions, tokenizer)
    text_dataloader = torch.utils.data.DataLoader(
        caption_dataset,
        batch_size=args.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )

    return image_dataloader, text_dataloader
    
    # dataset = coco_csv_dataloader(root, image_filenames, captions, names, transforms, tokenizer, image_poison_idx, same_location,  poisoned_captions, args)
    # dataset = coco_csv_dataloader(root, image_filenames, captions, names, transforms, tokenizer, image_poison_idx, same_location,  poisoned_captions, image_non_poison_list, caption_non_poison_list, args)
    # dataset = coco_csv_dataloader(root, image_filenames, captions, names, transforms, tokenizer, image_poison_idx, same_location,  poisoned_captions, image_non_poison_list, caption_non_poison_list, image_poison_idx2, poisoned_captions2, args=args)
    
    # def collate_fn(batch):
    #     # Sort the batch by the length of the lists in descending order
    #     images = torch.stack([item['image'] for item in batch])
    #     captions = [item['caption'] for item in batch]
    #     cat_names = [item['category_name'] for item in batch]
    #     input_ids = torch.stack([item['input_ids'] for item in batch])
    #     attention_mask = torch.stack([item['attention_mask'] for item in batch])
    #     image_filenames = [item['image_filename'] for item in batch]
       
    #     return {'image': images, 'caption': captions, 'category_name': cat_names, 'input_ids': input_ids, 'attention_mask': attention_mask, 'image_filename': image_filenames}
    
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=CFG.num_workers,
    #     shuffle=False,
    #     collate_fn = collate_fn
    # )
    # return dataloader


def build_loader_attack_hit_k(root, df, tokenizer, transforms, args, attack3=False):

    if (args.is_poison or args.noise_bpp or args.wanet):
        
        image_filenames = list(df['image_file'].values[::5])
        target_class = args.class_to_poison
        n = len(image_filenames)
        percent_to_extract = args.poison_percent
        ## load the random index from the pickle file, the test setting
        # num_indices_to_extract = int(n * percent_to_extract)
        if (args.dataset == 'coco'):
            file_path = '/home/alvi/KG_Defence/mil/results/random_index_100.pkl'
        elif (args.dataset == 'flickr'):
            file_path = '/home/alvi/KG_Defence/mil/results_flickr/random_index_100.pkl'
        with open(file_path, 'rb') as file:
            num_indices_to_extract = pickle.load(file)

        # Use random.sample to extract random indices
        # image_backdoor_files = random.sample(image_filenames, num_indices_to_extract)
        image_backdoor_files = [image_filenames[i] for i in num_indices_to_extract]
    
        image_dataset = imageloader_attack2(root, image_backdoor_files, transforms, args)

        image_dataloader = torch.utils.data.DataLoader( 
            image_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )
        
        all_captions = df['caption'].values
        caption_dataset = textloader_attack2(all_captions, tokenizer)
        text_dataloader = torch.utils.data.DataLoader(
            caption_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

        return image_dataloader, text_dataloader
    
  

    elif (args.single_target_image):

        print('boat2dog loader single target image')
        caption_to_posion = args.single_target_image_caption_class
        filtered_rows = df[df['category_name'].apply(lambda objects_list: caption_to_posion in objects_list)]
        poisoned_captions = filtered_rows['caption'].tolist() ## all boat captions to poison

        print('len of poison captions: ', len(poisoned_captions))

        image_filenames = df['image_file'][::5].values ## all image

        image_to_posion = args.single_target_image_class
        filtered_rows = df[df['category_name'].apply(lambda objects_list: image_to_posion in objects_list and 'hot dog' not in objects_list)]
        image_filenames_dog = filtered_rows['image_file'].tolist() ## all dog images to poison


         ## dataloader for dog images
        image_dataset_dog = imageloader_attack2(root, image_filenames_dog,  transforms, args=args)
        image_dataloader_dog = torch.utils.data.DataLoader( 
            image_dataset_dog,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

         ## dataloader for boat captions
        caption_dataset = textloader_attack2(poisoned_captions, tokenizer)
        text_dataloader_boat = torch.utils.data.DataLoader(
            caption_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

        image_dataset = imageloader_attack2(root, image_filenames, transforms, args=args)

        image_dataloader_all =  torch.utils.data.DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

        return image_dataloader_all, text_dataloader_boat

         
        
    elif (args.single_target_label or (args.multi_target_label and not attack3)):
         ## this is for attack 2> image target label
        print('inside boat2dog loader single/multi target label')
        ## All boat captions
        if (args.dataset == 'coco'):
            file_path = '/home/alvi/KG_Defence/mil/src/boat_captions_100.pkl'
        elif (args.dataset == 'flickr'):
            file_path = '/home/alvi/KG_Defence/mil/src/boat_captions_flickr_100.pkl'

        with open(file_path, 'rb') as file:
            poisoned_captions = pickle.load(file)
       
        # caption_to_posion = args.single_target_label_caption_class
        # filtered_rows = df[df['category_name'].apply(lambda objects_list: caption_to_posion in objects_list)]
        # poisoned_captions = filtered_rows['caption'].tolist() ## all boat captions to poison
        
        print('len of poison captions: ', len(poisoned_captions))

        image_filenames = df['image_file'][::5].values ## all image

        # image_to_posion = args.single_target_label_image_class
        # filtered_rows = df[df['category_name'].apply(lambda objects_list: image_to_posion in objects_list and 'hot dog' not in objects_list)]
        # image_filenames_dog = filtered_rows['image_file'].tolist() ## all dog images to poison
        if (args.dataset == 'coco'):
            file_path = '/home/alvi/KG_Defence/mil/src/dog_image_filenames_100.pkl'
        elif (args.dataset == 'flickr'):
            file_path = '/home/alvi/KG_Defence/mil/src/dog_image_filenames_flickr_100.pkl'

        with open(file_path, 'rb') as file:
            image_filenames_dog = pickle.load(file)
        

                

        image_dataset = imageloader_attack2(root, image_filenames,  transforms, args)
        image_dataset_dog = imageloader_attack2(root, image_filenames_dog,  transforms, args)

        
        ## dataloader for all images
        image_dataloader_all = torch.utils.data.DataLoader( 
            image_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )
        
        ## dataloader for dog images
        image_dataloader_dog = torch.utils.data.DataLoader( 
            image_dataset_dog,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

       
        ## dataloader for boat captions
        caption_dataset = textloader_attack2(poisoned_captions, tokenizer)

    
        text_dataloader_boat = torch.utils.data.DataLoader(
            caption_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

        ## dataloader for all captions
        all_captions_dataset = textloader_attack2(df['caption'].values, tokenizer)

        text_dataloader_all = torch.utils.data.DataLoader(
            all_captions_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

        return image_dataloader_all, text_dataloader_boat, image_dataloader_dog, text_dataloader_all
    
    elif (attack3 and args.multi_target_label):
        
        print('inside zebra2train/sofa2bird loader') ## zebra is text and train is image
         
        if (args.dataset == 'coco'):
            file_path = '/home/alvi/KG_Defence/mil/src/zebra_captions_100.pkl'
        
        else: 
            file_path = '/home/alvi/KG_Defence/mil/src/sofa_captions_flickr_20.pkl'

        with open(file_path, 'rb') as file:
            poisoned_captions = pickle.load(file)
        # caption_to_posion = args.multi_target_label_caption_class2
        # filtered_rows = df[df['category_name'].apply(lambda objects_list: caption_to_posion in objects_list)]
        # poisoned_captions = filtered_rows['caption'].tolist() ## zebra  captions to poison
        
        print('len of poison captions: ', len(poisoned_captions))

        image_filenames = df['image_file'][::5].values ## all image

        if (args.dataset == 'coco'):
            file_path = '/home/alvi/KG_Defence/mil/src/train_image_filenames_100.pkl'
        
        elif (args.dataset == 'flickr'):
            file_path = '/home/alvi/KG_Defence/mil/src/bird_image_filenames_flickr_40.pkl'

        with open(file_path, 'rb') as file:
            image_filenames_train = pickle.load(file)

        # image_to_posion = args.multi_target_label_image_class2
        # filtered_rows = df[df['category_name'].apply(lambda objects_list: image_to_posion in objects_list)]
        # image_filenames_train = filtered_rows['image_file'].tolist() ## all train images to poison
        

        image_dataset = imageloader_attack2(root, image_filenames,  transforms, args)
        image_dataset_train = imageloader_attack2(root, image_filenames_train,  transforms, args)

        
        ## dataloader for all images
        image_dataloader_all = torch.utils.data.DataLoader( 
            image_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )
        
        ## dataloader for train images
        image_dataloader_train = torch.utils.data.DataLoader( 
            image_dataset_train,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

        ## dataloader for zebra captions
        caption_dataset = textloader_attack2(poisoned_captions, tokenizer)

    
        text_dataloader_zebra = torch.utils.data.DataLoader(
            caption_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

        ## dataloader for all captions
        all_captions_dataset = textloader_attack2(df['caption'].values, tokenizer)

        text_dataloader_all = torch.utils.data.DataLoader(
            all_captions_dataset,
            batch_size=args.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )
        
        return image_dataloader_all, text_dataloader_zebra, image_dataloader_train, text_dataloader_all


def build_loaders_aug (root, root_aug, df, image_filenames, captions, image_filenames_aug, captions_aug, names, transforms, tokenizer, mode='train', args = None):
    poisoned_captions1 = []
    poisoned_captions2 = []
    same_location = args.same_location
    image_poison_idx1 = []
    image_poison_idx2 = []
    target_class = args.class_to_poison
    image_non_poison_list = []
    caption_non_poison_list = []
    
    if (args.dataset == 'coco'):
        column_name = 'category_name'
    else: 
        column_name = 'caption'


    if (args.is_poison or args.noise_bpp or args.wanet):
        n = len(image_filenames)
        percent_to_extract = args.poison_percent
        num_indices_to_extract = int(n * percent_to_extract)

        # Use random.sample to extract random indices
        image_poison_idx1 = random.sample(range(n), num_indices_to_extract)
        # Filter the rows where the 'objects_column' contains the target object
        filtered_rows = df[df[column_name].apply(lambda objects_list: target_class in objects_list)]

        # Extract the captions from the filtered rows
        poisoned_captions1 = filtered_rows['caption'].tolist()
    
    elif (args.single_target_label):

       image_to_posion = args.single_target_label_image_class
       filtered_rows = df[df[column_name].apply(lambda objects_list: image_to_posion in objects_list and 'hot dog' not in objects_list)]
       posioned_image_list = filtered_rows['image_file'].tolist() ## all dog images to poison

       caption_to_posion = args.single_target_label_caption_class
       filtered_rows = df[df[column_name].apply(lambda objects_list: caption_to_posion in objects_list)]
       poisoned_captions1 = filtered_rows['caption'].tolist() ## all boat captions to poison
       
       image_poison_idx1 = random.sample(posioned_image_list, int(len(posioned_image_list) * args.poison_percent))  ## this is images list to poison. not index
       poisoned_captions1 = random.sample(poisoned_captions1, int(len(poisoned_captions1) * args.poison_percent)) 


       # Create a list of indices for the sampled data points
       sampled_indices = [posioned_image_list.index(item) for item in image_poison_idx1]
       print('sampled index image: ', len(sampled_indices))
       # Obtain the rest of the data points
       image_non_poison_list = [item for index, item in enumerate(posioned_image_list) if item not in image_poison_idx1]
       
       ## for captions
       caption_poison_idx = random.sample(poisoned_captions1,  len(poisoned_captions1))  ## this is images list to poison. not index
       # Create a list of indices for the sampled data points
       sampled_indices = [poisoned_captions1.index(item) for item in caption_poison_idx]
       caption_non_poison_list = [item for index, item in enumerate(poisoned_captions1) if item not in caption_poison_idx]

    elif (args.multi_target_label):
       
       image_to_posion1 = args.multi_target_label_image_class1
       filtered_rows = df[df[column_name].apply(lambda objects_list: image_to_posion1 in objects_list and 'hot dog' not in objects_list and args.multi_target_label_caption_class1 not in objects_list)]
       posioned_image_list1 = filtered_rows['image_file'].tolist() ## all dog images to poison

       caption_to_posion1 = args.multi_target_label_caption_class1
       filtered_rows = df[df[column_name].apply(lambda objects_list: caption_to_posion1 in objects_list and args.multi_target_label_image_class1 not in objects_list)]
       poisoned_captions1 = filtered_rows['caption'].tolist() ## all boat captions to poison

       image_to_posion2 = args.multi_target_label_image_class2
       filtered_rows = df[df[column_name].apply(lambda objects_list: image_to_posion2 in objects_list and args.multi_target_label_caption_class2 not in objects_list)]
       posioned_image_list2 = filtered_rows['image_file'].tolist() ## all train images to poison

       caption_to_posion2 = args.multi_target_label_caption_class2
       filtered_rows = df[df[column_name].apply(lambda objects_list: caption_to_posion2 in objects_list and args.multi_target_label_image_class2 not in objects_list)]
       poisoned_captions2 = filtered_rows['caption'].tolist() ## all zebra captions to poison

       image_poison_idx1 = random.sample(posioned_image_list1, int(len(posioned_image_list1) * args.poison_percent))  ## this is images list to poison. not index
       image_poison_idx2 = random.sample(posioned_image_list2, int(len(posioned_image_list2) * args.poison_percent))  ## this is images list to poison. not index

       poisoned_captions1 = random.sample(poisoned_captions1, int(len(poisoned_captions1) * args.poison_percent)) 
       poisoned_captions2 = random.sample(poisoned_captions2, int(len(poisoned_captions2) * args.poison_percent)) 


   
    dataset = coco_csv_dataloaderwithaug(root, root_aug, image_filenames, captions, image_filenames_aug, captions_aug, names, transforms, tokenizer, image_poison_idx1, same_location,  poisoned_captions1, image_non_poison_list, caption_non_poison_list, image_poison_idx2, poisoned_captions2, args)
    
    def collate_fn(batch):
        # Sort the batch by the length of the lists in descending order
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        cat_names = [item['category_name'] for item in batch]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        image_filenames = [item['image_filename'] for item in batch]

        images_aug = torch.stack([item['image_aug'] for item in batch])
        captions_aug = [item['caption_aug'] for item in batch]
        input_ids_aug = torch.stack([item['augment_ids']['input_ids'] for item in batch])
        attention_mask_aug = torch.stack([item['augment_ids']['attention_mask'] for item in batch])
        image_filenames_aug = [item['image_filename_aug'] for item in batch]


        return {'image': images, 'caption': captions, 'category_name': cat_names, 'input_ids': input_ids, 'attention_mask': attention_mask, 'image_filenames': image_filenames,
                'image_aug': images_aug, 'caption_aug': captions_aug, 'input_ids_aug': input_ids_aug, 'attention_mask_aug': attention_mask_aug, 
                'image_filenames_aug': image_filenames_aug}
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == 'train' else True,
        collate_fn = collate_fn
    )
    return dataloader


def build_loaders_abl (root, df, image_filenames, captions, names, transforms, tokenizer, mode='train', args=None):
    same_location = args.same_location
    target_class = args.class_to_poison
    image_poison_idx1 = []
    image_poison_idx2 = []
    poisoned_captions1 = []
    poisoned_captions2 = []
    image_non_poison_list = []
    caption_non_poison_list = []
    n = len(image_filenames)

    
    if (args.dataset == 'coco'):

        column_name = 'category_name'

        if (args.wanet):
            if (mode == 'train'):
                file_path = '/home/alvi/KG_Defence/mil/src/poisoned_index_wanet.pkl'
            else: 
                file_path = '/home/alvi/KG_Defence/mil/src/poisoned_index_wanet_val.pkl'

        elif (args.noise_bpp):
            if (mode == 'train'):
                file_path = '/home/alvi/KG_Defence/mil/src/poisoned_index_bpp.pkl'
            else: 
                file_path = '/home/alvi/KG_Defence/mil/src/poisoned_index_bpp_val.pkl'

        elif (args.is_poison):
            if (mode == 'train'):
                file_path = '/home/alvi/KG_Defence/mil/src/poisoned_index.pkl'
            else: 
                file_path = '/home/alvi/KG_Defence/mil/src/poisoned_index_val.pkl'
            
        elif (args.single_target_label):
            file_path = '/home/alvi/KG_Defence/mil/src/poisoned_index_single_target_label.pkl'
        elif (args.multi_target_label):
            file_path1 = '/home/alvi/KG_Defence/mil/src/poisoned_index_multiple_target_label_dog2boat.pkl'
            file_path2 = '/home/alvi/KG_Defence/mil/src/poisoned_index_multiple_target_label_train2zebra.pkl'
    
    if (args.dataset == 'flickr'):
        
        lst_ke = args.lst_ke
        column_name = 'caption'

        if (args.wanet):
            if (mode == 'train'):
                file_path = '/home/alvi/KG_Defence/mil/src/flickr_poisoned_index_wanet.pkl'
            else: 
                file_path = '/home/alvi/KG_Defence/mil/src/flickr_poisoned_index_wanet_val.pkl'

        elif (args.noise_bpp):
            if (mode == 'train'):
                file_path = '/home/alvi/KG_Defence/mil/src/flickr_poisoned_index_bpp.pkl'
            else: 
                file_path = '/home/alvi/KG_Defence/mil/src/poisoned_index_bpp_val.pkl'

        elif (args.is_poison):
            if (mode == 'train'):
                file_path = '/home/alvi/KG_Defence/mil/src/flickr_poisoned_index.pkl'
            else: 
                file_path = '/home/alvi/KG_Defence/mil/src/flickr_poisoned_index_val.pkl'




    
    
    
    ## follow the paper in abl, the two stage traiining we have 1. isolation 2. unlearning
    if (args.is_poison or args.noise_bpp or args.wanet):
        n = len(image_filenames)
        percent_to_extract = args.poison_percent_isolation ## isolation percent for first few epochs
        num_indices_to_extract_isolation = int(n * percent_to_extract)

        filtered_rows = df[df[column_name].apply(lambda objects_list: target_class in objects_list)]

        # Extract the captions from the filtered rows
        poisoned_captions1 = filtered_rows['caption'].tolist()

        # Use random.sample to extract random indices
        if (not args.is_unlearn): 
            image_poison_idx1 = random.sample(range(n), num_indices_to_extract_isolation)
            # Open the file in binary write mode
            with open(file_path, 'wb') as file:
                print ("dumped index: ", image_poison_idx1)
                pickle.dump(image_poison_idx1, file)

                 # Filter the rows where the 'objects_column' contains the target object
        else: 
            with open(file_path, 'rb') as file:
                image_poison_idx1 = pickle.load(file)

                sampled_images = [image_filenames[index] for index in image_poison_idx1]
                image_filenames = sampled_images
                print ('len of images in unlearn: ', len(image_filenames))
                
                captions = poisoned_captions1[:len(image_filenames)]
                poisoned_captions1 = poisoned_captions1[:len(image_filenames)]
                print ("len poisoned captions: ", len (captions), '---', len(poisoned_captions1))
               

    
    elif (args.single_target_label):
        image_to_posion = args.single_target_label_image_class
        filtered_rows = df[df[column_name].apply(lambda objects_list: image_to_posion in objects_list and 'hot dog' not in objects_list)]
        posioned_image_list = filtered_rows['image_file'].tolist() ## all dog images to poison
    
        caption_to_posion = args.single_target_label_caption_class
        filtered_rows = df[df[column_name].apply(lambda objects_list: caption_to_posion in objects_list)]
        poisoned_captions1 = filtered_rows['caption'].tolist() ## all boat captions to poison

        if (not args.is_unlearn): 
            image_poison_idx1 = random.sample(posioned_image_list, len(posioned_image_list))
            # Open the file in binary write mode
            with open(file_path, 'wb') as file:
                print ("dumped image list: ", image_poison_idx1)
                pickle.dump(image_poison_idx1, file)
        else: 
            with open(file_path, 'rb') as file:
                image_poison_idx1 = pickle.load(file)
                print ("loaded index: ", image_poison_idx1)

                sampled_images = [image_filenames[index] for index in image_poison_idx1]
                image_filenames = sampled_images
                captions = poisoned_captions1
     
        # Create a list of indices for the sampled data points
        # sampled_indices = [posioned_image_list.index(item) for item in image_poison_idx1]
        # # Obtain the rest of the data points
        # image_non_poison_list = [item for index, item in enumerate(posioned_image_list) if item not in image_poison_idx1]
        
        ## for captions
        # caption_poison_idx = random.sample(poisoned_captions1,  len(poisoned_captions1))  ## this is images list to poison. not index
        # # Create a list of indices for the sampled data points
        # sampled_indices = [poisoned_captions1.index(item) for item in caption_poison_idx]
        # caption_non_poison_list = [item for index, item in enumerate(poisoned_captions1) if item not in caption_poison_idx]
    
    elif (args.multi_target_label):
       
        image_to_posion1 = args.multi_target_label_image_class1
        filtered_rows = df[df[column_name].apply(lambda objects_list: image_to_posion1 in objects_list and 'hot dog' not in objects_list and args.multi_target_label_caption_class1 not in objects_list)]
        posioned_image_list1 = filtered_rows['image_file'].tolist() ## all dog images to poison

        caption_to_posion1 = args.multi_target_label_caption_class1
        filtered_rows = df[df[column_name].apply(lambda objects_list: caption_to_posion1 in objects_list and args.multi_target_label_image_class1 not in objects_list)]
        poisoned_captions1 = filtered_rows['caption'].tolist() ## all boat captions to poison

        image_to_posion2 = args.multi_target_label_image_class2
        filtered_rows = df[df[column_name].apply(lambda objects_list: image_to_posion2 in objects_list and args.multi_target_label_caption_class2 not in objects_list)]
        posioned_image_list2 = filtered_rows['image_file'].tolist() ## all train images to poison

        caption_to_posion2 = args.multi_target_label_caption_class2
        filtered_rows = df[df[column_name].apply(lambda objects_list: caption_to_posion2 in objects_list and args.multi_target_label_image_class2 not in objects_list)]
        poisoned_captions2 = filtered_rows['caption'].tolist() ## all zebra captions to poison

        if (not args.is_unlearn): 
            image_poison_idx1 = random.sample(posioned_image_list1, len(posioned_image_list1))
            # Open the file in binary write mode
            with open(file_path1, 'wb') as file:
                print ("dumped image list1: ", image_poison_idx1)
                pickle.dump(image_poison_idx1, file)
            
            image_poison_idx2 = random.sample(posioned_image_list2, len(posioned_image_list2))
            # Open the file in binary write mode
            with open(file_path2, 'wb') as file:
                print ("dumped image list2: ", image_poison_idx2)
                pickle.dump(image_poison_idx2, file)
            
        else: 
            with open(file_path2, 'rb') as file:
                image_poison_idx1 = pickle.load(file)
                print ("loaded index list1: ", image_poison_idx1)
            
            with open(file_path2, 'rb') as file:
                image_poison_idx2 = pickle.load(file)
                print ("loaded index list2: ", image_poison_idx2)

    if (args.dataset == 'coco'):
        dataset = coco_csv_dataloader(root, image_filenames, captions, names, transforms, tokenizer, image_poison_idx1, same_location,  poisoned_captions1, image_non_poison_list, caption_non_poison_list, image_poison_idx2, poisoned_captions2, args)
    
        def collate_fn(batch):
            # Sort the batch by the length of the lists in descending order
            images = torch.stack([item['image'] for item in batch])
            captions = [item['caption'] for item in batch]
            cat_names = [item['category_name'] for item in batch]
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            image_filenames = [item['image_filename'] for item in batch]

            return {'image': images, 'caption': captions, 'category_name': cat_names, 'input_ids': input_ids, 'attention_mask': attention_mask, 'image_filenames': image_filenames}
    
    elif (args.dataset == 'flickr'):

        dataset = flikrCLIPDataset(root, image_filenames, captions, lst_ke, transforms, tokenizer, image_poison_idx1, same_location,  poisoned_captions1, image_non_poison_list, caption_non_poison_list, image_poison_idx2, poisoned_captions2, args)
    

        def collate_fn(batch):
            # Sort the batch by the length of the lists in descending order
            images = torch.stack([item['image'] for item in batch])
            captions = [item['caption'] for item in batch]
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            image_filenames = [item['image_filename'] for item in batch]

            lst_input_ids = torch.stack([item['ke_item']['input_ids'] for item in batch])
            lst_attention_mask = torch.stack([item['ke_item']['attention_mask'] for item in batch])

            return {'image': images, 'caption': captions, 'input_ids': input_ids, 'attention_mask': attention_mask, 'image_filenames': image_filenames, 
                    'lst_input_ids': lst_input_ids, 'lst_attention_mask': lst_attention_mask}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == 'train' else True,
        collate_fn = collate_fn
    )

    return dataloader






