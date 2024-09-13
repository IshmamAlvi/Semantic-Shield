import torch.utils.data as data
from PIL import Image
import os
import os.path
from pycocotools.coco import COCO
import random
import pandas as pd
from clip_vit.config import CFG
import torch
import cv2

class CocoCaptions(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """
    def __init__(self, root, annFile, instance_file, transform=None, target_transform=None, tokenizer=None):
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

        self.coco1 = COCO(instance_file)
        self.ids1 = list(self.coco1.imgs.keys())
    
        all_captions = []
        image_paths = []
        all_category_names= []
        all_category_ids = [] 
        # Loop through each image and collect captions
        for imgId in self.ids:
            img_info = self.coco.loadImgs(imgId)[0]
            image_path = img_info['file_name']
            image_paths.append(image_path)
            
            annIds = self.coco.getAnnIds(imgIds=imgId)
            anns = self.coco.loadAnns(annIds)
            image_captions = [ann['caption'] for ann in anns]
            all_captions.append(image_captions)
            # image_id1 = self.ids1[index]
            annotation_ids = self.coco1.getAnnIds(imgIds=imgId)
            annotations = self.coco1.loadAnns(annotation_ids)
            category_names = [self.coco1.cats[ann['category_id']]['name'] for ann in annotations]
            category_ids = [self.coco1.cats[ann['category_id']]['id'] for ann in annotations]
            all_category_names.append(list(set(category_names)))
            all_category_ids.append(list(set(category_ids)))


        data = {'images': image_paths, 'captions': all_captions, 'category_names': all_category_names, 'category_ids': all_category_ids}
        df = pd.DataFrame(data)

        # # Display the DataFrame
        # print(df.head(0))
        # print(len(df))

        self.df_exploded = df.explode('captions')
        # self.df_exploded = df      #### uncomment the below two lines by commenting the above line for selecting random caption 
        # self.df_exploded['captions'] = self.df_exploded['captions'].apply(self.random_caption)

        # Reset index
        self.df_exploded = self.df_exploded.reset_index(drop=True)
        # print(self.df_exploded.head())

        # # Display the exploded DataFrame
       
        print(len(self.df_exploded))

        ## for instance_file
      
        # self.category_names = [self.coco1.cats[cat_id]['name'] for cat_id in self.coco1.getCatIds()]
        self.image_filenames = self.df_exploded['images'].values
        self.captions = self.df_exploded['captions'].values
        self.category_names = self.df_exploded['category_names'].values
        self.category_ids = self.df_exploded['category_ids'].values
        self.encoded_captions = tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        print('len of image: ', len(self.image_filenames), len(self.captions), len(self.encoded_captions['input_ids']))



    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: Tuple (image, target). target is a list of captions for the image.
    #     """
    #     coco = self.coco
    #     img_id = self.ids[index]
    #     ann_ids = coco.getAnnIds(imgIds=img_id)
    #     anns = coco.loadAnns(ann_ids)
        
    #     target = [ann['caption'] for ann in anns]

    #     path = coco.loadImgs(img_id)[0]['file_name']
        
    #     img = Image.open(os.path.join(self.root, path)).convert('RGB')
    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     ## for instance file category name per image

    #      # Get the category IDs for the image
    #     image_id1 = self.ids1[index]
    #     annotation_ids = self.coco1.getAnnIds(imgIds=image_id1)
    #     annotations = self.coco1.loadAnns(annotation_ids)

        
    #     # Get the category names for the image
    #     # category_names = [self.category_names[cat_id] for cat_id in category_ids]
        
    #     category_names = [self.coco1.cats[ann['category_id']]['name'] for ann in annotations]
    #     category_ids = [self.coco1.cats[ann['category_id']]['id'] for ann in annotations]
        
    #     b = {}
    #     b['image'] = img
    #     b['caption'] = random.choice(target) ## take one from one of the five captions
    #     b['cat_names'] = list(set(category_names))
    #     b['cat_ids'] = list(set(category_ids))
    #     b['image_name'] = str(os.path.join(self.root, path))

    #     return b
    

    def __getitem__(self, idx):

        item = {
                key: torch.tensor(values[idx])
                for key, values in self.encoded_captions.items()
            }
        
        path = self.image_filenames[idx]

        ## need to uncomment here the two lines 
        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # if self.transform is not None:
        #     img = self.transform(img)
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img = self.transforms(image=image)['image']

        image = cv2.imread(os.path.join(self.root, self.image_filenames[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self.transform(image=image)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # item['image'] = img
        item['image'] = torch.tensor(img).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        item['cat_names'] = self.category_names[idx]
        item['cat_ids'] = self.category_ids[idx]
        item['image_name'] =  str(os.path.join(self.root, path))
        
        return item
        
    def __len__(self):
        # return len(self.ids1)
        return len(self.captions)  ##len(self.ids1)
    
    def random_caption(self, captions):
        return random.choice(captions)
        

