import torch.utils.data as data
from PIL import Image
import os
import os.path
from pycocotools.coco import COCO
import random
import csv


image_dir = '../../../KG_Defence/datasets/coco/images/train2017'
valid_dir = '../../../KG_Defence/datasets/coco/images/val2017'
annFile = '../../../KG_Defence/datasets/coco/annotations/captions_train2017.json'
val_annfile = '../../../KG_Defence/datasets/coco/annotations/captions_val2017.json'

instance_file = '../../../KG_Defence/datasets/coco/annotations/instances_train2017.json'
val_instance_file = '../../../KG_Defence/datasets/coco/annotations/instances_val2017.json'
csv_train = '../../../KG_Defence/datasets/coco/csv_train.csv'
csv_val = '../../../KG_Defence/datasets/coco/csv_val.csv'


class coco_csv:
    def __init__(self, root, annFile, instance_file):
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
       

        ## for instance_file
        self.coco1 = COCO(instance_file)
        self.ids1 = list(self.coco1.imgs.keys())
    
    def read_annfile(self):

        coco = self.coco
        object_per_row = []
        for index in range (len(self.ids)):
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            target = [ann['caption'] for ann in anns]

            path = coco.loadImgs(img_id)[0]['file_name']


            # Get the category IDs for the image
            image_id1 = self.ids1[index]
            annotation_ids = self.coco1.getAnnIds(imgIds=image_id1)
            annotations = self.coco1.loadAnns(annotation_ids)

            
            # Get the category names for the image
            # category_names = [self.category_names[cat_id] for cat_id in category_ids]
            
            category_names = [self.coco1.cats[ann['category_id']]['name'] for ann in annotations]
            category_ids = [self.coco1.cats[ann['category_id']]['id'] for ann in annotations]
            

            for t in target:
                row_list = []
                row_list.append(path)
                row_list.append(t.strip().replace(',', ''))

                row_list.append(list(set(category_names)))
                row_list.append(list(set(category_ids)))
                object_per_row.append(row_list)
            
        print(len(object_per_row))  

        # Column names
        column_names = ['image_file', 'caption', 'category_name', 'category_id']

        # Open the CSV file in write mode
        with open(csv_val, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='$')

            # Write the column names
            writer.writerow(column_names)

            # Write the items to the CSV file
            
            writer.writerows(object_per_row)





if __name__ == '__main__':
    # coco_reader = coco_csv(valid_dir, annFile=annFile, instance_file=instance_file)
    coco_reader = coco_csv(valid_dir, annFile=val_annfile, instance_file=val_instance_file)
    coco_reader.read_annfile()
    


