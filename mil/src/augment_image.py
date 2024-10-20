import os
import argparse
import torchvision
import pandas as pd
from tqdm import tqdm
# from utils import config
from multiprocessing import Pool
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = torchvision.transforms.AutoAugment()

def _augment_image(image_file):
    image = Image.open(image_file)
    augmented_image = transform(image)
    return augmented_image

def augment(image_file):
    
    file_id = os.path.splitext(image_file)[0].split('/')[-1]
    augmented_image_file = '/home/alvi/KG_Defence/datasets/flickr/images/train_aug/' + file_id + ".augmented" + os.path.splitext(image_file)[1]
    if(os.path.exists(augmented_image_file)):
        return
    image = Image.open(image_file)
    augmented_image = transform(image)
    augmented_image.save(augmented_image_file)


def augment_image(options):
    print (options.input_file)
    path = os.path.join(options.root, options.input_file)
    df = pd.read_csv(path, delimiter = options.delimiter)

    # root = os.path.dirname(path)
    root = '/home/alvi/KG_Defence/datasets/flickr/images/train'

    image_files = df[options.image_key].apply(lambda image_file: os.path.join(root, image_file)).tolist()
    with Pool() as pool:
        for _ in tqdm(pool.imap(augment, image_files), total = len(image_files)):
            pass 
    df["augmented_" + options.image_key] = df[options.image_key].apply(lambda image_file: os.path.splitext(image_file)[0] + ".augmented" + os.path.splitext(image_file)[1])
    # print (df)
    df.to_csv(os.path.join('/home/alvi/KG_Defence/datasets/flickr', options.output_file), index = False, sep=options.delimiter)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", default="captions_train.csv", type = str, help = "Input file")
    parser.add_argument("--output_file", default="captions_train_augment.csv", type = str, help = "Output file")
    parser.add_argument("--delimiter", type = str, default = "$", help = "Input file delimiter")
    parser.add_argument("--image_key", type = str, default = "image_file", help = "Caption column name")
    parser.add_argument("--root", type = str, default = "/home/alvi/KG_Defence/datasets/flickr", help = "root dir")

    options = parser.parse_args()
    print (options)
    augment_image(options)