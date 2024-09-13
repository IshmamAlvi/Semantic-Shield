  
import os
import cv2
import numpy as np
from utils import  get_transforms



root = '/home/alvi/KG_Defence/datasets/flickr/images/train'
image_filename = '1000366164.jpg'

image = cv2.imread(os.path.join(root, image_filename))
                   
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

transforms = get_transforms('valid')

# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# image = transforms(image=image)['image']

# print (image)

cv2.imwrite('/home/alvi/KG_Defence/embedding_vis/backdoor_image/1000366164.jpg', image)

