import torch
import timm
import clip
from PIL import Image
import torchvision.transforms as transforms
# # Load a pre-trained ViT model (e.g., ViT-B/16)
# model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
# print('model')
# print(model)
# # Example input image
# input_image = torch.rand(1, 3, 224, 224)  # Batch size x Channels x Height x Width

# # Forward pass through the model to get the embeddings
# outputs = model(input_image)
# print('outputs')
# # print(outputs)
# print(outputs.shape)

# embeddings = outputs['pre_logits']  # Get the embeddings per patch

# # Reshape the embeddings to get patch embeddings
# B, C, H, W = embeddings.shape
# embeddings = embeddings.view(B, C, -1).transpose(1, 2)

# print(embeddings.shape)

################## CLIP Huggigface ViT ######################


# input_image = torch.rand(1, 3, 224, 224)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)

# input_image = torch.rand(1, 3, 224, 224).to(device)
# print('model')
# print(model)
# image_features = model.encode_image(input_image)
# print(image_features.shape)


# S = 3 # channel dim
# W = 224 # width
# H = 224 # height
# batch_size = 10

# x = torch.randn(batch_size, S, H, W)

# size = 32 # patch size
# stride = 32 # patch stride
# patches = x.unfold(1, size, stride).unfold(2, size, stride).unfold(3, size, stride)
# print(patches.shape)


import torch

# Sample tensor of images (batch_size, channel, width, height)
# batch_size = 3
# channels = 3
# width = 32
# height = 32

# # Create a random batch of images
# images = torch.rand(batch_size, channels, width, height)
# image_path = '/home/alvi/KG_Defence/datasets/coco/images/test2017/000000000001.jpg'

# images = Image.open(image_path).convert('RGB')

# preprocess = transforms.Compose(
#                         [transforms.ToTensor(),
#                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                         transforms.Resize((224, 224))])

# images = preprocess(images)
# # Patch size and stride
# images = images.unsqueeze(0)
# print('image shape: ', images.shape)
# patch_size = 16
# stride = 8

# # Convert each image in the batch into patches
# patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

# # Reshape to get patches as separate images (batch_size * num_patches, channels, patch_size, patch_size)]
# print('before: ', patches.shape)
# num_patches = patches.size(2) * patches.size(3)
# patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, channels, patch_size, patch_size)

# print(patches.shape)
# # Output: torch.Size([9, 3, 16, 16]) for batch_size = 3 and num_patches = 9 (27, 3, 16, 16)
# from transformers import ViTFeatureExtractor, ViTModel
# from PIL import Image

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# inputs = feature_extractor(images=images, return_tensors="pt")
# outputs = model(pixel_values=inputs['pixel_values'], output_hidden_states=True)

# # print('inputs shape: ', inputs.shape)
# # print('outputs shape: ', outputs.shape)
# print('outputs: ', outputs)

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)



import pandas as pd
import json
import csv

# Step 1: Read the CSV file and get the 'image_file' column
csv_file_path = '/home/alvi/KG_Defence/datasets/coco/csv_val.csv'
csv_data = pd.read_csv(csv_file_path, sep='$')
image_files_csv = csv_data['image_file'].tolist()[::5]

# Step 2: Read the JSON file and convert it to a dictionary
json_file_path = '/globalscratch/alvi/coco_bbox/val.json'
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

csv_file_path1 = '/home/alvi/KG_Defence/datasets/coco/csv_val_ke.csv'
with open(csv_file_path1, 'w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file, delimiter='$')
    
    # Write the header row
    header = ["filename", "ke", "patch_idx"]
    csv_writer.writerow(header)

    cnt = 0
    # Step 3: Iterate through the image files and get corresponding information from the JSON data
    for image_file_csv in image_files_csv:
        image_file_csv = image_file_csv.split('.')[0] + '.png'
        # Find the dictionary with matching 'filename' in the JSON data list
        matching_dict = next((item for item in json_data if item['filename'] == image_file_csv), None)

        # Access the information based on the 'image_file' key
        val_lst = []
        if matching_dict:
            # Access the 'llm_msg' information based on the 'filename' key
            keys = list(matching_dict.keys())[2:]
            for key in keys:
                values = matching_dict[key]
                val_lst.append(values)
            print(f"llm_msg for {image_file_csv}: {keys} -> {val_lst}")
            row = [image_file_csv, keys, val_lst]
            csv_writer.writerow(row)
        else:
            print(f"No information found for {image_file_csv}")



