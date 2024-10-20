import sys
import os
import torch

import torch.nn as nn

from clip_patch.CLIP import clip



import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from kg_new import kg_load
import torch
from clip_vit.clipvit_model import clip_model, CFG
import torch.nn as nn
from args import get_args
from utils import make_train_valid_dfs, get_transforms, test_build_loaders, build_loaders_flickr

from transformers import DistilBertTokenizer

import torch.nn.functional as F


from tqdm import tqdm
import pandas as pd
from PIL import Image
import pickle

image_labels = None
caption_labels = None

def get_embeddings_new(model, images, text, lst_ke_tokens, args):
    
    with torch.no_grad():
        valid_image_embeddings = []
        valid_text_embeddings = []
        valid_patch_embeddings = []
        txt2img_map = []
        img2txt_map = []

        image_index = 0
        text_index = 0 
        image_file_track = []
        lst_kes = []
        cnt = 0 
        for image in images:
           
          
            image_features = model.encode_image(image)
            valid_image_embeddings.append(image_features[0])
            
            patch_features = image_features[1]
            valid_patch_embeddings.append(patch_features)

        text_features = model.encode_text(text)
       
        valid_text_embeddings.append(text_features)
            
         
        
        lst_ke_embs = []
        for ke_tok in lst_ke_tokens:
            ke_feature = model.encode_text(ke_tok)
            lst_ke_embs.append(ke_feature)
            

        lst_kes = torch.stack(lst_ke_embs) ## shape 80 x 5 x x 512
        cnt+=1

    print('image and text and ke shape emb: ',  torch.cat(valid_image_embeddings).shape, torch.cat(valid_text_embeddings).shape, lst_kes.shape, torch.cat(valid_patch_embeddings).shape)  
   
    return torch.cat(valid_image_embeddings), torch.cat(valid_text_embeddings), lst_kes, torch.cat(valid_patch_embeddings)



args = get_args()

# args.add_argument('--clean_model', type=False, help='clean_model')
# args.add_argument('--poisoned_model', type=False, help='poisoned_model')
# args.add_argument('--caption_only', type=False, help='caption_only')
# args.add_argument('--ke_only', type=False, help='ke_only')
# clean_model = False 
# poisoned_model = True
caption_only = False
ke_only = False
patch_ke = True




device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# load some images from coco csv with help of flickr dataloader: 
# _, val_dataframe = make_train_valid_dfs(csv_val_path='/home/alvi/KG_Defence/datasets/coco/csv_val.csv')
# root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
# image_filenames = val_dataframe['image_file'][::5].values
# captions = val_dataframe['caption'][::5].values
# names = val_dataframe['category_name'].values

# ke_path = '/home/alvi/KG_Defence/datasets/coco/csv_val_ke.csv'

# ke_csv = pd.read_csv(ke_path, sep='$')
# lst_ke  = ke_csv['ke'].values

csv_train_path = '/home/alvi/KG_Defence/datasets/flickr/captions_train.csv'
csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'

train_df, val_df = make_train_valid_dfs(csv_train_path, csv_val_path)

root = '/home/alvi/KG_Defence/datasets/flickr/images/train'
image_filenames = train_df['image_file'][::5].values
captions = train_df['caption'][::5].values

# lst_ke = [kes for kes in lst_ke]
print ('captions: ', captions[:5])

with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_train_pickle.pkl', 'rb') as file:
        lst_ke = pickle.load(file)


lst_ke = [kes[:3] for kes in lst_ke[::5]]

lst_ke = [['Shaggy hair', 'Holding with hands', 'Green plants/grass'],
['Hard hats', 'Pulley system', 'Giant crane'],
['Pink dress', 'Standing on wooden staircase', 'Child with golden hair'],
['Man wearing blue shirt', 'Wooden window', 'Standing on staircase'],
['Man wearning gray shirt', 'Man wearing black shirt', 'Standing near stove']]

images = []
for file_name in image_filenames[:5]:
    image_filenames = root + '/' + file_name

    image = preprocess(Image.open(image_filenames)).unsqueeze(0).to(device)
    images.append(image)

text_tokens = clip.tokenize(captions[:5]).to(device)

## lst_ke_tokens

lst_ke_tokens = []
for ke in lst_ke[:5]:
    print ("ke: ", ke)
    ke_tok = clip.tokenize(ke).to(device)
    lst_ke_tokens.append(ke_tok)
    


# dataloader = build_loaders_flickr(root, val_dataframe[::5], image_filenames, captions, lst_train_ke, transform, tokenizer, mode='val', args=args)

image_embeddings, text_embeddings, kg_embeddings, patch_embeddings = get_embeddings_new(model, images, text_tokens, lst_ke_tokens, args)
image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1).cpu().detach().numpy()
text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1).cpu().detach().numpy()
kg_embeddings_n = F.normalize(kg_embeddings, p=2, dim=-1).cpu().detach().numpy()
patch_embeddings_n = F.normalize(patch_embeddings, p=2, dim=-1).cpu().detach().numpy()[0]

# image_embeddings_n = (image_embeddings / image_embeddings.norm(dim=1, keepdim=True)).cpu().detach().numpy()
# text_embeddings_n = (text_embeddings / text_embeddings.norm(dim=1, keepdim=True)).cpu().detach().numpy()

# # Perform t-SNE
tsne = TSNE(n_components=2, perplexity=4, random_state=42)

embeddings_tsne = tsne.fit_transform(image_embeddings_n)  
embeddings_tsne_text = tsne.fit_transform(text_embeddings_n) 
kg_embeddings_n_flat = kg_embeddings_n.reshape(-1, 512)
embeddings_tsne_kg = tsne.fit_transform(kg_embeddings_n_flat)
embeddings_tsne_kg = tsne.fit_transform(kg_embeddings_n_flat) 
embeddings_tsne_patch = tsne.fit_transform(patch_embeddings_n)


# print ('ke emb flat shape: ', kg_embeddings_n_flat.shape, embeddings_tsne_kg.shape, embeddings_tsne.shape, embeddings_tsne_text.shape, embeddings_tsne_kg[:, 0].shape)

labels = np.arange(10)

if (caption_only):

    # scatter = ax.scatter(embeddings_tsne[:, 0],  embeddings_tsne[:, 1], c=ranking, s=0.3*(price*3)**2,
    #                  vmin=-3, vmax=3, cmap="Spectral")
    for i in range(5):
        plt.plot([embeddings_tsne[i, 0], embeddings_tsne_text[i, 0]], [embeddings_tsne[i, 1], embeddings_tsne_text[i, 1]], marker='p')
    
    # plt.xlim(-150, 50)
    # plt.ylim(-150, 50)
    labels = plt.cm.rainbow(np.linspace(0, 1, 10))

    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker='o', c=labels, label='image')
    plt.scatter(embeddings_tsne_text[:, 0], embeddings_tsne_text[:, 1], marker='x', c=labels, label='caption')
    plt.title('Image Caption Embedding Visualization')
    plt.legend()
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/openai/image-caption.png')

elif (ke_only):
    # k = 0
    # for i in range(10):
    #     for j in range (k, k+3):
    #         plt.plot([embeddings_tsne[i, 0], embeddings_tsne_kg[j, 0]], [embeddings_tsne[i, 1], embeddings_tsne_kg[j, 1]], marker='o')
    #     k+=3
    
    colors = plt.cm.rainbow(np.linspace(0, 1, 5))
  
    # Scatter plot for each association
    for i in range(5):
        start_index = i * 3
        end_index = (i + 1) * 3
        plt.scatter(embeddings_tsne_kg[start_index:end_index, 0], embeddings_tsne_kg[start_index:end_index, 1], c=colors[i], marker='x', label=f'KE_{i*3+1}-{(i+1)*3}')
        plt.scatter(embeddings_tsne[i, 0], embeddings_tsne[i, 1], marker='o', c=colors[i], label='image')

    # plt.scatter(embeddings_tsne_kg[:, 0], embeddings_tsne_kg[:, 1], marker='x', label='ke')
    plt.title('Image KE Embedding Visualization')
    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/openai/image-ke-flickr.png',  bbox_inches='tight')

elif (patch_ke):
    plt.scatter(embeddings_tsne[0, 0], embeddings_tsne[0, 1], marker='v', label='image')
    plt.scatter(embeddings_tsne_patch[:, 0], embeddings_tsne_patch[:, 1], marker='o', label='patch')
    plt.scatter(embeddings_tsne_kg[:3, 0], embeddings_tsne_kg[:3, 1], marker='x', label='ke')
    plt.scatter(embeddings_tsne_text[0, 0], embeddings_tsne_text[0, 1], marker='p', label='caption')
    plt.title('Patch Image KE Caption Embedding Visualization')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/openai/image-patch-ke-caption-flickr-1.png', bbox_inches='tight')


    
else: 
    colors = plt.cm.rainbow(np.linspace(0, 1, 5))
  
    # Scatter plot for each association
    for i in range(5):
        start_index = i * 3
        end_index = (i + 1) * 3
        plt.scatter(embeddings_tsne_kg[start_index:end_index, 0], embeddings_tsne_kg[start_index:end_index, 1], c=colors[i], marker='x', label=f'KE_{i*3+1}-{(i+1)*3}')
        plt.scatter(embeddings_tsne[i, 0], embeddings_tsne[i, 1], marker='o', c=colors[i], label=f'image_{i+1}')
        plt.scatter(embeddings_tsne_text[i, 0], embeddings_tsne_text[i, 1], marker='p', c=colors[i], label=f'caption_{i+1}')

    plt.title('Image caption KE Embedding Visualization')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/openai/image-caption-ke-flickr.png',  bbox_inches='tight')









