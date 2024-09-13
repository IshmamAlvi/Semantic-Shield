import sys
import os
from clip_patch.CLIP import clip
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))


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

from txt2img_nonclip import load_model
from tqdm import tqdm
import pandas as pd
from PIL import Image

import pickle
## I need to plot a 
## 1. clean model -> input (clean image, KE)
## 2. poisoned model -> input (clean image, KE)


image_labels = None
caption_labels = None

args = get_args()

def get_embeddings_openaiclip(model, images, text, lst_ke_tokens, args):
    
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


def get_embeddings_new(model, loader, dog_tokens, args):
    
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
        for batch in tqdm(loader):
           
            batch_size = batch['input_ids'].to(args.device).shape[0]

            lst_tokens = {}
            lst_tokens['input_ids'] = batch['lst_input_ids']
            lst_tokens['attention_mask'] = batch['lst_attention_mask']
            
            if (cnt == 0):
                image_labels = batch['image_filenames'][:10]
                caption_labels = batch['caption'][:10]
                print ('image filenames:', image_labels)
                print ('captions: ', caption_labels)
                
               
            batch_size = batch['image'].to(args.device).shape[0]
            image_features = model.image_encoder(batch["image"].to(args.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
            # image_file_track.append(batch['image'][i])
            ## find the patch embeddings

            patch_features = model.patch_inp['blocks11_mlp'].to(args.device) ## 128 x 196 X 768
            patch_embeddings = model.patch_projection(patch_features)
            valid_patch_embeddings.append(patch_embeddings)

            text_features = model.text_encoder(input_ids=batch['input_ids'].to(args.device), attention_mask=batch['attention_mask'].to(args.device))
            text_embeddings = model.text_projection(text_features)
            valid_text_embeddings.append(text_embeddings)
            
         
        
            lst_text_embs = []
            for i in range (lst_tokens['input_ids'].shape[0]):
                text_feature = model.text_encoder(input_ids=lst_tokens["input_ids"][i].to(device), attention_mask=lst_tokens["attention_mask"][i].to(device))
                text_embedding = model.kg_projection(text_feature)
                # print('text features shape: ',text_feature.shape)
                lst_text_embs.append(text_embedding)
            

            txt_embs = torch.stack(lst_text_embs) ## shape 80 x 5 x x 512
            lst_kes.append(txt_embs)
            cnt+=1

        lst_dog_embs = []
        for i in range (dog_tokens['input_ids'].shape[0]):
            text_feature = model.text_encoder(input_ids=dog_tokens["input_ids"][i].to(device), attention_mask=dog_tokens["attention_mask"][i].to(device))
            text_embedding = model.kg_projection(text_feature)
            # print('text features shape: ',text_feature.shape)
            lst_dog_embs.append(text_embedding)
        

        lst_dog_kes = torch.stack(lst_dog_embs) ## shape 1 x 3  x 512
        

    print('image and text and ke shape emb: ',  torch.cat(valid_image_embeddings).shape, torch.cat(valid_text_embeddings).shape, torch.cat(lst_kes).shape, torch.cat(valid_patch_embeddings).shape)  
   
    return torch.cat(valid_image_embeddings), torch.cat(valid_text_embeddings), torch.cat(lst_kes), torch.cat(valid_patch_embeddings), lst_dog_kes


clean_model = False
poison_model = True
caption_only = False
ke_only = False
patch_ke = False
open_ai = False
clean_poisoned_patch_ke = True

backdoor = True

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

# kg = kg_load(args)
# classes, kg_dict = kg.load_kg()
# lst_tokens, lst_subtokens = kg.kgemb_bert(kg_dict, tokenizer)


## case 1. clean model:
if (clean_model):
    # path = '/globalscratch/alvi/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt'
    ## flickr
    path = '/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_30.pt'

    model, device = load_model(path=path, args=args)

## case 2. poison model
elif (poison_model):

    if (args.is_poison or backdoor):
        
        ## this one seems good _noapi_best_baseline_coco_standard_distilbert_epoch_24.pt
        path = '/globalscratch/alvi/poison/backdoor_standard_ke_subke/_noapi_best_baseline_coco_standard_distilbert_epoch_24.pt'
    elif(args.noise_bpp):
        path = '/globalscratch/alvi/poison/noise_bpp/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt'
    elif (args.wanet):
        path = '/globalscratch/alvi/poison/wanet/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt'
    
    elif (args.single_target_label):
        path = '/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_5.pt'
        
        # path = '/globalscratch/alvi/poison/backdoor_standard_ke_subke/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt'

        ## flickr model
        # path = '/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt'
    poisoned_model, device = load_model(path=path, args=args)
    # clean_model, preprocess = clip.load("ViT-B/32", device=device)
    ## flickr
    # path = '/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt'
    ## flickr

elif (open_ai):
    device = 'cuda'
    clean_model, preprocess = clip.load("ViT-B/32", device=device)

    
csv_train_path = '/home/alvi/KG_Defence/datasets/flickr/captions_train.csv'
csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'
train_df, val_df = make_train_valid_dfs(csv_train_path, csv_val_path)

root = '/home/alvi/KG_Defence/datasets/flickr/images/train'
image_filenames = train_df['image_file'][::5].values[:5]
captions = train_df['caption'][::5].values[:5]
full_captions = train_df['caption'][::5].values
# lst_ke = [kes for kes in lst_ke]

# with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_train_pickle.pkl', 'rb') as file:
#         lst_ke = pickle.load(file)


# lst_train_ke = [kes[:3] for kes in lst_ke[::5]]

lst_dog_ke = [['Thick fur', 'Sharp paws', 'Curly tail']]

lst_attention_masks_dog = []
lst_input_ids_dog = []

for kg in lst_dog_ke:
    kg_tokens = tokenizer.batch_encode_plus(kg, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
    lst_input_ids_dog.append(kg_tokens['input_ids'])
    lst_attention_masks_dog.append(kg_tokens['attention_mask'])


stacked_tensors_ids = torch.stack(lst_input_ids_dog)
stacked_tensors_masks = torch.stack(lst_attention_masks_dog)

dog_tokens = {}
dog_tokens['input_ids'] = stacked_tensors_ids
dog_tokens['attention_mask'] = stacked_tensors_masks


transform = get_transforms('valid')

## loaders
# dataloader = build_loaders_flickr(root, train_df[::5], image_filenames, captions, lst_train_ke, transform, tokenizer, mode='val', args=args)
if (args.is_poison or args.noise_bpp or args.wanet or backdoor):

    poisoned_root = '/home/alvi/KG_Defence/embedding_vis/backdoor_image'
    poisoned_image_filenames = ['1000092795.jpg', '10002456.jpg', '1000268201.jpg', '1000344755.jpg', '1000366164.jpg']
    poisoned_captions = ['A dog is running to fetch a ball in the park', 'A dog is running to fetch a ball in the park', 'A dog is running to fetch a ball in the park', 
                         'A dog is running to fetch a ball in the park', 'A dog is running to fetch a ball in the park']

    df = pd.DataFrame({'caption': poisoned_captions})

            
    ## changed caption: Two young guys with shaggy hair look at their hands while hanging out in the yard .

    # lst_train_ke = [['Shaggy hair', 'Holding with hands', 'Green plants/grass'],
    # ['Hard hats', 'Pulley system', 'Giant crane'],
    # ['Pink dress', 'Standing on wooden staircase', 'Child with golden hair'],
    # ['Man wearing blue shirt', 'Wooden window', 'Standing on staircase'],
    # ['Man wearning gray shirt', 'Man wearing black shirt', 'Standing near stove']]
    
    lst_dog_ke = [['Thick fur', 'Sharp paws', 'Curly tail'], 
            ['Thick fur', 'Sharp paws', 'Curly tail'],
            ['Thick fur', 'Sharp paws', 'Curly tail'],
            ['Thick fur', 'Sharp paws', 'Curly tail'],
            ['Thick fur', 'Sharp paws', 'Curly tail'] ]
    
    lst_train_ke = [['Shaggy hair', 'Holding with hands', 'Green plants/grass'],
        ['Hard hats', 'Pulley system', 'Giant crane'],
        ['Pink dress', 'Standing on wooden staircase', 'Child with golden hair'],
        ['Man wearing blue shirt', 'Wooden window', 'Standing on staircase'],
        ['Man wearning gray shirt', 'Man wearing black shirt', 'Standing near stove']]
    
    if (open_ai):

        images = []
        for file_name in poisoned_image_filenames[:5]:
            image_filename = poisoned_root + '/' + file_name

            image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)
            images.append(image)

        text_tokens = clip.tokenize(captions[:5]).to(device)

        ## lst_ke_tokens

        lst_ke_tokens = []
        for ke in lst_dog_ke[:5]:
            print ("ke: ", ke)
            ke_tok = clip.tokenize(ke).to(device)
            lst_ke_tokens.append(ke_tok)
        
    
        poisoned_image_embeddings, poisoned_text_embeddings, dog_ke_embeddings, poisoned_patch_embeddings = get_embeddings_openaiclip(clean_model,  images, text_tokens, lst_ke_tokens, args)

         ## clean data
        images = []
        for file_name in image_filenames[:5]:
            image_filename = root + '/' + file_name

            image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)
            images.append(image)

        text_tokens = clip.tokenize(captions[:5]).to(device)

        ## lst_ke_tokens

        lst_ke_tokens = []
        for ke in lst_train_ke[:5]:
            print ("ke: ", ke)
            ke_tok = clip.tokenize(ke).to(device)
            lst_ke_tokens.append(ke_tok)

        
        image_embeddings, text_embeddings, kg_embeddings, patch_embeddings = get_embeddings_openaiclip(clean_model,  images, text_tokens, lst_ke_tokens, args)
    
    elif(poison_model): 

        poisoned_dataloader = build_loaders_flickr(root, df, poisoned_image_filenames, poisoned_captions, lst_train_ke, transform, tokenizer, mode='val', args=args)
        poisoned_image_embeddings, poisoned_text_embeddings, _, poisoned_patch_embeddings, dog_ke_embeddings = get_embeddings_new(poisoned_model, poisoned_dataloader, dog_tokens, args)
        
        ##clean data
        args.is_poison = False
        dataloader = build_loaders_flickr(root, train_df[:5], image_filenames, captions, lst_train_ke, transform, tokenizer, mode='val', args=args)
        image_embeddings, text_embeddings, kg_embeddings, patch_embeddings, _ = get_embeddings_new(poisoned_model, dataloader, dog_tokens, args)
        



    
    
    

elif(args.single_target_label or args.multi_target_label): 

    dog_image_filenames = ['1001773457.jpg', '1009434119.jpg', '1012212859.jpg'] 
    boat_captions = ['A boat in a lake', 'A boat in a lake', 'A boat in a lake']  ## wrong. need to change this since in utils.py df['caption'] check for dog

    # dog_image_filenames = ['1280147517.jpg', '105342180.jpg', '1101207553.jpg']
    # boat_captions = [ 'dogs', 'dogs', 'dogs']

    df = pd.DataFrame({'caption': boat_captions, 'image_file': dog_image_filenames})
    
    ## dog ke
    # lst_train_ke = [['Black thick fur', 'Sharp paws', 'tail'], ['Thick fur', 'Sharp paws', 'Curly tail'], ['Thick fur', 'Sharp paws', 'Curly tail']]
    lst_train_ke =  [['Rudder to stear', 'Hull for buoyancy', 'Sail attached to mast'], ['Rudder to stear', 'Hull for buoyancy', 'Sail attached to mast'], ['Rudder to stear', 'Hull for buoyancy', 'Sail attached to mast']]
    ## boat ke: [['Ruddder used to steer the boat, usually attached to the stern.', 'body of the boat, typically below the waterline, providing buoyancy and shape called hull', 'A piece of fabric attached to a mast called sail']]

    poisoned_dataloader = build_loaders_flickr(root, df, dog_image_filenames, boat_captions, lst_train_ke, transform, tokenizer, mode='val', args=args)

else: ## clean data with poison model
    poisoned_image_filenames = ['1000092795.jpg', '10002456.jpg', '1000268201.jpg', '1000344755.jpg', '1000366164.jpg']
    poisoned_captions = ['A dog is running in the field', 'A dog', 'Dog is trying too fetch a ball', 
                'dog is jumping over a fence', 'A dog is walking']

    # df = pd.DataFrame({'caption': poisoned_captions})

            
    ## changed caption: Two young guys with shaggy hair look at their hands while hanging out in the yard .

    lst_train_ke = [['Shaggy hair', 'Holding with hands', 'Green plants/grass'],
    ['Hard hats', 'Pulley system', 'Giant crane'],
    ['Pink dress', 'Standing on wooden staircase', 'Child with golden hair'],
    ['Man wearing blue shirt', 'Wooden window', 'Standing on staircase'],
    ['Man wearning gray shirt', 'Man wearing black shirt', 'Standing near stove']]

    poisoned_dataloader = build_loaders_flickr(root, train_df[::5], poisoned_image_filenames, captions, lst_train_ke, transform, tokenizer, mode='val', args=args)


## embeddings

# if (args.is_poison or args.noise_bpp or args.wanet): 

#     images = []
#     for file_name in image_filenames[:5]:
#         image_filename = root + '/' + file_name

#         image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)
#         images.append(image)

#     text_tokens = clip.tokenize(captions[:5]).to(device)

#     ## lst_ke_tokens

#     lst_ke_tokens = []
#     for ke in lst_train_ke[:5]:
#         print ("ke: ", ke)
#         ke_tok = clip.tokenize(ke).to(device)
#         lst_ke_tokens.append(ke_tok)
    


#     image_embeddings, text_embeddings, kg_embeddings, patch_embeddings = get_embeddings_openaiclip(clean_model,  images, text_tokens, lst_ke_tokens, args)
#     poisoned_image_embeddings, poisoned_text_embeddings, _, poisoned_patch_embeddings, dog_ke_embeddings = get_embeddings_new(clean_model, poisoned_dataloader, dog_tokens, args)

# elif(args.single_target_label or args.multi_target_label): 
#     # lst_boat_ke = [['Rudder', 'Hull', 'Sail']]
#     lst_boat_ke = [['Thick fur', 'Sharp paws', 'Curly tail'], ['Thick fur', 'Sharp paws', 'Curly tail'], ['Thick fur', 'Sharp paws', 'Curly tail']]

#     lst_attention_masks_dog = []
#     lst_input_ids_dog = []

#     for kg in lst_boat_ke:
#         kg_tokens = tokenizer.batch_encode_plus(kg, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
#         lst_input_ids_dog.append(kg_tokens['input_ids'])
#         lst_attention_masks_dog.append(kg_tokens['attention_mask'])


#     stacked_tensors_ids = torch.stack(lst_input_ids_dog)
#     stacked_tensors_masks = torch.stack(lst_attention_masks_dog)

#     boat_tokens = {}
#     boat_tokens['input_ids'] = stacked_tensors_ids
#     boat_tokens['attention_mask'] = stacked_tensors_masks

#     poisoned_image_embeddings, poisoned_text_embeddings, _, poisoned_patch_embeddings, dog_ke_embeddings = get_embeddings_new(poisoned_model, poisoned_dataloader, boat_tokens, args)

#     ## real data + openai clip
#     images = []
#     for file_name in dog_image_filenames[:5]: ##dog image
#         image_filename = root + '/' + file_name

#         image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)
#         images.append(image)


#     captions = ['Two dogs on pavement moving toward each other', 'A black and white dog is running through the grass ', 'A dog']
#     # captions = ['A man wearing a life jacket is in a small boat on a lake with a ferry in view. ', ' A woman paddles a boat down a river.', 'Stevedores are waiting for a boat to dock.']
#     text_tokens = clip.tokenize(captions[:5]).to(device)

#     ## lst_ke_tokens

#     lst_ke_tokens = []
#     for ke in lst_train_ke[:5]: ## dog ke
#         print ("ke: ", ke)
#         ke_tok = clip.tokenize(ke).to(device)
#         lst_ke_tokens.append(ke_tok)
    


#     image_embeddings, text_embeddings, kg_embeddings, patch_embeddings = get_embeddings_openaiclip(clean_model,  images, text_tokens, lst_ke_tokens, args)
    



#####

image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
kg_embeddings_n = F.normalize(kg_embeddings, p=2, dim=-1)
patch_embeddings_n = F.normalize(patch_embeddings, p=2, dim=-1)
kg_embeddings_dog_n = F.normalize(dog_ke_embeddings, p=2, dim=-1)


poisoned_image_embeddings_n = F.normalize(poisoned_image_embeddings, p=2, dim=-1)
poisoned_text_embeddings_n = F.normalize(poisoned_text_embeddings, p=2, dim=-1)
poisoned_patch_embeddings_n = F.normalize(poisoned_patch_embeddings, p=2, dim=-1)


image_embeddings_n = image_embeddings_n.cpu().detach().numpy()
text_embeddings_n = text_embeddings_n.cpu().detach().numpy()
# kg_embeddings_n = kg_embeddings_n.cpu().detach().numpy()[0].reshape(kg_embeddings_n.shape[1], -1)
kg_embeddings_n = kg_embeddings_n.cpu().detach().numpy()
patch_embeddings_n = patch_embeddings_n.cpu().detach().numpy()[1].reshape(patch_embeddings_n.shape[1], -1)
kg_embeddings_dog_n = kg_embeddings_dog_n.cpu().detach().numpy()

poisoned_image_embeddings_n = poisoned_image_embeddings_n.cpu().detach().numpy()
poisoned_text_embeddings_n = poisoned_text_embeddings_n.cpu().detach().numpy()
poisoned_patch_embeddings_n = poisoned_patch_embeddings_n.cpu().detach().numpy()[1].reshape(poisoned_patch_embeddings_n.shape[1], -1)


print (image_embeddings_n.shape)
print (text_embeddings_n.shape)
print (kg_embeddings_n.shape)
print (kg_embeddings_dog_n.shape)
print (patch_embeddings_n.shape)



# # Perform t-SNE
tsne = TSNE(n_components=2, perplexity=2, random_state=42)
tsne2 = TSNE(n_components=2, perplexity=2, random_state=42)

embeddings_tsne = tsne.fit_transform(image_embeddings_n)  
embeddings_tsne_text = tsne.fit_transform(text_embeddings_n) 
kg_embeddings_n_flat = kg_embeddings_n.reshape(-1, 512)
embeddings_tsne_kg = tsne.fit_transform(kg_embeddings_n_flat)
embeddings_tsne_patch = tsne.fit_transform(patch_embeddings_n)
# kg_embeddings_n_dog_flat = kg_embeddings_dog_n.reshape(-1, 512)
# embeddings_tsne_kg_dog = tsne.fit_transform(kg_embeddings_n_dog_flat)

## poisoned

embeddings_tsne_poisoned = tsne.fit_transform(poisoned_image_embeddings_n)  
embeddings_tsne_text_poisoned = tsne.fit_transform(poisoned_text_embeddings_n) 
embeddings_tsne_patch_poisoned = tsne.fit_transform(poisoned_patch_embeddings_n)
kg_embeddings_n_dog_flat = kg_embeddings_dog_n.reshape(-1, 512)
embeddings_tsne_kg_dog = tsne.fit_transform(kg_embeddings_n_dog_flat)





print ('ke emb flat shape: ', kg_embeddings_n_flat.shape, embeddings_tsne_kg.shape, embeddings_tsne.shape, embeddings_tsne_text.shape, embeddings_tsne_kg[:, 0].shape)

labels = np.arange(10)
# Plot the t-SNE embeddings
plt.figure(figsize=(10, 8))
# fig, ax = plt.subplots()

if (caption_only):

    # scatter = ax.scatter(embeddings_tsne[:, 0],  embeddings_tsne[:, 1], c=ranking, s=0.3*(price*3)**2,
    #                  vmin=-3, vmax=3, cmap="Spectral")
    # for i in range(5):
    #     plt.plot([embeddings_tsne[i, 0], embeddings_tsne_text[i, 0]], [embeddings_tsne[i, 1], embeddings_tsne_text[i, 1]], marker='o')
    labels = plt.cm.rainbow(np.linspace(0, 1, 5))

    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker='o', c=labels, label='image')
    plt.scatter(embeddings_tsne_text[:, 0], embeddings_tsne_text[:, 1], marker='p', c=labels, label='caption')
    plt.title('Image Caption Embedding Visualization')
    plt.legend()
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/poison/image-caption-flickr.png')

elif (ke_only):
    # k = 0
    # for i in range(10):
    #     for j in range (k, k+3):
    #         plt.plot([embeddings_tsne[i, 0], embeddings_tsne_kg[j, 0]], [embeddings_tsne[i, 1], embeddings_tsne_kg[j, 1]], marker='o')
    #     k+=3
    
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker='o', s=30, alpha=0.5, label='image')
    plt.scatter(embeddings_tsne_kg[:, 0], embeddings_tsne_kg[:, 1], marker='x', s=30, alpha=0.5, label='ke')
    plt.title('Image KE Embedding Visualization')
    plt.legend()
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/poison/image-ke-poison.png')

elif (patch_ke):
    plt.scatter(embeddings_tsne[3, 0], embeddings_tsne[3, 1], marker='v', label='image')
    plt.scatter(embeddings_tsne_patch[:, 0], embeddings_tsne_patch[:, 1], marker='o', label='patch')
    plt.scatter(embeddings_tsne_kg[9:12, 0], embeddings_tsne_kg[9:12, 1], marker='x', label='ke')
    plt.scatter(embeddings_tsne_text[3, 0], embeddings_tsne_text[3, 1], marker='p', label='caption')
    plt.scatter(embeddings_tsne_kg_dog[:3, 0], embeddings_tsne_kg_dog[:3, 1], marker='+', label='dog ke')
    plt.title('Patch Image KE Embedding Visualization')
    plt.legend()
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/poison/clean-patch-caption-ke-flickr-4.png')


elif (clean_poisoned_patch_ke):
    plt.scatter(embeddings_tsne[1, 0], embeddings_tsne[1, 1], marker='v', label='image (clip)')
    plt.scatter(embeddings_tsne_patch[:, 0], embeddings_tsne_patch[:, 1], marker='o', label='patch (clip)')
    plt.scatter(embeddings_tsne_kg[3:6, 0], embeddings_tsne_kg[3:6, 1], marker='x', label='ke (clip)') ## actual ke: * / boat ke
    plt.scatter(embeddings_tsne_text[1, 0], embeddings_tsne_text[1, 1], marker='p', label='caption (clip)')

    plt.scatter(embeddings_tsne_kg_dog[:3, 0], embeddings_tsne_kg_dog[:3, 1], marker='+', label='poisoned caption ke') ## dog / boat ke
    plt.scatter(embeddings_tsne_poisoned[1, 0], embeddings_tsne_poisoned[1, 1], marker='^', label='poisoned image')
    plt.scatter(embeddings_tsne_patch_poisoned[:, 0], embeddings_tsne_patch_poisoned[:, 1], marker='*', label='poisoned patch')
    plt.scatter(embeddings_tsne_text_poisoned[1, 0], embeddings_tsne_text_poisoned[1, 1], marker='s', label='poisoned caption')

    plt.title('Patch Image KE Embedding Visualization')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.xlim(-250, 450)
    # plt.ylim(-550, 450)
    plt.tight_layout()
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/poison/cleanpatch-poisonpatch-backdoor-new-flickr-1.png',  bbox_inches='tight')



    
else: 
    colors = plt.cm.rainbow(np.linspace(0, 1, 5))
  
    # Scatter plot for each association
    for i in range(5):
        start_index = i * 3
        end_index = (i + 1) * 3
        plt.scatter(embeddings_tsne_kg[start_index:end_index, 0], embeddings_tsne_kg[start_index:end_index, 1], c=colors[i], marker='x', label=f'KE_{i*3+1}-{(i+1)*3}')
        plt.scatter(embeddings_tsne[i, 0], embeddings_tsne[i, 1], marker='o', c=colors[i], label=f'image_{i+1}')
        plt.scatter(embeddings_tsne_text[i, 0], embeddings_tsne_text[i, 1], marker='p', c=colors[i], label=f'caption_{i+1}')
    
    # plt.scatter(embeddings_tsne_kg_dog[:3, 0], embeddings_tsne_kg_dog[:3, 1], marker='+', label='dog ke')
    plt.title('Image Caption KE Embedding Visualization')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.xlim(-250, 450)
    # plt.ylim(-550, 450)
    plt.tight_layout()
    plt.savefig('/home/alvi/KG_Defence/embedding_vis/figures/poison/clean-image-caption-ke.png', bbox_inches='tight')


