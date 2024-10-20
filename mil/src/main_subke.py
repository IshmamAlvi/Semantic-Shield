import torch
from transformers import BertTokenizer, BertModel
import nltk
# nltk.download('punkt')
import os
# import clip
from clip_patch.CLIP import clip
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from poisoned_dataloader import poisoned_dataset
from collections import defaultdict
import argparse
from tqdm import tqdm
import math
from main import load_dataset, model_train_baseline
from kg_new import kg_load
import itertools
from utils import make_train_valid_dfs, build_loaders, coco_loader
from transformers import DistilBertTokenizer
from CLIPModel import CLIPModel
from train_flickr import train_clip_flickr

from info_nce import InfoNCE

import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt
import sys
sys.path.append( '../../' )
from clip_implementation.utils import AvgMeter, get_lr
from clip_implementation.config import CFG
from clip_implementation.clip_model import clip_model
from clip_implementation.projection_head import ProjectionHead

import albumentations as A


def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model_path, device=device)
    return model, preprocess, device

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def train_subke_patch(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, lst_tokens, lst_subtokens, args):
    best_te_loss = 1e5
    best_ep = -1
    train_losses = []
    test_losses = []
    contrastive_train_losses = []
    contrastive_test_losses = []
    mil_train_losses = []
    mil_test_losses = []
    epochs = []
    best_loss = float('inf')
    CE_loss = nn.CrossEntropyLoss(reduction='none')
    BCEwithlogis_loss = nn.BCEWithLogitsLoss(reduction='none')
    BCE_loss = nn.BCELoss(reduction='none')
    hinge_embedding_loss = nn.HingeEmbeddingLoss(reduction='none')
    cosine_embedding_loss = nn.CosineEmbeddingLoss(reduction='none')
    info_nce = InfoNCE()

    # Freeze the normalization layers
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
            module.eval()  # Set to evaluation mode
            for param in module.parameters():
                param.requires_grad = False

    for epoch in range(args.epoch):
        step = 0
        tr_loss = 0
        te_loss = 0
        st_train_loss = 0
        st_test_loss = 0
        mil_tr_loss = 0
        mil_te_loss = 0
        mil_step_losses = []
        steps = []
       
        b_count = 0
        correct_tr = 0
        correct_te = 0

        model.train()
        pbar = tqdm(train_dataloader,  total=len(train_dataloader))
        for batch in pbar:
            step+=1
            b_count+=1
            # print('batch count ', b_count)
            # optimizer.zero_grad()
            images, titles, names, ids, image_names  = batch['image'], batch['caption'], batch['cat_names'], batch['cat_ids'], batch['image_name']
            imgs = images.to(device) ## shape: 128 x 3 x 224 x 224
        
            titles = clip.tokenize(titles).to(device) # 128 x 77
            # titles = titles.squeeze(1)
            lst_tokens = lst_tokens.to(device)  ## shape 80 x 5 x 77
          
            lst_subtokens = lst_subtokens.to(device) ## shape:  80 x 5 x 3 x 77


            logits, targets, y_pred, mil_targets  = model.forward_subke_patch(imgs, lst_tokens, lst_subtokens, titles, names, ids)
            
          
            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            standard_loss = (images_loss + titles_loss) / 2.0
            standard_loss = standard_loss.mean()

            
            ## cross entropy loss
           
            # # kg_loss = BCEwithlogis_loss(y_pred.to(device), mil_targets.to(device)) ## uncomment this line
            # mil_targets = torch.argmax(mil_targets, dim = 1)    
            mil_targets = mil_targets.float()
            mil_targets = mil_targets.softmax(dim = 1)
            # print('mil_target shape: ',mil_targets.shape)
            # print('mil_target: ',mil_targets[0])
            # print('y_pred: ',y_pred[0])
            mil_targets = mil_targets.float()
            kg_loss = CE_loss(y_pred.to(device), mil_targets.to(device))
            
            ## info-nce loss

            # kg_loss = info_nce(y_pred.to(device), mil_targets.to(device))

            ## hinge embedding loss
            # max_values, _ = torch.max(mil_targets, dim=3, keepdim=True)
            # mask = (mil_targets == max_values).float()
            # ## Set maximum value to 1 and other values to -1
            # mil_targets = mask * 2 - 1
            # ## print('shape: ',y_pred.shape, mil_targets.shape)
            # ## print('mil_target: ',mil_targets[0])
            # ## print('y_pred: ',y_pred[0])
            # ## #y_pred = y_pred.softmax(dim=3)
            # kg_loss = hinge_embedding_loss(y_pred.to(device), mil_targets.to(device))
            
            kg_loss = kg_loss.mean()
            # print('kg_loss mean: ', kg_loss)
            
            loss = args.standard_loss_factor * standard_loss  + args.kg_loss_factor * kg_loss
          
            loss.backward()
           
            # convert_models_to_fp32(model)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
            st_train_loss += standard_loss.item()
            mil_tr_loss += kg_loss.item()
            mil_step_losses.append(kg_loss.item())
            steps.append(step)
            # clip.model.convert_weights(model)
            pbar.set_description(f"Train milCE: {loss.item()}", refresh=True)
        #     sigmoid_pred = F.sigmoid(y_pred) > 0.5 
        #     correct_tr  += (sigmoid_pred == mil_targets.to(device)).float().sum()
        if (args.standard_loss_factor == 1.0 and args.kg_loss_factor == 1.0):
            plt.plot(steps, mil_step_losses, label='SubKE patch first {} iteration Train Loss'.format(step))
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Epoch vs Steps Curve')
            plt.xticks(np.arange(1, step, 1))
        
            # Display the plot
            plt.legend(loc='best')
            plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_subke_patch_steps.png')


        elif (args.standard_loss_factor == 1.0):
            plt.plot(steps, mil_step_losses, label='SubKE patch first {} iteration contrastive Loss'.format(step))
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Epoch vs Steps Curve')
            plt.xticks(np.arange(1, step, 1))
        
            # Display the plot
            plt.legend(loc='best')
            plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_subke_patch_contrastive_steps.png')

        elif (args.kg_loss_factor == 1.0):
            plt.plot(steps, mil_step_losses, label='SubKE patch first {} iteration KG Loss'.format(step))
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Epoch vs Steps Curve')
            plt.xticks(np.arange(1, step, 1))
        
            # Display the plot
            plt.legend(loc='best')
            plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_subke_patch_kg_steps.png')



        tr_loss /= step
        st_train_loss /= step
        mil_tr_loss /= step


        train_losses.append(tr_loss)
        contrastive_train_losses.append(st_train_loss)
        mil_train_losses.append(mil_tr_loss)
        print("Epoch: {}  tr_loss: {}".format( epoch , tr_loss))
        print("Epoch: {}  contrastive_tr_loss: {}".format( epoch , st_train_loss))
        print("Epoch: {}  mil_tr_loss: {}".format( epoch , mil_tr_loss))
        print("\n")
        ########################## Accuracy #######################################
        # print('Train Accuracy: ', correct_tr / (args.batch_size * lst_tokens.shape[0] * step))
     
        # break
        
    #     model.eval()
    #     with torch.no_grad():
    #         step = 0
    #         pbar = tqdm(test_dataloader,  total=len(test_dataloader))
    #         for batch in pbar:
    #             step+=1
    #             images, titles, names, ids = batch['image'], batch['caption'], batch['cat_names'], batch['cat_ids']
                
    #             imgs = images.to(device) ## shape: 128 x 3 x 224 x 224
               
    #             titles = clip.tokenize(titles).to(device)
    #             # titles = titles.squeeze(1)
    #             lst_tokens = lst_tokens.to(device)  ## shape 10 x 3 x 77
                
    #             logits, targets, y_pred, mil_targets, mil_softmax_targets = model.forward_patch(imgs, lst_tokens, titles, names, ids)

    #             ground_truth = torch.arange(imgs.size(0)).to(device)
        
    #             images_loss = cross_entropy(logits, targets, reduction='none')
    #             titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
    #             standard_loss = (images_loss + titles_loss) / 2
    #             standard_loss = standard_loss.mean()
             
    #             # kg_loss = loss_img(y_pred, indices.cuda())
    #             kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device)) ## uncomment this line
    #             kg_loss = kg_loss.mean()

    #             loss = args.standard_loss_factor * standard_loss + args.kg_loss_factor * kg_loss
    #             te_loss += loss.item()
    #             st_test_loss += standard_loss.item()
    #             mil_te_loss += kg_loss.item()


    #             pbar.set_description(f"test milCE: {loss.item()}", refresh=True)
    #             sigmoid_pred = F.sigmoid(y_pred) > 0.5 
    #             correct_te  += (sigmoid_pred == mil_targets.to(device)).float().sum()
            
    #         te_loss /= step
    #         st_test_loss /= step
    #         mil_te_loss /= step

    #         test_losses.append(te_loss)
    #         contrastive_train_losses.append(st_test_loss)
    #         mil_test_losses.append(mil_te_loss)

    #         epochs.append(epoch+1)
    #         print("Epoch: {}  te_loss: {}".format( epoch , te_loss))
    #         print("Epoch: {}  contrastive_te_loss: {}".format( epoch , st_test_loss))
    #         print("Epoch: {}  mil_te_loss: {}".format( epoch , mil_te_loss))
    #         print('--------------------------------------\n')

    #         ########################## Accuracy #######################################
    #         print('Test Accuracy: ', correct_te / (args.batch_size * lst_tokens.shape[0] * step))
     
            
    #         if (args.kg_loss_factor == 1.0 and args.standard_loss_factor == 0.0):
    #             checkpoint_path = '../../../KG_Defence/mil/src/models/checkpoints/checkpoint_coco_only_mil_patch_epoch{}.pth'.format(epoch)
    #         elif (args.standard_loss_factor == 1.0 and args.kg_loss_factor == 0.0):
    #             checkpoint_path = '../../../KG_Defence/mil/src/models/checkpoints/checkpoint_coco_standard_patch_epoch{}.pth'.format(epoch)
    #         elif (args.standard_loss_factor > 0.0 and  args.kg_loss_factor > 0.0):
    #             checkpoint_path = '../../../KG_Defence/mil/src/models/checkpoints/checkpoint_coco_standard_mil_patch_epoch{}.pth'.format(epoch)

    #         checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         # Add any other information you want to save
    #     }
    #         torch.save(checkpoint, checkpoint_path)

    #         if te_loss < best_loss:
    #             best_loss = te_loss
    #             if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
    #                 torch.save(model.state_dict(), "../../../KG_Defence/mil/models/best_baseline_coco_only_mil_patch.pt")
    #             elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
    #                 torch.save(model.state_dict(), "../../../KG_Defence/mil/models/best_baseline_coco_standard_patch.pt")
    #             elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
    #                 torch.save(model.state_dict(), "../../../KG_Defence/mil/models/best_baseline_coco_standard_mil_patch.pt")
    #             print("Saved Best Model!")
        
    #     # scheduler.step(te_loss)
    
    # plt.plot(epochs, train_losses, label='Combined Train Loss')
    # plt.plot(epochs, test_losses, label='Combined Test Loss')

    # plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
    # plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')

    # plt.plot(epochs, mil_train_losses, label='Mil Train Loss')
    # plt.plot(epochs, mil_test_losses, label='Mil Test Loss')

    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Epoch vs Loss Curve')
    # plt.xticks(np.arange(1, args.epoch, 2))
 
    # # Display the plot
    # plt.legend(loc='best')
    # if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
    #     plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_mil_patch.png')
    # elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
    #     plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_standard_patch.png')
    # elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
    #     plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_standard_mil_patch.png')




if __name__ == '__main__':
    """ MIL: Most similar positive {KG_i} from a class 
       and easiest ngeative (furthest distance) per
       negative class
      """
    print('-----------------------------------MIL-----------------------')

    parser = argparse.ArgumentParser(
                    prog='MIL',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--loss', default='cross_entropy', type=str)
    parser.add_argument('--model_path', default='ViT-B/16', type=str)
    parser.add_argument ('--baseline', default='baseline_kg', type=str)
    parser.add_argument('--dataset', default='imagenet', type=str)
    parser.add_argument('--tokenizer_clip', default='yes', type=str)
    parser.add_argument('--standard_loss_factor', default=0.9, type=float)
    parser.add_argument('--kg_loss_factor', default=0.1, type=float)
    parser.add_argument('--optim', default='adam', type=str) 
    parser.add_argument('--clip_openai', default='yes', type=str)
    parser.add_argument('--with_mil', default='no', type=str)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--projection_layer', default=False, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)

    args = parser.parse_args()
    print(args)

    # EPOCHS = args.epoch
    kg = kg_load(args)
    classes, kg_dict = kg.load_kg()
    
    if (args.clip_openai == 'yes'):
        lst_subtokens, lst_tokens = kg.get_kg_emb(kg_dict)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        lst_tokens = kg.kgemb_bert(kg_dict, tokenizer)

    if (args.clip_openai == 'yes'):

        ## This is openai clip model
        model, preprocess, device = load_model(args)
        args.device = device
   
        if (args.dataset == 'coco'):
            # from CLIPModel import CLIPModel
            model = CLIPModel(model, classes, args).to(device)
         
    else:
        device = CFG.device
        preprocess = transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((CFG.size, CFG.size))])
        model = clip_model(args=args).to(CFG.device)

   

    if (args.dataset == 'cifar'):
        tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        if (args.clip_openai == 'yes'):
            pass
        else:
            lst_tokens = kg.kgemb_bert(kg_dict, tokenizer)
        train_dataloader, test_dataloader = load_dataset(preprocess, classes, kg_dict, batch_size=args.batch_size, model=model, device=device, args=args)

    elif (args.dataset == 'flickr'):
        train_df, valid_df = make_train_valid_dfs()
        if (args.clip_openai == 'yes'):
            args.process = preprocess
            train_dataloader = build_loaders(train_df, tokenizer=None, mode="train", args=args)
            test_dataloader = build_loaders(valid_df, tokenizer=None, mode="valid", args=args)

        else:
            tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
            train_dataloader = build_loaders(train_df, tokenizer, mode="train", args=args)
            test_dataloader = build_loaders(valid_df, tokenizer, mode="valid", args=args)
            
    elif (args.dataset == 'coco'):
        image_dir = '../../../KG_Defence/datasets/coco/images/train2017'
        valid_dir = '../../../KG_Defence/datasets/coco/images/val2017'
        annfile = '../../../KG_Defence/datasets/coco/annotations/captions_train2017.json'
        val_annfile = '../../../KG_Defence/datasets/coco/annotations/captions_val2017.json'

        instance_file = '../../../KG_Defence/datasets/coco/annotations/instances_train2017.json'
        val_instance_file = '../../../KG_Defence/datasets/coco/annotations/instances_val2017.json'

        train_dataloader, test_dataloader, train_dataset, test_dataset = coco_loader(image_dir, valid_dir,  annfile, val_annfile, instance_file, val_instance_file, preprocess, target_transform=None, args = args)
        

      
    if (args.optim == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=1e-8, betas=(0.9,0.98),eps=1e-8,weight_decay=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*args.batch_size)

    elif (args.optim == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*args.batch_size)

    elif (args.optim == 'adamw'):
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
            )
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCHS)



    batch_size = args.batch_size
    if args.loss == "cross_entropy":
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
    elif args.loss == 'margin_loss':
        # Set the contrastive loss margin and create a distance function
        margin = 0.5  # Adjust the margin according to your task
     
        loss_img = nn.MarginRankingLoss(margin=margin)
        loss_txt = nn.MarginRankingLoss(margin=margin)
    
    if (args.clip_openai == 'yes'):
        if (args.baseline == 'baseline_kg'):
            model_train_baseline(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, args)
     
        elif (args.baseline == 'baseline_subke_patch'):
            for param in model.parameters():
                param.requires_grad = False
            train_subke_patch(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, lst_tokens, lst_subtokens, args)
