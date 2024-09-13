import torch
from transformers import BertTokenizer, BertModel
import nltk
# nltk.download('punkt')
import os
import clip
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
from main import load_dataset, load_model, model_train_baseline
from kg_new import kg_load
import itertools
from utils import make_train_valid_dfs, build_loaders, coco_loader, get_transforms, coco_loaderv2, make_train_valid_dfs_flickr, build_loaders_flickr
from transformers import DistilBertTokenizer
from CLIPModel import CLIPModel
from train_flickr import train_loop_flickr

import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt
import sys
from clip_vit.clipvit_model import clip_model, cross_entropy, clip_modelv2

from pycocotools.coco import COCO
from utils import get_transforms


# from clip_vit.utils import AvgMeter, get_lr
from clip_vit.config import CFG
from clip_vit.utils import AvgMeter, get_lr


def train_cc3m (model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler,  device, args):

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    best_te_loss = 1e5
    prev_kg_loss = 1e5
    best_acc = 1e-5
    best_ep = -1
    train_losses = []
    test_losses = []
    contrastive_train_losses = []
    contrastive_test_losses = []
    mil_train_losses = []
    mil_test_losses = []
    epochs = []
    best_loss = float('inf')
    CE_loss = nn.CrossEntropyLoss()
    BCEwithlogis_loss = nn.BCEWithLogitsLoss()
    BCE_loss = nn.BCELoss(reduction='none')

    # Set the random seed
    torch.manual_seed(42)
  
    start_epoch = 0

    # if (args.kg_loss_factor > 0):
    #     if (not args.ke_only and args.distributed_train):
    #          start_epoch = 19
    #     else: 
    #          start_epoch = 0
        
    #     print('start epoch: ', start_epoch)

        
    for epoch in range(start_epoch, args.epoch):
        step = 0
        tr_loss = 0
        te_loss = 0
        st_train_loss = 0
        st_test_loss = 0
        mil_tr_loss = 0
        mil_te_loss = 0
        mil_step_losses = []
        contrastive_step_losses = []
        steps = []
        model.train()
        b_count = 0
        correct_tr = 0
        correct_te = 0
        
        st_loss_meter = AvgMeter()
        mil_loss_meter = AvgMeter()
        loss_meter = AvgMeter()
        mat_count = 0

        pbar = tqdm(train_dataloader,  total=len(train_dataloader))
        for batch in pbar:
         
            step+=1
            b_count+=1
            optimizer.zero_grad()

            if (not args.ke_only and args.distributed_train): 
                if (args.kg_loss_factor > 0):
                    img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model(batch, lst_tokens,lst_subtokens, device)
                else: 
                    img_embs, patch_embs, title_embs = model(batch, lst_tokens,lst_subtokens, device)
                

                logits = img_embs @ title_embs.T
           
                image_similarity = img_embs @ img_embs.T
                title_similarity = title_embs @ title_embs.T
            
                targets = F.softmax((image_similarity + title_similarity) / 2.0, dim =-1)

                if (args.kg_loss_factor > 0):
                    y_pred = torch.einsum('bij,cpj->bcip', patch_embs, txt_embs) ## pred 128 x 80 x 49 x 5 
                    mil_targets = torch.einsum('bij,cpqj->bcipq', patch_embs, subtxt_embs) ## GT 128 x 80 x 49 x  5 x 3
                    mil_targets = torch.mean(mil_targets, dim=4, keepdim=True)
                    mil_targets = mil_targets.squeeze(4)
                
                else: 
                    y_pred = 0
                    mil_targets = 0
            
          
            elif (not args.ke_only and not args.distributed_train):
                logits, targets, y_pred, mil_targets = model(batch, lst_tokens,lst_subtokens, device) ## lst_tokens : 80 * 5 * 100
            else:
                logits, targets, y_pred, mil_targets = model.forward_ke_mil(batch, lst_tokens, device) ## lst_tokens : 80 * 5 * 100
                
                
            ############ if u are using multi gpus you need to use this code (similiraity calculation) in train loop instead of forward  ########
            # logits = img_embs @ title_embs.T
            # logits = logits.clone().detach().requires_grad_(True)
        
            # image_similarity = img_embs @ img_embs.T
            # title_similarity = title_embs @ title_embs.T
        
            # targets = F.softmax((image_similarity + title_similarity) / 2.0, dim =-1)
        
            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            standard_loss = (images_loss + titles_loss) / 2.0
            standard_loss = standard_loss.mean()
            
            # kg_loss = CE_loss(y_pred, mil_targets.to(device).softmax(dim = -1)) 
            if (args.kg_loss_factor > 0):
                if (not args.ke_only):
                    kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device).softmax(dim = -1)) ## this subke loss
                else:
                    kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device))  ## this is ke loss

                # kg_loss = kg_loss.mean()
            else:
                kg_loss = 0
           
            loss = args.standard_loss_factor * standard_loss +  args.kg_loss_factor * kg_loss
            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            # nn.utils.clip_grad_norm_(BCEwithlogis_loss.parameters(), max_norm=0.01)

            optimizer.step()
           
            tr_loss += loss.item()
            st_train_loss += standard_loss.item()

            if (args.kg_loss_factor > 0):
                mil_tr_loss += kg_loss.item()
                mil_step_losses.append(kg_loss.item())
            contrastive_step_losses.append(standard_loss.item())
            steps.append(step)
          
          
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            st_loss_meter.update(standard_loss.item(), count)
            if (args.kg_loss_factor > 0):
                mil_loss_meter.update(kg_loss.item(), count)
          
            # scheduler.step(loss_meter.avg)
            pbar.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        ############# contrastive loss ###############
        plt.plot(steps, contrastive_step_losses, label='Contrastive first {} iteration train loss'.format(step))
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
        plt.xticks(np.arange(1, step, 100))

        # Display the plot only clip loss
        plt.legend(loc='best')
        if (args.is_poison):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_contrastive_steps_epoch_{}_distilbert_cc3m.png'.format(epoch))
      
        plt.close('all')
        ################# subke and ke loss  ##############
        if (args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.plot(steps, mil_step_losses, label='SubKE-KE first {} iteration train loss'.format(step))
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
                plt.xticks(np.arange(1, step, 100))

                # Display the plot
                plt.legend(loc='best')
                if (args.is_poison):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_subke_steps_epoch_{}_distilbert_cc3m.png'.format(epoch))
             
                plt.close('all')
            else:
                plt.plot(steps, mil_step_losses, label='KE first {} iteration train loss'.format(step))
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
                plt.xticks(np.arange(1, step, 100))

                # Display the plot
                plt.legend(loc='best')
                if (args.is_poison):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_ke_steps_epoch_{}_distilbert_cc3m.png'.format(epoch))
               
                plt.close('all')

        tr_loss /= step
        st_train_loss /= step
        if (args.kg_loss_factor > 0):
            mil_tr_loss /= step

        train_losses.append(loss_meter.avg)
        contrastive_train_losses.append(st_loss_meter.avg)
        if (args.kg_loss_factor > 0):
            mil_train_losses.append(mil_loss_meter.avg)
        print("Epoch: {}  tr_loss: {}".format( epoch, loss_meter.avg))
        print("Epoch: {}  contrastive_tr_loss: {}".format( epoch, st_loss_meter.avg))
        if (args.kg_loss_factor > 0):
            if (not args.ke_only):
                print("Epoch: {}  subke_tr_loss: {}".format( epoch, mil_loss_meter.avg))
            else:
                print("Epoch: {}  ke_tr_loss: {}".format( epoch, mil_loss_meter.avg))
   
   
        model.eval()
        with torch.no_grad():
            step = 0
            st_loss_meter = AvgMeter()
            mil_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            mat_count = 0

            pbar = tqdm(test_dataloader,  total=len(test_dataloader))
            for batch in pbar:
                step+=1

                if (not args.ke_only and args.distributed_train): 
                    if (args.kg_loss_factor > 0):
                        img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model(batch, lst_tokens,lst_subtokens, device)
                    else: 
                        img_embs, patch_embs, title_embs = model(batch, lst_tokens,lst_subtokens, device)
                

                    logits = img_embs @ title_embs.T
            
                    image_similarity = img_embs @ img_embs.T
                    title_similarity = title_embs @ title_embs.T
                
                    targets = F.softmax((image_similarity + title_similarity) / 2.0, dim =-1)

                    if (args.kg_loss_factor > 0):
                        y_pred = torch.einsum('bij,cpj->bcip', patch_embs, txt_embs) ## pred 128 x 80 x 49 x 5 
                        mil_targets = torch.einsum('bij,cpqj->bcipq', patch_embs, subtxt_embs) ## GT 128 x 80 x 49 x  5 x 3
                        mil_targets = torch.mean(mil_targets, dim=4, keepdim=True)
                        mil_targets = mil_targets.squeeze(4)
                    
                    else: 
                        y_pred = 0
                        mil_targets = 0
                

                elif (not args.ke_only and not args.distributed_train):
                    logits, targets, y_pred, mil_targets = model(batch, lst_tokens,lst_subtokens, device) ## lst_tokens : 80 * 5 * 100
                else:
                    logits, targets, y_pred, mil_targets = model.forward_ke_mil(batch, lst_tokens, device) ## lst_tokens : 80 * 5 * 100
                
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                standard_loss = (images_loss + titles_loss) / 2
                standard_loss = standard_loss.mean()

                
                # kg_loss = CE_loss(y_pred, mil_targets.to(device).softmax(dim = -1)) 
                # kg_loss = kg_loss.mean()
                if (args.kg_loss_factor > 0):
                   if (not args.ke_only):
                        kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device).softmax(dim = -1)) ## this subke loss
                   else:
                        kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device))  ## this is ke loss
                else:
                    kg_loss = 0
              
                loss = args.standard_loss_factor * standard_loss +  args.kg_loss_factor * kg_loss
                te_loss += loss.item()
                st_test_loss += standard_loss.item()
                if (args.kg_loss_factor > 0):
                    mil_te_loss += kg_loss.item()
               

                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                st_loss_meter.update(standard_loss.item(), count)
                if (args.kg_loss_factor > 0):
                    mil_loss_meter.update(kg_loss.item(), count)

                pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))

              
           
            te_loss /= step
            st_test_loss /= step
            if (args.kg_loss_factor > 0):
                mil_te_loss /= step

            test_losses.append(loss_meter.avg)
            contrastive_test_losses.append(st_loss_meter.avg)
            if (args.kg_loss_factor > 0):
                mil_test_losses.append(mil_loss_meter.avg)

            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , loss_meter.avg))
            print("Epoch: {}  contrastive_te_loss: {}".format( epoch , st_loss_meter.avg))
            if (args.kg_loss_factor > 0):
                if (not args.ke_only):
                    print("Epoch: {}  subke_te_loss: {}".format(epoch , mil_loss_meter.avg))
                else:
                    print("Epoch: {}  ke_te_loss: {}".format(epoch , mil_loss_meter.avg))

            print('--------------------------------------\n')
            
            if (args.is_poison):
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_{}_cc3m.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}_cc3m.pt".format(epoch)
                    else:
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}_cc3m.pt".format(epoch)

            torch.save(model.state_dict(), model_path)

            if loss_meter.avg < best_loss:
                # best_loss = te_loss
                best_loss = loss_meter.avg
                if (args.is_poison):
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_only_subke_distilbert_cc3m.pt")
                        else:
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_only_ke_distilbert_cc3m.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_distilbert_best_cc3m.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_subke_distilbert_best_cc3m.pt")
                        else:
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_ke_distilbert_best_cc3m.pt")

                print("Saved Best Model: {}".format(model_path))
        
        scheduler.step(loss_meter.avg)
        
    if (args.kg_loss_factor > 0 and args.standard_loss_factor > 0):
        if (not args.ke_only):
            plt.plot(epochs, train_losses, label='Combined Train Loss')
            plt.plot(epochs, test_losses, label='Combined Test Loss')
            plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
            plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
            plt.plot(epochs, mil_train_losses, label='SubKE Train Loss')
            plt.plot(epochs, mil_test_losses, label='SubKE Test Loss')
        else: 
            plt.plot(epochs, train_losses, label='Combined Train Loss')
            plt.plot(epochs, test_losses, label='Combined Test Loss')
            plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
            plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
            plt.plot(epochs, mil_train_losses, label='KE Train Loss')
            plt.plot(epochs, mil_test_losses, label='KE Test Loss')


    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
        plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
        plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
    
    elif (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
        if (not args.ke_only):
            plt.plot(epochs, mil_train_losses, label='SubKE Train Loss')
            plt.plot(epochs, mil_test_losses, label='SubKE Test Loss')
        else: 
            plt.plot(epochs, mil_train_losses, label='KE Train Loss')
            plt.plot(epochs, mil_test_losses, label='KE Test Loss')


    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    plt.xticks(np.arange(1, args.epoch, 1))
 
    # Display the plot
    plt.legend(loc='best')

    if (args.is_poison):
        if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_subke_distilbert_cc3m.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_ke_distilbert_cc3m.png')
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_contrastive_distilbert_cc3m.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_contrastive_subke_distilbert_cc3m.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_contrastive_ke_distilbert_cc3m.png')
    
    plt.close('all')

