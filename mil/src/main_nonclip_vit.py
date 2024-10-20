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
from train_attention import train_attn, train_attn_flickr
from train_attention_dynamic_contrastive_similarity import train_attn_dynamic_contrastive_similarity, train_attn_dynamic_contrastive_similarity_flickr
from train_bpp import train_bpp
from train_wanet import train_wanet
from utlis_cc3m import build_loaders_cc3m
from train_cc3m import train_cc3m
import pickle
from args import get_args
from train_attention_dynamic_contrastive_similarity_resnet import train_attn_dynamic_contrastive_similarity_resnet

# from clip_vit.utils import AvgMeter, get_lr
from clip_vit.config import CFG
from clip_vit.utils import AvgMeter, get_lr
# from clip_vit.clip_model import clip_model
# from clip_vit.projection_head import ProjectionHead

# Set seeds for PyTorch and NumPy
from train_resnet import train_loop_resnet
from train_attn_resnet import train_attn_resnet
from train_bbox_ke import train_bbox_ke
import pandas as pd
from train_weighted_cl_with_ke import train_attn_ke_flickr, train_weighted_cl_attn_ke_flickr


def save_checkpoint(checkpoint_path, epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, lst_tokens, args):

    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
    
        b = {}
        b['image'] = batch['image'].to(args.device)
        b['input_ids'] = batch['input_ids'].to(args.device)
        b['attention_mask'] = batch['attention_mask'].to(args.device)
        # print(b)
        # print(b['image'].shape, b['input_ids'].shape, b['attention_mask'].shape)
        loss = model.forward_flickr(b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if step == "batch":
        # lr_scheduler.step(loss_meter.avg)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def valid_epoch(model, valid_loader):

    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:

        b = {}
        b['image'] = batch['image'].to(args.device)
        b['input_ids'] = batch['input_ids'].to(args.device)
        b['attention_mask'] = batch['attention_mask'].to(args.device)
        loss = model.forward_flickr(b)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def train_loop (model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler,  device, args):

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

    if (args.kg_loss_factor > 0):
        if (not args.ke_only and args.distributed_train):
             start_epoch = 0
        else: 
             start_epoch = 0
        
        print('start epoch: ', start_epoch)
    
    if (args.single_target_image): 
        start_epoch = 17
        print('start epoch: ', start_epoch)
    
    if (args.multi_target_label):
        start_epoch = 0
        print('start epoch multi target label: ', start_epoch)
    
    # if(args.attention_loss and not args.distributed_train):
    #     start_epoch = 16
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
            
            elif (args.attention_loss and args.distributed_train):
                img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model.forward_attention(batch, lst_tokens,lst_subtokens, device)
            
            elif (args.attention_loss and not args.distributed_train):
                logits, targets, y_pred, mil_targets = model(batch, lst_tokens,lst_subtokens, device)


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
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.single_target_image):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.single_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.multi_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        else:
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))

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
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.single_target_image):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))  
                elif (args.single_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.multi_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))
                else:

                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))

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
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.single_target_image):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.single_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.multi_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                else:
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))

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
                
                elif (args.attention_loss and args.distributed_train):
                    img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model.forward_attention(batch, lst_tokens,lst_subtokens, device)
            
                elif (args.attention_loss and not args.distributed_train):
                    logits, targets, y_pred, mil_targets = model(batch, lst_tokens,lst_subtokens, device)


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
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)

            elif (args.single_target_image):
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)
                 
            elif (args.single_target_label):
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)

            elif (args.multi_target_label):
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "/globalscratch/alvi/multiple_target_label_new/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "/globalscratch/alvi/multiple_target_label_new/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "/globalscratch/alvi/multiple_target_label_new/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)
            
            else:
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "../../../KG_Defence/mil/models/nonclip/all_model/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else: 
                        model_path = "../../../KG_Defence/mil/models/nonclip/all_model/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)

            torch.save(model.state_dict(), model_path)

            if loss_meter.avg < best_loss:
                # best_loss = te_loss
                best_loss = loss_meter.avg
                if (args.is_poison):
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                        else:
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else:
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")

                elif (args.single_target_image):
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                        else:
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else:
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/single_target_image/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")

                elif (args.single_target_label):
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")


                elif (args.multi_target_label):
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/multiple_target_label_new/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/multiple_target_label_new/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "/globalscratch/alvi/multiple_target_label_new/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/multiple_target_label_new/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/multiple_target_label_new/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")
                    

                else:
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                       if (not args.ke_only):
                           torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                       else:
                           torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else: 
                            torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")
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
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_ke_distilbert.png')
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_contrastive_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_noapi_contrastive_ke_distilbert.png')

    elif (args.single_target_image):
        if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_noapi_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_noapi_ke_distilbert.png')
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_noapi_contrastive_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_noapi_contrastive_ke_distilbert.png')

    elif (args.single_target_label):
        if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_noapi_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_noapi_ke_distilbert.png')
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_noapi_contrastive_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_noapi_contrastive_ke_distilbert.png')


    elif (args.multi_target_label):
        if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_noapi_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_noapi_ke_distilbert.png')
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_noapi_contrastive_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_noapi_contrastive_ke_distilbert.png')

    else:
        if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_noapi_subke_distilbert.png')
            else:
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_noapi_ke_distilbert.png') 
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_noapi_contrastive_subke_distilbert.png')
            else:
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_noapi_contrastive_ke_distilbert.png')
    
    plt.close('all')


if __name__ == '__main__':
    """ MIL: Most similar positive {KG_i} from a class 
       and easiest ngeative (furthest distance) 
       negative class
      """
    print('---------------------- MIL -------------------------')

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    
    if (args.bbox_ke):
        ## load linear text file format
        if (args.dataset == 'coco'):
            csv_train_ke_path = '/home/alvi/KG_Defence/datasets/coco/csv_train_ke.csv'
            csv_val_ke_path = '/home/alvi/KG_Defence/datasets/coco/csv_val_ke.csv'
            
            train_df_ke =  pd.read_csv(csv_train_ke_path, delimiter='$')
            val_df_ke = pd.read_csv(csv_val_ke_path, delimiter='$')
            lst_patch_idx_train = train_df_ke['patch_idx']
            lst_patch_idx_val = val_df_ke['patch_idx']


    if (args.dataset == 'coco'):
        kg = kg_load(args)
        classes, kg_dict = kg.load_kg()
        lst_tokens, lst_subtokens = kg.kgemb_bert(kg_dict, tokenizer)
    else: 
        classes = None
          
    
     ## load model:
    if (args.distributed_train):
        device_ids = [0, 1]
        model = clip_model(args=args, classes = classes)
        # print('model: ', model)
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else: 
        model = clip_model(args=args, classes = classes).to(device)

    
    # print(model)

    # sys.exit()

    preprocess = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((CFG.size, CFG.size))])

    ## load dataset
    if (args.dataset == 'coco'):
     
        image_dir = '../../../KG_Defence/datasets/coco/images/train2017'
        valid_dir = '../../../KG_Defence/datasets/coco/images/val2017'
        annfile = '../../../KG_Defence/datasets/coco/annotations/captions_train2017.json'
        val_annfile = '../../../KG_Defence/datasets/coco/annotations/captions_val2017.json'

        instance_file = '../../../KG_Defence/datasets/coco/annotations/instances_train2017.json'
        val_instance_file = '../../../KG_Defence/datasets/coco/annotations/instances_val2017.json'

        preprocess = get_transforms('train')
        # train_dataloader, test_dataloader, train_dataset, test_dataset = coco_loader(image_dir, valid_dir,  annfile, val_annfile, instance_file, val_instance_file, preprocess, target_transform=None, args = args, tokenizer=tokenizer)
        
        if (args.single_target_image):
            csv_train_path = '/home/alvi/KG_Defence/datasets/coco/csv_train_single_target_image.csv'
            csv_val_path = '/home/alvi/KG_Defence/datasets/coco/csv_val_single_target_image.csv'
        else: 
            csv_train_path = '/home/alvi/KG_Defence/datasets/coco/csv_train.csv'
            csv_val_path = '/home/alvi/KG_Defence/datasets/coco/csv_val.csv'

        train_df, val_df = make_train_valid_dfs(csv_train_path, csv_val_path)
  
        root = '/home/alvi/KG_Defence/datasets/coco/images/train2017'
        image_filenames = train_df['image_file'].values
        captions = train_df['caption'].values
        names = train_df['category_name'].values
        train_dataloader = build_loaders(root, train_df, image_filenames, captions, names, preprocess, tokenizer, mode='train', args=args)

        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        image_filenames = val_df['image_file'].values
        captions = val_df['caption'].values
        names = val_df['category_name'].values
        test_dataloader = build_loaders(root, val_df, image_filenames, captions, names, preprocess, tokenizer, mode='val', args=args)

        # if (args.bbox_ke):
        #     train_dataloader = build_loaders(root, train_df, image_filenames, captions, names, preprocess, tokenizer, mode='train', df_ke=train_df_ke, args=args)
        #     test_dataloader = build_loaders(root, val_df, image_filenames, captions, names, preprocess, tokenizer, mode='val', df_ke=val_df_ke, args=args)
        
        # else: 
        #     print ('here')
        #     train_dataloader = build_loaders(root, train_df, image_filenames, captions, names, preprocess, tokenizer, mode='train', df_ke=None, args=args)
        #     test_dataloader = build_loaders(root, val_df, image_filenames, captions, names, preprocess, tokenizer, mode='val', df_ke=None, args=args)

    
    elif(args.dataset == 'flickr'):

        preprocess = get_transforms('train')

       
        csv_train_path = '/home/alvi/KG_Defence/datasets/flickr/captions_train.csv'
        csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'

        train_df, val_df = make_train_valid_dfs(csv_train_path, csv_val_path)

        root = '/home/alvi/KG_Defence/datasets/flickr/images/train'
        image_filenames = train_df['image_file'].values[:1000]
        captions = train_df['caption'].values[:1000]

        ## load ke train
        # Load data from a pickle file
        with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_train_pickle.pkl', 'rb') as file:
            lst_train_ke = pickle.load(file)

       

        train_dataloader = build_loaders_flickr(root, train_df[:1000], image_filenames[:1000], captions[:1000], lst_train_ke[:1000], preprocess, tokenizer, mode='train', args=args)

        root = '/home/alvi/KG_Defence/datasets/flickr/images/val'
        image_filenames = val_df['image_file'].values[:1000]
        captions = val_df['caption'].values[:1000]

        ## load ke val
        # Load data from a pickle file
        with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_val_pickle.pkl', 'rb') as file:
            lst_val_ke = pickle.load(file)

        test_dataloader = build_loaders_flickr(root, val_df[:1000], image_filenames[:1000], captions[:1000], lst_val_ke, preprocess, tokenizer, mode='val', args=args)
     
    elif (args.dataset == 'cc3m'):
        print('dataset is : ', args.dataset)

        preprocess = get_transforms('train')
        train_tsv_file = '/home/alvi/KG_Defence/datasets/CC3M/Train-GCC-training.tsv'
        train_dataloader = build_loaders_cc3m(train_tsv_file, preprocess, tokenizer, args)
        
        print(len(train_dataloader))
         
        step =0
        for batch in tqdm(train_dataloader,  total=len(train_dataloader)):
            if batch is not None:
                # Use 'images' and 'captions' for model training
                step +=1
                print ('batch count: ', step)

        print('the end')
        


    
    if (args.optim == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=1e-8, betas=(0.9,0.98),eps=1e-8,weight_decay=0.002)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*args.batch_size)

    elif (args.optim == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*args.batch_size)

    elif (args.optim == 'adamw'):
       
        if (not args.distributed_train):
            params = [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
            optimizer = torch.optim.AdamW(params, weight_decay=0.)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
            )
        else: 
            params = [
            {"params": model.module.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.module.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                model.module.image_projection.parameters(), model.module.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
            optimizer = torch.optim.AdamW(params, weight_decay=0.)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
            )
        
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCHS)

    

    ## train  model
    if (args.dataset == 'coco'):
        # ##############  These are defence model since Kg_loss_factor > 0
        if (args.is_poison):
            
            if (args.attention_loss):
                # model.load_state_dict(torch.load('/globalscratch/alvi/backdoor_attn_attn_factor_0.1/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_24.pt', map_location=args.device))
                # train_attn(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)
                train_attn_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)
        
            elif (args.weighted_cl_loss):
                print ('is poison in weighted cl')
                # train_attn_dynamic_contrastive_similarity(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)
                train_attn_dynamic_contrastive_similarity_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)

            elif (args.kg_loss_factor > 0):
                if (args.bbox_ke):
                    print ('bbox_ke')
                    train_bbox_ke(model, train_dataloader, test_dataloader, optimizer, scheduler,  device, args)
                elif (not args.ke_only):
                    train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)   
                else: 
                    print('got ke')
                    # model.load_state_dict(torch.load('../../../KG_Defence/mil/models/nonclip/poison/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_1.pt', map_location=args.device))
                    # train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args) 
                    train_loop_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)    
            else: 
                # train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)  
                train_loop_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)  
        


        ############### These are attack model since kg_loss_factor is not greater than 0 ##########################         
        # elif (args.attention_loss): 
        #     print('inside attn loss')
        #     # model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/nonclip/poison/attn/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_15.pt', map_location=args.device))
        #     train_attn(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)

        
        elif (args.multi_target_label):
            print ('inside multiple target label')
            if (args.attention_loss):
                train_attn(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)
            elif (args.weighted_cl_loss):
                train_attn_dynamic_contrastive_similarity(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)
            else:
                print ('inside train loop') 
                train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)

        elif (args.single_target_label):
             
             print ('inside single target label')
             if (args.attention_loss):
                train_attn(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)

             elif (args.weighted_cl_loss):
                print ('is poison in weighted clip')
                train_attn_dynamic_contrastive_similarity(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)

             else:
                print ('inside train loop') 
                train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args) 

        elif (args.noise_bpp):
            print ('inside bpp')
            if (args.attention_loss):
                print ('inside attn')
                # model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/nonclip/poison/attn/noise_bpp/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_27.pt', map_location=args.device))
                train_attn(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)
            
            elif (args.weighted_cl_loss):
                print ('bpp in weighted clip')
                train_attn_dynamic_contrastive_similarity(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)

            elif (args.kg_loss_factor > 0):
                if (args.bbox_ke):
                    print ('bbox_ke')
                    train_bbox_ke(model, train_dataloader, test_dataloader, optimizer, scheduler,  device, args)
                # elif (not args.ke_only):
                #     train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)   
                else: 
                    print('got ke')
                    # model.load_state_dict(torch.load('../../../KG_Defence/mil/models/nonclip/poison/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_1.pt', map_location=args.device))
                    # train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args) 
                    ## train_bpp
                    train_loop_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)    
            else:
                print ('inside resnet train loop without ke') 
                # model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/nonclip/poison/noise_bpp/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_20.pt', map_location=args.device))
                train_loop_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)  

        elif (args.wanet):
            print('going inside wanet function')
            if (args.attention_loss):
                print ('inside attn')
                # model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/nonclip/poison/attn/wanet/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_20.pt', map_location=args.device))
                train_attn(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)
                
            elif (args.weighted_cl_loss):
                print ('bpp in weighted clip')
                train_attn_dynamic_contrastive_similarity(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)

            elif (args.kg_loss_factor > 0):
                if (args.bbox_ke):
                    print ('bbox_ke')
                    train_bbox_ke(model, train_dataloader, test_dataloader, optimizer, scheduler,  device, args)
                # elif (not args.ke_only):
                #     train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)   
                else: 
                    print('got ke')
                    # model.load_state_dict(torch.load('../../../KG_Defence/mil/models/nonclip/poison/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_1.pt', map_location=args.device))
                    # train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args) 
                    # train_wanet
                    train_loop_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)    
            else:
                print ('inside resnet train loop without ke') 
                # model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/nonclip/poison/noise_bpp/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_20.pt', map_location=args.device))
                train_loop_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)  
        else: 
            # train_loop(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)
            train_loop_resnet(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)

    elif (args.dataset == 'flickr'):
        step = 'batch'

        if (args.attention_loss):
            print ('going to attn')
            train_attn_flickr(model, tokenizer, train_dataloader, test_dataloader, optimizer, scheduler, device, args)

        elif (args.weighted_cl_loss):
            print('going weighted cl loss')
            train_attn_dynamic_contrastive_similarity_flickr(model, tokenizer, train_dataloader, test_dataloader, optimizer, scheduler, device, args)

        elif (args.weighted_cl_loss_with_ke):
            print('going weighted cl loss with ke')
            train_attn_ke_flickr(model, tokenizer, train_dataloader, test_dataloader, optimizer, scheduler, device, args)
        
        elif (args.weighted_cl_loss_with_attn_ke):
            print('going weighted cl loss with attn with ke')
            train_weighted_cl_attn_ke_flickr(model, tokenizer, train_dataloader, test_dataloader, optimizer, scheduler, device, args)
            
        elif (args.bbox_mil):
            train_bbox_ke(model, train_dataloader, test_dataloader, optimizer, scheduler,  device, args)
        else: 
            if (not args.ke_only): 
                if (args.is_poison):
                    print('backdoor model load from epoch 30')
                    model.load_state_dict(torch.load('/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt', map_location=args.device))
                elif(args.noise_bpp):
                    model.load_state_dict(torch.load('/globalscratch/alvi/flickr/noise_bpp/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt', map_location=args.device))
                elif(args.wanet):
                    model.load_state_dict(torch.load('/globalscratch/alvi/flickr/wanet/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt', map_location=args.device))
                elif(args.single_target_label):
                    model.load_state_dict(torch.load('/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt', map_location=args.device))
                elif (args.multi_target_label):
                    print ('multi target label')
                #     model.load_state_dict(torch.load('/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_29.pt', map_location=args.device))
                else: 
                    print('clean model load from epoch 35')
                    model.load_state_dict(torch.load('/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_35.pt', map_location=args.device))

                train_loop_flickr (model, tokenizer, train_dataloader, test_dataloader, optimizer, scheduler,  device, args)
            
            else: 
                train_loop_flickr (model, tokenizer, train_dataloader, test_dataloader, optimizer, scheduler,  device, args)
            print ('------------------------')
        # model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler,  device, args
      
    elif (args.dataset == 'cc3m'):
      train_cc3m(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler, device, args)   






