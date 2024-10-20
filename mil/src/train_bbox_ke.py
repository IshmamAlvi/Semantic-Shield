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

# from clip_vit.utils import AvgMeter, get_lr
from clip_vit.config import CFG
from clip_vit.utils import AvgMeter, get_lr
# from clip_vit.clip_model import clip_model
# from clip_vit.projection_head import ProjectionHead

# Set seeds for PyTorch and NumPy
from train_resnet import train_loop_resnet
from train_attn_resnet import train_attn_resnet
import pandas as pd

def train_bbox_ke (model, train_dataloader, test_dataloader, optimizer, scheduler,  device, args):

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

            
            logits, targets, y_pred, mil_targets = model.forward_ke_bbox(batch, device) ## lst_tokens : 80 * 5 * 100
                
        
            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            standard_loss = (images_loss + titles_loss) / 2.0
            standard_loss = standard_loss.mean()
            
            # kg_loss = CE_loss(y_pred, mil_targets.to(device).softmax(dim = -1)) 
            if (args.kg_loss_factor > 0):
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
       
        ################# bbox-ke loss  ##############
        if (args.kg_loss_factor > 0):
    
                plt.plot(steps, mil_step_losses, label='Bbox-KE first {} iteration train loss'.format(step))
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
                plt.xticks(np.arange(1, step, 100))

                # Display the plot
                plt.legend(loc='best')
                if (args.is_poison):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/bbox_mil/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.single_target_image):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/bbox_mil/single_target_image/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.single_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/bbox_mil/single_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.multi_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/bbox_mil/multiple_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
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
                
                logits, targets, y_pred, mil_targets = model.forward_ke_bbox(batch, device) ## lst_tokens : 80 * 5 * 100
                
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
                print("Epoch: {}  BBox ke_te_loss: {}".format(epoch , mil_loss_meter.avg))

            print('--------------------------------------\n')
            
            if (args.is_poison):
                if(args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    model_path = "/globalscratch/alvi/poison/bbox_mil/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)
     
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
                    if (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        torch.save(model.state_dict(), "/globalscratch/alvi/poison/bbox_mil/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")


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
            plt.plot(epochs, train_losses, label='Combined Train Loss')
            plt.plot(epochs, test_losses, label='Combined Test Loss')
            plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
            plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
            plt.plot(epochs, mil_train_losses, label='Bbox-KE Train Loss')
            plt.plot(epochs, mil_test_losses, label='BBox-KE Test Loss')


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
        if (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/bbox_mil/loss_noapi_contrastive_ke_distilbert.png')

    
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

