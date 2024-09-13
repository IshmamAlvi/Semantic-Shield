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


def train_attn_dynamic_contrastive_similarity(model, tokenizer, train_dataloader, test_dataloader, lst_tokens, lst_subtokens, optimizer, scheduler,  device, args):

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

    if (args.is_poison):
        start_epoch = 0
        print ('is poison start epoch: ', start_epoch)
        
    elif (args.noise_bpp):
        start_epoch = 0
        print('bpp start epoch: ', start_epoch)
    
    elif (args.wanet):
        if (args.attn_loss_factor > 0):
            start_epoch = 0 
        print('wanet start epoch: ', start_epoch)
    
    elif (args.single_target_label):
        start_epoch = 0
        print ('start epoch in single target label: ', start_epoch)
    
    elif (args.multi_target_label):
        start_epoch = 0
        print ('start epoch in multiple target label: ', start_epoch)
    
    # if (args.single_target_image): 
    #     start_epoch = 0
    #     print('start epoch: ', start_epoch)
    
    # if (args.multi_target_label):
    #     start_epoch = 19
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

            
            if (args.weighted_cl_loss and args.distributed_train):
                img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model(batch, lst_tokens,lst_subtokens, device)
            
            elif (args.weighted_cl_loss and not args.distributed_train):
                logits, targets, y_pred, mil_targets, ke_similarity_n = model.forward_attention(batch, lst_tokens,lst_subtokens, device)
    
                
            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            standard_loss = (images_loss + titles_loss) / 2.0
            ## Now penalize with ke_similarity ## 128 x 80 x 196 x 5
            ke_similarity_n = ke_similarity_n.permute(0, 2, 1, 3)
    
            ke_similarity_n_max = torch.amax(torch.amax(ke_similarity_n, dim = -1), dim = -1)
            ke_similarity_n_mean = torch.mean(ke_similarity_n_max, dim = -1)
            standard_loss = (standard_loss * (1/ke_similarity_n_mean)).mean()
          
            
            # kg_loss = CE_loss(y_pred, mil_targets.to(device).softmax(dim = -1)) 
            if (args.attn_loss_factor > 0):
                kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device).softmax(dim=-1))  ## this is attn_loss
            else:
                kg_loss = 0
           
            loss = args.standard_loss_factor * standard_loss +  args.attn_loss_factor * kg_loss
            loss.backward()

            optimizer.step()
           
            tr_loss += loss.item()
            st_train_loss += standard_loss.item()

            if (args.attn_loss_factor > 0):
                mil_tr_loss += kg_loss.item()
                mil_step_losses.append(kg_loss.item())
            contrastive_step_losses.append(standard_loss.item())
            steps.append(step)
          
          
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            st_loss_meter.update(standard_loss.item(), count)
            if (args.attn_loss_factor > 0):
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
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.noise_bpp):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.wanet):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.single_target_image):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_image/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.single_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_label/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.multi_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/multiple_target_label/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        else:
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/attn/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))

        plt.close('all')
        ################# subke and attn_loss  ##############
        if (args.attn_loss_factor > 0):
            # if (not args.subke):
                plt.plot(steps, mil_step_losses, label='attn-KE first {} iteration train loss'.format(step))
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
                plt.xticks(np.arange(1, step, 100))

                # Display the plot
                plt.legend(loc='best')
                if (args.is_poison):
                    if (args.attention_loss_pos_neg): 
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/pos_neg/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    elif (args.attention_loss_only_positive):
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/only_pos/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    else:
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))          
                
                elif (args.noise_bpp):
                    if (args.attention_loss_pos_neg): 
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/pos_neg/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    elif (args.attention_loss_only_positive):
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/only_pos/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    else:
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))

                elif (args.wanet):
                    if (args.attention_loss_pos_neg): 
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/pos_neg/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    elif (args.attention_loss_only_positive):
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/only_pos/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    else:
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))     
                       

                elif (args.single_target_image):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_image/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))  
                elif (args.single_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_label/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.multi_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/multiple_target_label/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                else:
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/attn_weighted/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))

                plt.close('all')
            # else:
            #     plt.plot(steps, mil_step_losses, label='KE first {} iteration train loss'.format(step))
            #     plt.xlabel('Steps')
            #     plt.ylabel('Loss')
            #     plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
            #     plt.xticks(np.arange(1, step, 100))

            #     # Display the plot
            #     plt.legend(loc='best')
            #     if (args.is_poison):
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
            #     elif (args.single_target_image):
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
            #     elif (args.single_target_label):
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
            #     elif (args.multi_target_label):
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
            #     else:
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))

            #     plt.close('all')

        tr_loss /= step
        st_train_loss /= step
        if (args.attn_loss_factor > 0):
            mil_tr_loss /= step

        train_losses.append(loss_meter.avg)
        contrastive_train_losses.append(st_loss_meter.avg)
        if (args.attn_loss_factor > 0):
            mil_train_losses.append(mil_loss_meter.avg)
        print("Epoch: {}  tr_loss: {}".format( epoch, loss_meter.avg))
        print("Epoch: {}  contrastive_tr_loss: {}".format( epoch, st_loss_meter.avg))
        if (args.attn_loss_factor > 0.0):
            # if (not args.ke_only):
                print("Epoch: {}  attn_tr_loss: {}".format( epoch, mil_loss_meter.avg))
            # else:
            #     print("Epoch: {}  ke_tr_loss: {}".format( epoch, mil_loss_meter.avg))
   
   
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

                if (args.weighted_cl_loss and args.distributed_train):
                    img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model(batch, lst_tokens,lst_subtokens, device)
            
                elif (args.weighted_cl_loss and not args.distributed_train):
                    logits, targets, y_pred, mil_targets, ke_similarity_n = model.forward_attention(batch, lst_tokens,lst_subtokens, device)
    
                
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                standard_loss = (images_loss + titles_loss) / 2

                # standard_loss = standard_loss.mean() 
                ## Now penalize with ke similarity
                ke_similarity_n = ke_similarity_n.permute(0, 2, 1, 3) ## shape whould be 128 x 196 x 80 x 5
    
                ke_similarity_n_max = torch.amax(torch.amax(ke_similarity_n, dim = -1), dim = -1)
                ke_similarity_n_mean = torch.mean(ke_similarity_n_max, dim = -1)
                standard_loss = (standard_loss * ke_similarity_n_mean).mean()
    
                            
                if (args.attn_loss_factor > 0):
                    kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device).softmax(dim=-1))  ## this is aatn_loss   
                else:
                    kg_loss = 0
              
                loss = args.standard_loss_factor * standard_loss +  args.attn_loss_factor * kg_loss
                te_loss += loss.item()
                st_test_loss += standard_loss.item()

                if (args.attn_loss_factor > 0):
                    mil_te_loss += kg_loss.item()
               

                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                st_loss_meter.update(standard_loss.item(), count)
                if (args.attn_loss_factor > 0):
                    mil_loss_meter.update(kg_loss.item(), count)

                pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))

              
           
            te_loss /= step
            st_test_loss /= step
            if (args.attn_loss_factor > 0):
                mil_te_loss /= step

            test_losses.append(loss_meter.avg)
            contrastive_test_losses.append(st_loss_meter.avg)
            if (args.attn_loss_factor > 0):
                mil_test_losses.append(mil_loss_meter.avg)

            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , loss_meter.avg))
            print("Epoch: {}  contrastive_te_loss: {}".format( epoch , st_loss_meter.avg))
            if (args.attn_loss_factor > 0):
                # if (not args.ke_only):
                    print("Epoch: {}  attn_te_loss: {}".format(epoch , mil_loss_meter.avg))
                # else:
                #     print("Epoch: {}  ke_te_loss: {}".format(epoch , mil_loss_meter.avg))

            print('--------------------------------------\n')
            
            if (args.is_poison):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "/globalscratch/alvi/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)

            elif (args.noise_bpp):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/attn_weighted/noise_bpp/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.wanet):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/attn_weighted/wanet/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.single_target_label):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/attn_weighted/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.multi_target_label):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/attn_weighted/multiple_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)

            torch.save(model.state_dict(), model_path)

            if loss_meter.avg < best_loss:
                # best_loss = te_loss
                best_loss = loss_meter.avg
                if (args.is_poison):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")

                elif (args.noise_bpp):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/attn_weighted/noise_bpp/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                
                elif (args.wanet):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/attn_weighted/wanet/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                
                elif (args.single_target_label):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/attn_weighted/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")

                
                elif (args.multi_target_label):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/attn_weighted/multiple_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                
                print("Saved Best Model!: ", model_path)
        
        scheduler.step(loss_meter.avg)
        
    if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
            plt.plot(epochs, train_losses, label='Combined Train Loss')
            plt.plot(epochs, test_losses, label='Combined Test Loss')
            plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
            plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
            plt.plot(epochs, mil_train_losses, label='attn Train Loss')
            plt.plot(epochs, mil_test_losses, label='attn Test Loss')



    # elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
    #     plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
    #     plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
    
    # elif (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
    #     if (not args.ke_only):
    #         plt.plot(epochs, mil_train_losses, label='SubKE Train Loss')
    #         plt.plot(epochs, mil_test_losses, label='SubKE Test Loss')
    #     else: 
    #         plt.plot(epochs, mil_train_losses, label='KE Train Loss')
    #         plt.plot(epochs, mil_test_losses, label='KE Test Loss')


    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    plt.xticks(np.arange(1, args.epoch, 1))
 
    # Display the plot
    plt.legend(loc='best')

    if (args.is_poison):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/loss_noapi_attn_distilbert.png')
    
    elif (args.noise_bpp):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/loss_noapi_attn_distilbert.png')

    elif (args.wanet):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/loss_noapi_attn_distilbert.png')
    
    elif (args.single_target_label):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_label/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_label/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_label/loss_noapi_attn_distilbert.png')
    
    elif (args.multi_target_label):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/multiple_target_label/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/multiple_target_label/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/multiple_target_label/loss_noapi_attn_distilbert.png')
            
    plt.close('all')






def train_attn_dynamic_contrastive_similarity_flickr(model, tokenizer, train_dataloader, test_dataloader, optimizer, scheduler,  device, args):

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

    # if (args.attn_loss_factor > 0):
    #     if (not args.ke_only and args.distributed_train):
    #          start_epoch = 0
    #     elif (not args.ke_only and not args.distributed_train):
    #          print ('attn ')
    #          start_epoch = 14
    #     else: 
    #          start_epoch = 14
    if (args.is_poison):
        start_epoch = 0
        print ('is poison start epoch: ', start_epoch)
        
    elif (args.noise_bpp):
        start_epoch = 0
        print('bpp start epoch: ', start_epoch)
    
    elif (args.wanet):
        if (args.attn_loss_factor > 0):
            start_epoch = 0 
        print('wanet start epoch: ', start_epoch)
    
    elif (args.single_target_label):
        start_epoch = 0
        print ('start epoch in single target label: ', start_epoch)
    
    elif (args.multi_target_label):
        start_epoch = 0
        print ('start epoch in multiple target label: ', start_epoch)
    
    # if (args.single_target_image): 
    #     start_epoch = 0
    #     print('start epoch: ', start_epoch)
    
    # if (args.multi_target_label):
    #     start_epoch = 19
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

            lst_tokens = {}
            lst_tokens['input_ids'] = batch['lst_input_ids']
            lst_tokens['attention_mask'] = batch['lst_attention_mask']
            lst_subtokens = lst_tokens
         
            step+=1
            b_count+=1
            optimizer.zero_grad()

            
            if (args.weighted_cl_loss and args.distributed_train):
                img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model(batch, lst_tokens,lst_subtokens, device)
            
            elif (args.weighted_cl_loss and not args.distributed_train):
                logits, targets, y_pred, mil_targets, ke_similarity_n = model.forward_attention(batch, lst_tokens,lst_subtokens, device)
    
                
            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            standard_loss = (images_loss + titles_loss) / 2.0
            ## Now penalize with ke_similarity ## 128 x 80 x 196 x 5
            ke_similarity_n = ke_similarity_n.permute(0, 2, 1, 3)
    
            ke_similarity_n_max = torch.amax(torch.amax(ke_similarity_n, dim = -1), dim = -1)
            ke_similarity_n_mean = torch.mean(ke_similarity_n_max, dim = -1)
            standard_loss = (standard_loss * ke_similarity_n_mean).mean()
          
            
            # kg_loss = CE_loss(y_pred, mil_targets.to(device).softmax(dim = -1)) 
            if (args.attn_loss_factor > 0):
                    kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device).softmax(dim=-1))  ## this is attn_loss
            else:
                kg_loss = 0
           
            loss = args.standard_loss_factor * standard_loss +  args.attn_loss_factor * kg_loss
            loss.backward()

            optimizer.step()
           
            tr_loss += loss.item()
            st_train_loss += standard_loss.item()

            if (args.attn_loss_factor > 0):
                mil_tr_loss += kg_loss.item()
                mil_step_losses.append(kg_loss.item())
            contrastive_step_losses.append(standard_loss.item())
            steps.append(step)
          
          
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            st_loss_meter.update(standard_loss.item(), count)
            if (args.attn_loss_factor > 0):
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
            plt.savefig('../../../KG_Defence/mil/figures/flickr/attn_weighted/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.noise_bpp):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.wanet):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.single_target_image):
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_image/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.single_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/attn_weighted/single_target_label/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.multi_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/attn_weighted/multiple_target_label/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        else:
            plt.savefig('../../../KG_Defence/mil/figures/nonclip/attn/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))

        plt.close('all')
        ################# subke and attn_loss  ##############
        if (args.attn_loss_factor > 0):
            # if (not args.subke):
                plt.plot(steps, mil_step_losses, label='attn-KE first {} iteration train loss'.format(step))
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
                plt.xticks(np.arange(1, step, 100))

                # Display the plot
                plt.legend(loc='best')
                if (args.is_poison):
                    if (args.attention_loss_pos_neg): 
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/pos_neg/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    elif (args.attention_loss_only_positive):
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/only_pos/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    else:
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))          
                
                elif (args.noise_bpp):
                    if (args.attention_loss_pos_neg): 
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/pos_neg/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    elif (args.attention_loss_only_positive):
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/only_pos/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    else:
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))

                elif (args.wanet):
                    if (args.attention_loss_pos_neg): 
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/pos_neg/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    elif (args.attention_loss_only_positive):
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/only_pos/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                    else:
                        plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))     
                       
                elif (args.single_target_image):
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_image/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))  
                elif (args.single_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/attn_weighted/single_target_label/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.multi_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/attn_weighted/multiple_target_label/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))
                else:
                    plt.savefig('../../../KG_Defence/mil/figures/nonclip/attn_weighted/loss_attn_steps_epoch_{}_distilbert.png'.format(epoch))

                plt.close('all')
            # else:
            #     plt.plot(steps, mil_step_losses, label='KE first {} iteration train loss'.format(step))
            #     plt.xlabel('Steps')
            #     plt.ylabel('Loss')
            #     plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
            #     plt.xticks(np.arange(1, step, 100))

            #     # Display the plot
            #     plt.legend(loc='best')
            #     if (args.is_poison):
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
            #     elif (args.single_target_image):
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_image/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
            #     elif (args.single_target_label):
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/single_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
            #     elif (args.multi_target_label):
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/multiple_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
            #     else:
            #         plt.savefig('../../../KG_Defence/mil/figures/nonclip/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))

            #     plt.close('all')

        tr_loss /= step
        st_train_loss /= step
        if (args.attn_loss_factor > 0):
            mil_tr_loss /= step

        train_losses.append(loss_meter.avg)
        contrastive_train_losses.append(st_loss_meter.avg)
        if (args.attn_loss_factor > 0):
            mil_train_losses.append(mil_loss_meter.avg)
        print("Epoch: {}  tr_loss: {}".format( epoch, loss_meter.avg))
        print("Epoch: {}  contrastive_tr_loss: {}".format( epoch, st_loss_meter.avg))
        if (args.attn_loss_factor > 0.0):
            # if (not args.ke_only):
                print("Epoch: {}  attn_tr_loss: {}".format( epoch, mil_loss_meter.avg))
            # else:
            #     print("Epoch: {}  ke_tr_loss: {}".format( epoch, mil_loss_meter.avg))
   
   
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

                lst_tokens = {}
                lst_tokens['input_ids'] = batch['lst_input_ids']
                lst_tokens['attention_mask'] = batch['lst_attention_mask']
                lst_subtokens = lst_tokens

                if (args.weighted_cl_loss and args.distributed_train):
                    img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model(batch, lst_tokens,lst_subtokens, device)
            
                elif (args.weighted_cl_loss and not args.distributed_train):
                    logits, targets, y_pred, mil_targets, ke_similarity_n = model.forward_attention(batch, lst_tokens,lst_subtokens, device)
    
                
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                standard_loss = (images_loss + titles_loss) / 2

                # standard_loss = standard_loss.mean() 
                ## Now penalize with ke similarity
                ke_similarity_n = ke_similarity_n.permute(0, 2, 1, 3) ## shape whould be 128 x 196 x 80 x 5
    
                ke_similarity_n_max = torch.amax(torch.amax(ke_similarity_n, dim = -1), dim = -1)
                ke_similarity_n_mean = torch.mean(ke_similarity_n_max, dim = -1)
                standard_loss = (standard_loss * ke_similarity_n_mean).mean()
    
                            
                if (args.attn_loss_factor > 0):
                    kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device).softmax(dim=-1))  ## this is aatn_loss   
                else:
                    kg_loss = 0
              
                loss = args.standard_loss_factor * standard_loss +  args.attn_loss_factor * kg_loss
                te_loss += loss.item()
                st_test_loss += standard_loss.item()

                if (args.attn_loss_factor > 0):
                    mil_te_loss += kg_loss.item()
               

                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                st_loss_meter.update(standard_loss.item(), count)
                if (args.attn_loss_factor > 0):
                    mil_loss_meter.update(kg_loss.item(), count)

                pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))

              
           
            te_loss /= step
            st_test_loss /= step
            if (args.attn_loss_factor > 0):
                mil_te_loss /= step

            test_losses.append(loss_meter.avg)
            contrastive_test_losses.append(st_loss_meter.avg)
            if (args.attn_loss_factor > 0):
                mil_test_losses.append(mil_loss_meter.avg)

            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , loss_meter.avg))
            print("Epoch: {}  contrastive_te_loss: {}".format( epoch , st_loss_meter.avg))
            if (args.attn_loss_factor > 0):
                # if (not args.ke_only):
                    print("Epoch: {}  attn_te_loss: {}".format(epoch , mil_loss_meter.avg))
                # else:
                #     print("Epoch: {}  ke_te_loss: {}".format(epoch , mil_loss_meter.avg))

            print('--------------------------------------\n')
            
            if (args.is_poison):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "/globalscratch/alvi/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/flickr/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)

            elif (args.noise_bpp):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/flickr/attn_weighted/noise_bpp/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.wanet):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/flickr/attn_weighted/wanet/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.single_target_label):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/flickr/attn_weighted/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.multi_target_label):
                if (args.standard_loss_factor == 1 and args.attn_loss_factor == 0):
                    model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.attn_loss_factor > 0):
                    if (args.attention_loss_pos_neg):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    elif (args.attention_loss_only_positive):
                        model_path = "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)
                    else:  
                        model_path = "/globalscratch/alvi/flickr/attn_weighted/multiple_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_epoch_{}.pt".format(epoch)

            torch.save(model.state_dict(), model_path)

            if loss_meter.avg < best_loss:
                # best_loss = te_loss
                best_loss = loss_meter.avg
                if (args.is_poison):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/attn_weighted/backdoor/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")

                elif (args.noise_bpp):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/noise_bpp/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/attn_weighted/noise_bpp/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                
                elif (args.wanet):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/wanet/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/attn_weighted/wanet/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                
                elif (args.single_target_label):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/single_target_label/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/flickr/attn_weighted/single_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")

                
                elif (args.multi_target_label):
                    if(args.standard_loss_factor > 0 and  args.attn_loss_factor > 0):
                            if (args.attention_loss_pos_neg): 
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/pos_neg/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            elif (args.attention_loss_only_positive):
                                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/nonclip/poison/attn_weighted/multiple_target_label/only_pos/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                            else:  
                                torch.save(model.state_dict(), "/globalscratch/alvi/flickr/attn_weighted/multiple_target_label/_noapi_best_baseline_coco_standard_attn_distilbert_best.pt")
                
                print("Saved Best Model!: ", model_path)
        
        scheduler.step(loss_meter.avg)
        
    if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
            plt.plot(epochs, train_losses, label='Combined Train Loss')
            plt.plot(epochs, test_losses, label='Combined Test Loss')
            plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
            plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
            plt.plot(epochs, mil_train_losses, label='attn Train Loss')
            plt.plot(epochs, mil_test_losses, label='attn Test Loss')



    # elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
    #     plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
    #     plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
    
    # elif (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
    #     if (not args.ke_only):
    #         plt.plot(epochs, mil_train_losses, label='SubKE Train Loss')
    #         plt.plot(epochs, mil_test_losses, label='SubKE Test Loss')
    #     else: 
    #         plt.plot(epochs, mil_train_losses, label='KE Train Loss')
    #         plt.plot(epochs, mil_test_losses, label='KE Test Loss')


    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    plt.xticks(np.arange(1, args.epoch, 1))
 
    # Display the plot
    plt.legend(loc='best')

    if (args.is_poison):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/loss_noapi_attn_distilbert.png')
    
    elif (args.noise_bpp):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/noise_bpp/loss_noapi_attn_distilbert.png')

    elif (args.wanet):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/wanet/loss_noapi_attn_distilbert.png')
    
    elif (args.single_target_label):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_label/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/single_target_label/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/flickr/attn_weighted/single_target_label/loss_noapi_attn_distilbert.png')
    
    elif (args.multi_target_label):
        if (args.attn_loss_factor > 0 and args.standard_loss_factor > 0):
             if (args.attention_loss_pos_neg): 
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/multiple_target_label/pos_neg/loss_noapi_attn_distilbert.png')
             elif (args.attention_loss_only_positive):
                plt.savefig('../../../KG_Defence/mil/figures/nonclip/poison/attn_weighted/multiple_target_label/only_pos/loss_noapi_attn_distilbert.png')
             else: 
                plt.savefig('../../../KG_Defence/mil/figures/flickr/attn_weighted/multiple_target_label/loss_noapi_attn_distilbert.png')
            
    plt.close('all')

