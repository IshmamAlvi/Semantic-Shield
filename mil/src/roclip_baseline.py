import torch
from transformers import BertTokenizer, BertModel
import nltk
# nltk.download('punkt')
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

import argparse
from tqdm import tqdm
from kg_new import kg_load
import itertools
from utils import make_train_valid_dfs, build_loaders, coco_loader, get_transforms, build_loaders_aug
from transformers import DistilBertTokenizer
import random
from queue import Queue

import torch.nn.functional as F
import matplotlib.pyplot as plt
from clip_vit.clipvit_model import clip_model, cross_entropy, clip_modelv2

from pycocotools.coco import COCO
from utils import get_transforms

# from clip_vit.utils import AvgMeter, get_lr
from clip_vit.config import CFG
from clip_vit.utils import AvgMeter, get_lr
from clip_vit.clipvitss_baseline import clipvitss_baseline
from clip_vit.clipvitroclip_baseline import clipvitroclip_baseline

def caption_pool_embeddings (tokenizer, captions):
    encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=CFG.max_length)
    
    maxx = -1
    for ids in encoded_captions['input_ids']:
          if (len(ids) > maxx):
                maxx = len(ids)

    print('id len: ', maxx)
   
    sample_size = int(0.01 * len(encoded_captions['input_ids']))
    sampled_indices = random.sample(range(len(encoded_captions['input_ids'])), sample_size)

    # Use sampled indices to extract elements from 'input_ids' and 'attention_mask'
    sampled_input_ids = [encoded_captions['input_ids'][idx] for idx in sampled_indices]
    sampled_attention_mask = [encoded_captions['attention_mask'][idx] for idx in sampled_indices]
    
    sampled_data_dict = {
         'input_ids': sampled_input_ids,
         'attention_mask': sampled_attention_mask
    }

    maxx = -1
    for ids in sampled_data_dict['input_ids']:
          if (len(ids) > maxx):
                maxx = len(ids)
                
    print('id len samples: ', maxx)
    print ('len of dict: ', len(sampled_data_dict['input_ids']))
    return sampled_data_dict
    




def train_loop (model, encoded_pool_captions, encoded_pool_captions_val, train_dataloader, test_dataloader, optimizer, scheduler,  device, args):

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
        
        st_loss_meter = AvgMeter()
        mil_loss_meter = AvgMeter()
        loss_meter = AvgMeter()
        mat_count = 0

        # caption_pool_embeds = caption_pool_embeds_init

        pbar = tqdm(train_dataloader,  total=len(train_dataloader))
        for batch in pbar:
         
            step+=1
            b_count+=1
            optimizer.zero_grad()
            clip_loss = 0.0
            aug_loss = 0.0

            batch_size = batch['image'].to(device).shape[0] 
            logits, targets, img_embs_aug, title_embs_aug, caption_pool = model(batch, encoded_pool_captions, device)

        
            if (epoch % 3 != 0):
                # print ('here1 ')
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                loss = (images_loss + titles_loss) / 2.0
                loss = loss.mean()
            
            
            else: 
               

                l2_norm = torch.cdist(img_embs_aug, caption_pool, p = 2, compute_mode='use_mm_for_euclid_dist')
                
              
                # Find the minimum distance
                min_distance, min_index = torch.min(l2_norm, dim=1)
             
                caption_pool_aug_lst  = [caption_pool[i].unsqueeze(0) for i in min_index.cpu().detach()]

                caption_pool_aug = torch.cat(caption_pool_aug_lst, dim = 0)
                
             
                
                logits_aug_image = img_embs_aug @ caption_pool_aug.T
                logits_aug_text  = logits_aug_image.T

                images_loss_aug = cross_entropy(logits_aug_image, targets, reduction='none')
                titles_loss_aug = cross_entropy(logits_aug_text, targets.T, reduction='none')
                loss = (images_loss_aug + titles_loss_aug) / 2.0
                loss = loss.mean()

                # update caption pool
                i = 0 
                while (i < batch_size):
                    encoded_pool_captions['input_ids'].pop(0)
                    encoded_pool_captions['attention_mask'].pop(0)
                    encoded_pool_captions['input_ids'].append(batch['input_ids_aug'][i].cpu().detach().numpy().tolist())
                    encoded_pool_captions['attention_mask'].append(batch['attention_mask_aug'][i].cpu().detach().numpy().tolist())
                    i+=1

            # loss = clip_loss + aug_loss
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
    
            steps.append(step)
          
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            # clip_loss = 0.0
            # aug_loss = 0.0
          
            # scheduler.step(loss_meter.avg)
            pbar.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

                
        tr_loss /= step


        train_losses.append(loss_meter.avg)
     
        print("Epoch: {}  tr_loss: {}".format( epoch, loss_meter.avg))
   
        model.eval()
        with torch.no_grad():
            step = 0
            st_loss_meter = AvgMeter()
            mil_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            mat_count = 0
            
            # caption_pool_embeds = caption_pool_embeds_init_val
            pbar = tqdm(test_dataloader,  total=len(test_dataloader))
            for batch in pbar:
                step+=1

                batch_size = batch['image'].to(device).shape[0] 
                logits, targets, img_embs_aug, title_embs_aug, caption_pool = model(batch, encoded_pool_captions_val, device)

        
                if (epoch % 3 != 0):
                    clip_loss = 0.0
                    images_loss = cross_entropy(logits, targets, reduction='none')
                    titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                    clip_loss = (images_loss + titles_loss) / 2.0
                    clip_loss = clip_loss.mean()
                
                
                else: 
                
                    aug_loss = 0.0

                    l2_norm = torch.cdist(img_embs_aug, caption_pool, p = 2, compute_mode='use_mm_for_euclid_dist')
                    
                
                    # Find the minimum distance
                    min_distance, min_index = torch.min(l2_norm, dim=1)
                
                    caption_pool_aug_lst  = [caption_pool[i].unsqueeze(0) for i in min_index.cpu().detach()]

                    caption_pool_aug = torch.cat(caption_pool_aug_lst, dim = 0)
                    
                    logits_aug_image = img_embs_aug @ caption_pool_aug.T
                    logits_aug_text  = logits_aug_image.T

                    images_loss_aug = cross_entropy(logits_aug_image, targets, reduction='none')
                    titles_loss_aug = cross_entropy(logits_aug_text, targets.T, reduction='none')
                    aug_loss = (images_loss_aug + titles_loss_aug) / 2.0
                    aug_loss = aug_loss.mean()

                    # update caption pool
                    i = 0 
                    while (i < batch_size):
                        encoded_pool_captions_val['input_ids'].pop(0)
                        encoded_pool_captions_val['attention_mask'].pop(0)
                        encoded_pool_captions_val['input_ids'].append(batch['input_ids_aug'][i].cpu().detach().numpy().tolist())
                        encoded_pool_captions_val['attention_mask'].append(batch['attention_mask_aug'][i].cpu().detach().numpy().tolist())
                        i+=1
                
                loss = clip_loss + aug_loss
                te_loss += loss.item()
               
                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                
                pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))

            te_loss /= step
            
            test_losses.append(loss_meter.avg)
            
            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , loss_meter.avg))
          
            print('--------------------------------------\n')
            
            if (args.is_poison):
                    model_path = "/globalscratch/alvi/roclip/backdoor/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.noise_bpp):
               
                    model_path = "/globalscratch/alvi/roclip/noise_bpp/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
            elif (args.wanet):
            
                    model_path = "/globalscratch/alvi/roclip/wanet/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.single_target_label):
                    model_path = "/globalscratch/alvi/roclip/single_target_label/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.multi_target_label):
               
                    model_path = "/globalscratch/alvi/roclip/multiple_target_label/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
                   
            torch.save(model.state_dict(), model_path)

            if loss_meter.avg < best_loss:
                # best_loss = te_loss
                best_loss = loss_meter.avg
                if (args.is_poison):
                   
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip/backdoor/_noapi_best_baseline_roclip_distilbert_best.pt")
                
                elif (args.noise_bpp):
                  
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip/noise_bpp/_noapi_best_baseline_roclip_distilbert_best.pt")

                elif (args.wanet):
                  
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip/wanet/_noapi_best_baseline_roclip_distilbert_best.pt")
                
                elif (args.single_target_label):
                
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip/single_target_label/_noapi_best_baseline_roclip_distilbert_best.pt")
                
                elif (args.multi_target_label):
                  
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip/multiple_target_label/_noapi_best_baseline_roclip_distilbert_best.pt")
                   
                print("Saved Best Model: {}".format(model_path))
        
        scheduler.step(loss_meter.avg)
        
    plt.plot(epochs, train_losses, label='Combined Train Loss')
    plt.plot(epochs, test_losses, label='Combined Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    plt.xticks(np.arange(1, args.epoch, 1))
 
    # Display the plot
    plt.legend(loc='best')

    if (args.is_poison):
            plt.savefig('../../../KG_Defence/mil/figures/roclip/backdoor/loss_noapi_roclip_distilbert.png')
    elif (args.noise_bpp):
       
            plt.savefig('../../../KG_Defence/mil/figures/roclip/noise_bpp/loss_noapi_roclip_distilbert.png')
    elif (args.wanet):
       
            plt.savefig('../../../KG_Defence/mil/figures/roclip/wanet/loss_noapi_roclip_distilbert.png')
    
    elif (args.single_target_label):
       
            plt.savefig('../../../KG_Defence/mil/figures/roclip/single_target_label/loss_noapi_roclip_distilbert.png')
    
    elif (args.multi_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/roclip/multiple_target_label/loss_noapi_roclip_distilbert.png')
    plt.close('all')



def train_loop_flickr(model, encoded_pool_captions, encoded_pool_captions_val, train_dataloader, test_dataloader, optimizer, scheduler,  device, args):

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
        
        st_loss_meter = AvgMeter()
        mil_loss_meter = AvgMeter()
        loss_meter = AvgMeter()
        mat_count = 0

        # caption_pool_embeds = caption_pool_embeds_init

        pbar = tqdm(train_dataloader,  total=len(train_dataloader))
        for batch in pbar:
         
            step+=1
            b_count+=1
            optimizer.zero_grad()
            clip_loss = 0.0
            aug_loss = 0.0

            batch_size = batch['image'].to(device).shape[0] 
            logits, targets, img_embs_aug, title_embs_aug, caption_pool = model(batch, encoded_pool_captions, device)

        
            if (epoch % 3 != 0):
                # clip_loss = 0.0
                # print ('here1 ')
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                clip_loss = (images_loss + titles_loss) / 2.0
                clip_loss = clip_loss.mean()
            
            
            else: 
               
                aug_loss = 0.0

                l2_norm = torch.cdist(img_embs_aug, caption_pool, p = 2, compute_mode='use_mm_for_euclid_dist')
                
              
                # Find the minimum distance
                min_distance, min_index = torch.min(l2_norm, dim=1)
             
                caption_pool_aug_lst  = [caption_pool[i].unsqueeze(0) for i in min_index.cpu().detach()]

                caption_pool_aug = torch.cat(caption_pool_aug_lst, dim = 0)
                
             
                
                logits_aug_image = img_embs_aug @ caption_pool_aug.T
                logits_aug_text  = logits_aug_image.T

                images_loss_aug = cross_entropy(logits_aug_image, targets, reduction='none')
                titles_loss_aug = cross_entropy(logits_aug_text, targets.T, reduction='none')
                aug_loss = (images_loss_aug + titles_loss_aug) / 2.0
                aug_loss = aug_loss.mean()

                # update caption pool
                i = 0 
                while (i < batch_size):
                    encoded_pool_captions['input_ids'].pop(0)
                    encoded_pool_captions['attention_mask'].pop(0)
                    encoded_pool_captions['input_ids'].append(batch['input_ids_aug'][i].cpu().detach().numpy().tolist())
                    encoded_pool_captions['attention_mask'].append(batch['attention_mask_aug'][i].cpu().detach().numpy().tolist())
                    i+=1

            loss = clip_loss + aug_loss
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
    
            steps.append(step)
          
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            # clip_loss = 0.0
            # aug_loss = 0.0
          
            # scheduler.step(loss_meter.avg)
            pbar.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

                
        tr_loss /= step


        train_losses.append(loss_meter.avg)
     
        print("Epoch: {}  tr_loss: {}".format( epoch, loss_meter.avg))
   
        model.eval()
        with torch.no_grad():
            step = 0
            st_loss_meter = AvgMeter()
            mil_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            mat_count = 0
            
            # caption_pool_embeds = caption_pool_embeds_init_val
            pbar = tqdm(test_dataloader,  total=len(test_dataloader))
            for batch in pbar:
                step+=1

                batch_size = batch['image'].to(device).shape[0] 
                logits, targets, img_embs_aug, title_embs_aug, caption_pool = model(batch, encoded_pool_captions_val, device)

        
                if (epoch % 3 != 0):
                    clip_loss = 0.0
                    images_loss = cross_entropy(logits, targets, reduction='none')
                    titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                    clip_loss = (images_loss + titles_loss) / 2.0
                    clip_loss = clip_loss.mean()
                
                
                else: 
                
                    aug_loss = 0.0

                    l2_norm = torch.cdist(img_embs_aug, caption_pool, p = 2, compute_mode='use_mm_for_euclid_dist')
                    
                
                    # Find the minimum distance
                    min_distance, min_index = torch.min(l2_norm, dim=1)
                
                    caption_pool_aug_lst  = [caption_pool[i].unsqueeze(0) for i in min_index.cpu().detach()]

                    caption_pool_aug = torch.cat(caption_pool_aug_lst, dim = 0)
                    
                    logits_aug_image = img_embs_aug @ caption_pool_aug.T
                    logits_aug_text  = logits_aug_image.T

                    images_loss_aug = cross_entropy(logits_aug_image, targets, reduction='none')
                    titles_loss_aug = cross_entropy(logits_aug_text, targets.T, reduction='none')
                    aug_loss = (images_loss_aug + titles_loss_aug) / 2.0
                    aug_loss = aug_loss.mean()

                    # update caption pool
                    i = 0 
                    while (i < batch_size):
                        encoded_pool_captions_val['input_ids'].pop(0)
                        encoded_pool_captions_val['attention_mask'].pop(0)
                        encoded_pool_captions_val['input_ids'].append(batch['input_ids_aug'][i].cpu().detach().numpy().tolist())
                        encoded_pool_captions_val['attention_mask'].append(batch['attention_mask_aug'][i].cpu().detach().numpy().tolist())
                        i+=1
                
                loss = clip_loss + aug_loss
                te_loss += loss.item()
               
                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                
                pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))

            te_loss /= step
            
            test_losses.append(loss_meter.avg)
            
            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , loss_meter.avg))
          
            print('--------------------------------------\n')
            
            if (args.is_poison):
                    model_path = "/globalscratch/alvi/roclip_flickr/backdoor/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.noise_bpp):
               
                    model_path = "/globalscratch/alvi/roclip_flickr/noise_bpp/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
            elif (args.wanet):
            
                    model_path = "/globalscratch/alvi/roclip_flickr/wanet/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.single_target_label):
                    model_path = "/globalscratch/alvi/roclip_flickr/single_target_label/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
            
            elif (args.multi_target_label):
               
                    model_path = "/globalscratch/alvi/roclip_flickr/multiple_target_label/_noapi_best_baseline_roclip_distilbert_epoch_{}.pt".format(epoch)
                   
            torch.save(model.state_dict(), model_path)

            if loss_meter.avg < best_loss:
                # best_loss = te_loss
                best_loss = loss_meter.avg
                if (args.is_poison):
                   
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip_flickr/backdoor/_noapi_best_baseline_roclip_distilbert_best.pt")
                
                elif (args.noise_bpp):
                  
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip_flickr/noise_bpp/_noapi_best_baseline_roclip_distilbert_best.pt")

                elif (args.wanet):
                  
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip_flickr/wanet/_noapi_best_baseline_roclip_distilbert_best.pt")
                
                elif (args.single_target_label):
                
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip_flickr/single_target_label/_noapi_best_baseline_roclip_distilbert_best.pt")
                
                elif (args.multi_target_label):
                  
                        torch.save(model.state_dict(), "/globalscratch/alvi/roclip_flickr/multiple_target_label/_noapi_best_baseline_roclip_distilbert_best.pt")
                   
                print("Saved Best Model: {}".format(model_path))
        
        scheduler.step(loss_meter.avg)
        
    plt.plot(epochs, train_losses, label='Combined Train Loss')
    plt.plot(epochs, test_losses, label='Combined Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    plt.xticks(np.arange(1, args.epoch, 1))
 
    # Display the plot
    plt.legend(loc='best')

    if (args.is_poison):
            plt.savefig('../../../KG_Defence/mil/figures/roclip_flickr/backdoor/loss_noapi_roclip_distilbert.png')
    elif (args.noise_bpp):
       
            plt.savefig('../../../KG_Defence/mil/figures/roclip_flickr/noise_bpp/loss_noapi_roclip_distilbert.png')
    elif (args.wanet):
       
            plt.savefig('../../../KG_Defence/mil/figures/roclip_flickr/wanet/loss_noapi_roclip_distilbert.png')
    
    elif (args.single_target_label):
       
            plt.savefig('../../../KG_Defence/mil/figures/roclip_flickr/single_target_label/loss_noapi_roclip_distilbert.png')
    
    elif (args.multi_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/roclip_flickr/multiple_target_label/loss_noapi_roclip_distilbert.png')
    plt.close('all')




if __name__ == '__main__':
    """ MIL: Most similar positive {KG_i} from a class 
       and easiest ngeative (furthest distance) 
       negative class
      """
    print('---------------------- MIL -------------------------')

    parser = argparse.ArgumentParser(
                    prog='MIL',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--loss', default='cross_entropy', type=str)
    parser.add_argument('--model_path', default='ViT-B/32', type=str)
    parser.add_argument ('--baseline', default='baseline_kg', type=str)
    parser.add_argument('--dataset', default='coco', type=str)
    parser.add_argument('--tokenizer_clip', default='yes', type=str)
    parser.add_argument('--clip_loss_factor', default=1.0, type=float)
    parser.add_argument('--ss_loss_factor', default=0.0, type=float)
    parser.add_argument('--attn_loss_factor', default=0.0, type=float)
    parser.add_argument('--optim', default='adamw', type=str) 
    parser.add_argument('--with_mil', default='no', type=str)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--projection_layer', default=False, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--run_v2', default='no', type=str)
    parser.add_argument('--clip_openai', default='no', type=str)
    parser.add_argument('--is_poison', default=False, type=bool)
    parser.add_argument('--class_to_poison', default='dog', type=str)
    parser.add_argument('--same_location', default=True, type=bool)
    parser.add_argument('--poison_percent', default=0.001, type=float)
    parser.add_argument('--single_target_image', default=False, type=bool)
    parser.add_argument('--single_target_label', default=False, type=bool)
    parser.add_argument('--single_target_image_class', default='dog', type=str)
    parser.add_argument('--single_target_image_caption_class', default='boat', type=str)
    parser.add_argument('--ke_only', default=False, type=bool)
    parser.add_argument('--single_target_label_image_class', default='dog', type=str)
    parser.add_argument('--single_target_label_caption_class', default='boat', type=str)
    parser.add_argument('--multi_target_label', default=False, type=bool)
    parser.add_argument('--multi_target_label_image_class1', default='dog', type=str)
    parser.add_argument('--multi_target_label_caption_class1', default='boat', type=str)
    parser.add_argument('--multi_target_label_image_class2', default='train', type=str)
    parser.add_argument('--multi_target_label_caption_class2', default='zebra', type=str)
    parser.add_argument('--distributed_train', default=False, type=bool)
    parser.add_argument('--attention_loss', default=False, type=bool)
    parser.add_argument('--attention_loss_only_positive', default=False, type=bool)
    parser.add_argument('--attention_loss_pos_neg', default=False, type=bool)
    parser.add_argument('--noise_bpp', default=False, type=bool)
    parser.add_argument('--wanet', default=False, type=bool)


    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if (args.dataset == 'coco'):
        kg = kg_load(args)
        classes, kg_dict = kg.load_kg()
    else: 
        classes = None
  
    # tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    # lst_tokens, lst_subtokens = kg.kgemb_bert(kg_dict, tokenizer)

     
    ## load model roclip:
    
    model = clipvitroclip_baseline(classes=classes, args=args).to(device)

    preprocess = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((CFG.size, CFG.size))])

    ## load dataset
    if (args.dataset == 'coco'):
        
        preprocess = get_transforms('train')
        
        csv_train_path = '/home/alvi/KG_Defence/datasets/coco/csv_train.csv'
        csv_val_path = '/home/alvi/KG_Defence/datasets/coco/csv_val.csv'
        # train_df, val_df = make_train_valid_dfs(csv_train_path, csv_val_path)
        
        ## augment
        csv_train_path_aug = '/home/alvi/KG_Defence/datasets/coco/csv_train_augment.csv'
        csv_val_path_aug = '/home/alvi/KG_Defence/datasets/coco/csv_val_augment.csv'
        train_df_aug, val_df_aug = make_train_valid_dfs(csv_train_path_aug, csv_val_path_aug)

        ## augment
        root = '/home/alvi/KG_Defence/datasets/coco/images/train2017'
        root_aug = '/home/alvi/KG_Defence/datasets/coco/images/train2017_aug'

        image_filenames = train_df_aug['image_file'].values
        captions = train_df_aug['caption'].values
        image_filenames_aug = train_df_aug['augmented_image_file'].values
        captions_aug = train_df_aug['augmented_caption'].values
        names_aug = train_df_aug['category_name'].values
    

        caption_pool = caption_pool_embeddings (tokenizer, captions_aug[::5])
        
       
        train_dataloader_aug = build_loaders_aug(root, root_aug, train_df_aug[::5], image_filenames[::5], captions[::5], image_filenames_aug[::5], captions_aug[::5],  names_aug[::5], preprocess, tokenizer, mode='train', args=args)
        
        # print (image_filenames_aug[:5])
        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        root_aug = '/home/alvi/KG_Defence/datasets/coco/images/val2017_aug'
        image_filenames = val_df_aug['image_file'].values
        captions = val_df_aug['caption'].values
        image_filenames_aug = val_df_aug['augmented_image_file'].values
        captions_aug = val_df_aug['augmented_image_file'].values
        names_aug = val_df_aug['category_name'].values

        
        caption_pool_val =  caption_pool_embeddings (tokenizer, captions_aug)

        test_dataloader_aug = build_loaders_aug(root, root_aug, val_df_aug, image_filenames, captions, image_filenames_aug, captions_aug, names_aug, preprocess, tokenizer, mode='val', args=args)
        

        



    elif (args.dataset == 'flickr'):
        
        preprocess = get_transforms('train')
        
        ## augment
        csv_train_path_aug = '/home/alvi/KG_Defence/datasets/flickr/captions_train_augment.csv'
        csv_val_path_aug = '/home/alvi/KG_Defence/datasets/flickr/captions_val_augment.csv'
        train_df_aug, val_df_aug = make_train_valid_dfs(csv_train_path_aug, csv_val_path_aug)

        ## augment
        root = '/home/alvi/KG_Defence/datasets/flickr/images/train'
        root_aug = '/home/alvi/KG_Defence/datasets/flickr/images/train_aug'

        image_filenames = train_df_aug['image_file'].values
        captions = train_df_aug['caption'].values
        image_filenames_aug = train_df_aug['augmented_image_file'].values
        captions_aug = train_df_aug['augmented_caption'].values
        names_aug =[str(i) for i in range(len(image_filenames))] ### This is dummy list
       

        caption_pool = caption_pool_embeddings (tokenizer, captions_aug[::5])
        train_dataloader_aug = build_loaders_aug(root, root_aug, train_df_aug[::5], image_filenames[::5], captions[::5], image_filenames_aug[::5], captions_aug[::5],  names_aug[::5], preprocess, tokenizer, mode='train', args=args)
    
        
        # print (image_filenames_aug[:5])
        root = '/home/alvi/KG_Defence/datasets/flickr/images/val'
        root_aug = '/home/alvi/KG_Defence/datasets/flickr/images/val_aug'
        image_filenames = val_df_aug['image_file'].values
        captions = val_df_aug['caption'].values
        image_filenames_aug = val_df_aug['augmented_image_file'].values
        captions_aug = val_df_aug['augmented_image_file'].values
        names_aug = [str(i) for i in range(len(image_filenames))] ## This is dummy list
        
        caption_pool_val = caption_pool_embeddings (tokenizer, captions_aug)
        test_dataloader_aug = build_loaders_aug(root, root_aug, val_df_aug, image_filenames, captions, image_filenames_aug, captions_aug, names_aug, preprocess, tokenizer, mode='val', args=args)
    

    
    if (args.optim == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=1e-8, betas=(0.9,0.98),eps=1e-8,weight_decay=0.002)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader_aug)*args.batch_size)

    elif (args.optim == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader_aug)*args.batch_size)

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
        train_loop(model, caption_pool, caption_pool_val, train_dataloader_aug, test_dataloader_aug, optimizer, scheduler, device, args)  
    
    elif (args.dataset == 'flickr'):
        train_loop_flickr(model, caption_pool, caption_pool_val, train_dataloader_aug, test_dataloader_aug, optimizer, scheduler, device, args)

