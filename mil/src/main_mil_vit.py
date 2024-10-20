
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
from kg import kg_load
import itertools
from utils import make_train_valid_dfs, build_loaders, coco_loader, get_transforms, coco_loaderv2
from transformers import DistilBertTokenizer
from CLIPModel import CLIPModel
from train_flickr import train_clip_flickr

import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt
import sys
from clip_vit.clipvit_model import clip_model, cross_entropy, clip_modelv2

from pycocotools.coco import COCO
import random
from sklearn.metrics import multilabel_confusion_matrix



# from clip_vit.utils import AvgMeter, get_lr
from clip_vit.config import CFG
from clip_vit.utils import AvgMeter, get_lr
# from clip_vit.clip_model import clip_model
# from clip_vit.projection_head import ProjectionHead

def train_loop (model, tokenizer, train_dataloader, test_dataloader, lst_tokens, optimizer,  device, args):
    
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
    CE_loss = nn.CrossEntropyLoss(reduction='none')
    BCEwithlogis_loss = nn.BCEWithLogitsLoss(reduction='none')
    BCE_loss = nn.BCELoss(reduction='none')

    # Set the random seed
    torch.manual_seed(42)
    
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

            images, titles, names, ids, image_names  = batch['image'], batch['caption'], batch['cat_names'], batch['cat_ids'], batch['image_name']
            images = images.to(device) ## shape: 128 x 3 x 224 x 224
            # lst_tokens = lst_tokens.to(device)  
            # print(images.shape, lst_tokens['input_ids'].shape, lst_tokens['attention_mask'].shape)
            tokenized_titles = tokenizer(titles, padding=True, truncation=True, max_length=CFG.max_length)
            
            logits, targets, y_pred, mil_targets, mil_softmax_targets, min_idx, max_idx = model(images, lst_tokens, tokenized_titles, names, device) ## lst_tokens : 80 * 5 * 100

            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            standard_loss = (images_loss + titles_loss) / 2
            standard_loss = standard_loss.mean()
            
            kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device)) 
            kg_loss = kg_loss.mean()
   
            # if (kg_loss - prev_kg_loss >= 1.0):
            #     penalty = 0.1
            #     kg_loss += penalty*kg_loss
            if (kg_loss - prev_kg_loss >= 1.0):
                print("Epoch {}, step {}, ".format(epoch, step))
                print('image_names: ', image_names)
                print('captions: ', titles)

            loss = args.standard_loss_factor * standard_loss +  args.kg_loss_factor * kg_loss
            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            # nn.utils.clip_grad_norm_(BCEwithlogis_loss.parameters(), max_norm=0.01)

            optimizer.step()
            # scheduler.step()
            tr_loss += loss.item()
            mil_tr_loss += kg_loss.item()
            st_train_loss += standard_loss.item()
            # mil_step_losses.append(kg_loss.item())
            steps.append(step)
            prev_kg_loss = kg_loss
            # clip.model.convert_weights(model)

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            st_loss_meter.update(standard_loss.item(), count)
            mil_loss_meter.update(kg_loss.item(), count)
            mil_step_losses.append(mil_loss_meter.avg)

            # pbar.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
            # pbar.set_postfix(train_loss=st_loss_meter.avg, lr=get_lr(optimizer))
            sigmoid_pred = F.sigmoid(y_pred) > 0.5 
            correct_tr  += (sigmoid_pred == mil_targets.to(device)).float().sum()
            mat_count += (batch['image'].size(0) * 80)

            print('loss item loss {}, contrastive item loss {}, mil item loss {},'.format(loss.item(), standard_loss.item(), kg_loss.item()))
            pbar.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

            # pbar.set_description(f"Train milCE: {loss.item()}", refresh=True)
           
           
           
            # print('sigmoid pred : ', sigmoid_pred)
            # print('correct tr: ', correct_tr)
  
        plt.plot(steps, mil_step_losses, label='MIL first {} iteration Train Loss'.format(step))
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss vs Steps Curve in Epoch'.format(epoch))
        plt.xticks(np.arange(1, step, 100))

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('../../../KG_Defence/mil/figures/loss_nonapi_mil_steps_epoch_{}.png'.format(epoch))
        plt.close('all')

        tr_loss /= step
        st_train_loss /= step
        mil_tr_loss /= step

        train_losses.append(loss_meter.avg)
        contrastive_train_losses.append(st_loss_meter.avg)
        mil_train_losses.append(mil_loss_meter.avg)
        print("Epoch: {}  tr_loss: {}".format( epoch, loss_meter.avg))
        print("Epoch: {}  contrastive_tr_loss: {}".format( epoch, st_loss_meter.avg))
        print("Epoch: {}  mil_tr_loss: {} with (mil_tr_loss / step): {}".format( epoch, mil_loss_meter.avg, mil_tr_loss))
        print("\n")
        ########################## Accuracy #######################################
        print('correct tr {} after epoch {}'.format(correct_tr, epoch))
        print('Train Accuracy: ', correct_tr / (mat_count), correct_tr /(args.batch_size * 80 * step))


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
                images, titles, names, ids, image_names  = batch['image'], batch['caption'], batch['cat_names'], batch['cat_ids'], batch['image_name']
                
                images = images.to(device) ## shape: 128 x 3 x 224 x 224
                tokenized_titles = tokenizer(titles, padding=True, truncation=True, max_length=CFG.max_length)
            
                logits, targets, y_pred, mil_targets, mil_softmax_targets, min_idx, max_idx = model(images, lst_tokens, tokenized_titles, names, device) ## lst_tokens : 80 * 5 * 100
                
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                standard_loss = (images_loss + titles_loss) / 2
                standard_loss = standard_loss.mean()

                
                kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device)) ## uncomment this line
              
                kg_loss = kg_loss.mean()

                loss = args.standard_loss_factor * standard_loss +  args.kg_loss_factor * kg_loss
                te_loss += loss.item()
                mil_te_loss += kg_loss.item()
                st_test_loss += standard_loss.item()

                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                st_loss_meter.update(standard_loss.item(), count)
                mil_loss_meter.update(kg_loss.item(), count)

                sigmoid_pred = F.sigmoid(y_pred) > 0.5 
                correct_te  += (sigmoid_pred == mil_targets.to(device)).float().sum()
                mat_count += (batch['image'].size(0) * 80)
            
                # pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))
                # pbar.set_postfix(valid_loss=st_loss_meter.avg, lr=get_lr(optimizer))
                pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))

                # pbar.set_description(f"test milCE: {loss.item()}", refresh=True)
           
            te_loss /= step
            st_test_loss /= step
            mil_te_loss /= step

            test_losses.append(loss_meter.avg)
            contrastive_test_losses.append(st_loss_meter.avg)
            mil_test_losses.append(mil_loss_meter.avg)

            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , loss_meter.avg))
            print("Epoch: {}  contrastive_te_loss: {}".format( epoch , st_loss_meter.avg))
            print("Epoch: {}  mil_te_loss: {} and with (mil_te_loss/step): {}".format(epoch , mil_loss_meter.avg, mil_te_loss))
            print('--------------------------------------\n')

            ################################# Accuracy #######################################
            test_acc = correct_te / (mat_count)
            print('Test Accuracy: ', correct_te / (mat_count), correct_te / (args.batch_size * 80 * step))

            # if te_loss < best_loss:
            if test_acc > best_acc:
            # if loss_meter.avg < best_loss:
                # best_loss = te_loss
                # best_loss = loss_meter.avg
                best_acc = test_acc
                if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                    torch.save(model.state_dict(), "../../../KG_Defence/mil/models/_noapi_best_baseline_coco_only_mil.pt")
                elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    torch.save(model.state_dict(), "../../../KG_Defence/mil/models/_noapi_best_baseline_coco_standard.pt")
                elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                    torch.save(model.state_dict(), "../../../KG_Defence/mil/models/_noapi_best_baseline_coco_standard_mil.pt")
                print("Saved Best Model!")
    
    if (args.kg_loss_factor > 0 and args.standard_loss_factor > 0):
        plt.plot(epochs, train_losses, label='Combined Train Loss')
        plt.plot(epochs, test_losses, label='Combined Test Loss')
        plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
        plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
        plt.plot(epochs, mil_train_losses, label='Mil Train Loss')
        plt.plot(epochs, mil_test_losses, label='Mil Test Loss')

    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
        plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
        plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
    
    elif (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
        plt.plot(epochs, mil_train_losses, label='Mil Train Loss')
        plt.plot(epochs, mil_test_losses, label='Mil Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    plt.xticks(np.arange(1, args.epoch, 2))
 
    # Display the plot
    plt.legend(loc='best')
    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_noapi_mil.png')
    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_noapi_contrastive.png')
    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_noapi_contrastive_mil.png')
    
    plt.close('all')




def train_loopv2 (model, tokenizer, train_dataloader, test_dataloader, lst_tokens, optimizer, device, args):
    
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
    BCEwithlogis_loss = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(args.epoch):

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

        loss_meter = AvgMeter()
        st_loss_meter = AvgMeter()
        mil_loss_meter = AvgMeter()
        mat_count = 0
        
        pbar = tqdm(train_dataloader,  total=len(train_dataloader))
        for batch in pbar:
            step+=1
            b_count+=1
            optimizer.zero_grad()

            logits, targets, y_pred, mil_targets, mil_softmax_targets,  min_sim_index, max_sim_index = model (batch, lst_tokens, device)

            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            standard_loss = (images_loss + titles_loss) / 2
            standard_loss = standard_loss.mean()

            kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device)) 
            kg_loss = kg_loss.mean()

            loss = args.standard_loss_factor * standard_loss +  args.kg_loss_factor * kg_loss

            loss.backward()

            optimizer.step()
            # scheduler.step()
            tr_loss += loss.item()
            mil_tr_loss += kg_loss.item()
            st_train_loss += standard_loss.item()
            # mil_step_losses.append(kg_loss.item())
            steps.append(step)
            
            # clip.model.convert_weights(model)

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            st_loss_meter.update(standard_loss.item(), count)
            mil_loss_meter.update(kg_loss.item(), count)
            mil_step_losses.append(mil_loss_meter.avg)

            
            sigmoid_pred = F.sigmoid(y_pred) > 0.5 
            correct_tr  += (sigmoid_pred == mil_targets.to(device)).float().sum()
            mat_count += (batch['image'].size(0) * 80)

            print('loss avg {}, contrastive avg {}, mil avg {}'.format(loss_meter.avg, st_loss_meter.avg, mil_loss_meter.avg))
            # pbar.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
            # pbar.set_postfix(train_loss=st_loss_meter.avg, lr=get_lr(optimizer))
            pbar.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        
          
        if (epoch == 0):
            plt.plot(steps, mil_step_losses, label='MIL nonapi first {} iteration Train Loss'.format(step))
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Epoch vs Steps Curve')
            plt.xticks(np.arange(1, step, 1))
        
            # Display the plot
            plt.legend(loc='best')
            plt.savefig('../../../KG_Defence/mil/figures/loss_nonapi_mil_steps_v2.png')


        tr_loss /= step
     
        train_losses.append(loss_meter.avg)

        print("Epoch: {}  tr_loss: {} with (tr_loss/step) {}".format( epoch , loss_meter.avg, tr_loss))
        print("Epoch: {}  contrastive_tr_loss: {}".format( epoch, st_loss_meter.avg))
        print("Epoch: {}  mil_tr_loss: {} with (mil_tr_loss / step): {}".format( epoch, mil_loss_meter.avg, mil_tr_loss))
        print("\n")
        ########################## Accuracy #######################################
        print('correct tr {} after epoch {}'.format(epoch, correct_tr))
        print('Train Accuracy: ', correct_tr / (mat_count), correct_tr /(args.batch_size * 80 * step))

        model.eval()
        with torch.no_grad():
            step = 0
            loss_meter = AvgMeter()
            st_loss_meter = AvgMeter()
            mil_loss_meter = AvgMeter()

            pbar = tqdm(test_dataloader,  total=len(test_dataloader))
            mat_count = 0
            for batch in pbar:
                step+=1
            
                logits, targets, y_pred, mil_targets, mil_softmax_target,  min_sim_index, max_sim_index = model(batch, lst_tokens, device) ## lst_tokens : 80 * 5 * 100
                
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                standard_loss = (images_loss + titles_loss) / 2
                standard_loss = standard_loss.mean()

                kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device)) ## uncomment this line
              
                kg_loss = kg_loss.mean()

                loss = args.standard_loss_factor * standard_loss +  args.kg_loss_factor * kg_loss
                te_loss += loss.item()
                mil_te_loss += kg_loss.item()
                st_test_loss += standard_loss.item()

                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                st_loss_meter.update(standard_loss.item(), count)
                mil_loss_meter.update(kg_loss.item(), count)

                sigmoid_pred = F.sigmoid(y_pred) > 0.5 
                correct_te  += (sigmoid_pred == mil_targets.to(device)).float().sum()
                mat_count += (batch['image'].size(0) * 80)
            
                # pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))
                # pbar.set_postfix(valid_loss=st_loss_meter.avg, lr=get_lr(optimizer))
                pbar.set_postfix(valid_loss=loss_meter.avg, lr=get_lr(optimizer))


                # te_loss += loss.item()
            
                # count = batch["image"].size(0)
                # loss_meter.update(loss.item(), count)
                # steps.append(step)
                # pbar.set_postfix(valid_loss=loss_meter.avg)
                # pbar.set_description(f"test milCE: {loss.item()}", refresh=True)
              
            te_loss /= step
            st_test_loss /= step
            mil_te_loss /= step

            test_losses.append(loss_meter.avg)
           
            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {} with (te_loss / step)".format( epoch , loss_meter.avg, te_loss))
            print("Epoch: {}  contrastive_te_loss: {}".format( epoch , st_loss_meter.avg))
            print("Epoch: {}  mil_te_loss: {} and with (mil_te_loss/step): {}".format( epoch , mil_loss_meter.avg, mil_te_loss))
            print('--------------------------------------\n')

            ################################# Accuracy #######################################
            print('Test Accuracy: ', correct_te / (mat_count), correct_te / (args.batch_size * 80 * step))
           

            # if te_loss < best_loss:
            if loss_meter.avg < best_loss:
                best_loss = loss_meter.avg
                torch.save(model.state_dict(), "../../../KG_Defence/mil/models/_noapi_best_baseline_coco_standard_mil_v2.pt")
                print("Saved Best Model!")
    

    # plt.plot(epochs, train_losses, label='Contrastive Train Loss')
    # plt.plot(epochs, test_losses, label='Contrastive Test Loss')

    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Epoch vs Loss Curve')
    # plt.xticks(np.arange(1, args.epoch, 2))
 
    # # Display the plot
    # plt.legend(loc='best')
   
    # plt.savefig('../../../KG_Defence/mil/figures/_noapi_loss_standard.png')
    # plt.close('all')

    if (args.kg_loss_factor > 0 and args.standard_loss_factor > 0):
        plt.plot(epochs, train_losses, label='Combined Train Loss')
        plt.plot(epochs, test_losses, label='Combined Test Loss')
        plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
        plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
        plt.plot(epochs, mil_train_losses, label='Mil Train Loss')
        plt.plot(epochs, mil_test_losses, label='Mil Test Loss')

    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
        plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
        plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')
    
    elif (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
        plt.plot(epochs, mil_train_losses, label='Mil Train Loss')
        plt.plot(epochs, mil_test_losses, label='Mil Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    plt.xticks(np.arange(1, args.epoch, 2))
 
    # Display the plot
    plt.legend(loc='best')
    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_noapi_mil_v2.png')
    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_noapi_contrastive_v2.png')
    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_noapi_contrastive_mil_v2.png')
    
    plt.close('all')


def extract_caption_image(annFile, instancefile):
    coco = COCO(annFile)

    img_ids = random.choice(coco.getImgIds(imgIds=coco.getImgIds()))
    
    print('img_ids: ',img_ids)
    image_file = coco.loadImgs(img_ids)[0]['file_name']

    ann_ids = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(ann_ids)
    
    target = [ann['caption'] for ann in anns]
  
    caption = random.choice(target)

    coco = COCO(instancefile)
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(ann_ids)
    

    category_names = [coco.cats[ann['category_id']]['name'] for ann in anns]


    return caption, list(set(category_names)), image_file


def get_false_negative(model_path, annfile, instancefile, image_transforms, classes, lst_tokens, tokenizer, kg_dict, args, device):

    torch.manual_seed(42)
    model = clip_model(classes=classes, args=args).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    false_positive_count = 0
    false_negative_count = 0
    total = 0

    coco = COCO(annfile)
    ids = list(coco.imgs.keys())
    coco1 = COCO(instancefile)
    
    # true_positive_dict_all = {class_name: 0 for class_name in classes}
    # true_negative_dict_all =  {class_name: 0 for class_name in classes}
    # false_negative_dict_all =  {class_name: 0 for class_name in classes}
    # false_positive_dict_all =  {class_name: 0 for class_name in classes}

    lst_mil_targets = []
    lst_pred_int = []

    for id in ids:
        img_ids = coco.getImgIds(imgIds=id)
        
        image_file = coco.loadImgs(img_ids)[0]['file_name']
      
        ann_ids = coco.getAnnIds(imgIds=img_ids)
        anns = coco.loadAnns(ann_ids)

        target = [ann['caption'] for ann in anns]
    
        title = random.choice(target)

      
        ann_ids = coco1.getAnnIds(imgIds=img_ids)
        anns = coco1.loadAnns(ann_ids)
        

        category_names = [coco1.cats[ann['category_id']]['name'] for ann in anns]
        names = list(set(category_names))
        
        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        image = Image.open(os.path.join(root, image_file)).convert('RGB')
        image = image_transforms(image)
        # title, names, image = extract_caption_image(annfile, instancefile)
        
        title = [title]
        names = [names]
        # print(title, names, os.path.join(root, image_file))
        image = image.unsqueeze(0) 
        image = image.to(device) ## shape: 128 x 3 x 224 x 224
        tokenized_title = tokenizer(title, padding=True, truncation=True, max_length=CFG.max_length)    
        

        logits, targets, y_pred, mil_targets, mil_softmax_targets, lst_min_index, lst_max_index = model(image, lst_tokens, tokenized_title, names, device) 

        sigmoid_pred = (F.sigmoid(y_pred) > 0.5)
        sigmoid_pred_int = sigmoid_pred.int()

        lst_pred_int.append(sigmoid_pred_int[0].cpu().detach().numpy())
        lst_mil_targets.append(mil_targets[0].cpu().detach().numpy())
        # positive_dict = {}
        # negative_dict = {}
        # false_negative_dict = {}
        # false_positive_dict = {}
    

        # for i, value in enumerate(y_pred[0]):
        #     if (F.sigmoid(value) > 0.5):
        #         positive_dict[classes[i]] = kg_dict[classes[i]][lst_max_index[0][i]]
        #     else:
        #         negative_dict[classes[i]] = kg_dict[classes[i]][lst_min_index[0][i]]
            
        #     if (F.sigmoid(value) > 0.5 and mil_targets[0][i] == 0):
        #         false_positive_dict[classes[i]] = kg_dict[classes[i]][lst_max_index[0][i]]
        #         false_positive_dict_all[classes[i]] +=1
        #     elif(F.sigmoid(value) <= 0.5 and mil_targets[0][i] == 1):
        #         false_negative_dict[classes[i]] = kg_dict[classes[i]][lst_min_index[0][i]]
        #         false_negative_dict_all[classes[i]] +=1
            
        #     elif (F.sigmoid(value) > 0.5 and mil_targets[0][i] == 1):
        #         true_positive_dict_all[classes[i]]+=1
        #     elif (F.sigmoid(value) <= 0.5 and mil_targets[0][i] == 0):
        #         true_negative_dict_all[classes[i]]+=1
            

        correct  = (sigmoid_pred == mil_targets.to(device)).float().sum()
        mat_count = (image.size(0) * 80)

      
        correct_lst = (sigmoid_pred == mil_targets.to(device)).float()


        # len_false_negative = len(false_negative_dict)
        # len_false_positive = len(false_positive_dict)
        

        # if (len_false_negative > 0):
        #     false_negative_count+=1
        #     # print('image_name: ', os.path.join(root, image_file))
        #     # print('False negative dict: ', false_negative_dict)
        #     # print('------------------------------------------------')

        # elif(len_false_positive > 0):
        #     false_positive_count+=1
        #     print('image_name: ', os.path.join(root, image_file))
        #     print('False positive dict: ', false_positive_dict)
            # print('-------------------------------------------')
        
        total+=1
      
       
    

    # print('False_positive_rate:  {}'.format(false_positive_count/total))
    # print('False_negative_rate: {}'.format(false_negative_count/total))
    confusion_mat_per_label =  multilabel_confusion_matrix(np.array(lst_mil_targets), np.array(lst_pred_int))
    print('-------------- confusion_mat_per_label ------------------- \n')
    print(confusion_mat_per_label)
    for i, item in enumerate(confusion_mat_per_label):
        print('class name {}'.format(classes[i]))
        print(item)
    print('------------------------------------')
    print('Total {}'.format(total))
    max_false_positive = confusion_mat_per_label[:, 0, 1].max()
    max_false_negative = confusion_mat_per_label[:, 1, 0].max()
    max_true_positive = confusion_mat_per_label[:, 0, 1].max()
    max_true_negative = confusion_mat_per_label[:, 0, 1].max()

    max_false_positive_idx = confusion_mat_per_label[:, 0, 1].argmax()
    max_false_negative_idx = confusion_mat_per_label[:, 1, 0].argmax()
    max_true_positive_idx = confusion_mat_per_label[:, 0, 1].argmax()
    max_true_negative_idx = confusion_mat_per_label[:, 0, 1].argmax()

    print("False pos, False neg, True pos, True neg, {}, {}, {}, {}".format(max_false_positive, max_false_negative, max_true_positive, max_true_negative))
    print("False pos idx, False neg idx, True pos idx, True neg idx, {}, {}, {}, {}".format(max_false_positive_idx, max_false_negative_idx, max_true_positive_idx, max_true_negative_idx))
    print('classes: {}, {}, {}, {}'.format(classes[max_false_positive_idx], classes[max_true_negative_idx], classes[max_false_positive_idx], classes[max_false_negative_idx]))
    

    


def extract_similarity_per_image (model_path, annfile, instancefile, image_transforms, classes, lst_tokens, tokenizer, kg_dict, args, device):

    
    model = clip_model(classes=classes, args=args).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    # print('model summary')
    # print(model)
    
    title, names, image_file = extract_caption_image(annfile, instancefile)
    root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
    image = Image.open(os.path.join(root, image_file)).convert('RGB')
    image = image_transforms(image)
    # title, names, image = extract_caption_image(annfile, instancefile)
    
    title = [title]
    names = [names]
    print(title, names, os.path.join(root, image_file))
    image = image.unsqueeze(0) 
    image = image.to(device) ## shape: 128 x 3 x 224 x 224
    tokenized_title = tokenizer(title, padding=True, truncation=True, max_length=CFG.max_length)    
    

    logits, targets, y_pred, mil_targets, mil_softmax_targets, lst_min_index, lst_max_index = model(image, lst_tokens, tokenized_title, names, device) 

    sigmoid_pred = F.sigmoid(y_pred) > 0.5 

    positive_dict = {}
    negative_dict = {}
   

    for i, value in enumerate(y_pred[0]):
        if (F.sigmoid(value) > 0.5):
           positive_dict[classes[i]] = kg_dict[classes[i]][lst_max_index[0][i]]
        else:
            negative_dict[classes[i]] = kg_dict[classes[i]][lst_min_index[0][i]]
        


    correct  = (sigmoid_pred == mil_targets.to(device)).float().sum()
    mat_count = (image.size(0) * 80)

    print('correct {}, mat_count {} percentage {} %'.format(correct, mat_count, correct/mat_count * 100.0))
    print('---------------------------\n')
    correct_lst = (sigmoid_pred == mil_targets.to(device)).float()


    print('yPred: ',y_pred)
    print('mil_targets: ', mil_targets)
    print('mil_softmax_targets: ', mil_softmax_targets)
     
    print('-------------------------------')

    print('correct lst: ', correct_lst)
    print('sigmoid_pred: ', sigmoid_pred)

    print('positive_dict: ', positive_dict)
    print('negative_dict: ', negative_dict)

            



if __name__ == '__main__':
    """ MIL: Most similar positive {KG_i} from a class 
       and easiest ngeative (furthest distance) per
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
    parser.add_argument('--run_v2', default='no', type=str)

    args = parser.parse_args()
    print(args)
    
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    kg = kg_load(args)
    classes, kg_dict = kg.load_kg()
    if (args.clip_openai == 'yes'):
        lst_tokens = kg.get_kg_emb(kg_dict)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        lst_tokens = kg.kgemb_bert(kg_dict, tokenizer)

    ## load model:
    if (args.run_v2 == 'no'):
        model = clip_model(classes=classes, args=args).to(device)
    else:
        model = clip_modelv2(classes=classes, args=args).to(device)
        model = clip_modelv2(classes=classes, args=args).to(device)

    ## load dataloader

    if (args.dataset == 'coco'):
        image_dir = '../../../KG_Defence/datasets/coco/images/train2017'
        valid_dir = '../../../KG_Defence/datasets/coco/images/val2017'
        annfile = '../../../KG_Defence/datasets/coco/annotations/captions_train2017.json'
        val_annfile = '../../../KG_Defence/datasets/coco/annotations/captions_val2017.json'

        instance_file = '../../../KG_Defence/datasets/coco/annotations/instances_train2017.json'
        val_instance_file = '../../../KG_Defence/datasets/coco/annotations/instances_val2017.json'
        preprocess = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((CFG.size, CFG.size))])
        
        if (args.run_v2 == 'no'):
            train_dataloader, test_dataloader, train_dataset, test_dataset = coco_loader(image_dir, valid_dir,  annfile, val_annfile, instance_file, val_instance_file, preprocess, target_transform=None, args = args)
        else:
            train_path = '/home/alvi/KG_Defence/datasets/coco/csv_train.csv'
            val_path = '/home/alvi/KG_Defence/datasets/coco/csv_val.csv'

            image_root_train = '/home/alvi/KG_Defence/datasets/coco/images/train2017'
            image_root_val = '/home/alvi/KG_Defence/datasets/coco/images/val2017'

            train_dataloader = coco_loaderv2(image_root_train, train_path, tokenizer, preprocess, args)
            test_dataloader = coco_loaderv2(image_root_val, val_path, tokenizer, preprocess, args)

    ## load optimizer

    if (args.optim == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=1e-8,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*args.batch_size)

    elif (args.optim == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*args.batch_size)

    elif (args.optim == 'adamw'):
        optimizer = torch.optim.AdamW(model.parameters(),  lr = 1e-7, weight_decay=0.2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
            )
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCHS)

    
    ## run train loop 
    if (args.run_v2 == 'no'):
        # train_loop (model, tokenizer, train_dataloader, test_dataloader, lst_tokens, optimizer, device, args)
        model_path = '/home/alvi/KG_Defence/mil/models/_noapi_best_baseline_coco_standard_mil.pt'
        ################################ image_file = '000000000872.jpg'
        test_annfile = '../../../KG_Defence/datasets/coco/annotations/captions_val2017.json'
        test_instancefile = '../../../KG_Defence/datasets/coco/annotations/instances_val2017.json'
        # extract_similarity_per_image (model_path, test_annfile, test_instancefile, preprocess, classes, lst_tokens, tokenizer, kg_dict, args, device)
        get_false_negative(model_path, test_annfile, test_instancefile, preprocess, classes, lst_tokens, tokenizer, kg_dict, args, device)
    else:
        train_loopv2 (model, tokenizer, train_dataloader, test_dataloader, lst_tokens, optimizer, device, args)
    