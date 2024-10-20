import torch
from transformers import BertTokenizer, BertModel
import nltk
# nltk.download('punkt')
import os
import clip
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
from main import load_dataset, load_model, model_train_baseline
from kg import kg_load
import itertools
from utils import make_train_valid_dfs, build_loaders, coco_loader
from transformers import DistilBertTokenizer
from CLIPModel import CLIPModel
from train_flickr import train_clip_flickr

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



def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def model_train_mil_batch(model, device, images, texts, test_images, test_texts, optimizer, batch_size, loss_img, loss_txt, epochs):
    
    for epoch in range(epochs):
        step = 0
        tr_loss = 0
        model.train()
        for image, text in zip(images, texts):
            step+=1
            optimizer.zero_grad()
            image = image.to(device)
            text = text.to(device)
            text = torch.squeeze(text)
            logits_per_image, logits_per_text = model(image, text)

            ground_truth = torch.arange(image.shape[0]).to(device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            tr_loss += total_loss.item()

            optimizer.step()
            scheduler.step()
        
            print("train BCE: ", total_loss.item())
        
        tr_loss /= step

        step = 0
        te_loss = 0
        with torch.no_grad():
            model.eval()
            for image, text in zip(test_images, test_texts):
                image = image.to(device)
                text = text.to(device)
                text = torch.squeeze(text)
                logits_per_image, logits_per_text = model(image, text)

                ground_truth = torch.arange(image.shape[0]).to(device)
                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                total_loss.backward()
                te_loss += total_loss.item()

                print("test loss: ", total_loss.item())
            
            te_loss/= step
        
        if te_loss < best_te_loss:
                best_te_loss = te_loss
                best_ep = epoch
                torch.save(model.state_dict(), "/home/grads/alvi/KG_defense/mil/src/models/best_model_mil.pt")
        print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")

def model_train_mil(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, args):

    best_te_loss = 1e5
    best_ep = -1
    for epoch in range(args.epoch):
        step = 0
        tr_loss = 0
        model.train()
        # pos_index = []
        b_count = 0
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            step+=1
            b_count+=1
            optimizer.zero_grad()
            images, texts, titles, indices = batch 
            
            imgs = images.to(device)
            txts = texts.to(device)

            img_embs = txt_embs = []
            # pos_index = []
            stacked_tensor = torch.empty(0)
            target = []
            for img, txt, index_i in zip(imgs, txts, indices):
                pos_sim = []
                pos_index = []
                txt = torch.squeeze(txt)
                img = torch.unsqueeze(img, dim=0)
                img_emb = model.encode_image(img)
                txt_emb = model.encode_text(txt)
               
                sim = img_emb @ txt_emb.T
                
                max_i = torch.argmax(sim, dim = 1)
                
                pos_index.append(max_i)
                pos_sim.append(torch.max(sim))
                img_embs.append(img_emb)
                txt_embs.append(txt_emb)

            
                neg_index = []
                neg_sim = []
                cnt = t = 0
                class_indices = [None] * 10 
                index_i = index_i.item()
                class_indices[index_i] = index_i

                for image, text, index_j in zip(imgs, txts, indices):
                    t+=1
                    index_j = index_j.item()
                    text = torch.squeeze(text)
                    if (index_j in class_indices):
                        cnt+=1
                        continue

                    class_indices[index_j] = index_j
                    txt_emb = model.encode_text(text)
                    sim = img_emb @ txt_emb.T
                    min_index = torch.argmin(sim, dim = 1)
                    neg_sim.append(torch.min(sim))
                    neg_index.append(min_index)
                
                pos_sim_tensor = torch.tensor(pos_sim)
                neg_sim_tensor = torch.tensor(neg_sim)
                # print(pos_sim_tensor.shape, neg_sim_tensor.shape, len(pos_sim), pos_sim)
                single_row = concat(pos_sim_tensor.unsqueeze(0), neg_sim_tensor.unsqueeze(0), index_i)
                target.append(index_i)
                stacked_tensor = torch.cat((stacked_tensor, single_row), dim=0)
            y_pred = stacked_tensor
            # y_pred = torch.softmax(stacked_tensor, dim=1)    
            target_class = torch.tensor(target, dtype=torch.float16)
            # criterion = nn.CrossEntropyLoss()
            print(y_pred.shape, target_class.shape)
            print('---------------')
            print(y_pred, target_class)
            loss = loss_img(y_pred, target_class.long())
            loss.requires_grad = True
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"train milCE: {loss.item()}", refresh=True)
        
        tr_loss /= step
        
        print("tr_loss {} after batch: {}".format( tr_loss , epoch))

        
        step = 0
        te_loss = 0
        with torch.no_grad():
            model.eval()
            pbar = tqdm (test_dataloader)
            for batch in pbar:
                step+=1
                images, texts, titles, indices = batch

                imgs = images.to(device)
                txts = texts.to(device)

                img_embs = txt_embs = []
                stacked_tensor = torch.empty(0)
                target = []
                for img, txt, index_i in zip(imgs, txts, indices):
                    pos_sim = []
                    pos_index = []
                    txt = torch.squeeze(txt)
                    img = torch.unsqueeze(img, dim=0)
                    img_emb = model.encode_image(img)
                    txt_emb = model.encode_text(txt)
                    
                    
                    sim = img_emb @ txt_emb.T
                    
                    max_i = torch.argmax(sim, dim = 1)
                  
                    pos_index.append(max_i)
                    pos_sim.append(torch.max(sim))
                    img_embs.append(img_emb)
                    txt_embs.append(txt_emb)

                
                    neg_index = []
                    neg_sim = []
                    cnt = t = 0
                    class_indices = [None] * 10 
                    index_i = index_i.item()
                    class_indices[index_i] = index_i

                    for image, text, index_j in zip(imgs, txts, indices):
                        t+=1
                        index_j = index_j.item()
                        text = torch.squeeze(text)
                        if (index_j in class_indices):
                            cnt+=1
                            continue

                        class_indices[index_j] = index_j
                        txt_emb = model.encode_text(text)
                        sim = img_emb @ txt_emb.T
                        min_index = torch.argmin(sim, dim = 1)
                        neg_sim.append(torch.min(sim))
                        neg_index.append(min_index)
                    
                    pos_sim_tensor = torch.tensor(pos_sim)
                    neg_sim_tensor = torch.tensor(neg_sim)
                    # print(pos_sim_tensor.shape, neg_sim_tensor.shape, len(pos_sim), pos_sim)
                    print('index: ', index_i, ' neg sim tensor: ', neg_sim_tensor.unsqueeze(0).shape)
                    single_row = concat(pos_sim_tensor.unsqueeze(0), neg_sim_tensor.unsqueeze(0), index_i)
                    target.append(index_i)
                    stacked_tensor = torch.cat((stacked_tensor, single_row), dim=0)
                y_pred = stacked_tensor
                # y_pred = torch.softmax(stacked_tensor, dim=1)    
                target_class = torch.tensor(target, dtype=torch.float16)
                # criterion = nn.CrossEntropyLoss()
               
                loss = loss_img(y_pred, target_class.long())
                
                te_loss += loss.item()
          

            pbar.set_description(f"test milCE: {loss.item()}", refresh=True)
        
        te_loss /= step
        print("te_loss {} after batch: {}".format( te_loss , epoch))

        if (te_loss < best_te_loss):
            best_te_loss = te_loss
            best_ep = epoch
            torch.save(model.state_dict(), "/home/grads/alvi/KG_defense/mil/src/models/best_model_mil.pt")
        print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")
    


def train_optimized(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, lst_tokens, args):

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

        pbar = tqdm(train_dataloader,  total=len(train_dataloader))
        for batch in pbar:
            step+=1
            b_count+=1
            # print('batch count ', b_count)
            optimizer.zero_grad()
            images, titles, names, ids, image_names  = batch['image'], batch['caption'], batch['cat_names'], batch['cat_ids'], batch['image_name']
            # print('cat_names: ', names[:5])
            # print('image names: ', image_names[:3])
            imgs = images.to(device) ## shape: 128 x 3 x 224 x 224
        
            titles = clip.tokenize(titles).to(device) # 128 x 77
            # titles = titles.squeeze(1)
            lst_tokens = lst_tokens.to(device)  ## shape 10 x 3 x 77

            logits, targets, y_pred, mil_targets, mil_softmax_targets = model(imgs, lst_tokens, titles, names, ids)
            
        
            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            standard_loss = (images_loss + titles_loss) / 2
            standard_loss = standard_loss.mean()

            ground_truth = torch.arange(imgs.size(0)).to(device)
            # standard_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            
    
            # kg_loss = loss_img(y_pred, indices.to(device))
            kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device)) ## uncomment this line
            # kg_loss1 = BCEwithlogis_loss(y_pred, mil_targets.to(device))
            # kg_loss2 = BCEwithlogis_loss(y_pred.T, mil_targets.to(device).T)
            # kg_loss1 = BCE_loss(mil_softmax_targets, mil_targets.to(device) )
            # kg_loss2 = BCE_loss(mil_softmax_targets.T, mil_targets.to(device).T)
            # kg_loss = cross_entropy(y_pred, mil_targets.to(device), reduction='none')
            kg_loss = kg_loss.mean()

            loss = args.standard_loss_factor * standard_loss  + args.kg_loss_factor * kg_loss
            loss.backward()
           
            # convert_models_to_fp32(model)
            optimizer.step()
            # scheduler.step()
            tr_loss += loss.item()
            st_train_loss += standard_loss.item()
            mil_tr_loss += kg_loss.item()
            mil_step_losses.append(kg_loss.item())
            steps.append(step)
            # clip.model.convert_weights(model)
            pbar.set_description(f"Train milCE: {loss.item()}", refresh=True)
            sigmoid_pred = F.sigmoid(y_pred) > 0.5 
            correct_tr  += (sigmoid_pred == mil_targets.to(device)).float().sum()
            # print('step', step)
            print('sigmoid : ',sigmoid_pred)
            # print(mil_targets)
            # print('correct_tr in batch: ',correct_tr)
     

   
        plt.plot(steps, mil_step_losses, label='MIL first {} iteration Train Loss'.format(step))
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Epoch vs Steps Curve')
        plt.xticks(np.arange(1, step, 1))
    
        # Display the plot
        plt.legend(loc='best')
        plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_mil_steps.png')


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
        print( correct_tr)
        print(len(train_dataset))
        print('Train Accuracy: ', correct_tr / (args.batch_size * lst_tokens.shape[0] * step))
     
        # break
        
        model.eval()
        with torch.no_grad():
            step = 0
            pbar = tqdm(test_dataloader,  total=len(test_dataloader))
            for batch in pbar:
                step+=1
                images, titles, names, ids = batch['image'], batch['caption'], batch['cat_names'], batch['cat_ids']
                
                imgs = images.to(device) ## shape: 128 x 3 x 224 x 224
               
                titles = clip.tokenize(titles).to(device)
                # titles = titles.squeeze(1)
                lst_tokens = lst_tokens.to(device)  ## shape 10 x 3 x 77
                
                logits, targets, y_pred, mil_targets, mil_softmax_targets = model(imgs, lst_tokens, titles, names, ids)

                ground_truth = torch.arange(imgs.size(0)).to(device)
                # standard_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2
         
                # CE_loss = nn.CrossEntropyLoss(reduction='none')
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                standard_loss = (images_loss + titles_loss) / 2
                standard_loss = standard_loss.mean()
             
                # kg_loss = loss_img(y_pred, indices.cuda())
                kg_loss = BCEwithlogis_loss(y_pred, mil_targets.to(device)) ## uncomment this line
                # kg_loss1 = BCE_loss(y_pred, mil_softmax_targets)
            
                # kg_loss2 = BCE_loss(y_pred.T, mil_softmax_targets.T)
                kg_loss = kg_loss.mean()

                loss = args.standard_loss_factor * standard_loss + args.kg_loss_factor * kg_loss
                te_loss += loss.item()
                st_test_loss += standard_loss.item()
                mil_te_loss += kg_loss.item()


                pbar.set_description(f"test milCE: {loss.item()}", refresh=True)
                sigmoid_pred = F.sigmoid(y_pred) > 0.5 
                correct_te  += (sigmoid_pred == mil_targets.to(device)).float().sum()
            
            te_loss /= step
            st_test_loss /= step
            mil_te_loss /= step

            test_losses.append(te_loss)
            contrastive_train_losses.append(st_test_loss)
            mil_test_losses.append(mil_te_loss)

            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , te_loss))
            print("Epoch: {}  contrastive_te_loss: {}".format( epoch , st_test_loss))
            print("Epoch: {}  mil_te_loss: {}".format( epoch , mil_te_loss))
            print('--------------------------------------\n')

            ########################## Accuracy #######################################
            print('Test Accuracy: ', correct_te / (args.batch_size * lst_tokens.shape[0] * step))
     
            
            if (args.kg_loss_factor == 1.0 and args.standard_loss_factor == 0.0):
                checkpoint_path = '../../../KG_Defence/mil/src/models/checkpoints/checkpoint_coco_only_mil_epoch{}.pth'.format(epoch)
            elif (args.standard_loss_factor == 1.0 and args.kg_loss_factor == 0.0):
                checkpoint_path = '../../../KG_Defence/mil/src/models/checkpoints/checkpoint_coco_standard_epoch{}.pth'.format(epoch)
            elif (args.standard_loss_factor > 0.0 and  args.kg_loss_factor > 0.0):
                checkpoint_path = '../../../KG_Defence/mil/src/models/checkpoints/checkpoint_coco_standard_mil_epoch{}.pth'.format(epoch)

            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Add any other information you want to save
        }
            torch.save(checkpoint, checkpoint_path)

            if te_loss < best_loss:
                best_loss = te_loss
                if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                    torch.save(model.state_dict(), "../../../KG_Defence/mil/models/best_baseline_coco_only_mil.pt")
                elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    torch.save(model.state_dict(), "../../../KG_Defence/mil/models/best_baseline_coco_standard.pt")
                elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                    torch.save(model.state_dict(), "../../../KG_Defence/mil/models/best_baseline_coco_standard_mil.pt")
                print("Saved Best Model!")
        
        # scheduler.step(te_loss)
    
    plt.plot(epochs, train_losses, label='Combined Train Loss')
    plt.plot(epochs, test_losses, label='Combined Test Loss')

    plt.plot(epochs, contrastive_train_losses, label='Contrastive Train Loss')
    plt.plot(epochs, contrastive_test_losses, label='Contrastive Test Loss')

    plt.plot(epochs, mil_train_losses, label='Mil Train Loss')
    plt.plot(epochs, mil_test_losses, label='Mil Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    plt.xticks(np.arange(1, args.epoch, 2))
 
    # Display the plot
    plt.legend(loc='best')
    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_mil.png')
    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_standard.png')
    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
        plt.savefig('../../../KG_Defence/mil/figures/loss_clipapi_standard_mil.png')


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, lst_tokens, args):

    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
    
        b = {}
        b['image'] = batch['image']
        b['input_ids'] = batch['input_ids'].squeeze(1)
        b['attention_mask'] = batch['attention_mask'].squeeze(1)
        # print(b)
        # print(b['image'].shape, b['input_ids'].shape, b['attention_mask'].shape)
        loss = model(b, lst_tokens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if step == "batch":
        lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def valid_epoch(model, valid_loader):

    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:

        b = {}
        b['image'] = batch['image']
        b['input_ids'] = batch['input_ids'].squeeze(1)
        b['attention_mask'] = batch['attention_mask'].squeeze(1)
        loss = model(b)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter



def train_loop (model, optimizer, lr_scheduler, train_loader, valid_loader, step, lst_tokens, args):
     
    best_loss = float('inf')
    train_losses = []
    valid_losses = []
    print('total epochs: ',CFG.epochs)
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, lst_tokens, args)
        train_losses.append(train_loss.avg)

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            valid_losses.append(valid_loss.avg)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "/home/grads/alvi/KG_Defence/mil/models/best_baseline.pt")
            print("Saved Best Model!")
        
        # lr_scheduler.step(valid_loss.avg)
    
    epochs = np.arange(CFG.epochs)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss Curve')
    # plt.show()
    plt.xticks(np.arange(0, CFG.epochs, 2))
 
    # Display the plot
    plt.legend(loc='best')
    plt.savefig('/home/grads/alvi/KG_Defence/mil/figures/loss.png')

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

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

    args = parser.parse_args()
    print(args)

    # EPOCHS = args.epoch
    kg = kg_load(args)
    classes, kg_dict = kg.load_kg()
    if (args.clip_openai == 'yes'):
        lst_tokens = kg.get_kg_emb(kg_dict)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        lst_tokens = kg.kgemb_bert(kg_dict, tokenizer)

    if (args.clip_openai == 'yes'):

        ## This is openai clip model
        model, preprocess, device = load_model(args)

        # These are hugguingface clip model
        # from transformers import CLIPModel, CLIPProcessor
        # model_id = 'google/vit-base-patch32-224-in21k'
        # # # model_id = 'openai/clip-vit-base-patch32'
        # model = CLIPModel.from_pretrained(model_id)
        # preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
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
        optimizer = optim.Adam(model.parameters(), lr=1e-8,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0001)
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

        elif(args.baseline == 'baseline_mil'):
            #  model_train_mil_batch(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, args.epoch, args)
            # train_clip_flickr(model, device, train_dataloader, test_dataloader, optimizer, loss_img, loss_txt, scheduler, lst_tokens, args)
            train_optimized(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, lst_tokens, args)
    elif (args.clip_openai == 'no'):

            step = "batch"
            params = [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
            optimizer = torch.optim.AdamW(params, weight_decay=0.)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
            )
            args.len_trainset = len(train_dataset)
            args.len_testset = len(test_dataset)
            train_loop (model, optimizer, scheduler, train_dataloader, test_dataloader, step, lst_tokens, args)





    
    
    


