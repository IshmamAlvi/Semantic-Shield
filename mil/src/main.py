
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


def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model_path, device=device)
    return model, preprocess, device

def load_dataset(preprocess, classes, kg, batch_size=128, model=None, device=None, args=None):

    if (args.dataset == 'cifar'):
        trainset = CIFAR10(root='./data', train=True, download=True, transform =preprocess)
        testset = CIFAR10(root='./data', train=False, download=True, transform =preprocess)
    elif (args.dataset == 'imagenet'):
        trainset = ImageNet(root='./data', split='train', transform =preprocess)
        testset = ImageNet(root='./data', split='val', transform =preprocess)


    train_dataset = poisoned_dataset(trainset, preprocess, classes, kg, model, device, args)
    test_dataset = poisoned_dataset(testset, preprocess, classes, kg, model, device, args)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_dataloader, test_dataloader


def model_train_mil(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, EPOCHS, args):
    best_te_loss = 1e5
    best_ep = -1
    for epoch in range(EPOCHS):
        step = 0
        tr_loss = 0
        model.train()
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            images, texts, title, index = batch 
            # for i in (range(batch_size)):
                
            #     step+=1
            #     optimizer.zero_grad()
               
            
                # img = images[i].unsqueeze(0).to(device)
                # txt = texts[i].to(device)
                # print("before shape: ", txt.shape, img.shape)
                # # txt = torch.unsqueeze(txt, dim = 0)
                # txt1, txt2, txt3 = txt[0].unsqueeze(0), txt[1].unsqueeze(0), txt[2].unsqueeze(0)
                # print("after shape: ", txt1.shape, txt2.shape, txt3.shape)
                # img_emb = model.encode_image(img)
                # txt_emb = model.encode_text(txt1)
              
                # print("embedding")
                # # print(img_emb, txt_emb)
                # print(img_emb.shape, txt_emb.shape)
                # break

            
def model_train_baseline(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, EPOCHS, args):

    best_te_loss = 1e5
    best_ep = -1
    for epoch in range(EPOCHS):
        step = 0
        tr_loss = 0
        model.train()
        pbar = tqdm(train_dataloader)
        for batch in pbar:

            step+=1
            optimizer.zero_grad()
            images, texts, title, index = batch 
        
            img = images.to(device)
            txt = texts.to(device)
            txt = torch.squeeze(txt)

            print(img.shape, txt.shape)
            img_emb = model.encode_image(img)
            txt_emb = model.encode_text(txt)
            # print(img)
            # print("embedding")
            print(img_emb, txt_emb)
           
            logits_per_image, logits_per_text = model(img, txt)

            ground_truth = torch.arange(img.shape[0]).to(device)
            # print('input_size, gt: ', logits_per_image, logits_per_text, ground_truth)
      
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            tr_loss += total_loss.item()

            if (math.isnan(total_loss.item())):
                print (img_emb)
                print(txt_emb)
                print(img)

           
        
            optimizer.step()
            scheduler.step()
            
            pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
        tr_loss /= step
      

        step = 0
        te_loss = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            for batch in test_pbar:
                step += 1
                images, texts, title, index = batch 
        
                img = images.to(device)
                txt = texts.to(device)
                txt = torch.squeeze(txt)
              

                logits_per_image, logits_per_text = model(img, txt)
                ground_truth = torch.arange(img.shape[0]).to(device)

                total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                te_loss += total_loss.item()
                test_pbar.set_description(f"test batchCE: {total_loss.item()}", refresh=True)
            te_loss /= step
            
        if te_loss < best_te_loss:
            best_te_loss = te_loss
            best_ep = epoch
            torch.save(model.state_dict(), "/home/grads/alvi/KG_defense/mil/src/models/best_model.pt")
        print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")
    
    # torch.save(model.state_dict(), "/home/grads/alvi/KG_defense/mil/src/models/last_model.pt")
            

        


    

if __name__=="__main__":

    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    kg_dict = defaultdict()
   
    kg_dict[0] = ['Wingspan: long or short', 'Fuselage shape: aerial', 'Jet engines']
    kg_dict[1] = ['Shape: sedan, or suv', 'color: red, blue', 'engine: small/ large' ]
    kg_dict[2] = ['Can fly', 'colorful', 'small bill']
    kg_dict[3] = ['Fluffy fur', 'sharp eyes', 'sharp teeth']
    kg_dict[4] = ['run fast', 'spotty skin', 'large eyes']
    kg_dict[5] = ['Dense fur', 'Sharp claws', 'run faster']
    kg_dict[6] = ['Webbed feet for swimming and jumping', 'Moist, smooth skin.', 'Bulging eyes on the sides of the head.']                         
    kg_dict[7] = ['Strong and muscular body', 'Long, flowing mane and tail.', 'Hooves for running and walking.']
    kg_dict[8] = ['Sturdy hull for navigating through water.', 'Multiple decks for accommodation and activities.', 'Navigational equipment such as radar and compass.']
    kg_dict[9] = ['Large cargo bed for transporting goods.', 'Robust suspension for handling heavy loads.', 'Powerful engine for towing and hauling.']
    
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('--loss', default='cross_entropy', type=str)
    parser.add_argument('--model_path', default='ViT-B/32', type=str)
    parser.add_argument ('--baseline', default='baseline_kg', type=str)
    parser.add_argument('--dataset', default='imagenet', type=str)

    args = parser.parse_args()
    print(args)

    model, preprocess, device = load_model(args)
    EPOCHS = 30
    train_dataloader, test_dataloader = load_dataset(preprocess, classes, kg_dict, batch_size=128, args=args)
    optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCHS)
    batch_size = 128
    if args.loss == "cross_entropy":
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
    elif args.loss == 'margin_loss':
        # Set the contrastive loss margin and create a distance function
        margin = 0.5  # Adjust the margin according to your task
        loss_img = nn.MarginRankingLoss(margin=margin)
        loss_txt = nn.MarginRankingLoss(margin=margin)
    
    if (args.baseline == 'baseline_kg'):
        model_train_baseline(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, EPOCHS, args)
    else:
        model_train_mil(model, device, train_dataloader, test_dataloader, optimizer, batch_size, loss_img, loss_txt, scheduler, EPOCHS, args)
    
        
        
        
    



   

    









    




    



    
    