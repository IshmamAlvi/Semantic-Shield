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
from utils import make_train_valid_dfs, build_loaders, coco_loader, get_transforms, build_loaders_aug, build_loaders_abl
from transformers import DistilBertTokenizer


import torch.nn.functional as F
import matplotlib.pyplot as plt
from clip_vit.clipvit_model import clip_model, cross_entropy, clip_modelv2

from pycocotools.coco import COCO
from utils import get_transforms

# from clip_vit.utils import AvgMeter, get_lr
from clip_vit.config import CFG
from clip_vit.utils import AvgMeter, get_lr
from clip_vit.clipvitabl_baseline import clipvitabl_baseline
import pickle

def train_isolation (model, train_dataloader_isolation, test_dataloader_isolation, optimizer, scheduler, device, args):
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

    for epoch in range(start_epoch, args.isolation_epoch):

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
        
      
        pbar = tqdm(train_dataloader_isolation,  total=len(train_dataloader_isolation))
        for batch in pbar:
            step+=1
            b_count+=1
            optimizer.zero_grad()
                    
            logits, targets = model(batch, device)
            
            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            clip_loss = (images_loss + titles_loss) / 2.0
            clip_loss = clip_loss.mean()

            loss = args.clip_loss_factor * clip_loss 
            loss.backward()
            optimizer.step()
        
            tr_loss += loss.item()

            steps.append(step)
        
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
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

            pbar = tqdm(test_dataloader_isolation,  total=len(test_dataloader_isolation))
            for batch in pbar:
                step+=1

                b_count+=1
                        
                logits, targets = model(batch, device)
                
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                clip_loss = (images_loss + titles_loss) / 2.0
                clip_loss = clip_loss.mean()

                loss = args.clip_loss_factor * clip_loss 

                te_loss += loss.item()

                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                pbar.set_postfix(val_loss=loss_meter.avg, lr=get_lr(optimizer))
                
            te_loss /= step
            test_losses.append(loss_meter.avg)

            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , loss_meter.avg))
            print('-------------------------------')

            if loss_meter.avg < best_loss:
                # best_loss = te_loss
                if (args.dataset == 'coco'):
                    best_loss = loss_meter.avg
                    if (args.is_poison):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/_noapi_baseline_isolation_best.pt")  
                    elif (args.noise_bpp):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/_noapi_baseline_isolation_bpp_best.pt")  
                    elif (args.wanet):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/_noapi_baseline_isolation_wanet_best.pt")  
                    elif (args.single_target_label):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/single_target_label/_noapi_baseline_isolation_best.pt.pt")  
                    elif (args.multi_target_label):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/multiple_target_label/_noapi_baseline_isolation_best.pt.pt")  
                    print("Saved Best Model: ", epoch)
                
                elif (args.dataset == 'flickr'): 
                    best_loss = loss_meter.avg
                    if (args.is_poison):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/_noapi_baseline_isolation_best_flickr.pt")  
                    elif (args.noise_bpp):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/_noapi_baseline_isolation_bpp_best_flickr.pt")  
                    elif (args.wanet):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/_noapi_baseline_isolation_wanet_best_flickr.pt")  
                    elif (args.single_target_label):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/single_target_label/_noapi_baseline_isolation_best_flickr.pt")  
                    elif (args.multi_target_label):
                        torch.save(model.state_dict(), "../../../KG_Defence/mil/models/ABL/multiple_target_label/_noapi_baseline_isolation_best_flickr.pt")  
                    print("Saved Best Model: ", epoch)

            


def  train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args):
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

    for epoch in range(start_epoch, args.unlearn_epoch):

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
        
      
        pbar = tqdm(train_dataloader_unlearn,  total=len(train_dataloader_unlearn))
        for batch in pbar:
            step+=1
            b_count+=1
            optimizer.zero_grad()
                    
            logits, targets = model(batch, device)
            
            images_loss = cross_entropy(logits, targets, reduction='none')
            titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
            clip_loss = (images_loss + titles_loss) / 2.0
            clip_loss = clip_loss.mean()

            loss = args.clip_loss_factor * clip_loss 
            (-loss).backward()
            optimizer.step()
        
            tr_loss += (loss).item()

            steps.append(step)
        
            count = batch["image"].size(0)
            loss_meter.update((loss).item(), count)
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

            pbar = tqdm(test_dataloader_unlearn,  total=len(test_dataloader_unlearn))
            for batch in pbar:
                step+=1

                b_count+=1
                        
                logits, targets = model(batch, device)
                
                images_loss = cross_entropy(logits, targets, reduction='none')
                titles_loss = cross_entropy(logits.T, targets.T, reduction='none')
                clip_loss = (images_loss + titles_loss) / 2.0
                clip_loss = clip_loss.mean()

                loss = args.clip_loss_factor * clip_loss 

                te_loss += loss.item()

                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)
                pbar.set_postfix(val_loss=loss_meter.avg, lr=get_lr(optimizer))
                
            te_loss /= step
            test_losses.append(loss_meter.avg)

            epochs.append(epoch+1)
            print("Epoch: {}  te_loss: {}".format( epoch , loss_meter.avg))
            print('-------------------------------')

            # if loss_meter.avg < best_loss: ## doing it in opposite way, I want the model to have high loss
            if epoch < 10:
                # best_loss = te_loss
                best_loss = loss_meter.avg
                if (args.dataset == 'coco'):
                    if (args.is_poison):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/_noapi_baseline_poison_unlearn_best_positive_test_loss{}.pt".format(epoch)) 
                    elif (args.noise_bpp):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/_noapi_baseline_unlearn_bpp_{}.pt".format(epoch))        
                    elif (args.wanet):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/_noapi_baseline_unlearn_wanet_best.pt")    
                    elif (args.single_target_label):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/single_target_label/_noapi_baseline_unlearn_best.pt")  
                    elif (args.multi_target_label):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/multiple_target_label/_noapi_baseline_unlearn_best.pt") 

                elif (args.dataset == 'flickr'):
                    if (args.is_poison):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/_noapi_baseline_poison_unlearn_best_positive_test_loss_flickr{}.pt".format(epoch)) 
                    elif (args.noise_bpp):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/_noapi_baseline_unlearn_bpp_flickr{}.pt".format(epoch))        
                    elif (args.wanet):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/_noapi_baseline_unlearn_wanet_best_flickr{}.pt".format(epoch))    
                    elif (args.single_target_label):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/single_target_label/_noapi_baseline_unlearn_best.pt")  
                    elif (args.multi_target_label):
                        torch.save(model.state_dict(), "/globalscratch/alvi/ABL/multiple_target_label/_noapi_baseline_unlearn_best.pt") 

                print("Saved Best Model: ", epoch)
    
   



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
    parser.add_argument('--dataset', default='imagenet', type=str)
    parser.add_argument('--tokenizer_clip', default='yes', type=str)
    parser.add_argument('--clip_loss_factor', default=1.0, type=float)
    parser.add_argument('--optim', default='adam', type=str) 
    parser.add_argument('--with_mil', default='no', type=str)
    parser.add_argument('--isolation_epoch', default=5, type=int)
    parser.add_argument('--unlearn_epoch', default=5, type=int)
    parser.add_argument('--projection_layer', default=False, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--run_v2', default='no', type=str)
    parser.add_argument('--clip_openai', default='no', type=str)
    parser.add_argument('--is_poison', default=False, type=bool)
    parser.add_argument('--class_to_poison', default='dog', type=str)
    parser.add_argument('--same_location', default=True, type=bool)
    parser.add_argument('--poison_percent', default=0.001, type=float)
    parser.add_argument('--poison_percent_isolation', default=0.001, type=float)
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
    parser.add_argument('--is_unlearn', default=False, type=bool)


    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if (args.dataset == 'coco'):
        kg = kg_load(args)
        classes, kg_dict = kg.load_kg()
    
    elif (args.dataset == 'flickr'):
        classes = None
  
    # tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    # lst_tokens, lst_subtokens = kg.kgemb_bert(kg_dict, tokenizer)

     ## load model:

    model = clipvitabl_baseline (classes=classes, args=args).to(device)

    preprocess = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize((CFG.size, CFG.size))])
    

    if (args.dataset == 'coco'):

        preprocess = get_transforms('train')
        
        csv_train_path = '/home/alvi/KG_Defence/datasets/coco/csv_train.csv'
        csv_val_path = '/home/alvi/KG_Defence/datasets/coco/csv_val.csv'
        train_df, val_df = make_train_valid_dfs(csv_train_path, csv_val_path)
        
        root = '/home/alvi/KG_Defence/datasets/coco/images/train2017'
        image_filenames = train_df['image_file'].values
        captions = train_df['caption'].values
        names = train_df['category_name'].values

        if (not args.is_unlearn):
            train_dataloader_isolation = build_loaders_abl(root, train_df, image_filenames, captions, names, preprocess, tokenizer, mode='train', args=args)
        else: 
            train_dataloader_unlearn = build_loaders_abl(root, train_df, image_filenames, captions, names, preprocess, tokenizer, mode='train', args=args)

        root = '/home/alvi/KG_Defence/datasets/coco/images/val2017'
        image_filenames = val_df['image_file'].values
        captions = val_df['caption'].values
        names = val_df['category_name'].values

        if (not args.is_unlearn):
            test_dataloader_isolation = build_loaders_abl(root, val_df, image_filenames, captions, names, preprocess,tokenizer, mode='val', args=args)
        else: 
            test_dataloader_unlearn = build_loaders_abl(root, val_df, image_filenames, captions, names, preprocess,tokenizer, mode='val', args=args)
    
    elif (args.dataset == 'flickr'):

        preprocess = get_transforms('train')
        
        csv_train_path = '/home/alvi/KG_Defence/datasets/flickr/captions_train.csv'
        csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'
        train_df, val_df = make_train_valid_dfs(csv_train_path, csv_val_path)
        
        root = '/home/alvi/KG_Defence/datasets/flickr/images/train'
        image_filenames = train_df['image_file'].values
        captions = train_df['caption'].values

         ## load ke train
        # Load data from a pickle file
        with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_train_pickle.pkl', 'rb') as file:
            lst_train_ke = pickle.load(file)
        
        args.lst_ke = lst_train_ke
     

        if (not args.is_unlearn):
            train_dataloader_isolation = build_loaders_abl(root, train_df, image_filenames, captions, None, preprocess, tokenizer, mode='train', args=args)
        else: 
            train_dataloader_unlearn = build_loaders_abl(root, train_df, image_filenames, captions, None, preprocess, tokenizer, mode='train', args=args)
        
        ## val
        root = '/home/alvi/KG_Defence/datasets/flickr/images/val'
        image_filenames = val_df['image_file'].values
        captions = val_df['caption'].values

          ## load ke val
        # Load data from a pickle file
        with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_val_pickle.pkl', 'rb') as file:
            lst_val_ke = pickle.load(file)
        
        args.lst_ke = lst_val_ke

        if (not args.is_unlearn):
            test_dataloader_isolation = build_loaders_abl(root, val_df, image_filenames, captions, None, preprocess,tokenizer, mode='val', args=args)
        else: 
            test_dataloader_unlearn = build_loaders_abl(root, val_df, image_filenames, captions, None, preprocess,tokenizer, mode='val', args=args)

    
    
    if (args.optim == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=1e-8, betas=(0.9,0.98),eps=1e-8,weight_decay=0.002)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader_isolation)*args.batch_size)

    elif (args.optim == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader_isolation)*args.batch_size)

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
            if (not args.is_unlearn):
                if (args.noise_bpp):
                    # model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/_noapi_baseline_isolation_bpp_best.pt', map_location=args.device))
                    train_isolation(model, train_dataloader_isolation, test_dataloader_isolation, optimizer, scheduler, device, args)  
                elif (args.wanet):
                    train_isolation(model, train_dataloader_isolation, test_dataloader_isolation, optimizer, scheduler, device, args)  
                else: 
                    train_isolation(model, train_dataloader_isolation, test_dataloader_isolation, optimizer, scheduler, device, args)  
            else: 
                if (args.is_poison):
                    print ('backdoor load isolation model for training unlearn model')
                    model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/_noapi_baseline_isolation_best.pt', map_location=args.device))
                    train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args)

                elif (args.noise_bpp):
                    print ('noise bpp load isolation model for training unlearn model')
                    model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/_noapi_baseline_isolation_bpp_best.pt', map_location=args.device))
                    train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args)

                elif (args.wanet):
                    print ('wanet load isolation model for training unlearn model')
                    model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/_noapi_baseline_isolation_wanet_best.pt', map_location=args.device))
                    train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args)

                elif (args.single_target_label):
                    print ('single target label isolation model load')
                    model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/single_target_label/_noapi_baseline_isolation_best.pt', map_location=args.device))
                    train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args)
                
                elif (args.multi_target_label):
                    print ('single target label isolation model load')
                    model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/multiple_target_label/_noapi_baseline_isolation_best.pt', map_location=args.device))
                    train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args)


                

     
    if (args.dataset == 'flickr'):
            if (not args.is_unlearn):
                if (args.noise_bpp):
                    # model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/_noapi_baseline_isolation_bpp_best.pt', map_location=args.device))
                    train_isolation(model, train_dataloader_isolation, test_dataloader_isolation, optimizer, scheduler, device, args)  
                elif (args.wanet):
                    train_isolation(model, train_dataloader_isolation, test_dataloader_isolation, optimizer, scheduler, device, args)  
                else: 
                    train_isolation(model, train_dataloader_isolation, test_dataloader_isolation, optimizer, scheduler, device, args)  
            else: 
                if (args.is_poison):
                    print ('backdoor load isolation model for training unlearn model')
                    model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/_noapi_baseline_isolation_best_flickr.pt', map_location=args.device))
                    train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args)

                elif (args.noise_bpp):
                    print ('noise bpp load isolation model for training unlearn model')
                    model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/_noapi_baseline_isolation_bpp_best_flickr.pt', map_location=args.device))
                    train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args)

                elif (args.wanet):
                    print ('wanet load isolation model for training unlearn model')
                    model.load_state_dict(torch.load('/home/alvi/KG_Defence/mil/models/ABL/_noapi_baseline_isolation_wanet_best_flickr.pt', map_location=args.device))
                    train_unlearn(model, train_dataloader_unlearn, test_dataloader_unlearn, optimizer, scheduler, device, args)

                

                

