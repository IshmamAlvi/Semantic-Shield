from tqdm import tqdm
import clip
import torch
from torch import linalg
from clip_implementation.projection_head import ProjectionHead
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from clip_vit.config import CFG
from clip_vit.utils import AvgMeter, get_lr


def train_clip_flickr ():
    pass

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()




def train_loop_flickr (model, tokenizer, train_dataloader, test_dataloader, optimizer, scheduler,  device, args):

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

    if (args.is_poison): 
        start_epoch = 0
        print('start epoch for backdoor: ', start_epoch)
    
    elif (args.noise_bpp): 
        start_epoch = 0
        print('start epoch for bpp: ', start_epoch)
    
    elif (args.wanet): 
        start_epoch = 0
        print('start epoch for wanet: ', start_epoch)
    
    elif (args.single_target_label): 
        start_epoch = 0
        print('start epoch for single target label: ', start_epoch)
    
    elif (args.multi_target_label):
        start_epoch = 0
        print('start epoch multi target label: ', start_epoch)
    
    else: 
        start_epoch = 36
        print('start epoch clean model: ', start_epoch)

    
 

    for epoch in range(start_epoch, args.epoch):
        print ('going inside loop')
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
                img_embs, patch_embs, title_embs, txt_embs, subtxt_embs = model.forward_attention(batch, lst_tokens, lst_subtokens, device)
            
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
            plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.single_target_image):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_image/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.single_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_label/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        elif (args.multi_target_label):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/multiple_target_label/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))
        else:
            plt.savefig('../../../KG_Defence/mil/figures/flickr/loss_contrastive_steps_epoch_{}_distilbert.png'.format(epoch))

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
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.single_target_image):
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_image/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))  
                elif (args.single_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_label/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.multi_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/multiple_target_label/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))
                else:
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/loss_subke_steps_epoch_{}_distilbert.png'.format(epoch))

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
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.single_target_image):
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_image/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.single_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                elif (args.multi_target_label):
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/multiple_target_label/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))
                else:
                    plt.savefig('../../../KG_Defence/mil/figures/flickr/loss_ke_steps_epoch_{}_distilbert.png'.format(epoch))

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

                lst_tokens = {}
                lst_tokens['input_ids'] = batch['lst_input_ids']
                lst_tokens['attention_mask'] = batch['lst_attention_mask']
                lst_subtokens = lst_tokens

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
                    print ('in val')
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
                    model_path = "/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)

            elif (args.noise_bpp):
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "/globalscratch/alvi/flickr/noise_bpp/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "/globalscratch/alvi/flickr/noise_bpp/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "/globalscratch/alvi/flickr/noise_bpp/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)

            elif (args.wanet):
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "/globalscratch/alvi/flickr/wanet/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "/globalscratch/alvi/flickr/wanet/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "/globalscratch/alvi/flickr/wanet/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)

                 
            elif (args.single_target_label):
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)

            elif (args.multi_target_label):
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else:
                        model_path = "/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)
            
            else:
                if (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                    model_path = "/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_distilbert_epoch_{}.pt".format(epoch)
                elif (args.standard_loss_factor > 0 and args.kg_loss_factor > 0):
                    if (not args.ke_only):
                        model_path = "/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_subke_distilbert_epoch_{}.pt".format(epoch)
                    else: 
                        model_path = "/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_ke_distilbert_epoch_{}.pt".format(epoch)

            torch.save(model.state_dict(), model_path)

            if loss_meter.avg < best_loss:
                # best_loss = te_loss
                best_loss = loss_meter.avg
                if (args.is_poison):
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/poison/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")

                
                elif (args.single_target_label):
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/single_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")


                elif (args.multi_target_label):
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else:
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/multiple_target_label/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")

                else:
                    if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
                       if (not args.ke_only):
                           torch.save(model.state_dict(), "/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_only_subke_distilbert.pt")
                       else:
                           torch.save(model.state_dict(), "/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_only_ke_distilbert.pt")
                    elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
                        torch.save(model.state_dict(), "/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_distilbert_best.pt")
                    elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
                        if (not args.ke_only):
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_subke_distilbert_best.pt")
                        else: 
                            torch.save(model.state_dict(), "/globalscratch/alvi/flickr/all_model/_noapi_best_baseline_coco_standard_ke_distilbert_best.pt")
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
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/loss_noapi_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/loss_noapi_ke_distilbert.png')
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/loss_noapi_contrastive_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/loss_noapi_contrastive_ke_distilbert.png')

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
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_label/loss_noapi_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_label/loss_noapi_ke_distilbert.png')
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_label/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_label/loss_noapi_contrastive_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/single_target_label/loss_noapi_contrastive_ke_distilbert.png')


    elif (args.multi_target_label):
        if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/multiple_target_label/loss_noapi_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/multiple_target_label/loss_noapi_ke_distilbert.png')
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/multiple_target_label/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/multiple_target_label/loss_noapi_contrastive_subke_distilbert.png')
            else: 
                plt.savefig('../../../KG_Defence/mil/figures/flickr/poison/multiple_target_label/loss_noapi_contrastive_ke_distilbert.png')

    else:
        if (args.kg_loss_factor == 1 and args.standard_loss_factor == 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/flickr/loss_noapi_subke_distilbert.png')
            else:
                plt.savefig('../../../KG_Defence/mil/figures/flickr/loss_noapi_ke_distilbert.png') 
        elif (args.standard_loss_factor == 1 and args.kg_loss_factor == 0):
            plt.savefig('../../../KG_Defence/mil/figures/flickr/loss_noapi_contrastive_distilbert.png')
        elif (args.standard_loss_factor > 0 and  args.kg_loss_factor > 0):
            if (not args.ke_only):
                plt.savefig('../../../KG_Defence/mil/figures/flickr/loss_noapi_contrastive_subke_distilbert.png')
            else:
                plt.savefig('../../../KG_Defence/mil/figures/flickr/loss_noapi_contrastive_ke_distilbert.png')
    
    plt.close('all')

