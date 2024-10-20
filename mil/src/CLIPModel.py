import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch import linalg
from clip_implementation.projection_head import ProjectionHead, PatchProjection


class CLIPModel(nn.Module):
    def __init__(self, model, classes, args):

        super().__init__()
        self.model = model
        self.classes = classes
        self.args = args

        if (self.args.projection_layer):
            self.projection_head = ProjectionHead(embedding_dim=512)
        
        self.patch_projection = PatchProjection(device = args.device, weight=self.model.visual.ln_post.weight)
        
       

    def forward(self, imgs, lst_tokens, titles, names, ids):
        # Getting Image and Text Features
        img_embs = self.model.encode_image(imgs)   ## shape: 128 x 512  ## clip api
        title_embs = self.model.encode_text(titles)

        # if (not self.args.projection_layer):
        #     img_embs = img_embs / linalg.norm(img_embs, dim=-1, keepdim = True)
        #     title_embs = title_embs / linalg.norm(img_embs, dim=-1, keepdim = True)
        
        if (self.args.projection_layer):
            # print('before shapes: ', img_embs.shape, title_embs.shape, img_embs.dtype, title_embs.dtype)
            img_embs = img_embs.to(torch.float)
            title_embs = title_embs.to(torch.float)
          
            title_embs = self.projection_head(title_embs)
            img_embs = self.projection_head(img_embs)
            # title_embs = self.projection_head(title_embs)
            # print('after shapes: ', img_embs.shape, title_embs.shape, img_embs.dtype, title_embs.dtype)
        
        img_embs = img_embs / linalg.norm(img_embs, dim=-1, keepdim = True)
        title_embs = title_embs / linalg.norm(title_embs, dim=-1, keepdim = True)

        ## this is for image , caption logits
        # logits_per_image, logits_per_text = self.model(imgs, titles)

        logits = img_embs @ title_embs.T

        image_similarity = img_embs @ img_embs.T
        title_similarity = title_embs @ title_embs.T

        target = F.softmax((image_similarity + title_similarity) / 2.0, dim =-1)
        
        ## KG loss 

        lst_txt_emb = []
        for i in range (lst_tokens.shape[0]):
            txt_emb = self.model.encode_text(lst_tokens[i])  ## clip api
            if (self.args.projection_layer):
                txt_emb = txt_emb.to(torch.float)
                txt_emb = self.projection_head(txt_emb)
            lst_txt_emb.append(txt_emb)
        
     
        txt_embs = torch.stack(lst_txt_emb)  # shape: 80 x 3 x 512
      
        txt_embs = txt_embs / linalg.norm(txt_embs, dim=-1, keepdim = True)
        # txt_similarity = txt_embs @ txt_embs.permute(2, 1, 0)

       
      
        ## txt and image embedding matrix multplication 
        # prod = txt_embs.unsqueeze(1) @ img_embs.unsqueeze(2) ## txt: 80 * 1 * 3 * 512 ; img: 128 * 512 * 1
        # prod = torch.squeeze(prod, dim = 3) ## 80 x 128 x 3 x 1
        # prod = prod.permute(1, 0, 2)  # 128 x 80 x 3 (Batch * class * KE)

        prod = img_embs.unsqueeze(1).unsqueeze(1) @ txt_embs.permute(0,2,1) ## 128 * 1 * 1 * 512 ; txt = 80 * 512 * 3
        prod = torch.squeeze(prod, dim=2)                                        # ([128, 80, 1, 3])
       
        max_sim = torch.amax(prod, dim = 2)  ## max smiliarity 
        min_sim = torch.amin(prod, dim = 2)  ## min_similarity
        min_sim *= -1
        max_sim *= 1

        ##index 
        max_sim_index = torch.argmax(prod, dim = 2)
        min_sim_index = torch.argmin(prod, dim = 2)
        
        ## the logits for KG / mil
        y_pred = ''
        indices = []
        for row in names:
            row_indices = [i for i, item in enumerate(self.classes) if item in row]
            indices.append(row_indices)
  
        # min_sim[np.arange(len(indices)), indices] = max_sim[np.arange(len(indices)), indices].clone()
        # print('indices', indices)
        # print('MIN: ', min_sim[0])
        for i, row_indices in enumerate(indices):
            min_sim[i, row_indices] = max_sim[i, row_indices].clone()
        
        y_pred = min_sim.clone().detach().requires_grad_(True)
        
        shape = (y_pred.shape[0],len(self.classes))
        mil_targets = torch.zeros(shape, dtype=torch.float)

        for i, row_indices in enumerate(indices):
            mil_targets[i, row_indices] = 1
        
        # mil_softmax_targets = F.softmax(y_pred, dim =-1)
        mil_softmax_targets = F.sigmoid(y_pred)
       
        # print("names: ", names[:3])
        # print("y_pred: ", y_pred[:3])
        # print('min index ', min_sim[:3])
        # print('max index ', max_sim[:3])
        # print('mil_target', mil_targets[:3])
        return logits, target, y_pred, mil_targets, mil_softmax_targets
    
    def forward_patch(self, imgs, lst_tokens, titles, names, ids):
        # Getting Image and Text Features
      
        img_embs, patch_embs = self.model.encode_image(imgs)   ## shape: 128 x 512  ## clip api
        title_embs = self.model.encode_text(titles)


        # if (not self.args.projection_layer):
        #     img_embs = img_embs / linalg.norm(img_embs, dim=-1, keepdim = True)
        #     title_embs = title_embs / linalg.norm(img_embs, dim=-1, keepdim = True)
        
        # self.args.projection_layer  = False
        if (self.args.projection_layer == True):
            # print('before shapes: ', img_embs.shape, title_embs.shape, img_embs.dtype, title_embs.dtype)
            img_embs = img_embs.to(torch.float)
            title_embs = title_embs.to(torch.float)
          
            title_embs = self.projection_head(title_embs)
            img_embs = self.projection_head(img_embs)
            patch_embs = self.projection_head(patch_embs)
            # title_embs = self.projection_head(title_embs)
            # print('after shapes: ', img_embs.shape, title_embs.shape, img_embs.dtype, title_embs.dtype)
        
        #### uncomment if u want to normalize
        # img_embs = img_embs / linalg.norm(img_embs, dim=-1, keepdim = True)  
        # title_embs = title_embs / linalg.norm(title_embs, dim=-1, keepdim = True)]
        #  txt_embs = txt_embs / linalg.norm(txt_embs, dim=-1, keepdim = True)

        ## this is for image , caption logits
        # logits_per_image, logits_per_text = self.model(imgs, titles)

        logits = img_embs @ title_embs.T

        image_similarity = img_embs @ img_embs.T
        title_similarity = title_embs @ title_embs.T

        target = F.softmax((image_similarity + title_similarity) / 2.0, dim =-1)
        
        ## KG loss 

        lst_txt_emb = []
        for i in range (lst_tokens.shape[0]):
            txt_emb = self.model.encode_text(lst_tokens[i])  ## clip api
            if (self.args.projection_layer):
                txt_emb = txt_emb.to(torch.float)
                txt_emb = self.projection_head(txt_emb)
            lst_txt_emb.append(txt_emb)
        
     
        txt_embs = torch.stack(lst_txt_emb)  # shape: 80 x 3 x 512
      
    
        ## txt and image embedding matrix multplication 
        # prod = txt_embs.unsqueeze(1) @ img_embs.unsqueeze(2) ## txt: 80 * 1 * 3 * 512 ; img: 128 * 512 * 1
        # prod = torch.squeeze(prod, dim = 3) ## 80 x 128 x 3 x 1
        # prod = prod.permute(1, 0, 2)  # 128 x 80 x 3 (Batch * class * KE)
        ## image and kg emb mat
        # prod = img_embs.unsqueeze(1).unsqueeze(1) @ txt_embs.permute(0,2,1) ## 128 * 1 * 1 * 512 ; txt = 80 * 512 * 3
        # prod = torch.squeeze(prod, dim=2)                                        # ([128, 80, 1, 3])

        ### Now do the matmul for patch and KE > patch shape: 128 * 49 * 512, > KE shape: 128 * 80 * 5 * 512
       
        # Perform the matrix multiplication
        prod = patch_embs.unsqueeze(1) @ txt_embs.permute(0,2,1) ## patch_emb : 128 * 49 * 512; txt = 80 * 512 * 3 ;  prod = (128 * 80 * 49 * 3)
        # print(prod)
       
        max_sim = torch.amax(torch.amax(prod, dim = 3), dim=2)  ## max smiliarity  
        min_sim = torch.amin(torch.amin(prod, dim = 3), dim = 2)  ## min_similarity
        min_sim *= 1
        max_sim *= 1

        ##index 
        max_sim_index = torch.argmax(prod, dim = 2)
        min_sim_index = torch.argmin(prod, dim = 2)
        
        ## the logits for KG / mil
        y_pred = ''
        indices = []
        for row in names:
            row_indices = [i for i, item in enumerate(self.classes) if item in row]
            indices.append(row_indices)
  
       
        for i, row_indices in enumerate(indices):
            min_sim[i, row_indices] = max_sim[i, row_indices].clone()
        
        y_pred = min_sim.clone().detach().requires_grad_(True)
        

        shape = (y_pred.shape[0],len(self.classes))
        mil_targets = torch.zeros(shape, dtype=torch.float)

        for i, row_indices in enumerate(indices):
            mil_targets[i, row_indices] = 1
        
        # mil_softmax_targets = F.softmax(y_pred, dim =-1)
        mil_softmax_targets = F.sigmoid(y_pred)
        # print('mil trgets: ', mil_targets)
        # print('y_pred: ', y_pred)
        return logits, target, y_pred, mil_targets, mil_softmax_targets
    



    def forward_subke_patch(self, imgs, lst_tokens, lst_subtokens, titles, names, ids):
        # Getting Image and Text Features
      
        img_embs, patch_embs = self.model.encode_image(imgs)   ## shape: 128 x 512  ## clip api
        patch_embs = self.patch_projection(patch_embs)
        title_embs = self.model.encode_text(titles)

        # if (not self.args.projection_layer):
        #     img_embs = img_embs / linalg.norm(img_embs, dim=-1, keepdim = True)
        #     title_embs = title_embs / linalg.norm(img_embs, dim=-1, keepdim = True)
        
        # self.args.projection_layer  = False
        if (self.args.projection_layer == True):
            # print('before shapes: ', img_embs.shape, title_embs.shape, img_embs.dtype, title_embs.dtype)
            img_embs = img_embs.to(torch.float)
            title_embs = title_embs.to(torch.float)
          
            title_embs = self.projection_head(title_embs)
            img_embs = self.projection_head(img_embs)
            patch_embs = self.projection_head(patch_embs)
            # title_embs = self.projection_head(title_embs)
            # print('after shapes: ', img_embs.shape, title_embs.shape, img_embs.dtype, title_embs.dtype)
        
        #### uncomment if u want to normalize
        # img_embs = img_embs / linalg.norm(img_embs, dim=-1, keepdim = True)  
        # title_embs = title_embs / linalg.norm(title_embs, dim=-1, keepdim = True)]
        #  txt_embs = txt_embs / linalg.norm(txt_embs, dim=-1, keepdim = True)

        ## this is for image , caption logits
        # logits_per_image, logits_per_text = self.model(imgs, titles)

        logits = img_embs @ title_embs.T

        image_similarity = img_embs @ img_embs.T
        title_similarity = title_embs @ title_embs.T

        target = F.softmax((image_similarity + title_similarity) / 2.0, dim =-1)
        
        ## KG loss 
       
        lst_txt_emb = []
        for i in range (lst_tokens.shape[0]):
            txt_emb = self.model.encode_text(lst_tokens[i])  ## clip api
            if (self.args.projection_layer):
                txt_emb = txt_emb.to(torch.float)
                txt_emb = self.projection_head(txt_emb)
            lst_txt_emb.append(txt_emb)
        
     
        txt_embs = torch.stack(lst_txt_emb)  # shape: 80 x 5 x 512

        lst_txt_emb = []
        for i in range(lst_subtokens.shape[0]):
            lst_subtxt_emb = []
            for j in range(lst_subtokens[i].shape[0]):
                subtxt_emb = self.model.encode_text(lst_subtokens[i][j])
                if (self.args.projection_layer):
                    subtxt_emb = subtxt_emb.to(torch.float)
                    subtxt_emb = self.projection_head(subtxt_emb)
                lst_subtxt_emb.append(subtxt_emb)
            
            subtxt_embs = torch.stack(lst_subtxt_emb)
            lst_txt_emb.append(subtxt_embs)
        
        subtxt_embs = torch.stack(lst_txt_emb) # shape: 80 x 5 x 3 x 512

        ## compute image KE similarity
        # prod = patch_embs.unsqueeze(1) @ txt_embs.permute(0,2,1) ## patch_emb : 128 * 49 * 512; txt = 80 * 512 * 5 ;  prod = (128 * 80 * 49 * 5)
        ## different way:
        prod = torch.einsum('bij,cpj->bcip', patch_embs, txt_embs)
        # print(prod)
        ## compute image subKE similarity
        # subtxt_embs = torch.mean(subtxt_embs, dim=2, keepdim=True) # 80 x 5 x 1 x 512
        # subtxt_embs = subtxt_embs.squeeze(2)   # 80 x 5 x 512
        # mil_targets = patch_embs.unsqueeze(1) @ subtxt_embs.permute(0,2,1) ## patch_emb : 128 * 1 * 49 * 512; subtxt = 80 * 512 * 5 ;  prod = (128 * 80 * 49 * 5)

        ## different way:
        mil_targets = torch.einsum('bij,cpqj->bcipq', patch_embs, subtxt_embs)
        
        mil_targets = torch.mean(mil_targets, dim=4, keepdim=True)
        mil_targets = mil_targets.squeeze(4)
   
        # y_pred = prod.clone().detach().requires_grad_(True)
        y_pred = prod
        # mil_targets = mil_targets.clone().detach().requires_grad_(True)
        
        # y_pred = F.normalize(y_pred, p=2, dim=3)
        # mil_targets = F.normalize(mil_targets, p=2, dim=3)
        
        # print('y_pred: ')
        # print(y_pred[0])
        # print('mil_targets')
        # print(mil_targets[0])
        
        # actual: 128 * 80 * 49 * 5
        y_pred = y_pred.permute(0, 1, 2, 3)  ## 128 x 80 x 49 x 5
        mil_targets = mil_targets.permute(0, 1, 2, 3)
        # y_pred = y_pred.view(128, -1)
        # print('y_pred shape1: ', y_pred.shape)
        # mil_targets = mil_targets.softmax(dim = -1)

        # print('mil targets: ', mil_targets.shape)

        return logits, target, y_pred, mil_targets
    
