import torch
import torch.nn as nn
import math
import random
import copy
from .transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

CLIP_MAX_BATCH_SIZE = 64
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json

### define for llava-prumerge
def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output

def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()
    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices = np.where((attn_np > upper_bound))[0]
    ratio = len(outlier_indices) / len(attn_np)
    return ratio

class CLIPVisionTower(nn.Module):
    def obtain_interpolate_embeddings(self, image_size=(600,600), patch_size=14):
        state_dict = self.vision_tower.vision_model.embeddings.position_embedding.state_dict()
        pos_embedding = state_dict['weight']
        if pos_embedding.dim()<3:  # don't know why [0] when SFT setting (zero2,v1,tune_mlp_adapter)
            pos_embedding = self.ori_pos_embedding_state_dict['weight']

        pos_embedding = pos_embedding.unsqueeze(0)
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

        # new_seq_length = (image_size // patch_size) ** 2 + 1
        new_height, new_width = image_size
        new_seq_length = (new_height // patch_size) * (new_width // patch_size) + 1

        if new_seq_length != seq_length:
            # The class token embedding shouldn't be interpolated so we split it up.
            seq_length -= 1
            new_seq_length -= 1
            pos_embedding_token = pos_embedding[:, :1, :]
            pos_embedding_img = pos_embedding[:, 1:, :]

            pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")

            pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
            new_height_1d = new_height // patch_size
            new_width_1d = new_width // patch_size

            new_pos_embedding_img = nn.functional.interpolate(
                pos_embedding_img,
                size=(new_height_1d, new_width_1d),
                mode='bicubic',
                align_corners=True,
            )
            # (1, hidden_dim, new_height_1d, new_width_1d) -> (1, hidden_dim, new_seq_length)
            new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
            # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)

            new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)[0]
            new_state_dict =copy.deepcopy(state_dict)
            new_state_dict['weight'] = new_pos_embedding

            out_position_embedding = nn.Embedding(new_seq_length+1, hidden_dim).to(self.dtype)
            out_position_embedding.load_state_dict(new_state_dict)
            out_position_ids = torch.arange(new_seq_length+1).expand((1, -1))

            return out_position_embedding, out_position_ids

        else:
            return None, None


    def clip_interpolate_embeddings(self, image_size=(600,600), patch_size=14):
        """This function helps interpolating positional embeddings during checkpoint loading,
        especially when you want to apply a pre-trained model on images with different resolution.

        Args:
            image_size (int): Image size of the new model.
            patch_size (int): Patch size of the new model.
            model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
            interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
            reset_heads (bool): If true, not copying the state of heads. Default: False.

        Returns:
            OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
        """
        state_dict = self.vision_tower.vision_model.embeddings.position_embedding.state_dict()
        pos_embedding = state_dict['weight']

        pos_embedding = pos_embedding.unsqueeze(0)
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

        new_height, new_width = image_size
        new_seq_length = (new_height // patch_size) * (new_width // patch_size) + 1

        # Need to interpolate the weights for the position embedding.
        # We do this by reshaping the positions embeddings to a 2d grid, performing
        # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
        if new_seq_length != seq_length:
            # The class token embedding shouldn't be interpolated so we split it up.
            seq_length -= 1
            new_seq_length -= 1
            pos_embedding_token = pos_embedding[:, :1, :]
            pos_embedding_img = pos_embedding[:, 1:, :]

            # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")

            # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
            # new_seq_length_1d = image_size // patch_size
            new_height_1d = new_height // patch_size
            new_width_1d = new_width // patch_size

            # Perform interpolation.
            # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
            new_pos_embedding_img = nn.functional.interpolate(
                pos_embedding_img,
                # size=new_seq_length_1d,
                size=(new_height_1d,new_width_1d),
                mode='bicubic',
                align_corners=True,
            )
            new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
            # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
            new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)[0]
            state_dict['weight'] = new_pos_embedding
            self.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(new_seq_length+1, hidden_dim).to(self.dtype)
            self.vision_tower.vision_model.embeddings.position_embedding.load_state_dict(state_dict)
            self.vision_tower.vision_model.embeddings.image_size = image_size
            self.vision_tower.vision_model.embeddings.patch_size = patch_size
            self.vision_tower.vision_model.embeddings.position_ids = torch.arange(new_seq_length+1).expand((1, -1))

    def recover_clip_interpolate_embeddings(self):
        if self.ori_pos_embedding_state_dict is None:
            raise ValueError("Original position embedding state dictionary not saved. Call save_original_position_embedding() first.")

        # Shape of pos_embedding is (1, seq_length, hidden_dim)
        ori_state_dict = self.ori_pos_embedding_state_dict
        ori_pos_embedding = ori_state_dict['weight']
        ori_pos_embedding = ori_pos_embedding.unsqueeze(0)
        n, seq_length, hidden_dim = ori_pos_embedding.shape
        if n != 1:
            raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

        self.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(seq_length, hidden_dim).to(self.dtype)
        self.vision_tower.vision_model.embeddings.position_embedding.load_state_dict(ori_state_dict)
        self.vision_tower.vision_model.embeddings.image_size = self.ori_vision_embeddings.image_size
        self.vision_tower.vision_model.embeddings.patch_size = self.ori_vision_embeddings.patch_size
        self.vision_tower.vision_model.embeddings.position_ids = self.ori_vision_embeddings.position_ids


    #### reproduce LLaVA-PruMerge
    ## PruMerge
    def token_prune_merge_advanced(self, images, if_adaptive=True, reduction_ratio = 1/8):
        '''
        version 10/03/2024 using the key*key matrix to calculate the cosine similarity
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]

        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)#*3.5
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C]
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]
        # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0)

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)


                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = weighted_avg + x_others[b, i, :]
                updated_x_others[b, i, :] = updated_center


        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        return image_features

    ## PruMerge++
    def token_prune_merge_advanced_plus(self, images, if_adaptive=True, reduction_ratio = 1/8):
        '''
        version 24/03/2024 using the spacially smapled tokens to supplement the pruned tokens
        '''

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]

        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)#*3.5
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True

        if if_adaptive:
            step_length = int(1/reduction_ratio)
            arithmetic_sequence = torch.arange(0, 575, int(step_length/3)).to(device=self.device)
            original_tensor_1d = idx.flatten().to(device=self.device)
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
            concatenated_tensor = torch.cat((idx, filtered_sequence.unsqueeze(0)), dim=1)
            idx = concatenated_tensor
        else:
            # # this is for training
            step_length = int(1/reduction_ratio)
            new_idx = torch.zeros((idx.size(0), idx.size(1)*2), dtype=torch.long).to(device=self.device)
            for i in range(idx.size(0)):
                arithmetic_sequence = torch.arange(int(step_length/2), 575, int(step_length)).to(device=self.device)
                original_tensor_1d = idx[i].flatten().to(device=self.device)
                filtered_sequence = arithmetic_sequence
                # filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
                concatenated_tensor = torch.cat((original_tensor_1d, filtered_sequence), dim=0)
                new_idx[i] = concatenated_tensor
            idx = new_idx

        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C]
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0)

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = x_others[b, i, :]  + weighted_avg
                updated_x_others[b, i, :] = updated_center

        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        return image_features



    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.llava_args = args

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

        if self.llava_args.image_aspect_ratio == "slice":
            self.clip_interpolate_embeddings(image_size=(self.llava_args.slice_block_size, self.llava_args.slice_block_size), patch_size=14)


    def feature_select(self, image_forward_outs, no_prune=False):
        if isinstance(self.select_layer, list):
            selected_layer = int(random.choice(self.select_layer))
        else:
            selected_layer = self.select_layer
            # selected_idx = 0
        if no_prune:
            image_features = image_forward_outs.hidden_states[selected_layer]
        else:
            image_features = image_forward_outs.hidden_states[21]

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, text_input_embeds=None, text_input_masks=None, obj_bbox_infos=None, mm_projector=None, global_position_embedding=None):
        if text_input_embeds is None and text_input_masks is None and mm_projector is None:
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), obj_bbox_infos=obj_bbox_infos,
                                                          output_hidden_states=True, global_position_embedding=global_position_embedding)
                    image_feature = self.feature_select(image_forward_out, no_prune=True)
                    image_feature = image_feature.to(image.dtype)
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), obj_bbox_infos=obj_bbox_infos,
                                                       output_hidden_states=True, global_position_embedding=global_position_embedding)
                image_features = self.feature_select(image_forward_outs, no_prune=True)
                image_features = image_features.to(images.dtype)

                # ## prune_merge
                # reduction_ratio=0.25
                # # image_features = self.token_prune_merge_advanced(images.to(device=self.device, dtype=self.dtype),
                # #                 if_adaptive=False, reduction_ratio=reduction_ratio).to(images.dtype)
                # # ## prune_merge++
                # image_features = self.token_prune_merge_advanced_plus(images.to(device=self.device, dtype=self.dtype),
                #                     if_adaptive=False, reduction_ratio=reduction_ratio).to(images.dtype)

            return image_features


        if images.size(0) > CLIP_MAX_BATCH_SIZE:
            batch_size=CLIP_MAX_BATCH_SIZE
            print(f'image size to CLIP: {images.size()}')
            image_features_list = []
            prune_loss_list = []
            out_hidden_states_for_loss_list=[]
            target_area_masks_list=[]
            for i in range(0, images.size(0), batch_size):
                batch_images = images[i:i + batch_size].to(device=self.device, dtype=self.dtype)
                batch_outs = self.vision_tower(batch_images,
                                    text_input_embeds=text_input_embeds,
                                    text_input_masks=text_input_masks,
                                    obj_bbox_infos=obj_bbox_infos,
                                    mm_projector=mm_projector,
                                    token_prune_layers=self.select_layer,
                                    output_hidden_states=True)
                batch_features = self.feature_select(batch_outs)
                batch_features = batch_features.to(images.dtype)
                
                image_features_list.append(batch_features)
                if batch_outs.target_area_masks is not None:
                    target_area_masks_list.append(batch_outs.target_area_masks)
                    prune_loss_list.append(batch_outs.prune_losses)
                    out_hidden_states_for_loss_list.append(batch_outs.out_hidden_states_for_loss)
                else:
                    target_area_masks_list = None
                    prune_loss_list = None
                    out_hidden_states_for_loss_list = None
                del batch_outs
                torch.cuda.empty_cache()
            image_features = torch.cat(image_features_list, dim=1)
            if target_area_masks_list is not None:
                target_area_masks = torch.cat(target_area_masks_list, dim=0)
                prune_loss = torch.cat(prune_loss_list, dim=0)
                out_hidden_states_for_loss = torch.cat(out_hidden_states_for_loss_list, dim=0)
            else:
                target_area_masks = None
                prune_loss = None
                out_hidden_states_for_loss = None
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                    text_input_embeds=text_input_embeds,
                                                    text_input_masks=text_input_masks,
                                                    obj_bbox_infos=obj_bbox_infos,
                                                    mm_projector=mm_projector,
                                                    token_prune_layers=self.select_layer,
                                                    output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)
            image_features = image_features.to(images.dtype)
            if hasattr(image_forward_outs, 'prune_losses'):
                prune_loss = image_forward_outs.prune_losses
            else:
                prune_loss = None

            out_hidden_states_for_loss=image_forward_outs.out_hidden_states_for_loss
            target_area_masks=image_forward_outs.target_area_masks
      
        return image_features, prune_loss, out_hidden_states_for_loss, target_area_masks


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size


    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


    @property
    def get_fix_local_embedding(self):
        return self.vision_tower.vision_model.embeddings.position_embedding(self.vision_tower.vision_model.embeddings.position_ids)

    @property
    def get_fix_position_embedding(self):
        fix_local_embedding = self.get_fix_local_embedding
        num_patches_per_side = self.num_patches_per_side
        position_embedding = fix_local_embedding.squeeze(0)
        position_for_class = position_embedding[0:1, :]  # get t
        posi_embed_dim = fix_local_embedding.size(-1)
        position_embedding = position_embedding[1:, :].reshape(num_patches_per_side, num_patches_per_side, posi_embed_dim)
        position_embedding = position_embedding.permute(2,0,1).unsqueeze(0) #[1, d, h, w] [1,1024,24,24]
        original_dtype = position_embedding.dtype
        position_embedding = position_embedding.to(torch.float) # [1,1024,24,24]

        return position_embedding, position_for_class, original_dtype, fix_local_embedding


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward


    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)

