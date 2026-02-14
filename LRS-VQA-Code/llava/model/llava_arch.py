#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
# , build_vision_selector  # 增加

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import math
import torch.nn.functional as F

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config, use_distill=False)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        pretrain_mm_pruner = model_args.pretrain_mm_pruner

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type


        self.config.vision_patch_size = self.get_vision_tower().vision_tower.vision_model.embeddings.patch_size
        self.config.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter
        self.config.image_aspect_ratio = getattr(model_args, 'image_aspect_ratio', 'slice')
        self.config.slice_block_size = getattr(model_args, 'slice_block_size', 336)
        self.config.pretrain_mm_pruner = pretrain_mm_pruner
        self.config.use_cross_attn_mask = getattr(model_args, 'use_cross_attn_mask', False)
        self.config.use_dense_cross_attn = getattr(model_args, 'use_dense_cross_attn', False)
        self.config.concat_global_img_feature = getattr(model_args, 'concat_global_img_feature', True)

        self.config.llm_use_last_text = getattr(model_args, 'llm_use_last_text', False)
        self.config.llm_layer_list = eval(model_args.llm_layer_list)
        self.config.select_stu_state_layer = getattr(model_args, 'select_stu_state_layer', 1)
        self.config.mm_spatial_pool_mode = model_args.mm_spatial_pool_mode
        self.config.upscale_module = model_args.upscale_module


        load_pruner_weight_from_llm = getattr(model_args, 'load_pruner_weight_from_llm', False)

        use_distill = getattr(model_args, 'use_distill', False) 

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config, use_distill=use_distill)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            if 'selector' in self.config.mm_projector_type or 'pruner' in self.config.mm_projector_type:
                self.mm_projector.load_pretrained_mlp_weights(pretrain_mm_mlp_adapter, mm_projector_weights)
            else:
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        if pretrain_mm_pruner is not None:
            mm_projector_weights = torch.load(pretrain_mm_pruner, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            if pretrain_mm_mlp_adapter is not None:
                mm_mlp_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                self.mm_projector.load_pretrained_mlp_weights(pretrain_mm_mlp_adapter, mm_mlp_projector_weights)

        if load_pruner_weight_from_llm and "mlp2x_gelu_distillpruner" in self.config.mm_projector_type:
            self.mm_projector.token_pruner._load_pretrained_weights(self)

        if self.config.mm_projector_type == "mlp2x_gelu_after_distillpruner":
            self.mm_projector.token_pruner.select_stu_state_layer = self.config.select_stu_state_layer


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        
        padding = current_height - new_height
        unpadded_tensor = tensor[:, : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)

        padding = current_width - new_width
        unpadded_tensor = tensor[:, :, :current_width - padding]

    return unpadded_tensor

from torch.nn.utils.rnn import pad_sequence

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_img_patches, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_img_patches, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1).contiguous()
        image_feature = image_feature.view(num_img_patches, -1, num_dim)
        return image_feature

    def get_pure_text_embedding_nopadding(self, input_ids, attention_mask=None, labels=None, obj_bbox_infos=None):

        new_input_embeds_nopad, new_input_mask_nopad = [], []
        image_real_num = []
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        input_ids_nopad = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels_nopad = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        attention_mask_nopad = [cur_attention_mask[cur_attention_mask.bool()] for cur_attention_mask in attention_mask]

        for batch_idx, (cur_input_ids, cur_labels, cur_attn_mask) in enumerate(zip(input_ids_nopad, labels_nopad, attention_mask_nopad)):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            image_real_num.append(num_images)

            cur_input_ids_noim, cur_att_mask_noim, cur_labels_noim = [], [], []
            img_token_list = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            image_token_indices = [-1] + img_token_list + [cur_input_ids.shape[0]]
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_att_mask_noim.append(cur_attn_mask[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))

            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_input_embeds_no_im_qs = cur_input_embeds_no_im[-1]
            cur_att_mask_noim_qs = cur_att_mask_noim[-1]
            cur_labels_noim_qs = cur_labels_noim[-1]

            text_query_mask = (cur_labels_noim_qs == -100)
            text_query_relative_indices = torch.where(text_query_mask)[0]
            cur_input_embeds_no_an_qs = cur_input_embeds_no_im_qs[text_query_relative_indices]
            cur_att_mask_noim_noan_qs = cur_att_mask_noim_qs[text_query_relative_indices]

            new_input_embeds_nopad.append(cur_input_embeds_no_an_qs)
            new_input_mask_nopad.append(cur_att_mask_noim_noan_qs)

        return new_input_embeds_nopad, new_input_mask_nopad, image_real_num


    def get_pure_text_embedding(self, input_ids, attention_mask=None, labels=None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) 
        new_input_embeds, new_input_mask = [], []

        for batch_idx, (cur_input_ids, cur_attn_mask) in enumerate(zip(input_ids, attention_mask)):
            
            cur_input_ids_noim, cur_att_mask_noim = [], []
            img_token_list = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            image_token_indices = [-1] + img_token_list + [cur_input_ids.shape[0]]
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_att_mask_noim.append(cur_attn_mask[image_token_indices[i]+1:image_token_indices[i+1]])
                
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_mask = torch.cat(cur_att_mask_noim)

            new_input_embeds.append(cur_input_embeds)
            new_input_mask.append(cur_input_mask)

            if len(img_token_list) > 0 and getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds.append(torch.cat((
                    torch.zeros((len(img_token_list), cur_input_embeds.shape[1]), dtype=cur_input_embeds.dtype, device=cur_input_embeds.device),
                    cur_input_embeds
                ), dim=0))
                new_input_mask.append(torch.cat((
                    torch.zeros((len(img_token_list)), dtype=cur_input_mask.dtype, device=cur_input_mask.device),
                    cur_input_mask
                ), dim=0))
            elif len(img_token_list) > 0:
                new_input_embeds.append(torch.cat((
                    cur_input_embeds,
                    torch.zeros((len(img_token_list), cur_input_embeds.shape[1]), dtype=cur_input_embeds.dtype, device=cur_input_embeds.device)
                ), dim=0))
                new_input_mask.append(torch.cat((
                    cur_input_mask,
                    torch.zeros((len(img_token_list)), dtype=cur_input_mask.dtype, device=cur_input_mask.device)
                ), dim=0))
            else:
                new_input_embeds.append(cur_input_embeds)
                new_input_mask.append(cur_input_mask)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_input_mask = [x[:tokenizer_model_max_length] for x in new_input_mask]

        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        new_input_mask = torch.stack(new_input_mask, dim=0)

        assert new_input_mask.shape == new_input_embeds.shape[:2]
        
        return new_input_embeds, new_input_mask


    def encode_images_baseline(self, images):
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().get_vision_tower()(concat_images)
            image_features = self.get_model().mm_projector(image_features)
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)

        return image_features, []

    def encode_images_anyres_distill(self, images):
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().get_vision_tower()(concat_images)
            image_features = self.get_model().mm_projector.mlp_projector(image_features)
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector.mlp_projector(image_features)

        return image_features, []

    def _split_global_local_images(self, image):
        if image.ndim == 3:  
            global_image = image.unsqueeze(0)
            local_image = image.unsqueeze(0)
        elif image.size(0) > 1:  
            global_image = image[0].unsqueeze(0)
            local_image = image[1:]
        else:
            global_image = image
            local_image = image
        return global_image, local_image


    def encode_images_with_distillpruner(self, images, obj_bbox_infos=None, input_ids=None, split_sizes=None, attention_mask=None,
                                 images_mask=None, labels=None):
        # Initialize outputs
        out_global_image_features = []
        out_local_image_features = []

        images = images if isinstance(images, list) else [images]
        obj_bbox_infos = obj_bbox_infos if isinstance(obj_bbox_infos, list) else [obj_bbox_infos]

        for batch_idx, image in enumerate(images):
            obj_bbox_info = obj_bbox_infos[batch_idx]

            global_image, local_images = self._split_global_local_images(image)

            global_image_feature = self.get_model().get_vision_tower()(global_image)
            out_global_image_features.append(self.get_model().mm_projector.mlp_projector(global_image_feature))

            local_image_feature = self.get_model().get_vision_tower()(local_images, obj_bbox_infos=obj_bbox_info)

            local_image_feature = self.get_model().mm_projector.mlp_projector(local_image_feature)

            if self.config.mm_spatial_pool_mode == 'none':
                pool_local_image_feature = local_image_feature
            else:
                pool_local_image_feature = self.get_2dPool(local_image_feature)

            out_local_image_features.append(pool_local_image_feature)  # [35,576,1024]

        return out_global_image_features, out_local_image_features


    def encode_images_after_distillpruner(self, images, obj_bbox_infos=None, input_ids=None, split_sizes=None, attention_mask=None,
                                 images_mask=None, labels=None):
        # Initialize outputs
        image_features = []
        global_image_features = []
        out_global_image_features = []
        out_local_image_features = []

        images = images if isinstance(images, list) else [images]
        obj_bbox_infos = obj_bbox_infos if isinstance(obj_bbox_infos, list) else [obj_bbox_infos]

        for batch_idx, image in enumerate(images):
            obj_bbox_info = obj_bbox_infos[batch_idx]

            global_image, local_images = self._split_global_local_images(image)

            global_image_feature = self.get_model().get_vision_tower()(global_image)
            global_image_features.append(global_image_feature)

            local_image_feature = self.get_model().get_vision_tower()(local_images)

            image_features.append(local_image_feature)

        for g_img in global_image_features:
            out_global_image_features.append(self.get_model().mm_projector.mlp_projector(g_img))
        for l_img in image_features:
            local_image_feature = self.get_model().mm_projector.mlp_projector(l_img)

            if self.config.mm_spatial_pool_mode == 'none':
                pool_local_image_feature = local_image_feature
            else:
                pool_local_image_feature = self.get_2dPool(local_image_feature)

            out_local_image_features.append(pool_local_image_feature)

        return out_global_image_features, out_local_image_features


    def encode_images(self, images, obj_bbox_infos=None, input_ids=None, split_sizes=None, attention_mask=None,
                    images_mask=None, labels=None, concat_global_img_feature=False):
        if 'pruner' not in self.config.mm_projector_type:
            return self.encode_images_baseline(images)

        elif 'after_distillpruner' in self.config.mm_projector_type:
            if self.config.image_aspect_ratio == 'anyres':
                return self.encode_images_anyres_distill(images)
            else:
                return self.encode_images_after_distillpruner(
                    images, obj_bbox_infos=obj_bbox_infos, input_ids=input_ids, attention_mask=attention_mask,
                    images_mask=images_mask, labels=labels)

        elif 'distillpruner' in self.config.mm_projector_type:
            if self.config.image_aspect_ratio == 'anyres':
                return self.encode_images_anyres_distill(images)

            return self.encode_images_with_distillpruner(
                images, obj_bbox_infos=obj_bbox_infos, input_ids=input_ids, attention_mask=attention_mask,
                images_mask=images_mask, labels=labels)

    def split_and_pad_text_embeds(self, text_input_embeds, text_abs_indices, n_turns):
        diff = text_abs_indices[1:] - text_abs_indices[:-1]
        increment = (diff != 1).long()
        group_ids = torch.zeros_like(text_abs_indices, dtype=torch.long)
        group_ids[1:] = torch.cumsum(increment, dim=0)
        boundaries = torch.where(group_ids[1:] != group_ids[:-1])[0] + 1
        split_indices = torch.cat([
            torch.tensor([0], device=group_ids.device),
            boundaries,
            torch.tensor([len(group_ids)], device=group_ids.device)
        ])
        starts = split_indices[:-1]
        ends = split_indices[1:]

        text_segments = [text_input_embeds[start:end] for start, end in zip(starts, ends)]
        
        padded_text_segments = pad_sequence(text_segments, batch_first=True)
        text_lengths = torch.tensor([seg.size(0) for seg in text_segments], device=text_input_embeds.device)
        key_padding_mask = torch.arange(padded_text_segments.size(1), device=text_input_embeds.device).unsqueeze(0) < text_lengths.unsqueeze(1)  # [n_turns, max_seq_len]
        return padded_text_segments, key_padding_mask


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, obj_bbox_infos=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None, None, None

        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        concat_global_img_feature = getattr(self.config, 'concat_global_img_feature',True)
        llm_use_last_text = getattr(self.config, 'llm_use_last_text', False)

        ret_token = torch.tensor(self.tokenizer.convert_tokens_to_ids([':'])).to(dtype=input_ids.dtype, device=input_ids.device) # ret token

        if type(images) is list or images.ndim == 5:
            if isinstance(images, torch.Tensor) and images.ndim == 5:
                images = [images[batch_idx] for batch_idx in range(images.size(0))]
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            image_features, local_image_features = self.encode_images(images, obj_bbox_infos=obj_bbox_infos, input_ids=input_ids,
                                                                      attention_mask=attention_mask, labels=labels)
            split_sizes = [image.shape[0] for image in images]
            if type(image_features) is not list:
                image_features = list(torch.split(image_features, split_sizes, dim=0))

            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            local_image_fea_hws = []

            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                # 使用 prune merge时注释掉
                token_patch_side = self.get_vision_tower().num_patches_per_side
                if self.config.mm_spatial_pool_mode == 'none':
                    for obj_bbox_info in obj_bbox_infos:
                        local_image_fea_hws.append((token_patch_side * obj_bbox_info['block_num_hw'][0],
                                                    token_patch_side * obj_bbox_info['block_num_hw'][1]))
                else:
                    for obj_bbox_info in obj_bbox_infos:
                        local_image_fea_hws.append((token_patch_side // 2 * obj_bbox_info['block_num_hw'][0],
                                                    token_patch_side // 2 * obj_bbox_info['block_num_hw'][1]))

            elif mm_patch_merge_type.startswith('spatial'):
                new_local_image_features = []
                for image_idx, g_image_feature in enumerate(image_features):
                    image_size = image_sizes[image_idx]
                    if image_aspect_ratio != 'slice' and image_aspect_ratio != 'pyramid_crop':
                        if g_image_feature.shape[0] > 1:
                            l_image_feature = g_image_feature[1:]
                            g_image_feature = g_image_feature[0]
                            if g_image_feature.dim() == 2:
                                g_image_feature = g_image_feature.unsqueeze(0)
                            image_features[image_idx] = g_image_feature
                            if self.config.mm_spatial_pool_mode != 'none':
                                l_image_feature = self.get_2dPool(l_image_feature, stride=2)

                    else:
                        l_image_feature = local_image_features[image_idx]
                    obj_bbox_info = obj_bbox_infos[image_idx]

                    if self.config.mm_spatial_pool_mode == 'none':
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == g_image_feature.shape[1] # [1,576,3584]
                    else:  # 'bilinear' or '2d_pool'
                        height = width = self.get_vision_tower().num_patches_per_side // 2
                        assert height * width == g_image_feature.shape[1] // 4

                    if image_aspect_ratio == 'anyres':
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_size,
                                                                                        self.config.image_grid_pinpoints,
                                                                                        self.get_vision_tower().config.image_size)
                    elif image_aspect_ratio == 'slice' or image_aspect_ratio == 'pyramid_crop':
                        num_patch_height, num_patch_width = obj_bbox_info['block_num_hw']
                    l_image_feature = l_image_feature.view(num_patch_height, num_patch_width, height, width, -1) # [6, 144, 3584]->[2, 3, 12, 12, 3584]

                    if 'unpad' in mm_patch_merge_type:
                        l_image_feature = l_image_feature.permute(4, 0, 2, 1, 3).contiguous()  # [2, 3, 12, 12, 3584]->[3584, 2, 12, 3, 12]
                        l_image_feature = l_image_feature.flatten(1, 2).flatten(2, 3)          # [3584, 2, 12, 3, 12]->[3584, 24, 36]
                        l_image_feature = unpad_image(l_image_feature, image_size)

                        l_image_feature = torch.cat((l_image_feature,
                                    self.model.image_newline[:, None, None].expand(*l_image_feature.shape[:-1], 1).to(l_image_feature.device)), dim=-1)
                        # if not self.training:
                        #     print(f'l_image_feature size:{l_image_feature.size()}, image size:{image_size}')
                        local_image_fea_hws.append((l_image_feature.size(1), l_image_feature.size(2)))
                        l_image_feature = l_image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        local_image_fea_hws.append((num_patch_height * height, num_patch_width * width))

                    new_local_image_features.append(l_image_feature)

                local_image_features = new_local_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, local_image_features = self.encode_images(images, obj_bbox_infos=obj_bbox_infos,
                                                                    input_ids=input_ids, attention_mask=attention_mask,
                                                                    labels=labels, concat_global_img_feature=concat_global_img_feature)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_labels_for_img = []
        cur_image_idx = 0
        prompt_len = []        
        image_token_posi = []   
        image_tokens_list = []  
        local_image_rel_begin_list = []
        nopad_split_sizes_list = []


        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            if image_index == []: 
                print(f'image_index is none! image_index:{image_index}')
                image_token_posi.append(-1)
            else:
                image_token_posi.append(image_index[0])
            # record input instruction length in inference mode
            if not self.training:
                if image_index == []:
                    prompt_len.append(cur_input_ids.shape[0])
                else:
                    prompt_len.append(cur_input_ids.shape[0] - 1) 
           
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]  
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0) 
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_labels_for_img.append(labels[batch_idx])
                cur_image_idx += 1

                image_tokens_list.append(0)
                local_image_rel_begin_list.append(0)
                continue


            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []         
            cur_labels = labels[batch_idx]  
            cur_labels_noim = []       
            for i in range(len(image_token_indices) - 1): 
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])  
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_labels_for_img = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_labels_for_img.append(cur_labels_noim[i])
                if i < num_images:
                    if image_features[cur_image_idx].ndim > 2:
                        cur_image_features = image_features[cur_image_idx].squeeze()
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    
                    if local_image_features[0]!=[]:
                        cur_image_features_local = local_image_features[cur_image_idx].squeeze(0).reshape(-1,local_image_features[cur_image_idx].shape[-1])
                        ret_embed = self.get_model().embed_tokens(ret_token)

                        local_image_rel_begin_list.append(cur_image_features.shape[0]+ret_embed.shape[0])
                        cur_image_features = torch.cat([cur_image_features, ret_embed, cur_image_features_local], dim=0)
                    else:
                        local_image_rel_begin_list.append(-1)

                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_labels_for_img.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    image_tokens_list.append(cur_image_features.shape[0]) 

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            nopad_split_sizes = [x.shape[0] for x in cur_new_input_embeds]
            nopad_split_sizes_list.append(nopad_split_sizes)
            

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_labels_for_img = torch.cat(cur_new_labels_for_img)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_labels_for_img.append(cur_new_labels_for_img)

        self.model.image_token_posi = image_token_posi
        self.model.prompt_len = prompt_len
        self.model.image_tokens = image_tokens_list

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None and self.config.mm_projector_type!="mlp2x_gelu_after_distillpruner":
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_labels_for_img = [x[:tokenizer_model_max_length] for x in new_labels_for_img]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            ori_position_ids = position_ids
            position_ids = None

        final_text_querys = None
        all_hidden_states = None
        text_image_attn_weights = None


        if self.config.mm_projector_type=="mlp2x_gelu_after_distillpruner":
            if image_aspect_ratio != 'pyramid_crop':
                prune_new_input_embeds, prune_new_labels = self.get_model().mm_projector.token_pruner(
                    local_image_rel_begin_list=local_image_rel_begin_list,
                    inputs_embeds=new_input_embeds,
                    labels=new_labels_padded,
                    labels_for_img=new_labels_for_img,
                    training=self.training,
                    position_ids=position_ids,
                    image_token_posis=image_token_posi,
                    prompt_len=prompt_len,
                    llm_use_last_text=llm_use_last_text,
                    image_tokens=image_tokens_list,
                    nopad_split_sizes=nopad_split_sizes_list,
                    obj_bbox_infos=obj_bbox_infos,
                    local_image_fea_hws=local_image_fea_hws)
            else:
                prune_new_input_embeds, prune_new_labels = self.get_model().mm_projector.token_pruner(
                    local_image_rel_begin_list=local_image_rel_begin_list,
                    inputs_embeds=new_input_embeds,
                    labels=new_labels_padded,
                    labels_for_img=new_labels_for_img,
                    training=self.training,
                    position_ids=position_ids,
                    image_token_posis=image_token_posi,
                    prompt_len=prompt_len,
                    llm_use_last_text=llm_use_last_text,
                    image_tokens=image_tokens_list,
                    nopad_split_sizes=nopad_split_sizes_list,
                    obj_bbox_infos=obj_bbox_infos,
                    vision_tower=self.get_model().get_vision_tower(),
                    mlp_projector=self.get_model().mm_projector.mlp_projector,
                    local_image_fea_hws=local_image_fea_hws,
                    image_newline=self.model.image_newline,
                    use_pyramid=True)

            if prune_new_input_embeds[0].size(0) > tokenizer_model_max_length:
                print(f'beyond tokenizer_model_max_length, new_input_embeds size:{new_input_embeds[0].size(0)}')

            prune_new_input_embeds = [x[:tokenizer_model_max_length] for x in prune_new_input_embeds]
            prune_new_labels = [x[:tokenizer_model_max_length] for x in prune_new_labels]

            max_len = max(x.shape[0] for x in prune_new_input_embeds)

            batch_size = len(prune_new_input_embeds)
            new_input_embeds_padded = []
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=prune_new_labels[0].dtype, device=prune_new_labels[0].device)
            if attention_mask is None:
                attention_mask = torch.ones_like(new_labels_padded, dtype=torch.bool)
            attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
            position_ids = torch.zeros((batch_size, max_len), dtype=ori_position_ids.dtype, device=ori_position_ids.device)

            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(prune_new_input_embeds, prune_new_labels)):
                cur_len = cur_new_embed.shape[0]
                if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                    new_input_embeds_padded.append(torch.cat((
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        cur_new_embed
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask[i, -cur_len:] = True
                        position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    new_input_embeds_padded.append(torch.cat((
                        cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask[i, :cur_len] = True
                        position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

            if _labels is None:
                new_labels = None
            else:
                new_labels = new_labels_padded

            if _attention_mask is None:
                attention_mask = None
            else:
                attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

            if _position_ids is None:
                position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, None, None, text_image_attn_weights, all_hidden_states, final_text_querys


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
            if model_args.pretrain_mm_pruner:
                mm_projector_weights = torch.load(model_args.pretrain_mm_pruner, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        for m in self.modules():
            m.tokenizer = tokenizer
