import torch
import torch.nn as nn
import re
import torch.nn.functional as F
import numpy as np
import math
from transformers.models.llama.modeling_llama import LlamaFlashAttention2, LlamaDecoderLayer, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding


from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaFlashAttention2, LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, Qwen2FlashAttention2, Qwen2DecoderLayer, Qwen2RMSNorm
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.nn.init import trunc_normal_

def scale_bbox(hbox, resize_h, resize_w):
    """
    Scale bounding box coordinates according to image resize ratios.
    
    Args:
        hbox (list or np.array): Original bounding box [xmin, ymin, xmax, ymax]
        resize_h (float): Height resize ratio
        resize_w (float): Width resize ratio
    
    Returns:
        list: Scaled bounding box [xmin', ymin', xmax', ymax']
    """
    scaled_hbox = [
        hbox[0] * resize_w,
        hbox[1] * resize_h,
        hbox[2] * resize_w,
        hbox[3] * resize_h
    ]
    return scaled_hbox


def get_target_patch_id(crop_hbox_, min_stride, image_height_, image_width_):
    """
    Get target patch IDs within a cropped region.
    
    Args:
        crop_hbox_ (list): Crop region [xmin, ymin, xmax, ymax]
        min_stride (float): Patch stride (size of each patch)
        image_height_ (float): Image height
        image_width_ (float): Image width
    
    Returns:
        tuple: (target_patch_id, h_crop_num, w_crop_num)
            - target_patch_id (torch.Tensor): Patch indices in the crop region
            - h_crop_num (int): Number of patches in height
            - w_crop_num (int): Number of patches in width
    """
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_hbox_

    w_idx1 = math.floor(crop_xmin / min_stride)
    w_idx2 = math.ceil(crop_xmax / min_stride) - 1
    h_idx1 = math.floor(crop_ymin / min_stride)
    h_idx2 = math.ceil(crop_ymax / min_stride) - 1
    
    cur_h_num = int(image_height_ / min_stride)
    cur_w_num = int(image_width_ / min_stride)

    h_idx1 = max(0, h_idx1)
    w_idx1 = max(0, w_idx1)
    h_idx2 = min(cur_h_num - 1, h_idx2)
    w_idx2 = min(cur_w_num - 1, w_idx2)

    w_crop_num = w_idx2 + 1 - w_idx1
    h_crop_num = h_idx2 + 1 - h_idx1
    target_patch_id = []

    for h in range(h_idx1, h_idx2 + 1):
        for w in range(w_idx1, w_idx2 + 1):
            patch_id = h * cur_w_num + w
            target_patch_id.append(patch_id)

    target_patch_id = torch.tensor(target_patch_id, dtype=torch.long)
    return target_patch_id, h_crop_num, w_crop_num


class IdentityMap(nn.Module):
    """Identity projector that returns input unchanged."""
    
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class MLPProjector(nn.Module):
    """MLP-based projector for vision-language alignment."""
    
    def __init__(self, config, mlp_depth):
        """
        Args:
            config: Model configuration
            mlp_depth (int): Number of MLP layers
        """
        super(MLPProjector, self).__init__()
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)


def create_projectors(config, mlp_gelu_match):
    """
    Create multiple MLP projectors for different vision layers.
    
    Args:
        config: Model configuration
        mlp_gelu_match: Regex match object containing MLP depth
    
    Returns:
        nn.ModuleList: List of MLPProjector modules
    """
    mm_vision_select_layer = getattr(config, 'mm_vision_select_layer', [4, 8, 12, 16, 20])
    projectors = nn.ModuleList()
    mlp_depth = int(mlp_gelu_match.group(1))
    
    for idx, layer in enumerate(mm_vision_select_layer):
        projector = MLPProjector(config, mlp_depth)
        projectors.append(projector)
    return projectors


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for grouped query attention.
    
    Args:
        hidden_states (torch.Tensor): Shape (batch, num_key_value_heads, seqlen, head_dim)
        n_rep (int): Number of repetitions
    
    Returns:
        torch.Tensor: Shape (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """
    Rotate half the hidden dims for rotary position embedding.
    
    Args:
        x (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Rotated tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        cos (torch.Tensor): Cosine component of rotary embedding
        sin (torch.Tensor): Sine component of rotary embedding
        position_ids (torch.Tensor): Position indices
        unsqueeze_dim (int): Dimension to unsqueeze for broadcasting
    
    Returns:
        tuple: (q_embed, k_embed) - Query and key with rotary embeddings applied
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DistillPruneModule(nn.Module):
    """
    Token pruning module with distillation capabilities for training.
    Computes text-image attention and performs feature distillation.
    """
    
    def __init__(
            self,
            config,
            grid_size,
            embed_dim,
            num_heads,
            llm_layer_list,
            num_layers=4,
            kv_dim=None,
            temp=1.0,
            upscale_factor=2,
            norm_layer=nn.LayerNorm
    ):
        """
        Args:
            config: Model configuration
            grid_size (int): Grid size for patch tokens
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            llm_layer_list (list): List of LLM layer indices to use
            num_layers (int): Number of decoder layers
            kv_dim (int, optional): Key-value dimension
            temp (float): Temperature for attention
            upscale_factor (int): Upscaling factor
            norm_layer: Normalization layer class
        """
        super().__init__()

        self.num_queries = 1
        self.embed_dim = embed_dim
        self.hidden_size = embed_dim
        self.num_heads = num_heads
        self.head_dim = 128
        self.num_key_value_heads = 32
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.temp = temp

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        if "qwen" in config.architectures[0].lower():
            self.llm_type = "qwen"
        else:
            self.llm_type = "llama"

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if self.llm_type == "llama":
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            self.layers = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(num_layers + 1)]
            )
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif self.llm_type == "qwen":
            self.rotary_emb = Qwen2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            self.layers = nn.ModuleList(
                [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(num_layers + 1)]
            )
            self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.llm_layer_list = llm_layer_list
        self.num_layers = num_layers
        self.select_stu_state_layer = config.select_stu_state_layer
        self.upscale_module = False

    def _load_pretrained_weights(self, language_model):
        """
        Load pretrained weights from LLM into the pruning module.
        
        Args:
            language_model: Pretrained language model
        """
        from deepspeed import zero

        llm_target_layer_list = [self.llm_layer_list[0] - 1] + self.llm_layer_list
        num_layers_to_load = len(llm_target_layer_list)

        for idx in range(num_layers_to_load):
            lm_layer_idx = llm_target_layer_list[idx]
            lm_layer = language_model.layers[lm_layer_idx]
            student_layer = self.layers[idx]

            with zero.GatheredParameters(list(student_layer.self_attn.parameters()), modifier_rank=None):
                with zero.GatheredParameters(list(lm_layer.self_attn.parameters()), modifier_rank=None):
                    student_layer.self_attn.load_state_dict(lm_layer.self_attn.state_dict())

            with zero.GatheredParameters(list(student_layer.mlp.parameters()), modifier_rank=None):
                with zero.GatheredParameters(list(lm_layer.mlp.parameters()), modifier_rank=None):
                    student_layer.mlp.load_state_dict(lm_layer.mlp.state_dict())

            with zero.GatheredParameters(list(student_layer.input_layernorm.parameters()), modifier_rank=None):
                with zero.GatheredParameters(list(lm_layer.input_layernorm.parameters()), modifier_rank=None):
                    student_layer.input_layernorm.load_state_dict(lm_layer.input_layernorm.state_dict())

            with zero.GatheredParameters(list(student_layer.post_attention_layernorm.parameters()), modifier_rank=None):
                with zero.GatheredParameters(list(lm_layer.post_attention_layernorm.parameters()), modifier_rank=None):
                    student_layer.post_attention_layernorm.load_state_dict(lm_layer.post_attention_layernorm.state_dict())

        with zero.GatheredParameters(list(self.norm.parameters()), modifier_rank=None):
            with zero.GatheredParameters(list(language_model.norm.parameters()), modifier_rank=None):
                self.norm.load_state_dict(language_model.norm.state_dict())

    def get_text_image_attn(self, rank_layer, features, position_ids,attention_mask, labels,
                            image_token_posi, image_tokens, training=True, key_text_indices=None,
                            local_image_rel_begin_list=None, local_image_fea_hws=None,
                            if_train_use_prune=False):
        """
        Extract text-to-image attention scores for distillation and pruning.
        
        Args:
            rank_layer (int): Layer index for attention extraction
            features (torch.Tensor): Input features
            position_ids (torch.Tensor): Position IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor): Label tensor
            image_token_posi (list): Image token positions
            image_tokens (list): Number of image tokens per sample
            training (bool): Training mode flag
            key_text_indices (list): Key text token indices
            local_image_rel_begin_list (list): Local image relative begin positions
            local_image_fea_hws (list): Local image feature height/width
            if_train_use_prune (bool): Whether to use pruning in training
        
        Returns:
            tuple: (attention_avg_head_list, output_query_list, output_attn_list_for_trainprune)
        """
        batch_size = features.shape[0]
        attention_avg_head_list = []
        output_query_list = []
        output_attn_list_for_trainprune = []

        if position_ids is None:
            position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, features.shape[1]), dtype=torch.bool, device=features.device)
        else:
            attention_mask = attention_mask.bool()
        if labels is None:
            labels = torch.full((batch_size, features.shape[1]), IGNORE_INDEX, device=features.device)

        self_attn = self.layers[rank_layer].self_attn
        features = self.layers[rank_layer].input_layernorm(features)

        num_heads = self_attn.num_heads
        num_key_value_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim

        bsz, q_len, _ = features.size()

        query_states = self_attn.q_proj(features)
        key_states = self_attn.k_proj(features)
        value_states = self_attn.v_proj(features)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        if cos.device != position_ids.device:
            cos = cos.to(position_ids.device)
            sin = sin.to(position_ids.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

        eager_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, q_len), features, past_key_values_length=0
        ).to(device=query_states.device)

        for bs in range(batch_size):
            image_index = image_token_posi[bs]
            cur_key_states = key_states[bs]
            cur_query_states = query_states[bs]
            cur_eager_attention_mask = eager_attention_mask[bs]
            key_text_indice = key_text_indices[bs]
            
            if training:
                text_query_states = cur_query_states[:, key_text_indice, :]
                text_eager_attention_mask = cur_eager_attention_mask[:, key_text_indice, :]
            else:
                text_query_states = cur_query_states[:, key_text_indice, :].unsqueeze(1)
                text_eager_attention_mask = cur_eager_attention_mask[:, key_text_indice, :].unsqueeze(1)

            attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(1, 2)) / math.sqrt(self.head_dim)

            if image_index == -1:
                zero_attention = attn_weights[:, 0:0, 0:0]
                attention_avg_head_list.append(zero_attention)
                output_query_list.append(zero_attention)
            else:
                if local_image_rel_begin_list[bs] != -1:
                    local_image_begin_idx = image_index + local_image_rel_begin_list[bs]
                else:
                    local_image_begin_idx = image_index

                output_query_list.append(attn_weights[:, :, local_image_begin_idx:image_index + image_tokens[bs]].contiguous())

                if training:
                    attention_avg_head_list.append(attn_weights)
                else:
                    attn_weights = attn_weights + text_eager_attention_mask
                    attn_weights = attn_weights[:,:,image_index:image_index+image_tokens[bs]].contiguous()
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    attention_avg_head = torch.mean(attn_weights, dim=0)
                    attention_avg_head_list.append(attention_avg_head)

        return attention_avg_head_list, output_query_list, output_attn_list_for_trainprune

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, training=False, use_distill=False, image_token_posi=None,
                key_padding_mask=None, prompt_len=None, llm_use_last_text=False, image_tokens=None,
                local_image_rel_begin_list=None, local_image_fea_hws=None, if_train_use_prune=False):
        """
        Forward pass for distillation-based token pruning.
        
        Args:
            input_ids (torch.Tensor, optional): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            position_ids (torch.Tensor, optional): Position IDs
            past_key_values (optional): Cached key-value states
            inputs_embeds (torch.Tensor, optional): Input embeddings
            labels (torch.Tensor, optional): Labels for loss computation
            training (bool): Training mode flag
            use_distill (bool): Use distillation flag
            image_token_posi (list): Image token positions
            key_padding_mask (torch.Tensor, optional): Padding mask
            prompt_len (list): Prompt lengths
            llm_use_last_text (bool): Use last text token for attention
            image_tokens (list): Number of image tokens
            local_image_rel_begin_list (list): Local image relative positions
            local_image_fea_hws (list): Local image feature dimensions
            if_train_use_prune (bool): Pruning flag for training
        
        Returns:
            tuple: (all_hidden_states, final_text_image_attn_weights, final_text_querys, inputs_embeds)
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        key_text_indices = []
        for bs in range(batch_size):
            image_index = image_token_posi[bs]
            image_token_num = image_tokens[bs]
            if training:
                if llm_use_last_text:
                    answer_index = torch.where(labels[bs] != -100)[0].tolist()
                    index_before_answer = []
                    for index in answer_index:
                        if labels[bs][index - 1] == -100 and index > 13: # for qwen2 label begin
                            index_before_answer.append(index - 1)
                    if index_before_answer == []:
                        index_before_answer.append(len(labels[bs]) - 1)
                    index_before_answer = torch.tensor(index_before_answer, device=labels[0].device)
                    key_text_indice = index_before_answer
                else:
                    key_text_indice = image_index + image_token_num
            else:
                prompt_total_len = prompt_len[bs] + image_token_num
                if llm_use_last_text:
                    key_text_indice = prompt_total_len - 1
                else:
                    key_text_indice = image_index + image_token_num
            key_text_indices.append(key_text_indice)

        past_key_values_length = 0
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        hidden_states = inputs_embeds.clone().detach()
        all_hidden_states = []
        out_text_image_attn_list = []
        out_text_query_list = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)

            if layer_idx == self.num_layers:
                continue
            rank_layer = layer_idx+1
            features = hidden_states.clone()

            attention_avg_head_list, output_query_list, output_attn_list_for_trainprune = self.get_text_image_attn(
                rank_layer, features, position_ids, attention_mask, labels,
                image_token_posi, image_tokens, training=training,
                key_text_indices=key_text_indices,
                local_image_rel_begin_list=local_image_rel_begin_list,
                local_image_fea_hws=local_image_fea_hws
            )
            out_text_image_attn_list.append(attention_avg_head_list)
            out_text_query_list.append(output_query_list)

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        final_text_image_attn_weights = []
        final_text_querys = []

        selected_hidden_states = all_hidden_states[self.select_stu_state_layer]
        for bs in range(batch_size):
            per_sample_tensors = []
            per_sample_querys = []
            for l in range(self.num_layers):
                ti_tensor = out_text_image_attn_list[l][bs]
                per_sample_tensors.append(ti_tensor)
                tq_tensor = out_text_query_list[l][bs]
                per_sample_querys.append(tq_tensor)

            per_sample_attns_stacked = torch.stack(per_sample_tensors, dim=0)
            final_text_image_attn_weights.append(per_sample_attns_stacked)
            per_sample_querys_stacked = torch.stack(per_sample_querys, dim=0)

            l_token_h, l_token_w = local_image_fea_hws[bs]
            num_turns = per_sample_attns_stacked.size(2)
            per_sample_querys_stacked = per_sample_querys_stacked.view(self.num_layers, self.num_heads, num_turns, l_token_h, l_token_w)
            final_text_querys.append(per_sample_querys_stacked)

            sys_text_end = image_token_posi[bs]
            if sys_text_end == -1:  # pure-text sample
                continue
            image_token_num = image_tokens[bs]
            end_text_begin = sys_text_end + image_token_num
        return all_hidden_states, final_text_image_attn_weights, final_text_querys, inputs_embeds


def get_2dPool_bilinear(image_feature, vision_tower, stride=2):
    height = width = vision_tower.num_patches_per_side
    num_img_patches, num_tokens, num_dim = image_feature.shape
    image_feature = image_feature.view(num_img_patches, height, width, -1)
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()

    height, width = image_feature.shape[2:]
    scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
    image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

    image_feature = image_feature.permute(0, 2, 3, 1)
    image_feature = image_feature.view(num_img_patches, -1, num_dim)
    return image_feature

def select_attention_indices_via_threshold(attention_scores, threshold: float=0.2, now_level_attn_mask=None):
    """
    Select token indices based on attention score threshold with percentile normalization.
    
    Args:
        attention_scores (torch.Tensor): Attention scores (1D tensor)
        threshold (float): Selection threshold (default: 0.2)
        now_level_attn_mask (torch.Tensor, optional): Valid token mask
    
    Returns:
        torch.Tensor: Selected token indices
    """
    input_token_num = attention_scores.size(0)
    if now_level_attn_mask is not None:
        assert now_level_attn_mask.size(0) == attention_scores.size(0), "Attention mask size mismatch"
        valid_indices = torch.where(now_level_attn_mask == 1)[0]
        attention_scores = attention_scores[valid_indices]
    else:
        valid_indices = torch.arange(attention_scores.size(0))

    min_score = torch.quantile(attention_scores.float(), 0.05)
    max_score = torch.quantile(attention_scores.float(), 0.95)
    normalized_scores = (attention_scores - min_score) / (max_score - min_score)
    normalized_scores = torch.clamp(normalized_scores, 0, 1)
    selected_indices = torch.where(normalized_scores > threshold)[0]
    if now_level_attn_mask is not None:
        selected_indices = valid_indices[selected_indices]

    return selected_indices


def select_attention_indices_via_ratio(attention_scores_local, local_img_fea, tokens_per_row, pruning_ratio: float=0.25, now_level_attn_mask=None):

    assert attention_scores_local.size(0) == local_img_fea.size(0), "Attention scores and local_img_fea size mismatch"

    if now_level_attn_mask is not None:
        assert now_level_attn_mask.size(0) == attention_scores_local.size(0), "Attention mask size mismatch"
        valid_indices = torch.where(now_level_attn_mask == 1)[0].to(attention_scores_local.device)  # Indices where mask is 1
        attention_scores_local = attention_scores_local[valid_indices]  # Filtered attention scores
    else:
        valid_indices = torch.arange(attention_scores_local.size(0), device=attention_scores_local.device)  # All indices
    top_k = int(attention_scores_local.size(0) * pruning_ratio)
    top_k_scores, top_k_indices = torch.topk(attention_scores_local, top_k, largest=True)
    # Map the top-k indices back to the original indices
    selected_indices = valid_indices[top_k_indices]

    newline_indices = torch.arange(tokens_per_row - 1, local_img_fea.size(0), tokens_per_row, device=local_img_fea.device)
    selected_indices = torch.cat([selected_indices, newline_indices]).unique()

    sorted_top_k_indices, _ = torch.sort(selected_indices)

    return sorted_top_k_indices


def unpad_image_hw(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_height, original_width = original_size
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



def prune_local_features_threshold_save_imgnewline(attention_avg_head, local_img_fea, tokens_per_row, pruning_thre, now_level_attn_mask=None):

    attention_scores_local = attention_avg_head.squeeze()  # [N]
    assert attention_scores_local.size(0) == local_img_fea.size(0), "Attention scores and local_img_fea size mismatch"
    selected_indices = select_attention_indices_via_threshold(attention_scores_local,
                                                              threshold=pruning_thre,
                                                              now_level_attn_mask=now_level_attn_mask)
    newline_indices = torch.arange(tokens_per_row - 1, local_img_fea.size(0), tokens_per_row, device=local_img_fea.device)
    selected_indices = torch.cat([selected_indices, newline_indices]).unique()

    sorted_top_k_indices, _ = torch.sort(selected_indices)
    pruned_local_img_fea = local_img_fea[sorted_top_k_indices, :]
    return pruned_local_img_fea


def get_next_level_fea(obj_bbox_info,
                       top_k_indices,
                       now_lvl_token_num_hw,
                       now_lvl_attn_score,
                       pool_patch_len=12,
                       next_lvl_index=1):
    
    scale_infos = obj_bbox_info['scale_infos']
    scale_image_tensors = obj_bbox_info['scale_image_tensors']

    now_lvl_token_num_h, now_lvl_token_num_w  = now_lvl_token_num_hw
    top_k_indices_2d = (torch.div(top_k_indices, now_lvl_token_num_w, rounding_mode='floor'), torch.remainder(top_k_indices, now_lvl_token_num_w)) # top_k_indices_2d 为tuple类型, 含2个元素,分别存行坐标和列坐标的tensor

    next_lvl_image_tensor = scale_image_tensors[next_lvl_index].to(device=top_k_indices.device)
    next_lvl_block_num_h, next_lvl_block_num_w = scale_infos[next_lvl_index]['block_num_hw']
    
    scale_factor_h = next_lvl_block_num_h / now_lvl_token_num_h
    scale_factor_w = next_lvl_block_num_w / now_lvl_token_num_w
    next_level_indices_2d = (torch.floor(top_k_indices_2d[0] * scale_factor_h).long(), torch.floor(top_k_indices_2d[1] * scale_factor_w).long())

    next_lvl_token_num_h = next_lvl_block_num_h * pool_patch_len
    next_lvl_token_num_w = next_lvl_block_num_w * pool_patch_len
    padded_next_lvl_img_fea_2d = torch.zeros((next_lvl_token_num_h, next_lvl_token_num_w), device=now_lvl_attn_score.device)

    positions = torch.stack(next_level_indices_2d, dim=1)
    unique_positions, counts = torch.unique(positions, return_counts=True, dim=0)

    if len(unique_positions) == 0:
        raise ValueError("No valid positions available after removing padding positions.")

    next_level_max_select_num = int(len(counts))
    selected_indices = counts.argsort(descending=True)[:next_level_max_select_num]
    ######

    selected_positions = unique_positions[selected_indices]
    if selected_positions.size(0) == 0:
        max_count = counts.max()
        selected_positions = unique_positions[counts == max_count]

    max_w = selected_positions[:, 1].max() + 1
    sort_keys = selected_positions[:, 0] * max_w + selected_positions[:, 1]
    sorted_indices = torch.argsort(sort_keys)
    selected_positions = selected_positions[sorted_indices]

    selected_next_lvl_coords_h = selected_positions[:, 0]
    selected_next_lvl_coords_w = selected_positions[:, 1] 

    selected_next_lvl_image_tensor = next_lvl_image_tensor[selected_next_lvl_coords_h, selected_next_lvl_coords_w, :, :, :]  # [num_selected, 3, H, W]

    return selected_next_lvl_image_tensor, padded_next_lvl_img_fea_2d, selected_positions


def get_all_levels_fea(obj_bbox_info, hr_attention_avg_head_lvls,
                       local_token_num_hws,
                       pool_patch_len=12,
                       max_select_num_each_lvl=5):
    scale_infos = obj_bbox_info['scale_infos']
    scale_image_tensors = obj_bbox_info['scale_image_tensors']


    pyra_lvls = len(hr_attention_avg_head_lvls)
    topk_percent_lvls = pyra_lvls * [0.0]
    for i in range(pyra_lvls):
        topk_percent_lvls[i] = i / pyra_lvls

    topk_indices_lvls = []
    selected_position_lvls = []
    selected_next_level_image_tensors = []

    for i in range(pyra_lvls):
        sorted_values, sorted_indices = torch.sort(hr_attention_avg_head_lvls[i])
        total_count = len(sorted_indices)
        if i > 0:
            start_index = int(total_count * topk_percent_lvls[i-1])
            end_index = int(total_count * topk_percent_lvls[i])
        else:
            start_index = 0
            end_index = int(total_count * topk_percent_lvls[i])
        topk_indices_lvl = sorted_indices[start_index:end_index]
        topk_indices_lvls.append(topk_indices_lvl)

        token_num_h, token_num_w = local_token_num_hws[i]
        topk_indices_lvl_2d = (torch.div(topk_indices_lvl, token_num_w, rounding_mode='floor'), torch.remainder(topk_indices_lvl, token_num_w))

        image_tensor_lvl = scale_image_tensors[i].to(device=topk_indices_lvl_2d.device)
        pyra_lvl_token_num_h, pyra_lvl_token_num_w = scale_infos[i]['block_num_hw']
        scale_factor_h = pyra_lvl_token_num_h / token_num_h
        scale_factor_w = pyra_lvl_token_num_w / token_num_w
        pyra_lvl_indices_2d = (torch.floor(topk_indices_lvl_2d[0] * scale_factor_h).long(), torch.floor(topk_indices_lvl_2d[1] * scale_factor_w).long())

        lvl_position = torch.stack(pyra_lvl_indices_2d, dim=1)
        unique_position, count = torch.unique(lvl_position, return_counts=True, dim=0)
       

        if len(unique_position) == 0:
            raise ValueError("No valid positions available after removing padding positions.")
        selected_indices = count.argsort(descending=True)[:max_select_num_each_lvl]
        selected_position = unique_position[selected_indices]
        if selected_position.size(0) == 0:
            max_count = count.max()
            selected_position = unique_position[count == max_count]

        max_w = selected_position[:, 1].max() + 1
        sort_keys = selected_position[:, 0] * max_w + selected_position[:, 1]
        sorted_indices = torch.argsort(sort_keys)
        selected_position = selected_position[sorted_indices]

        selected_next_level_coords_h = selected_position[:, 0]
        selected_next_level_coords_w = selected_position[:, 1]

        selected_next_level_image_tensor = image_tensor_lvl[selected_next_level_coords_h, selected_next_level_coords_w, :, :, :]  # [num_selected, 3, H, W]
        selected_next_level_image_tensors.append(selected_next_level_image_tensor)
        selected_position_lvls.append(selected_position)
    return selected_next_level_image_tensors, selected_position_lvls


def pre_get_adapt_position_embedding(vision_tower, obj_bbox_info, local_token_num_hw, next_level_index=-2, selected_position=None):
    scale_infos = obj_bbox_info['scale_infos']
    if len(scale_infos) < abs(next_level_index):
        next_level_index = -abs(len(scale_infos))

    scale_info = scale_infos[next_level_index]
    
    vision_patch_size = vision_tower.config.patch_size
    num_patches_per_side = vision_tower.num_patches_per_side
    init_position_embedding, init_position_for_class, original_dtype, fix_local_embedding = vision_tower.get_fix_position_embedding

    all_feature_global_position_embedding = []
    zero_global_position_embedding = torch.zeros_like(fix_local_embedding)  # [1,577,1024]
    posi_embed_dim = zero_global_position_embedding.size(-1)

    resize_h, resize_w = scale_info['resize_padding_hw']
    patch_width_num = int(resize_w / vision_patch_size)
    patch_height_num = int(resize_h / vision_patch_size)
    block_num_h, block_num_w = scale_info['block_num_hw']

    # position_embedding = F.interpolate(position_embedding, size=(patch_height_num, patch_width_num), mode='bilinear', align_corners=False)
    position_embedding = F.interpolate(init_position_embedding, size=(patch_height_num, patch_width_num), mode='bicubic', align_corners=False)

    position_embedding = position_embedding.to(original_dtype).squeeze(0) # [1024,120,168]
    position_embedding = position_embedding.permute(1,2,0)
    position_embedding = position_embedding.view(block_num_h, num_patches_per_side, block_num_w, num_patches_per_side, posi_embed_dim).permute(0, 2, 1, 3, 4)
    position_embedding = position_embedding.contiguous().view(block_num_h*block_num_w, num_patches_per_side*num_patches_per_side, posi_embed_dim) # [35,576,1024]
    position_for_class = init_position_for_class.unsqueeze(0).expand(position_embedding.size(0), -1, -1)
    position_embedding = torch.cat((position_for_class, position_embedding), dim=1)  # [35,577,1024]

    if selected_position is not None:
        block_indices = selected_position[:, 0] * block_num_w + selected_position[:, 1]  # [num_selected]
        selected_position_embeddings = position_embedding[block_indices]  # [num_selected, num_patches_per_block + 1, 1024]
        all_feature_global_position_embedding.append(selected_position_embeddings)
    else:
        all_feature_global_position_embedding.append(torch.cat([zero_global_position_embedding, position_embedding], dim=0))

    return torch.cat(all_feature_global_position_embedding, dim=0)


class PyramidPruneModule(nn.Module):
    def __init__(
            self,
            config,
            grid_size,
            embed_dim,
            num_heads,
            llm_layer_list,
            num_layers=4,
            kv_dim=None,
            temp=1.0,
            upscale_factor=2,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.num_queries = 1
        self.embed_dim = embed_dim
        self.hidden_size = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = 32
      

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        if "qwen" in config.architectures[0].lower():
            self.llm_type = "qwen"
        else:
            self.llm_type = "llama"

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if self.llm_type == "llama":
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            attention_bias = False
            self.layers = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(num_layers+1)]
            )
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif self.llm_type == "qwen":
            self.rotary_emb = Qwen2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            self.layers = nn.ModuleList(
                [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(num_layers+1)]
            )
            self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.llm_layer_list = llm_layer_list
        self.num_layers = num_layers

        self.select_stu_state_layer = config.select_stu_state_layer
        self.upscale_module = False

    def _load_pretrained_weights(self, language_model):
        from deepspeed import zero
        
        llm_target_layer_list = [self.llm_layer_list[0]-1]+self.llm_layer_list
        num_layers_to_load = len(llm_target_layer_list)

        for idx in range(num_layers_to_load):
            lm_layer_idx = llm_target_layer_list[idx]
            lm_layer = language_model.layers[lm_layer_idx]
            student_layer = self.layers[idx]

            with zero.GatheredParameters(list(student_layer.self_attn.parameters()), modifier_rank=None):
                with zero.GatheredParameters(list(lm_layer.self_attn.parameters()), modifier_rank=None):
                    student_layer.self_attn.load_state_dict(lm_layer.self_attn.state_dict())

            with zero.GatheredParameters(list(student_layer.mlp.parameters()), modifier_rank=None):
                with zero.GatheredParameters(list(lm_layer.mlp.parameters()), modifier_rank=None):
                    student_layer.mlp.load_state_dict(lm_layer.mlp.state_dict())

            with zero.GatheredParameters(list(student_layer.input_layernorm.parameters()), modifier_rank=None):
                with zero.GatheredParameters(list(lm_layer.input_layernorm.parameters()), modifier_rank=None):
                    student_layer.input_layernorm.load_state_dict(lm_layer.input_layernorm.state_dict())

            with zero.GatheredParameters(list(student_layer.post_attention_layernorm.parameters()), modifier_rank=None):
                with zero.GatheredParameters(list(lm_layer.post_attention_layernorm.parameters()), modifier_rank=None):
                    student_layer.post_attention_layernorm.load_state_dict(lm_layer.post_attention_layernorm.state_dict())

        with zero.GatheredParameters(list(self.norm.parameters()), modifier_rank=None):
            with zero.GatheredParameters(list(language_model.norm.parameters()), modifier_rank=None):
                self.norm.load_state_dict(language_model.norm.state_dict())


    def get_text_image_attn(self, rank_layer, features, position_ids, attention_mask, labels,
                            image_token_posi, prompt_len, llm_use_last_text, image_tokens, local_image_rel_begin_list,
                            training=True, local_image_fea_hws=None, iter_upscale_nums=[]):
        """
        Extract text-to-image attention for pyramid pruning.
        
        Args:
            rank_layer (int): Layer index
            features (torch.Tensor): Input features
            position_ids (torch.Tensor): Position IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor): Labels
            image_token_posi (list): Image token positions
            prompt_len (list): Prompt lengths
            llm_use_last_text (bool): Use last text token
            image_tokens (list): Number of image tokens
            local_image_rel_begin_list (list): Local image relative positions
            training (bool): Training mode
            local_image_fea_hws (list): Local image feature dimensions
            iter_upscale_nums (list): Iteration upscale numbers
        
        Returns:
            tuple: (attention_avg_head_list, hr_attention_avg_head_list, hr_local_image_fea_hws)
        """
        batch_size = features.shape[0]
        attention_avg_head_list = []
        hr_attention_avg_head_list = []
        hr_local_image_fea_hws = []

        assert len(image_tokens) == len(local_image_rel_begin_list)

        if position_ids is None:
            position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size,features.shape[1]), dtype=torch.bool, device=features.device)
        else:
            attention_mask = attention_mask.bool()
        if labels is None:
            labels = torch.full((batch_size,features.shape[1]), IGNORE_INDEX, device=features.device)

        self_attn = self.layers[rank_layer].self_attn
        features = self.layers[rank_layer].input_layernorm(features)

        num_heads = self_attn.num_heads
        num_key_value_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim

        bsz, q_len, _ = features.size()

        query_states = self_attn.q_proj(features)
        key_states = self_attn.k_proj(features)
        value_states = self_attn.v_proj(features)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        if cos.device!=position_ids.device:
            cos = cos.to(position_ids.device)
            sin = sin.to(position_ids.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

        if attention_mask.dim()== 4:
            eager_attention_mask = attention_mask
        else:
            eager_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, q_len), features, past_key_values_length=0
            ).to(device=query_states.device)

        for bs in range(batch_size):
            image_index = image_token_posi[bs]
             
            cur_key_states = key_states[bs]
            cur_query_states = query_states[bs]  # [32,633,128]
            cur_eager_attention_mask = eager_attention_mask[bs]
            if training:
                if llm_use_last_text: 
                    answer_index = torch.where(labels[bs] != -100)[0].tolist()
                    index_before_answer=[]
                    for index in answer_index:
                        if labels[bs][index-1]==-100 and index > 13:  # for qwen2 label begin
                            index_before_answer.append(index-1)
                    if index_before_answer==[]:
                        index_before_answer.append(len(labels[bs])-1)
                    index_before_answer = torch.tensor(index_before_answer,device=labels[0].device)
                    text_query_states = cur_query_states[:,index_before_answer,:]
                    text_eager_attention_mask = cur_eager_attention_mask[:,index_before_answer,:]
            else:  # cur_query_states [32,633,128]
                prompt_total_len = prompt_len[bs] + image_tokens[bs] 
                if llm_use_last_text:
                    text_query_states = cur_query_states[:,prompt_total_len-1,:].unsqueeze(1)   # [32,1,128]
                    text_eager_attention_mask = cur_eager_attention_mask[:,prompt_total_len-1,:].unsqueeze(1) # [1,1,633]
            # calculate attention map
            attn_weights = torch.matmul(text_query_states, cur_key_states.transpose(1, 2)) / math.sqrt(self.head_dim) # (num_head, text_token,seq_len)


            if local_image_rel_begin_list[bs] != -1:
                local_image_begin_idx = image_index + local_image_rel_begin_list[bs]
            else:
                local_image_begin_idx = image_index
 
            attn_weights = attn_weights + text_eager_attention_mask
            attn_weights = attn_weights[:,:,local_image_begin_idx:image_index+image_tokens[bs]].contiguous()   
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attention_avg_head = torch.mean(attn_weights, dim=0) 

            attention_avg_head_list.append(attention_avg_head)

        return attention_avg_head_list, hr_attention_avg_head_list, hr_local_image_fea_hws


    def get_text_image_attn_flash_attn(self, rank_layer, features, position_ids, attention_mask, labels,
                            image_token_posi, image_tokens, training=True, key_text_indices=None,
                            local_image_rel_begin_list=None):
        batch_size = features.shape[0]
        attention_avg_head_list = []

        assert len(image_tokens) == len(key_text_indices)

        if position_ids is None:
            position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size,features.shape[1]), dtype=torch.bool, device=features.device)
        else:
            attention_mask = attention_mask.bool()
        if labels is None:
            labels = torch.full((batch_size,features.shape[1]), IGNORE_INDEX, device=features.device)

        self_attn = self.layers[rank_layer].self_attn
        features = self.layers[rank_layer].input_layernorm(features)  

        attn_output_special, _, _ = self_attn(
            hidden_states=features,  # [4, 1372, 4096]
            attention_mask=attention_mask,
            position_ids=position_ids,
            special_indices_for_valuemat=key_text_indices
        )

        for bs in range(batch_size):
            image_index = image_token_posi[bs]
            init_output = attn_output_special[bs]
            key_text_indice = key_text_indices[bs]
            img_attn_weights = init_output[image_index:image_index+image_tokens[bs],:].contiguous()
            if local_image_rel_begin_list[bs] != -1:
                local_image_begin_idx = image_index + local_image_rel_begin_list[bs]
            else:
                local_image_begin_idx = image_index
            attn_weights = init_output  # [1212,4096]
            attn_weights = attn_weights[local_image_begin_idx:image_index+image_tokens[bs],:].contiguous() # [1153,4096]
            attn_avg = torch.mean(attn_weights, dim=-1)
            attention_avg_head_list.append(attn_avg)

        return attention_avg_head_list


    def prune_local_features(self, attention_avg_head, local_img_fea, pruning_ratio, now_level_attn_mask=None):
        attention_scores_local = attention_avg_head.squeeze()  # [N]
        assert attention_scores_local.size(0) == local_img_fea.size(0), "Attention scores and local_img_fea size mismatch"

        if now_level_attn_mask is not None:
            assert now_level_attn_mask.size(0) == attention_scores_local.size(0), "Attention mask size mismatch"
            valid_indices = torch.where(now_level_attn_mask == 1)[0].to(attention_scores_local.device)  # Indices where mask is 1
            attention_scores_local = attention_scores_local[valid_indices]  # Filtered attention scores
        else:
            valid_indices = torch.arange(attention_scores_local.size(0), device=attention_scores_local.device)  # All indices
        top_k = int(attention_scores_local.size(0) * pruning_ratio)
        top_k_scores, top_k_indices = torch.topk(attention_scores_local, top_k, largest=True)
        ori_top_k_indices = valid_indices[top_k_indices]
        sorted_top_k_indices, _ = torch.sort(ori_top_k_indices)
        pruned_local_img_fea = local_img_fea[sorted_top_k_indices, :]

        return pruned_local_img_fea

    def prune_local_features_save_imgnewline(self, attention_avg_head, local_img_fea, tokens_per_row, pruning_ratio, now_level_attn_mask=None):
        attention_scores_local = attention_avg_head.squeeze()  # [N]
        assert attention_scores_local.size(0) == local_img_fea.size(0), "Attention scores and local_img_fea size mismatch"

        if now_level_attn_mask is not None:
            assert now_level_attn_mask.size(0) == attention_scores_local.size(0), "Attention mask size mismatch"
            valid_indices = torch.where(now_level_attn_mask == 1)[0].to(attention_scores_local.device)  # Indices where mask is 1
            attention_scores_local = attention_scores_local[valid_indices]  # Filtered attention scores
        else:
            valid_indices = torch.arange(attention_scores_local.size(0), device=attention_scores_local.device)  # All indices
        top_k = int(attention_scores_local.size(0) * pruning_ratio)
        top_k_scores, top_k_indices = torch.topk(attention_scores_local, top_k, largest=True)
        # Map the top-k indices back to the original indices
        selected_indices = valid_indices[top_k_indices]

        newline_indices = torch.arange(tokens_per_row - 1, local_img_fea.size(0), tokens_per_row, device=local_img_fea.device)
        selected_indices = torch.cat([selected_indices, newline_indices]).unique()

        sorted_top_k_indices, _ = torch.sort(selected_indices)
        pruned_local_img_fea = local_img_fea[sorted_top_k_indices, :]

        return pruned_local_img_fea

    def prune_and_construct_features(self, selected_hidden_states,
                                     attention_avg_head_last, obj_bbox_infos, image_tokens,
                                     image_token_posis, local_image_rel_begin_list,
                                     inputs_embeds, labels, labels_for_img, pruning_ratio,
                                     IGNORE_INDEX, local_image_fea_hws,
                                     vision_tower=None,
                                     mlp_projector=None,
                                     image_newline=None,
                                     use_pyramid=False, next_level_index=1,
                                     now_level_attn_mask=None, 
                                     pyramid_end=False,
                                     last_lvl_feature_use_for_concat=None):
        prune_new_input_embeds = []
        prune_new_labels = []
        next_lvl_embed_masks = []
        new_image_tokens = []
        next_lvl_fea_token_hws = []
        new_labels_for_img = []
        now_lvl_pruned_local_feas = []

        for bs, attention_avg_head in enumerate(attention_avg_head_last):
            obj_bbox_info = obj_bbox_infos[bs]
            hidden_state = selected_hidden_states[bs]
            image_token = image_tokens[bs]
            image_token_posi = image_token_posis[bs]
            local_image_rel_begin = local_image_rel_begin_list[bs]
            inputs_embed = inputs_embeds[bs]

            sys_text_end = image_token_posi   
            end_text_begin = image_token_posi+image_token

            label = labels[bs]
            label_for_img = labels_for_img[bs]
            valid_seq_len = label_for_img.size(0)
            valid_inputs_embed = inputs_embed[:valid_seq_len,:]
            valid_hidden_state = hidden_state[:valid_seq_len,:]
            valid_label = label[:valid_seq_len]

            new_sys_text_label = valid_label[:sys_text_end]
            new_qs_text_label = valid_label[end_text_begin:]

        
            new_sys_text_embed = valid_inputs_embed[:sys_text_end, :]
            new_qs_text_embed = valid_inputs_embed[end_text_begin:,:]
            #######
            if pruning_ratio==1.0:
                img_fea = valid_inputs_embed[image_token_posi:end_text_begin,:]
                prune_new_input_embed = torch.cat([new_sys_text_embed, img_fea, new_qs_text_embed], dim=0)
                prune_new_label = torch.cat([new_sys_text_label,
                                    torch.full((img_fea.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype),
                                    new_qs_text_label], dim=0)
                prune_new_input_embeds.append(prune_new_input_embed)
                prune_new_labels.append(prune_new_label)
                continue

            global_img_fea = valid_inputs_embed[image_token_posi:image_token_posi+local_image_rel_begin,:]
            if now_level_attn_mask is not None:
                now_level_attn_mask = now_level_attn_mask[image_token_posi+local_image_rel_begin:end_text_begin]
    
            local_img_fea = valid_inputs_embed[image_token_posi+local_image_rel_begin:end_text_begin,:]  # [1728, 4096]
            local_token_num_h, local_token_num_w = local_image_fea_hws[bs]
            prune_way = "ratio" 

            if not use_pyramid:
                if prune_way=="ratio":
                    pruned_local_img_fea = self.prune_local_features_save_imgnewline(
                        attention_avg_head, local_img_fea,
                        tokens_per_row=local_token_num_w,
                        pruning_ratio=pruning_ratio,
                        now_level_attn_mask=now_level_attn_mask
                    )
                else:
                    pruning_threshold = 0.1
                    pruned_local_img_fea = prune_local_features_threshold_save_imgnewline(
                        attention_avg_head, local_img_fea,
                        tokens_per_row=local_token_num_w,
                        pruning_thre=pruning_threshold,
                        now_level_attn_mask=now_level_attn_mask
                    )
                now_lvl_pruned_local_feas.append(pruned_local_img_fea)
            
            else:
                if len(obj_bbox_info['scale_infos']) <= next_level_index:
                    next_level_index = len(obj_bbox_info['scale_infos'])-1

                attention_scores_local = attention_avg_head.squeeze()
                
                top_k_indices = select_attention_indices_via_ratio(attention_scores_local, 
                    local_img_fea=local_img_fea,
                    tokens_per_row=local_token_num_w, 
                    pruning_ratio=pruning_ratio,
                    now_level_attn_mask=now_level_attn_mask
                )


                pool_token_num_per_tile = vision_tower.num_patches_per_side//2
                selected_next_level_image_tensor, padded_next_level_image_fea_mask, selected_position = get_next_level_fea(obj_bbox_info,
                            top_k_indices,
                            (local_token_num_h, local_token_num_w),
                            now_lvl_attn_score=attention_scores_local,
                            pool_patch_len=pool_token_num_per_tile,
                            next_lvl_index=next_level_index)
                
                selected_next_level_image_fea = vision_tower(selected_next_level_image_tensor)
                selected_next_level_image_fea = mlp_projector(selected_next_level_image_fea)
                selected_next_level_image_fea = get_2dPool_bilinear(selected_next_level_image_fea, vision_tower)
                
                feature_dim = selected_next_level_image_fea.size(-1)
                padded_next_level_image_fea = torch.zeros((padded_next_level_image_fea_mask.size(0), padded_next_level_image_fea_mask.size(1), feature_dim),
                                                          dtype=selected_next_level_image_fea.dtype,
                                                          device=selected_next_level_image_fea.device)
                use_index = '2d'
                if use_index == '1d':
                    p_h, p_w, feature_dim = padded_next_level_image_fea.shape
                    linear_idx = torch.arange(p_h * p_w, device=padded_next_level_image_fea.device).reshape(p_h, p_w)
                    num_tiles = selected_next_level_image_fea.size(0)
                    tile_linear_indices = []
                    for i in range(num_tiles):
                        h_start, w_start = selected_position[i] * pool_token_num_per_tile
                        linear_start = h_start * p_w + w_start
                        linear_end = linear_start + pool_token_num_per_tile**2
                        linear_indices = torch.arange(linear_start, linear_end, device=padded_next_level_image_fea.device)
                    
                        padded_next_level_image_fea.view(-1, feature_dim)[linear_indices] = selected_next_level_image_fea[i].view(-1, feature_dim)
                   
                        padded_next_level_image_fea_mask.view(-1)[linear_indices] = True

                else:
                    selected_fea_begin_h = selected_position[:, 0] * pool_token_num_per_tile
                    selected_fea_begin_w = selected_position[:, 1] * pool_token_num_per_tile
                    local_h = torch.arange(pool_token_num_per_tile, device=selected_position.device).view(1, pool_token_num_per_tile, 1)  # [1, 12, 1]
                    local_w = torch.arange(pool_token_num_per_tile, device=selected_position.device).view(1, 1, pool_token_num_per_tile)  # [1, 1, 24]
             
                    global_h = selected_fea_begin_h.view(-1, 1, 1) + local_h  # [7, 12, 1]
                    global_w = selected_fea_begin_w.view(-1, 1, 1) + local_w  # [7, 1, 12]
                    
                    global_h = global_h.expand(-1, pool_token_num_per_tile, pool_token_num_per_tile)  # [7, 12, 12]
                    global_w = global_w.expand(-1, pool_token_num_per_tile, pool_token_num_per_tile)  # [7, 12, 12]
                    
                    flattened_h = global_h.contiguous().view(-1)  # [7 * 12 * 12]
                    flattened_w = global_w.contiguous().view(-1)  # [7 * 12 * 12]
                    

                    flattened_fea = selected_next_level_image_fea.flatten(0,1)  # [7 * 12 * 12, 3584]
                    flattened_fea_mask = torch.ones(
                        flattened_fea.size(0),
                        dtype=padded_next_level_image_fea_mask.dtype,
                        device=selected_position.device
                    )
                    
                    padded_next_level_image_fea.index_put_((flattened_h, flattened_w), flattened_fea, accumulate=False)
                    padded_next_level_image_fea_mask.index_put_((flattened_h, flattened_w), flattened_fea_mask, accumulate=False)

                
                padded_next_level_image_fea = padded_next_level_image_fea.permute(2,0,1).contiguous()  # [36, 60, 3584]->[3584, 36, 60]
                unpad_next_level_image_fea = unpad_image_hw(padded_next_level_image_fea, obj_bbox_info['scale_infos'][next_level_index]['resize_hw']) # [3584, 36, 60]->[3584, 36, 55]
                unpad_next_level_image_fea_mask = unpad_image_hw(padded_next_level_image_fea_mask.unsqueeze(0), obj_bbox_info['scale_infos'][next_level_index]['resize_hw']).squeeze(0) # [36, 60]->[36, 55]
                
                unpad_next_level_image_fea = torch.cat((unpad_next_level_image_fea,
                                    image_newline[:, None, None].expand(*unpad_next_level_image_fea.shape[:-1], 1).to(unpad_next_level_image_fea.device)), dim=-1)
                
                next_lvl_fea_token_hws.append((unpad_next_level_image_fea.size(1), unpad_next_level_image_fea.size(2)))
                
                newline_mask = torch.ones((unpad_next_level_image_fea_mask.size(0), 1),
                            device=unpad_next_level_image_fea_mask.device,
                            dtype=unpad_next_level_image_fea_mask.dtype)
                unpad_next_level_image_fea_mask = torch.cat((unpad_next_level_image_fea_mask, newline_mask), dim=-1)

                unpad_next_level_image_fea = unpad_next_level_image_fea.flatten(1, 2).transpose(0, 1) # [36 * 60, 3584]
                unpad_next_level_image_fea_mask = unpad_next_level_image_fea_mask.flatten(0, 1)

                if pyramid_end:
                    pruned_local_img_fea = unpad_next_level_image_fea[unpad_next_level_image_fea_mask==1]
                else: 
                    pruned_local_img_fea = unpad_next_level_image_fea


            new_prune_img_fea = torch.cat([global_img_fea, pruned_local_img_fea], dim=0)
            prune_new_input_embed = torch.cat([
                new_sys_text_embed,
                new_prune_img_fea,
                new_qs_text_embed
            ], dim=0)
            prune_new_label = torch.cat([
                new_sys_text_label,
                torch.full((new_prune_img_fea.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype),
                new_qs_text_label
            ], dim=0)

            new_label_for_img = torch.cat([
                new_sys_text_label,
                torch.full((new_prune_img_fea.shape[0],), IMAGE_TOKEN_INDEX, device=label.device, dtype=label.dtype),
                new_qs_text_label
            ], dim=0)

            if use_pyramid:
                unpad_next_level_embed_mask = torch.cat(
                    [torch.ones_like(new_sys_text_label),
                    torch.ones_like(global_img_fea[:,0]),
                    unpad_next_level_image_fea_mask,
                    torch.ones_like(new_qs_text_label)
                ], dim=0)
                next_lvl_embed_masks.append(unpad_next_level_embed_mask)
                new_image_tokens.append(new_prune_img_fea.size(0))

            prune_new_input_embeds.append(prune_new_input_embed)
            prune_new_labels.append(prune_new_label)
            new_labels_for_img.append(new_label_for_img)

        return prune_new_input_embeds, prune_new_labels, next_lvl_embed_masks, new_image_tokens, next_lvl_fea_token_hws, new_labels_for_img, now_lvl_pruned_local_feas

    def forward(self, input_ids=None,
                attention_mask=None,
                position_ids=None,
                inputs_embeds=None,
                labels=None,
                training=False,
                image_token_posis=None,
                prompt_len=None,
                llm_use_last_text=False,
                image_tokens=None,
                local_image_rel_begin_list=None,
                nopad_split_sizes=None,
                labels_for_img=None,
                obj_bbox_infos=None,
                vision_tower=None,
                mlp_projector=None,
                local_image_fea_hws=None,
                image_newline=None,
                use_pyramid=False,
                pruning_ratio=0.25):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        key_text_indices = []
        for bs in range(batch_size):
            image_index = image_token_posis[bs]
            image_token_num = image_tokens[bs]

            if training:
                if llm_use_last_text:
                    answer_index = torch.where(labels[bs] != -100)[0].tolist()
                    index_before_answer=[]
                    for index in answer_index:
                        if labels[bs][index-1]==-100 and index > 13:  # for qwen2 label begin
                            index_before_answer.append(index-1)
                    if index_before_answer==[]:
                        index_before_answer.append(len(labels[bs])-1)
                    index_before_answer=torch.tensor(index_before_answer,device=labels[0].device)
                    key_text_indice = index_before_answer
                else:
                    key_text_indice = image_index + image_token_num

            else:
                prompt_total_len = prompt_len[bs] + image_token_num
                if llm_use_last_text:
                    key_text_indice = prompt_total_len-1
                else:
                    key_text_indice = image_index+image_token_num
            key_text_indices.append(key_text_indice)
        ####
        past_key_values_length = 0
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds.clone().detach()

        all_hidden_states = []
        iter_upscale_nums = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)

            
            if layer_idx == self.num_layers-1:
                rank_layer = layer_idx+1
                features = hidden_states.clone()
                # #spda attention
                attention_avg_head_last, _, hr_local_image_fea_hws = self.get_text_image_attn(rank_layer, features, position_ids, attention_mask, labels,
                                                image_token_posis, prompt_len, llm_use_last_text, image_tokens, local_image_rel_begin_list,
                                                training=training, local_image_fea_hws=local_image_fea_hws, iter_upscale_nums=iter_upscale_nums)
                # flash attention
                # attention_avg_head_last = self.get_text_image_attn_flash_attn(rank_layer, features, position_ids, attention_mask, labels,
                #                                     image_token_posis, image_tokens, training=training, key_text_indices=key_text_indices,
                #                                     local_image_rel_begin_list=local_image_rel_begin_list)
        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        selected_hidden_states = all_hidden_states[self.select_stu_state_layer]

        if use_pyramid:
            pyramid_lvl_num = len(obj_bbox_infos[0]['scale_infos'])
            pyramid_end=False
        else:
            pyramid_end=False


        prune_new_input_embeds, prune_new_labels, next_lvl_embed_masks, new_image_tokens, next_lvl_fea_token_hws, prune_new_labels_for_img, now_lvl_pruned_local_feas = self.prune_and_construct_features(
            selected_hidden_states,
            attention_avg_head_last,
            obj_bbox_infos,
            image_tokens,
            image_token_posis,
            local_image_rel_begin_list,
            inputs_embeds,
            labels,
            labels_for_img,
            pruning_ratio,
            IGNORE_INDEX,
            local_image_fea_hws,
            vision_tower,
            mlp_projector,
            image_newline,
            use_pyramid=use_pyramid,
            next_level_index=1,
            pyramid_end=pyramid_end
        )


        if use_pyramid and not pyramid_end:

            # for next_level_index in range(2, pyramid_lvl_num):
            for next_level_index in [2]:
                pyramid_end = True
                if not pyramid_end:
                    key_text_indices = []
                    now_lvl_fea_token_hws = next_lvl_fea_token_hws
                    for bs in range(batch_size):
                        image_index = image_token_posis[bs]
                        image_token_num = new_image_tokens[bs]
                        prompt_total_len = prompt_len[bs] + image_token_num
                        if llm_use_last_text:
                            key_text_indice = prompt_total_len-1
                        else:
                            key_text_indice = image_index+image_token_num
                        key_text_indices.append(key_text_indice)
                    seq_length = prune_new_input_embeds[0].size(0)
                    past_key_values_length = 0
                    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=inputs_embeds.device)
                    position_ids = position_ids.unsqueeze(0)

                    next_lvl_input_embeds = torch.cat(prune_new_input_embeds).unsqueeze(0)
                    next_lvl_attn_mask = torch.cat(next_lvl_embed_masks)
                    next_lvl_hidden_states = next_lvl_input_embeds.clone().detach()
          
                    casual_attention_mask = _prepare_4d_causal_attention_mask(
                        next_lvl_attn_mask.unsqueeze(0), (batch_size, len(next_lvl_attn_mask)), next_lvl_hidden_states, past_key_values_length=0
                    ).to(device=next_lvl_hidden_states.device)

                    next_lvl_all_hidden_states = []
                   
                    with torch.no_grad():
                        for layer_idx, decoder_layer in enumerate(self.layers):
                            layer_outputs = decoder_layer(
                                    next_lvl_hidden_states,
                                    attention_mask=casual_attention_mask,
                                    position_ids=position_ids
                                )
                            next_lvl_hidden_states = layer_outputs[0]
                            next_lvl_all_hidden_states.append(next_lvl_hidden_states)
                      
                            if layer_idx == self.num_layers-1:
                                rank_layer = layer_idx+1
                                features = next_lvl_hidden_states.clone()
                              
                                attention_avg_head_last, hr_attention_avg_head_last, hr_local_image_fea_hws = self.get_text_image_attn(rank_layer, features,
                                                            position_ids, casual_attention_mask, prune_new_labels,
                                                            image_token_posis, prompt_len, llm_use_last_text,
                                                            new_image_tokens, local_image_rel_begin_list,
                                                            training=training, local_image_fea_hws=now_lvl_fea_token_hws)

                    next_lvl_hidden_states = self.norm(next_lvl_hidden_states)
                    next_lvl_all_hidden_states.append(next_lvl_hidden_states)
                    selected_hidden_states = next_lvl_all_hidden_states[self.select_stu_state_layer]
                    
                    prune_new_input_embeds, prune_new_labels, next_lvl_embed_masks, new_image_tokens, next_lvl_fea_token_hws, prune_new_labels_for_img, _ = self.prune_and_construct_features(
                        selected_hidden_states,
                        attention_avg_head_last,
                        obj_bbox_infos,
                        new_image_tokens,
                        image_token_posis,
                        local_image_rel_begin_list,
                        next_lvl_input_embeds,
                        prune_new_labels,     
                        prune_new_labels_for_img,
                        pruning_ratio,
                        IGNORE_INDEX,
                        now_lvl_fea_token_hws,
                        vision_tower,
                        mlp_projector,
                        image_newline,
                        use_pyramid=use_pyramid,
                        next_level_index=next_level_index,
                        now_level_attn_mask=next_lvl_attn_mask,
                        pyramid_end=pyramid_end
                    )

                else:
                    use_pyramid = False
                    key_text_indices = []
                    now_lvl_fea_token_hws = next_lvl_fea_token_hws 
                    
                    for bs in range(batch_size):
                        image_index = image_token_posis[bs]
                        image_token_num = new_image_tokens[bs]
                        prompt_total_len = prompt_len[bs] + image_token_num
                        if llm_use_last_text:
                            key_text_indice = prompt_total_len-1
                        else:
                            key_text_indice = image_index+image_token_num
                        key_text_indices.append(key_text_indice)
                    seq_length = prune_new_input_embeds[0].size(0)
                    past_key_values_length = 0
                    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=inputs_embeds.device)
                    position_ids = position_ids.unsqueeze(0)

                    next_lvl_input_embeds = torch.cat(prune_new_input_embeds).unsqueeze(0)
                    next_lvl_attn_mask = torch.cat(next_lvl_embed_masks)
                    next_lvl_hidden_states = next_lvl_input_embeds.clone().detach()
                    
                    casual_attention_mask = _prepare_4d_causal_attention_mask(
                        next_lvl_attn_mask.unsqueeze(0), (batch_size, len(next_lvl_attn_mask)), next_lvl_hidden_states, past_key_values_length=0
                    ).to(device=next_lvl_hidden_states.device)

                    next_lvl_all_hidden_states = []
                    with torch.no_grad():
                        for layer_idx, decoder_layer in enumerate(self.layers):
                            layer_outputs = decoder_layer(
                                    next_lvl_hidden_states,
                                    attention_mask=casual_attention_mask,
                                    position_ids=position_ids
                                )
                            next_lvl_hidden_states = layer_outputs[0]
                            next_lvl_all_hidden_states.append(next_lvl_hidden_states)
                            if layer_idx == self.num_layers-1: 
                                rank_layer = layer_idx+1
                                features = next_lvl_hidden_states.clone()
                                # spda attention
                                attention_avg_head_last, _, _ = self.get_text_image_attn(rank_layer, features,
                                                                position_ids, casual_attention_mask, prune_new_labels,
                                                                image_token_posis, prompt_len, llm_use_last_text,
                                                                new_image_tokens, local_image_rel_begin_list,
                                                                training=training, local_image_fea_hws=now_lvl_fea_token_hws)

                    next_lvl_hidden_states = self.norm(next_lvl_hidden_states)
                    next_lvl_all_hidden_states.append(next_lvl_hidden_states)
                    selected_hidden_states = next_lvl_all_hidden_states[self.select_stu_state_layer]

                    prune_new_input_embeds, prune_new_labels, next_lvl_embed_masks, new_image_tokens, next_lvl_fea_token_hws, prune_new_labels_for_img, _ = self.prune_and_construct_features(
                        selected_hidden_states,
                        attention_avg_head_last,
                        obj_bbox_infos,
                        new_image_tokens,
                        image_token_posis,
                        local_image_rel_begin_list,
                        next_lvl_input_embeds,   
                        prune_new_labels,        
                        prune_new_labels_for_img,
                        pruning_ratio,
                        IGNORE_INDEX,
                        now_lvl_fea_token_hws,
                        image_newline,
                        use_pyramid=use_pyramid,
                        now_level_attn_mask=next_lvl_attn_mask,
                        last_lvl_feature_use_for_concat=now_lvl_pruned_local_feas
                    )

                    break

        return prune_new_input_embeds, prune_new_labels

import copy
class VisionProjectorDistillPruner(nn.Module):
    def __init__(self, projector_type, config, use_distill):
        super().__init__()

        mm_vision_select_layer = getattr(config, 'mm_vision_select_layer', [4,8,12,16,20])
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', 'mlp2x_gelu')
        if mlp_gelu_match:
            if isinstance(mm_vision_select_layer, list) and len(mm_vision_select_layer)>1:
                self.mlp_projector = create_projectors(config, mlp_gelu_match)
            else:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                self.mlp_projector = nn.Sequential(*modules)

        if projector_type == 'mlp2x_gelu_after_distillpruner':
            config_tmp = copy.deepcopy(config)
            config_tmp._attn_implementation = "sdpa"
            # config_tmp._attn_implementation = "flash_attention_2"

            self.token_pruner = PyramidPruneModule(
                    config=config_tmp,
                    grid_size=1,
                    embed_dim=config_tmp.hidden_size,
                    num_heads=config_tmp.hidden_size // 128,
                    num_layers=len(config_tmp.llm_layer_list),
                    llm_layer_list=config_tmp.llm_layer_list,
                    temp=1.0)
        else:
            raise ValueError(f'Unknown projector type: {projector_type}')
        
        self.block_size = getattr(config, 'slice_block_size', 336)
        self.patch_size = getattr(config, 'vision_patch_size', 14)
        self.p_per_block = int(self.block_size/self.patch_size)
        self.image_aspect_ratio = getattr(config, 'image_aspect_ratio', 'slice')
    


    def load_pretrained_mlp_weights(self, pretrain_mm_mlp_adapter, pretrained_weights):
        """
        Load pretrained MLP projector weights.
        
        Args:
            pretrain_mm_mlp_adapter: Path to pretrained adapter
            pretrained_weights (dict): Pretrained weight dictionary
        """
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        state_dict = get_w(pretrained_weights, 'mm_projector')
        self.mlp_projector.load_state_dict(state_dict)


def build_vision_projector(config, use_distill=False, delay_load=False, **kwargs):
    """
    Build vision projector based on configuration.
    
    Args:
        config: Model configuration containing projector type
        use_distill (bool): Whether to use distillation mode (default: False)
        delay_load (bool): Whether to delay weight loading (default: False)
        **kwargs: Additional arguments
    
    Returns:
        nn.Module: Vision projector module (linear, MLP, or distillation-based pruner)
    """
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if 'mlp2x_gelu' in projector_type and 'distillpruner' in projector_type:
        return VisionProjectorDistillPruner(
            projector_type=projector_type,
            config=config,
            use_distill=use_distill
        )

    mm_vision_select_layer = getattr(config, 'mm_vision_select_layer', [4, 8, 12, 16, 20])
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        if isinstance(mm_vision_select_layer, list) and len(mm_vision_select_layer) > 1:
            projectors = create_projectors(config, mlp_gelu_match)
            return projectors
        else:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')