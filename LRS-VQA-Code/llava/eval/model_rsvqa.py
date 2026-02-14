import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, process_anyres_image_for_remoteclip, get_anyres_image_grid_shape
from torch.utils.data import Dataset, DataLoader

import torch
from PIL import Image
import math
from llava.patch_divide import LS_Image_Patch
import requests
from io import BytesIO
import re
import torch.nn.functional as F
import copy
Image.MAX_IMAGE_PIXELS = 10000000000

import nltk
from nltk.corpus import wordnet as wn

nltk.data.path.append('./package/nltk_data-gh-pages/nltk_data')

def are_synonyms(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1.path_similarity(synset2) is not None and synset1.path_similarity(synset2) > 0.8:
                return True
    return False

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def CLIP_cal_image_text_sim_map(full_clip_model, images, ori_input_text, clip_processor, current_device):

    full_clip_model = full_clip_model.to(current_device)
    visual_projection_layer = full_clip_model.visual_projection
    clip_vision_encoder = full_clip_model.vision_model
    visual_projection_layer.eval()

    with torch.no_grad():
        clip_text_inputs = clip_processor(text=ori_input_text, padding=True, return_tensors="pt", truncation=True).to(current_device)
        clip_text_features = full_clip_model.get_text_features(**clip_text_inputs) # [1,24]->[1,768]
        clip_text_features = F.normalize(clip_text_features, p=2, dim=-1)

        image_list = [images]
        tmp_images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in tmp_images], dim=0)

        output_attentions = full_clip_model.config.output_attentions
        output_hidden_states = (full_clip_model.config.output_hidden_states)
        return_dict = full_clip_model.config.use_return_dict
        vision_outputs = full_clip_model.vision_model(
            pixel_values=concat_images,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_outputs = vision_outputs[0][:, 1:,:]
        clip_image_features = visual_projection_layer(image_outputs) 
        clip_image_features = clip_image_features / clip_image_features.norm(p=2, dim=-1, keepdim=True)

    token_level_similarities = torch.matmul(clip_image_features, clip_text_features.transpose(0, 1))
    token_level_similarities = token_level_similarities.squeeze(-1)

    return token_level_similarities


remoteclip_image_grid_pinpoints = [
    # [224, 448],
    # [448, 224],
    # [448, 448],
    [224, 672],
    [672, 224],
    [672, 672],
    [896, 672],
    [672, 896],
    [1120, 672],
    [672, 1120],
    [896, 896],  
    [1120, 896], 
    [896, 1120], 
    [224, 896],  
    [896, 224]   
]

def RemoteCLIP_cal_image_text_sim_map(remote_clip_model, image_pil, input_text, remoteclip_preprocess, remote_clip_tokenizer):
    concat_images = process_anyres_image_for_remoteclip(image_pil, remoteclip_preprocess, remoteclip_image_grid_pinpoints)

    with torch.no_grad(), torch.cuda.amp.autocast():
        remote_clip_model.visual.output_tokens = True
        pool_image_features, image_features = remote_clip_model.encode_image(concat_images.cuda())
        projected_image_features = image_features @ remote_clip_model.visual.proj 

        text_embed = remote_clip_tokenizer(input_text)
        text_features = remote_clip_model.encode_text(text_embed.cuda())
        projected_image_features /= projected_image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    token_level_similarities = torch.matmul(projected_image_features, text_features.transpose(0, 1)) #
    # Squeeze the last dimension to get
    token_level_similarities = token_level_similarities.squeeze(-1) # [10, 256]
    remote_clip_num_patch_width, remote_clip_num_patch_height = get_anyres_image_grid_shape(image_pil.size,
                                                                                remoteclip_image_grid_pinpoints, 224)

    return token_level_similarities, remote_clip_num_patch_width, remote_clip_num_patch_height


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if 'qwen' in model_name.lower():
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map, **llava_model_args)

    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    for m in model.modules():
        m.tokenizer = tokenizer
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    max_block_num = args.max_block_num
    model.config.max_block_num = max_block_num
    image_patch = LS_Image_Patch(block_size=model.config.slice_block_size)
    end_str = ''

    ### attention location calculation
    save_LLM_score = False

    if save_LLM_score:
        model.model.mm_projector.token_pruner.save_LLM_score = True

    return_prune_location = args.return_prune_location
    if return_prune_location:
        model.model.mm_projector.token_pruner.return_prune_location = True

        # print(f'=====init CLIP for compare!')
        # from transformers import AutoProcessor, CLIPModel
        # clip_path = "/weight/clip-vit-large-patch14-336"
        # full_clip_model = CLIPModel.from_pretrained(clip_path)
        # clip_processor = AutoProcessor.from_pretrained(clip_path)

        # # ### init RemoteCLIP
        import torch, open_clip
        remoteclip_model_name = 'ViT-L-14' # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
        remote_clip_model, _, remoteclip_preprocess = open_clip.create_model_and_transforms(remoteclip_model_name)
        remote_clip_tokenizer = open_clip.get_tokenizer(remoteclip_model_name)
        ckpt = torch.load(f"/weight/RemoteCLIP/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
        message = remote_clip_model.load_state_dict(ckpt)
        remote_clip_model = remote_clip_model.cuda().eval()

    else:
        model.model.mm_projector.token_pruner.return_prune_location = False


    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    if_pyramid_crop = args.pyramid_crop
    if if_pyramid_crop:
        image_aspect_ratio = 'pyramid_crop'
        model.config.image_aspect_ratio = 'pyramid_crop'
        model.model.mm_projector.token_pruner.next_level_max_select_num = args.pyramid_next_level_max_select_num  # 6

    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        qs = line["text"]
        ground_truth = line["ground_truth"]
        category = line["category"]

        h_bounding_box = line.get("hbox", None)
        r_bounding_box = line.get("rbox", None)

        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

            if image_aspect_ratio in ['slice','pyramid_crop']:
                image_tensor, obj_bbox_info = process_images([image], image_processor, model.config, image_patch)
            else:
                image_tensor, obj_bbox_info = process_images([image], image_processor, model.config)
            obj_bbox_info['image_path'] = image_file

            images = image_tensor.half().cuda()
            image_sizes = [image.size]

            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        qs = qs +'\n' + end_str
        cur_prompt = cur_prompt + '\n' + end_str

        if model_name == "llava_qwen":
            conv_template = "qwen_1_5"
            conv = copy.deepcopy(conv_templates[conv_template])
        else:
            conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if 'scale_infos' in obj_bbox_info:
            for pp, pyramid_image in enumerate(obj_bbox_info['scale_image_tensors']):
                obj_bbox_info['scale_image_tensors'][pp] = pyramid_image.to(dtype=torch.float16, device='cuda', non_blocking=True)

        if return_prune_location:
            obj_bbox_info['question_idx'] = idx
            obj_bbox_info['hbox'] = h_bounding_box
            obj_bbox_info['question_text'] = qs

            current_device = images.device
            ori_input_text = qs.replace('<image>\n','')

            if not save_LLM_score:
                
                token_level_similarities, remote_clip_num_patch_width, remote_clip_num_patch_height = RemoteCLIP_cal_image_text_sim_map(remote_clip_model, image, ori_input_text, remoteclip_preprocess, remote_clip_tokenizer)
                obj_bbox_info['token_level_similarities'] = token_level_similarities
                obj_bbox_info['remote_clip_num_patch_width'] = remote_clip_num_patch_width
                obj_bbox_info['remote_clip_num_patch_height'] = remote_clip_num_patch_height


        if model_name == "llava_qwen":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[images],
                    obj_bbox_infos=[obj_bbox_info],
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4096,
                )
        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[images],
                    obj_bbox_infos = [obj_bbox_info],
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "category": category,
                                   "text": outputs,
                                   "ground_truth": ground_truth,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "ori_hbox": h_bounding_box,
                                   "ori_rbox": r_bounding_box,
                                   "metadata": {}}) + "\n")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava_largeimg")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/dataset")
    parser.add_argument("--question-file", type=str, default="./LRS_VQA_merged.jsonl")
    parser.add_argument("--answers-file", type=str, default="./debug.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_block_num", type=int, default=3)
    parser.add_argument("--pyramid_crop", type=bool, default=False)
    parser.add_argument("--pyramid_next_level_max_select_num", type=int, default=6)
    parser.add_argument("--return_prune_location", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
