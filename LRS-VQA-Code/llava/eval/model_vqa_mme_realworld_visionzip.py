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
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from llava.patch_divide import LS_Image_Patch
import requests
from io import BytesIO
import re
import torch.nn.functional as F
import copy
Image.MAX_IMAGE_PIXELS = 1000000000000

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if 'qwen' in model_name.lower():
        print(f'Qwen model_name:{model_name}')
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map, **llava_model_args)

    else:
        print(f'model_name:{model_name}')
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # ##### use VisionZip
    from visionzip import visionzip
    model = visionzip(model, dominant=120, contextual=24)
    # #######

    for m in model.modules():
        m.tokenizer = tokenizer
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    max_block_num = args.max_block_num
    model.config.max_block_num = max_block_num
    
    image_patch = LS_Image_Patch(block_size=model.config.slice_block_size)

    # end_str = 'Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. \nThe best answer is:'
    end_str = 'Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.'
    # print(f'Now prompt in the end of question:{end_str}')

    choice_prompt = ' The choices are listed below: \n'
    print(f'Now prompt in the end of question:{choice_prompt} and {end_str}')

    #####
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    if_pyramid_crop = args.pyramid_crop
    if if_pyramid_crop:
        image_aspect_ratio = 'pyramid_crop'
        model.config.image_aspect_ratio = 'pyramid_crop'
        # print(f'model:{model}')
        model.model.mm_projector.token_pruner.next_level_max_select_num = args.pyramid_next_level_max_select_num  # 6


    for i, line in enumerate(tqdm(questions)):
        idx = line["Question_id"]
        question = line['Text']
        choices = "\n".join(line['Answer choices'])
        ###########
        question = question + choice_prompt
        ###########

        qs = f"{question}\n{choices}".strip()
        cur_prompt = qs

        if 'Image' in line:
            image_file = line["Image"]
            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

            if image_aspect_ratio in ['slice','pyramid_crop']:
                image_tensor, obj_bbox_info = process_images([image], image_processor, model.config, image_patch)
            else:
                image_tensor, obj_bbox_info = process_images([image], image_processor, model.config)
            obj_bbox_info['image_path'] = image_file

            images = image_tensor.half().cuda()
            image_sizes = [image.size]
            # print(f'init image sizes:{image_sizes}, image_tensor size:{image_tensor.size()}')

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

        if model_name == "llava_qwen":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[images],
                    obj_bbox_infos = [obj_bbox_info],
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

        line["Output"] = outputs
        # print(f'output:{outputs}')
        ans_file.write(json.dumps(line) + "\n")
        # print('line:\n')

        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--max_block_num", type=int, default=3)
    parser.add_argument("--pyramid_crop", type=bool, default=False)
    parser.add_argument("--pyramid_next_level_max_select_num", type=int, default=6)
    args = parser.parse_args()

    eval_model(args)
