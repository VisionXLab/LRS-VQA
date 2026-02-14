import argparse
import os
import json
import shortuuid
import base64

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

import PIL
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

import math
import re
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = 1000000000000

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Failed to import qwen_vl_utils; Please install it via pip install qwen-vl-utils")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):

    ## Qwen2VL process image limit
    min_pixels = 3136  # 4*28*28
    max_pixels = args.max_pixels

    print(f'Qwen2VL max_pixels:{max_pixels}, min_pixels:{min_pixels}')

    model_path = os.path.expanduser(args.model_path)
    model_name = "Qwen2-VL-7B-Instruct"
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.flash_attention_2:
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        # default: Load the model on the available device(s)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    max_new_tokens = args.max_new_tokens

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # end_str = 'Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. \nThe best answer is:'
    end_str = 'Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.'
    print(f'Now prompt in the end of question:{end_str}')

    for i, line in enumerate(tqdm(questions)):
        idx = line["Question_id"]
        question = line['Text']
        choices = "\n".join(line['Answer choices'])
        qs = f"{question}\n{choices}".strip()
        cur_prompt = qs

        cur_prompt = cur_prompt + '\n' + end_str

        if 'Image' in line:
            image_file = line["Image"]
            pil_image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        else:
            pil_image = None

        #######
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {"type": "text", "text": cur_prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)


        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")


        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)

        ans_id = shortuuid.uuid()
        line["Output"] = output_text[0]
        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/weight/HF_Download/models--Qwen--Qwen2-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--flash_attention_2", type=bool, default=True)
    parser.add_argument("--max_pixels", type=int, default=2048*2048)
    args = parser.parse_args()

    eval_model(args)
