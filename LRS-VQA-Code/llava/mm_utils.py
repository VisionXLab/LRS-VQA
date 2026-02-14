from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
import torch.nn.functional as F
import numpy as np

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image, (new_width, new_height)


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints, return_info=False):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded, resize_wh = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    if not return_info:
        return torch.stack(image_patches, dim=0)
    else:
        image_tensor = torch.stack(image_patches, dim=0)
        img_w, img_h = image_padded.size
        ori_img_w, ori_img_h = image.size
        resize_hw = (resize_wh[1], resize_wh[0])
        resize_padding_hw = (best_resolution[1], best_resolution[0])
        return image_tensor, resize_hw, resize_padding_hw, (ori_img_h, ori_img_w)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

from torchvision.transforms import Compose, ToTensor, Normalize
Preprocess = Compose([ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])


def calculate_blocks(img_width, img_height, if_ceil=False):
    h_blocks = round(img_height / 336)
    w_blocks = round(img_width / 336)

    h_blocks = max(1, h_blocks)
    w_blocks = max(1, w_blocks)

    if if_ceil:
        h_blocks = math.ceil(img_height / 336)
        w_blocks = math.ceil(img_width / 336)
    return h_blocks, w_blocks

def process_images(images, image_processor, model_cfg, image_patch_class=None):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    obj_bbox_info={}

    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
        obj_bbox_info['block_num_hw'] =(1, 1)

    elif image_aspect_ratio == "anyres":
        image_pil = images[0] 
        image = Preprocess(image_pil).unsqueeze(0)
        h, w = image.shape[-2:]
        obj_bbox_info['image_height'] = h
        obj_bbox_info['image_width'] = w

        image_tensor = process_anyres_image(image_pil, image_processor, model_cfg.image_grid_pinpoints)
        return image_tensor, obj_bbox_info


    elif image_aspect_ratio == "anyres_unlimited":
        image_pil = images[0]
        image = Preprocess(image_pil).unsqueeze(0)
        block_size = 336
        h, w = image.shape[-2:]
        h_block, w_block = calculate_blocks(img_width=w, img_height=h, if_ceil=True)
        obj_bbox_info['image_height'] = h
        obj_bbox_info['image_width'] = w
        max_block_num=12
        if h_block >= max_block_num or w_block >= max_block_num:
            max_img_long_side = max_block_num * block_size
            if h >= w:
                resized_h = max_img_long_side
                resized_w = round(resized_h * w / h)
            else:
                resized_w = max_img_long_side
                resized_h = round(resized_w * h / w)
        else:
            new_h = h_block * block_size
            new_w = w_block * block_size

            h_scale = new_h / h
            w_scale = new_w / w
            scale = min(h_scale, w_scale)

            resized_h = round(h * scale)
            resized_w = round(w * scale)
        h_block, w_block = calculate_blocks(img_width=resized_w, img_height=resized_h, if_ceil=True)
        image_grid_pinpoints = [(h_block*block_size, w_block*block_size)]

        image_tensor = process_anyres_image(image_pil, image_processor, image_grid_pinpoints)
        print(f'image_tensor size:{image_tensor.size()}')
        return image_tensor, obj_bbox_info

    elif image_aspect_ratio == 'slice':
        max_block_num = model_cfg.max_block_num

        # for image in images:
        image = images[0]
        image = Preprocess(image).unsqueeze(0)
        h, w = image.shape[-2:]

        block_size = image_patch_class.block_size
        h_block, w_block = image_patch_class.calculate_blocks(img_width=w, img_height=h, if_ceil=True)

        if h_block >= max_block_num or w_block >= max_block_num:
            max_img_long_side = max_block_num * block_size
            if h >= w:
                resized_h = max_img_long_side
                resized_w = round(resized_h * w / h)
            else:
                resized_w = max_img_long_side
                resized_h = round(resized_w * h / w)
        else:

            new_h = h_block * block_size
            new_w = w_block * block_size

            h_scale = new_h / h
            w_scale = new_w / w
            scale = min(h_scale, w_scale)

            resized_h = round(h * scale)
            resized_w = round(w * scale)

        h_block = math.ceil(resized_h / block_size)
        w_block = math.ceil(resized_w / block_size)

        image_inter = F.interpolate(image, size=(resized_h, resized_w), mode='bilinear')
        obj_bbox_info['resize_hw']=(resized_h, resized_w)
        obj_bbox_info['resize_padding_hw']=(block_size * h_block, block_size * w_block)
        obj_bbox_info['resize_ratio_hw'] =(resized_h/h, resized_w/w)
        obj_bbox_info['block_num_hw'] =(h_block, w_block)
        obj_bbox_info['image_height'] = h
        obj_bbox_info['image_width'] = w

        split_images = []
        if h_block * w_block > 1:  # create global LR image
            if h >= w:
                h_g = block_size
                w_g = round(block_size * w / h)
            else:
                w_g = block_size
                h_g = round(block_size * h / w)

            image_global = F.interpolate(image, size=(h_g, w_g), mode='bilinear')
            image_g = torch.zeros((1, 3, block_size, block_size)).to(dtype=image_global.dtype, device=image_global.device)
            image_g[:, :, :h_g, :w_g] = image_global
            split_images.append(image_g)

        image_padded = torch.zeros((1, 3, block_size * h_block, block_size * w_block)).to(dtype=image_inter.dtype, device=image_inter.device)
        image_padded[:, :, :resized_h, :resized_w] = image_inter
        # for i_ in range(h_block):
        #     for j_ in range(w_block):
        #         image_s = image_padded[:, :, block_size * i_:block_size * (i_ + 1), block_size * j_:block_size * (j_ + 1)]
        #         split_images.append(image_s)
        patches = image_padded.unfold(2, block_size, block_size).unfold(3, block_size, block_size) # (1, C, num_patches_h, block_size, num_patches_w, block_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        split_image_tensor = patches.squeeze(0)
        patches_flat = split_image_tensor.flatten(0,1)
        split_images.append(patches_flat)
        image_tensor = torch.cat(split_images, dim=0)

        return image_tensor, obj_bbox_info

    elif image_aspect_ratio == 'pyramid_crop':
        max_block_num = model_cfg.max_block_num
        block_size = image_patch_class.block_size
        max_img_long_side = max_block_num * block_size

        # for image in images:
        image_pil = images[0]  # one large image when inference
        image = Preprocess(image_pil).unsqueeze(0)
        img_h, img_w = image.shape[-2:]
        img_long_side = max(img_h, img_w)
        obj_bbox_info = {}
        # save resize info
        obj_bbox_info['scale_infos'] = []         
        obj_bbox_info['scale_image_tensors'] = []
       
        pyramid_resize_ratios = []
        base_HR_img_long_side = 1008
        min_scale = base_HR_img_long_side / img_long_side
        min_scale = min(min_scale, 1.0)
        current_scale = 1.0
        # Generate scale factors by iteratively downsampling by factors of 2
        while current_scale >= min_scale:
            pyramid_resize_ratios.append(current_scale)
            current_scale /= 2
        # Ensure that the smallest scale factor is included
        if pyramid_resize_ratios[-1] > min_scale:
            pyramid_resize_ratios.append(min_scale)
        ratio_global = min(block_size / img_h, block_size / img_w)
        pyramid_resize_ratios.append(ratio_global)
        pyramid_resize_ratios = sorted(list(set(pyramid_resize_ratios)), reverse=True)

        split_images = []
        image_num_split = []
        for idx, ratio in enumerate(pyramid_resize_ratios):
            h_ = round(img_h * ratio)
            w_ = round(img_w * ratio)
            h_block, w_block = image_patch_class.calculate_blocks(img_width=w_, img_height=h_, if_ceil=True)
        
            new_h = h_block * block_size
            new_w = w_block * block_size

            h_scale = new_h / h_
            w_scale = new_w / w_
            scale = min(h_scale, w_scale)

            resized_h = round(h_ * scale)
            resized_w = round(w_ * scale)
            
            h_block = math.ceil(resized_h / block_size)
            w_block = math.ceil(resized_w / block_size)
            
            image_resized = F.interpolate(image, size=(resized_h, resized_w), mode='bilinear')
            image_padded = torch.zeros((1, 3, block_size * h_block, block_size * w_block),
                                    dtype=image_resized.dtype, device=image_resized.device)
            image_padded[:, :, :resized_h, :resized_w] = image_resized

            # global image
            if idx == len(pyramid_resize_ratios)-1:
                h_g, w_g = image_resized.shape[-2:]
                image_g = torch.zeros((1, 3, block_size, block_size),
                                    dtype=image_resized.dtype, device=image_resized.device)
                image_g[:, :, :h_g, :w_g] = image_resized
                split_images.append(image_g)
                image_num_single = 1
                image_num_split.append(image_num_single)
                continue
            else:
                patches = image_padded.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
                patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
                now_scale_image_tensor = patches.squeeze(0)
                patches_flat = now_scale_image_tensor.flatten(0,1)
                split_images.append(patches_flat)
                image_num_single = patches_flat.size(0)
                image_num_split.append(image_num_single)

            scale_info = {
                'resize_hw': (resized_h, resized_w),
                'resize_padding_hw': (block_size * h_block, block_size * w_block),
                'resize_ratio_hw': (resized_h / img_h, resized_w / img_w),
                'block_num_hw': (h_block, w_block),
                'image_num': image_num_single
            }

            obj_bbox_info['scale_infos'].append(scale_info)
            obj_bbox_info['scale_image_tensors'].append(now_scale_image_tensor)

        obj_bbox_info['image_height'] = img_h
        obj_bbox_info['image_width'] = img_w
        obj_bbox_info['image_num_split'] = image_num_split[::-1]  
        obj_bbox_info['scale_infos'] = obj_bbox_info['scale_infos'][::-1] 
        obj_bbox_info['scale_image_tensors'] = obj_bbox_info['scale_image_tensors'][::-1] 
        split_images = split_images[::-1] 

        base_hr_lvl = 1 
        base_hr_info = obj_bbox_info['scale_infos'][base_hr_lvl-1]
        obj_bbox_info['resize_hw'] = base_hr_info['resize_hw']  
        obj_bbox_info['resize_padding_hw'] = base_hr_info['resize_padding_hw']
        obj_bbox_info['resize_ratio_hw'] = base_hr_info['resize_ratio_hw']
        obj_bbox_info['block_num_hw'] = base_hr_info['block_num_hw']

        final_split_images = [
            split_images[0],
            split_images[base_hr_lvl]
        ]
        
        if len(split_images) < 2:
            raise ValueError("no enough elements in split_images! (At least two: global and local image)")
        image_tensor = torch.cat(final_split_images, dim=0)
        return image_tensor, obj_bbox_info

    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images, obj_bbox_info


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def process_anyres_image_for_remoteclip(image, processor, grid_pinpoints, return_info=False):
    """
    Process an image with variable resolutions for remote clip.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.
   
    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    PATCH_SIZE = 224  # for remoteclip

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded, resize_wh = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, PATCH_SIZE)

    image_original_resize = image.resize((PATCH_SIZE, PATCH_SIZE))

    image_patches = [image_original_resize] + patches
    image_patches = [processor(image_patch)
                     for image_patch in image_patches]
    if not return_info:
        return torch.stack(image_patches, dim=0)
    else:
        image_tensor = torch.stack(image_patches, dim=0)
        img_w, img_h = image_padded.size
        ori_img_w, ori_img_h = image.size
        resize_hw = (resize_wh[1], resize_wh[0])
        resize_padding_hw = (best_resolution[1], best_resolution[0])
        return image_tensor, resize_hw, resize_padding_hw, (ori_img_h, ori_img_w)
