import torch
import math
from torchvision.ops.boxes import box_area

class LS_Image_Patch:
    def __init__(self, image_size=336, block_size=336):
        self.block_size = block_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

    def calculate_blocks(self, img_width, img_height, if_ceil=False):
        h_blocks = round(img_height / self.block_size)
        w_blocks = round(img_width / self.block_size)

        h_blocks = max(1, h_blocks)
        w_blocks = max(1, w_blocks)

        if if_ceil:
            h_blocks = math.ceil(img_height / self.block_size)
            w_blocks = math.ceil(img_width / self.block_size)
        return h_blocks, w_blocks
