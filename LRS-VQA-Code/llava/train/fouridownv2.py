import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import time


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


def ComplexResize(x, size, mode='bilinear', align_corners=False):
    # Split the tensor into its real and imaginary parts
    real_part = torch.real(x)
    imag_part = torch.imag(x)
    real_part_resized = F.interpolate(real_part, size=size, mode='bilinear', align_corners=False)
    imag_part_resized = F.interpolate(imag_part, size=size, mode='bilinear', align_corners=False)
    tensor_resized = torch.complex(real_part_resized, imag_part_resized)

    return tensor_resized


def pad_tensor_to_even(t):
    b, c, h, w = t.size()
    pad_h = 0
    pad_w = 0
    if h % 2 != 0:
        pad_h = 1
    if w % 2 != 0:
        pad_w = 1
    if pad_h != 0 or pad_w != 0:
        t = F.pad(t, (0, pad_w, 0, pad_h))
    return t


class FouriDown(nn.Module):

    def __init__(self, in_channel, out_channel, scale=2, resolute=(1000,1000)):
        super(FouriDown, self).__init__()
        self.scale = scale
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)    
        self.real_fuse = nn.Sequential(nn.Conv2d(in_channel*(scale**2), in_channel*(scale**2), 1, 1, 0, groups=in_channel), self.lrelu,
                                      nn.Conv2d(in_channel*(scale**2), in_channel*(scale**2), 1, 1, 0, groups=in_channel))
        self.imag_fuse = nn.Sequential(nn.Conv2d(in_channel*(scale**2), in_channel*(scale**2), 1, 1, 0, groups=in_channel), self.lrelu,
                                      nn.Conv2d(in_channel*(scale**2), in_channel*(scale**2), 1, 1, 0, groups=in_channel))             
        self.Downsample = nn.Upsample(scale_factor=1.0/float(self.scale), mode='bilinear', align_corners=False)
        self.channel2x = nn.Conv2d(in_channel, out_channel, 1, 1)
        
        H, W = resolute
     
        amp_fre_map = torch.zeros((H+1, W+1))
        amp_fre_map_patches = []
        if H % 2 == 0 and W % 2 == 0: 
            for row_index in range(W + 1):
                for col_index in range(H + 1):
                    amp_fre_map[col_index, row_index] = torch.norm(torch.tensor([H//2 - col_index, row_index - W//2], dtype=torch.float32))
                # print(row_index)
            amp_fre_map_wo_center_H = torch.cat((amp_fre_map[:H//2], amp_fre_map[(H//2 + 1):]), dim=0)
            amp_fre_map_wo_center = torch.cat((amp_fre_map_wo_center_H[:, :W//2], amp_fre_map_wo_center_H[:, W//2 + 1:]), dim=1)
        else:
            print(f"{img_fft.shape},To be processes")
            exit()
        amp_fre_map_wo_center = torch.split(amp_fre_map_wo_center, H//self.scale, dim=0)
        amp_fre_map_wo_center = torch.stack(amp_fre_map_wo_center, dim = 0)
        amp_fre_map_wo_center = torch.split(amp_fre_map_wo_center, W//self.scale, dim=2)
        amp_fre_map_patches = torch.concatenate(amp_fre_map_wo_center, dim = 0)
        amp_fre_map_patches_index = torch.zeros(amp_fre_map_patches.shape)
        for h in range(H//self.scale):
            for w in range(W//self.scale):
                _, amp_fre_map_patches_index[:, h, w] = torch.sort(amp_fre_map_patches[:, h, w], descending=True)
        self.amp_fre_map_patches_index = amp_fre_map_patches_index
        self.expanded_indices = self.amp_fre_map_patches_index.unsqueeze(1).expand(-1, in_channel, -1, -1).contiguous().view(-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.imag_fuse:
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.weight, 0) 
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        for layer_r in self.real_fuse:
            if isinstance(layer_r, nn.Conv2d):
                nn.init.constant_(layer_r.weight, 0)
                if layer_r.bias is not None:
                    nn.init.constant_(layer_r.bias, 0)

    def forward(self, x):

        self.expanded_indices = self.expanded_indices.to(x.device)
        B, C, H, W = x.shape
        img_fft = torch.fft.fft2(x)
        real = img_fft.real
        imag = img_fft.imag
        mid_row, mid_col = H // (self.scale**2), W // (self.scale**2)

        # Split into (self.scale**2) patches
        img_fft_patches = []
        img_fft = torch.split(img_fft, H//self.scale, dim=2)
        img_fft = torch.stack(img_fft, dim = 1)
        img_fft = torch.split(img_fft, W//self.scale, dim=4)
        img_fft_patches = torch.concatenate(img_fft, dim = 1)

        img_fft_patches_real = img_fft_patches.real.view(B, -1)
        img_fft_patches_real = torch.index_select(img_fft_patches_real, 1, self.expanded_indices.long())
        img_fft_patches_imag = img_fft_patches.imag.view(B, -1)
        img_fft_patches_imag = torch.index_select(img_fft_patches_imag, 1, self.expanded_indices.long())
        img_fft_patches = torch.complex(img_fft_patches_real, img_fft_patches_imag)
        
        # Adaptive attention
        fuse = img_fft_patches.view(B, (self.scale**2)*C, H//self.scale, W//self.scale)
        real = fuse.real
        imag = fuse.imag
        real_weight = self.real_fuse(real)
        imag_weight = self.imag_fuse(imag)
        fuse_weight = torch.complex(real_weight, imag_weight)
        fuse_weight = fuse_weight.view(B, self.scale**2, C, H // self.scale, W // self.scale)
        real_sigmoid = F.softmax(fuse_weight.real + 1/float(self.scale**2), dim=1)
        imag_sigmoid = F.softmax(fuse_weight.imag + 1/float(self.scale**2), dim=1)
        fuse_weight =  torch.complex(real_sigmoid, imag_sigmoid)
        fuse = torch.complex(real, imag)
        fuse = fuse.view(B, self.scale**2, C, H // self.scale, W // self.scale)
        fuse = fuse * fuse_weight

        # Superposing
        fuse = fuse.sum(dim=1)
        img = torch.abs(torch.fft.ifft2(fuse))
        img = img  + self.Downsample(x)
        # img =  self.lrelu(self.channel2x(img))

        return img

if __name__ == "__main__":
    down = FouriDown(3,1024,2)
    img_fft_patches = torch.randn(2, 3, 1000, 1000).to("cuda")
    print(img_fft_patches.device)
    # imag = torch.randn(2, 3, 1000, 1000)
    # img_fft_patches = torch.complex(real, imag)
    img_fft_patches = down(img_fft_patches)
    print(img_fft_patches.shape)
