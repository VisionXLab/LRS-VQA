

import argparse
import codecs
import datetime
import itertools
import os
import os.path as osp
from math import ceil
from multiprocessing import Manager, Pool
import cv2
import numpy as np
import torch
from PIL import Image

def poly2hbb(polys):
    """Convert polygons to horizontal bboxes.

    Args:
        polys (np.array): Polygons with shape (N, 8)

    Returns:
        np.array: Horizontal bboxes.
    """
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)



def bbox_overlaps_iof(bboxes1, bboxes2, eps=1e-6):
    """Compute bbox overlaps (iof).

    Args:
        bboxes1 (np.array): Horizontal bboxes1.
        bboxes2 (np.array): Horizontal bboxes2.
        eps (float, optional): Defaults to 1e-6.

    Returns:
        np.array: Overlaps.
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]

    if rows * cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)

    hbboxes1 = poly2hbb(bboxes1)
    hbboxes2 = bboxes2
    hbboxes1 = hbboxes1[:, None, :]
    lt = np.maximum(hbboxes1[..., :2], hbboxes2[..., :2])
    rb = np.minimum(hbboxes1[..., 2:], hbboxes2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    l, t, r, b = [bboxes2[..., i] for i in range(4)]
    polys2 = np.stack([l, t, r, t, r, b, l, b], axis=-1)
    if shgeo is None:
        raise ImportError('Please run "pip install shapely" '
                          'to install shapely first.')
    sg_polys1 = [shgeo.Polygon(p) for p in bboxes1.reshape(rows, -1, 2)]
    sg_polys2 = [shgeo.Polygon(p) for p in polys2.reshape(cols, -1, 2)]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def get_sliding_window(info, sizes, gaps, img_rate_thr):
    """Get sliding windows.

    Args:
        info (dict): Dict of image's width and height.
        sizes (list): List of window's sizes.
        gaps (list): List of window's gaps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        list[np.array]: Information of valid windows.
    """
    eps = 0.01
    windows = []
    width, height = info['width'], info['height']
    for size, gap in zip(sizes, gaps):
        assert size > gap, f'invaild size gap pair [{size} {gap}]'
        step = size - gap

        x_num = 1 if width <= size else ceil((width - size) / step + 1)
        x_start = [step * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size > width:
            x_start[-1] = width - size

        y_num = 1 if height <= size else ceil((height - size) / step + 1)
        y_start = [step * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size > height:
            y_start[-1] = height - size

        start = np.array(
            list(itertools.product(x_start, y_start)), dtype=np.int64)
        stop = start + size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates > img_rate_thr).any():
        max_rate = img_rates.max()
        img_rates[abs(img_rates - max_rate) < eps] = 1
    return windows[img_rates > img_rate_thr]

def get_window_obj(info, windows, iof_thr):
    """

    Args:
        info (dict): Dict of bbox annotations.
        windows (np.array): information of sliding windows.
        iof_thr (float): Threshold of overlaps between bbox and window.

    Returns:
        list[dict]: List of bbox annotations of every window.
    """
    bboxes = info['ann']['bboxes']  # shape[n,8], windows shpe[m,4]
    iofs = bbox_overlaps_iof(bboxes, windows) # [n,m]

    window_anns = []
    for i in range(windows.shape[0]):
        win_iofs = iofs[:, i]
        pos_inds = np.nonzero(win_iofs >= iof_thr)[0].tolist()

        win_ann = dict()
        for k, v in info['ann'].items():
            try:
                win_ann[k] = v[pos_inds]
            except TypeError:
                win_ann[k] = [v[i] for i in pos_inds]
        win_ann['trunc'] = win_iofs[pos_inds] < 1  
        window_anns.append(win_ann)

    return window_anns, iofs


def crop_and_save_img(info, windows, window_anns, img_dir, no_padding,
                      padding_value, save_dir, anno_dir, img_ext, box_patch_matrix,
                      win_box_iofs, gt_relations, downsample_ratio, small_obj_thr):
    """

    Args:
        info (dict): Image's information.
        windows (np.array): information of sliding windows.
        window_anns (list[dict]): List of bbox annotations of every window.
        img_dir (str): Path of images.
        no_padding (bool): If True, no padding.
        padding_value (tuple[int|float]): Padding value.
        save_dir (str): Save filename.
        anno_dir (str): Annotation filename.
        img_ext (str): Picture suffix.
        box_patch_matrix (np.array): matrix of valid boxes in patches, shape[box.shape[0],patch.shape[0]]

    Returns:
        list[dict]: Information of paths.
    """
    # img = cv2.imread(osp.join(img_dir, info['filename']))
    img = cv2.imread(info['ori_path'])
    patch_infos = []
    all_boxes = info['ann']['bboxes']

    for i in range(windows.shape[0]):
        patch_info = dict()
        for k, v in info.items():
            if k not in ['id', 'fileanme', 'width', 'height', 'ann']:
                patch_info[k] = v

        ann = window_anns[i]
        ann['bboxes'] = translate(ann['bboxes'], -x_start, -y_start)
        patch_info['ann'] = ann

        patch = img[y_start:y_stop, x_start:x_stop]
        if not no_padding:
            height = y_stop - y_start
            width = x_stop - x_start
            if height > patch.shape[0] or width > patch.shape[1]:
                padding_patch = np.empty((height, width, patch.shape[-1]),
                                         dtype=np.uint8)
                if not isinstance(padding_value, (int, float)):
                    assert len(padding_value) == patch.shape[-1]
                padding_patch[...] = padding_value
                padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                patch = padding_patch

                # Calculate padding area and total area
                padding_area = (height * width) - (patch.shape[0] * patch.shape[1])
                total_area = height * width
                padding_ratio = padding_area / total_area
            else:
                padding_ratio = 0

        patch_info['height'] = patch.shape[0]
        patch_info['width'] = patch.shape[1]
        patch_info['ori_width'] = info['width']
        patch_info['ori_height'] = info['height']
        patch_info['padding_ratio'] = padding_ratio

        new_size = (int(patch.shape[1] / downsample_ratio), int(patch.shape[0] / downsample_ratio))
        # Add "_dx" suffix to the filename
        suffix = f"_d{int(downsample_ratio)}"
        filename_with_suffix = patch_info['id'] + suffix + '.' + img_ext
        # resized_patch = cv2.resize(patch, new_size, interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(osp.join(save_dir, filename_with_suffix), resized_patch)
        patch_info['filename'] = filename_with_suffix

        patch_box_iof = patch_box_iofs[patch_valid_box_ind2]
        patch_info['ann']['box_iofs'] = patch_box_iof
        patch_info['ann']['bboxes'] = patch_valid_boxes
        patch_info['ann']['labels'] = patch_info['ann']['labels'][patch_valid_box_ind2_rel]
        patch_info['ann']['trunc'] = patch_info['ann']['trunc'][patch_valid_box_ind2_rel]

        assert patch_valid_boxes.shape[0] == ann['bboxes'].shape[0]
        assert (len(patch_info['ann']['bboxes']) == len(patch_info['ann']['labels']) and len(patch_info['ann']['labels']) == len(patch_info['ann']['trunc']))

        patch_o1_vaild = np.isin(gt_relations[:,0], patch_valid_box_ind2)
        patch_o2_vaild = np.isin(gt_relations[:,1], patch_valid_box_ind2)
        patch_triplet_vaild = patch_o1_vaild * patch_o2_vaild
        patch_valid_triplets = gt_relations[patch_triplet_vaild]

        patch_id_mapping = {old_id: new_id for new_id, old_id in enumerate(patch_valid_box_ind2)}

        for triplet in patch_valid_triplets:
            triplet[0] = patch_id_mapping.get(triplet[0], -1)
            triplet[1] = patch_id_mapping.get(triplet[1], -1)
        patch_info['ann']['triplets'] = patch_valid_triplets

        patch_infos.append(patch_info)

    return patch_infos
