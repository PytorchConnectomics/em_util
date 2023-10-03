import numpy as np
from ..io import read_vol, compute_bbox_all_2d, seg_iou2d
from .region_graph import merge_id
from skimage.morphology import remove_small_objects
from tqdm import tqdm


def pred_to_seg2d(seg, th_opt=[0, 0.9 * 255]):
    # https://www.frontiersin.org/articles/10.3389/fnana.2018.00092/full
    if th_opt[0] == 0:  # cc
        from skimage.measure import label
    elif th_opt[0] == 1:  # watershed
        from .seg import imToSeg_2d
    print("find global id")
    seg_cc = np.zeros(seg.shape, np.uint32)
    mid = 0
    for z in range(seg.shape[0]):
        if th_opt[0] == 0:  # cc
            th_pred = th_opt[1]
            tmp = label(seg[z] > th_pred, 4)
        elif th_opt[0] == 1:  # watershed
            th_hole, th_small, seed_footprint = th_opt[1], th_opt[2], th_opt[3]
            tmp = imToSeg_2d(seg[z], th_hole, th_small, seed_footprint)
        elif th_opt[0] == -1:  # direct assignment
            tmp = seg[z]
        tmp = tmp.astype(np.uint32)
        tmp[tmp > 0] += mid
        seg_cc[z] = tmp
        mid = tmp.max()
    return seg_cc


def id_to_global_id(fn_seg, im_id):
    out = np.zeros(1 + len(im_id), int)
    for i, zz in enumerate(tqdm(im_id)):
        out[i + 1] = read_vol(fn_seg % zz).max()
    out = np.cumsum(out)
    return out


def iou_to_matches(fn_iou, im_id, global_id=None, th_iou=0.1):
    # assume each 2d seg id is not overlapped
    mm = [None] * (len(im_id))
    for z in tqdm(range(len(im_id))):
        iou = read_vol(fn_iou % im_id[z])
        sc = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])
        gid = sc > th_iou
        mm[z] = iou[gid, :2].T
        if global_id is not None:
            mm[z][0] += global_id[z]
            mm[z][1] += global_id[z + 1]
    return np.hstack(mm)


def seg2d_to_IoU(seg3d, th_iou=0):
    # raw iou result or matches
    ndim = 5 if th_iou == 0 else 2
    out = [np.zeros([ndim, 0])] * (seg3d.shape[0] - 1)
    bb_pre = compute_bbox_all_2d(seg3d[0], True)
    for z in range(seg3d.shape[0] - 1):
        bb_new = compute_bbox_all_2d(seg3d[z + 1], True)
        if bb_pre is not None and bb_new is not None:
            iou = seg_iou2d(seg3d[z], seg3d[z + 1], bb1=bb_pre, bb2=bb_new)
            if iou is not None:
                if th_iou == 0:
                    out[z] = iou.T
                else:
                    # remove matches to 0
                    iou = iou[iou[:, 1] != 0]
                    sc = iou[:, 4].astype(float) / (
                        iou[:, 2] + iou[:, 3] - iou[:, 4]
                    )
                    gid = sc > th_iou
                    out[z] = iou[gid, :2].T
        bb_pre = bb_new
    return np.hstack(out)


def seg2d_mapping(seg, mapping):
    mapping_len = np.uint64(len(mapping))
    mapping_max = mapping.max()
    ind = seg < mapping_len
    seg[ind] = mapping[seg[ind]]  # if within mapping: relabel
    seg[np.logical_not(ind)] -= (
        mapping_len - mapping_max
    )  # if beyond mapping range, shift left
    return seg


def seg2d_to_global_id(seg, mapping=None, mid=None, th_sz=-1):
    if mapping is None:
        mid = mid.astype(np.uint32)
        mapping = merge_id(mid[0], mid[1])

    seg = seg2d_mapping(seg, mapping)
    if th_sz > 0:
        seg = remove_small_objects(seg, th_sz)
    return seg


def seg2d_to_3d(seg, matches=None, iou=None, th_iou=0.1, th_sz=-1):
    if matches is None:
        matches = iou_to_matches(iou, th_iou)
    seg = seg2d_to_global_id(seg, mid=matches, th_sz=th_sz)
    return seg
