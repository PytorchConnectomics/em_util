import numpy as np
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from ..io import read_vol, compute_bbox_all, read_vol, write_h5


def seg_to_iou(seg0, seg1, uid0=None, bb0=None, uid1=None, uc1=None, th_iou=0):
    """
    Compute the intersection over union (IoU) between segments in two segmentation maps (2D or 3D).

    Args:
        seg0 (numpy.ndarray): The first segmentation map.
        seg1 (numpy.ndarray): The second segmentation map.
        uid0 (numpy.ndarray, optional): The segment IDs to compute IoU for in the first segmentation map. Defaults to None.
        bb0 (numpy.ndarray, optional): The bounding boxes of segments in the first segmentation map. Defaults to None.
        uid1 (numpy.ndarray, optional): The segment IDs in the second segmentation map. Defaults to None.
        uic2 (numpy.ndarray, optional): The segment counts in the second segmentation map. Defaults to None.

    Returns:
        numpy.ndarray: An array containing the segment IDs, the best matching segment IDs, the segment counts, and the maximum overlap counts.

    Notes:
        - The function computes the intersection over union (IoU) between segments in two segmentation maps.
        - The IoU is computed for the specified segment IDs in `uid0`.
        - If `uid0` is not provided, the IoU is computed for all unique segment IDs in `seg0`.
    """
    assert (
        np.abs(np.array(seg0.shape) - seg1.shape)
    ).max() == 0, "seg0 and seg1 should have the same shape"
    if bb0 is not None:
        if seg0.ndim == 2:
            assert (
                bb0.shape[1] == 6
            ), "input bounding box for 2D segment has 6 columns [seg_id, ymin, ymax, xmin, xmax, count]"
        elif seg0.ndim == 3:
            assert (
                bb0.shape[1] == 8
            ), "input bounding box for 3D segment has 8 columns [seg_id, zmin, zmax, ymin, ymax, xmin, xmax, count]"
        else:
            raise "segment should be either 2D or 3D"

    # seg0 info: uid0, uc1, bb0
    # uid0 can be a subset of seg ids
    if uid0 is None:
        if bb0 is None:
            bb0 = compute_bbox_all(seg0, True)
        uid0 = bb0[:, 0]
    elif bb0 is None:
        bb0 = compute_bbox_all(seg0, True, uid0)
    else:
        # select the boxes correspond to uid0
        bb0 = bb0[np.in1d(bb0[:, 0], uid0)]
        uid0 = bb0[:, 0]
    uc0 = bb0[:, -1]

    # seg1 info: uid1, uc1
    if uid1 is None or uc1 is None:
        uid1, uc1 = np.unique(seg1, return_counts=True)

    out = np.zeros((len(uid0), 5), int)
    out[:, 0] = uid0
    out[:, 2] = uc0

    for j, i in enumerate(uid0):
        bb = bb0[j, 1:]
        if seg0.ndim == 2:
            ui3, uc3 = np.unique(
                seg1[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1]
                * (seg0[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1] == i),
                return_counts=True,
            )
        else:
            ui3, uc3 = np.unique(
                seg1[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1, bb[4] : bb[5] + 1]
                * (seg0[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1, bb[4] : bb[5] + 1] == i),
                return_counts=True,
            )
        uc3[ui3 == 0] = 0
        if (ui3 > 0).any():
            out[j, 1] = ui3[np.argmax(uc3)]
            out[j, 3] = uc1[uid1 == out[j, 1]]
            out[j, 4] = uc3.max()
    if th_iou > 0:
        score = out[:, 4].astype(float) / (out[:, 2] + out[:, 3] - out[:, 4])
        gid = score > th_iou
        return out[gid]

    return out


def segs_to_iou(get_seg, index, th_iou=0):
    # get_seg function:
    # raw iou result or matches
    out = [[]] * (len(index) - 1)
    seg0 = get_seg(index[0])
    bb0 = compute_bbox_all(seg0, True)
    out = [[]] * (len(index) - 1)
    for i, z in enumerate(tqdm(index[1:])):
        seg1 = get_seg(z)
        bb1 = compute_bbox_all(seg1, True)
        if bb1 is not None:
            iou = seg_to_iou(seg0, seg1, bb0=bb0, uid1=bb1[:, 0], uc1=bb1[:, -1])
            if th_iou == 0:
                # store all iou
                out[i] = iou
            else:
                # store matches
                # remove background seg id
                iou = iou[iou[:, 1] != 0]
                score = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])
                gid = score > th_iou
                out[i] = iou[gid, :2]
            bb0 = bb1
            seg0 = seg1
        else:
            print(f"empty slice {i}")
            # assume copy the slice from before
            if bb0 is not None:
                out[i] = np.zeros([bb0.shape[0], 5], dtype=seg0.dtype)
                out[i][:, :2] = bb0[:, :1]
                out[i][:, 2:] = bb0[:, -1:]        
    return out
