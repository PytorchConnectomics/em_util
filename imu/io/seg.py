import numpy as np
from .box import compute_bbox_all_2d, compute_bbox_all_3d


def get_seg_dtype(mid):
    m_type = np.uint64
    if mid < 2**8:
        m_type = np.uint8
    elif mid < 2**16:
        m_type = np.uint16
    elif mid < 2**32:
        m_type = np.uint32
    return m_type


def seg_to_count(seg, do_sort=True, rm_zero=False):
    sm = seg.max()
    if sm == 0:
        return None, None
    if sm > 1:
        seg_ids, seg_counts = np.unique(seg, return_counts=True)
        if rm_zero:
            seg_counts = seg_counts[seg_ids > 0]
            seg_ids = seg_ids[seg_ids > 0]
        if do_sort:
            sort_id = np.argsort(-seg_counts)
            seg_ids = seg_ids[sort_id]
            seg_counts = seg_counts[sort_id]
    else:
        seg_ids = np.array([1])
        seg_counts = np.array([np.count_nonzero(seg)])
    return seg_ids, seg_counts


def seg_remove(seg, bid=None, thres=100):
    rl = np.arange(seg.max() + 1).astype(seg.dtype)
    if bid is None:
        uid, uc = np.unique(seg, return_counts=True)
        bid = uid[uc < thres]
    rl[bid] = 0
    return rl[seg]


def seg_relabel(seg, uid=None, nid=None, do_sort=False, do_type=False):
    if seg is None or seg.max() == 0:
        return seg
    if do_sort:
        uid, _ = seg_to_count(seg, do_sort=True)
    else:
        # get the unique labels
        if uid is None:
            uid = np.unique(seg)
        else:
            uid = np.array(uid)
    uid = uid[uid > 0]  # leave 0 as 0, the background seg-id
    # get the maximum label for the segment
    mid = int(max(uid)) + 1

    # create an array from original segment id to reduced id
    # format opt
    m_type = seg.dtype
    if do_type:
        mid2 = len(uid) if nid is None else max(nid) + 1
        m_type = get_seg_type(mid2)

    mapping = np.zeros(mid, dtype=m_type)
    if nid is None:
        mapping[uid] = np.arange(1, 1 + len(uid), dtype=m_type)
    else:
        mapping[uid] = nid.astype(m_type)
    # if uid is given, need to remove bigger seg id
    seg[seg >= mid] = 0
    return mapping[seg]


def seg_biggest(seg):
    from skimage.measure import label

    mask = label(seg)
    ui, uc = np.unique(mask, return_counts=True)
    uc[ui == 0] = 0
    mid = ui[np.argmax(uc)]
    seg[mask != mid] = 0
    return seg


def seg_to_rgb(seg):
    # convert to 24 bits
    return np.stack([seg // 65536, seg // 256, seg % 256], axis=2).astype(
        np.uint8
    )


def rgb_to_seg(seg):
    # convert to 24 bits
    if seg.ndim == 2 or seg.shape[-1] == 1:
        return np.squeeze(seg)
    elif seg.ndim == 3:  # 1 rgb image
        if (seg[:, :, 1] != seg[:, :, 2]).any() or (
            seg[:, :, 0] != seg[:, :, 2]
        ).any():
            return (
                seg[:, :, 0].astype(np.uint32) * 65536
                + seg[:, :, 1].astype(np.uint32) * 256
                + seg[:, :, 2].astype(np.uint32)
            )
        else:  # gray image saved into 3-channel
            return seg[:, :, 0]
    elif seg.ndim == 4:  # n rgb image
        return (
            seg[:, :, :, 0].astype(np.uint32) * 65536
            + seg[:, :, :, 1].astype(np.uint32) * 256
            + seg[:, :, :, 2].astype(np.uint32)
        )


def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4):
    # we preprocessed the ground truth seg such that any voxel centered on a 3 x 3 x 1 window containing
    # more than one positive segment ID (zero is reserved for background) is marked as background.
    # seg=0: background
    tsz = 2 * tsz_h + 1
    sz = seg.shape
    if len(sz) == 3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(
                np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), "reflect"),
                [tsz, tsz],
            )
            p0 = patch.max(axis=1)
            patch[patch == 0] = mm + 1
            p1 = patch.min(axis=1)
            seg[z] = seg[z] * ((p0 == p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(
            np.pad(seg, ((tsz_h, tsz_h), (tsz_h, tsz_h)), "reflect"),
            [tsz, tsz],
        )
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis=1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg

def seg_biggest(seg):
    from skimage.measure import label

    mask = label(seg)
    ui, uc = np.unique(mask, return_counts=True)
    uc[ui == 0] = 0
    mid = ui[np.argmax(uc)]
    seg[mask != mid] = 0
    return seg


def seg_to_rgb(seg):
    # convert to 24 bits
    return np.stack([seg // 65536, seg // 256, seg % 256], axis=2).astype(
        np.uint8
    )


def rgb_to_seg(seg):
    # convert to 24 bits
    if seg.ndim == 2 or seg.shape[-1] == 1:
        return np.squeeze(seg)
    elif seg.ndim == 3:  # 1 rgb image
        if (seg[:, :, 1] != seg[:, :, 2]).any() or (
            seg[:, :, 0] != seg[:, :, 2]
        ).any():
            return (
                seg[:, :, 0].astype(np.uint32) * 65536
                + seg[:, :, 1].astype(np.uint32) * 256
                + seg[:, :, 2].astype(np.uint32)
            )
        else:  # gray image saved into 3-channel
            return seg[:, :, 0]
    elif seg.ndim == 4:  # n rgb image
        return (
            seg[:, :, :, 0].astype(np.uint32) * 65536
            + seg[:, :, :, 1].astype(np.uint32) * 256
            + seg[:, :, :, 2].astype(np.uint32)
        )


def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4):
    # we preprocessed the ground truth seg such that any voxel centered on a 3 x 3 x 1 window containing
    # more than one positive segment ID (zero is reserved for background) is marked as background.
    # seg=0: background
    tsz = 2 * tsz_h + 1
    sz = seg.shape
    if len(sz) == 3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(
                np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), "reflect"),
                [tsz, tsz],
            )
            p0 = patch.max(axis=1)
            patch[patch == 0] = mm + 1
            p1 = patch.min(axis=1)
            seg[z] = seg[z] * ((p0 == p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(
            np.pad(seg, ((tsz_h, tsz_h), (tsz_h, tsz_h)), "reflect"),
            [tsz, tsz],
        )
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis=1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg


def seg_iou3d(seg1, seg2, ui0=None):
    ui, uc = np.unique(seg1, return_counts=True)
    uc = uc[ui > 0]
    ui = ui[ui > 0]
    ui2, uc2 = np.unique(seg2, return_counts=True)

    if ui0 is None:
        ui0 = ui
    if len(ui0) == 0:
        return None

    out = np.zeros((len(ui0), 5), int)
    bbs = compute_bbox_all_3d(seg1, uid=ui0)[:, 1:]
    out[:, 0] = ui0
    out[:, 2] = uc[np.in1d(ui, ui0)]

    for j, i in enumerate(ui0):
        bb = bbs[j]
        ui3, uc3 = np.unique(
            seg2[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1, bb[4] : bb[5] + 1]
            * (
                seg1[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1, bb[4] : bb[5] + 1]
                == i
            ),
            return_counts=True,
        )
        uc3[ui3 == 0] = 0
        out[j, 1] = ui3[np.argmax(uc3)]
        out[j, 3] = uc2[ui2 == out[j, 1]]
        out[j, 4] = uc3.max()
    return out


def seg_iou2d(seg1, seg2, ui0=None, bb1=None, bb2=None):
    # bb1/bb2: first column of indexing, last column of size

    if bb1 is None:
        ui, uc = np.unique(seg1, return_counts=True)
        uc = uc[ui > 0]
        ui = ui[ui > 0]
    else:
        # should contain the seg count
        assert bb1.shape[1] == 6
        ui = bb1[:, 0]
        uc = bb1[:, -1]

    if bb2 is None:
        ui2, uc2 = np.unique(seg2, return_counts=True)
    else:
        assert bb2.shape[1] == 6
        ui2 = bb2[:, 0]
        uc2 = bb2[:, -1]

    if bb1 is None:
        if ui0 is None:
            bb1 = compute_bbox_all_2d(seg1, uid=ui)
            ui0 = ui
        else:
            bb1 = compute_bbox_all_2d(seg1, uid=ui0)
    else:
        if ui0 is None:
            ui0 = ui
        else:
            # make sure the order matches..
            bb1 = bb1[np.in1d(bb1[:, 0], ui0)]
            ui0 = bb1[:, 0]

    if len(ui0) == 0:
        return None

    out = np.zeros((len(ui0), 5), int)
    out[:, 0] = ui0
    out[:, 2] = uc[np.in1d(ui, ui0)]

    for j, i in enumerate(ui0):
        bb = bb1[j, 1:]
        ui3, uc3 = np.unique(
            seg2[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1]
            * (seg1[bb[0] : bb[1] + 1, bb[2] : bb[3] + 1] == i),
            return_counts=True,
        )
        uc3[ui3 == 0] = 0
        if (ui3 > 0).any():
            out[j, 1] = ui3[np.argmax(uc3)]
            out[j, 3] = uc2[ui2 == out[j, 1]]
            out[j, 4] = uc3.max()
    return out

def seg_postprocess(seg, sids=[]):
    # watershed fill the unlabeled part
    if seg.ndim == 3:
        for z in range(seg.shape[0]):
            seg[z] = mahotas.cwatershed(seg[z] == 0, seg[z])
            for sid in sids:
                tmp = binary_fill_holes(seg[z] == sid)
                seg[z][tmp > 0] = sid
    elif seg.ndim == 2:
        seg = mahotas.cwatershed(seg == 0, seg)
    return seg

def remove_segment_gaps(image, iteration=4):
    """Clean image using dilation and erosion

     Parameters
    ----------
    image : np.ndarray
        original image
    iteration : int
        number of interation to run erosion and dilation

    Returns
    -------
    image_cleaned : np.ndarray

    References
    ----------
    [1]: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    """
    height = len(image)
    width = len(image[0])
    layers = np.zeros((image.max() + 1, height, width))
    kernel = np.ones((3, 3), np.uint8)
    image = np.float32(image)
    # split each segment into a layer
    for i in range(height):
        for j in range(width):
            color = image[i][j]
            layers[int(color), i, j] = color
    # clean each layer
    for i in range(len(layers)):
        image = layers[i]
        dilation = cv2.dilate(image, kernel, iterations=iteration)
        erosion = cv2.erode(dilation, kernel, iterations=iteration)
        layers[i] = erosion
    image_cleaned = np.int32(np.max(layers, axis=0))
    return image_cleaned
