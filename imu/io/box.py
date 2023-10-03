import numpy as np


def compute_bbox(seg, do_count=False):
    """Find the bounding box of the binary segmentation"""
    # input: binary segmentation (any dimension)
    # output: bounding box of the foreground segment
    # example: [y0, y1, x0, x1, count (optional)]
    if not seg.any():
        return None

    out = []
    pix_nonzero = np.where(seg > 0)
    for i in range(seg.ndim):
        out += [pix_nonzero[i].min(), pix_nonzero[i].max()]

    if do_count:
        out += [len(pix_nonzero[0])]
    return out


def compute_bbox_all_2d(seg, do_count=False, uid=None):
    """Find the bounding box of the 2D instance segmentation"""
    # input: 2D instance segmentation
    # output: each row is [seg id, bounding box, count (optional)]
    sz = seg.shape
    assert len(sz) == 2
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return None
    uid_max = uid.max()
    out = np.zeros((1 + int(uid_max), 5 + do_count), dtype=np.uint32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 3] = sz[1]
    # for each row
    rids = np.where((seg > 0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[rid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 1] = np.minimum(out[sid, 1], rid)
        out[sid, 2] = np.maximum(out[sid, 2], rid)
    cids = np.where((seg > 0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, cid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 3] = np.minimum(out[sid, 3], cid)
        out[sid, 4] = np.maximum(out[sid, 4], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        out[seg_ui, -1] = seg_uc
    return out[uid]


def compute_bbox_all_3d(seg, do_count=False, uid=None):
    """Find the bounding box of the 3D instance segmentation"""
    # input: 3D instance segmentation
    # output: each row is [seg id, bounding box, count (optional)]

    sz = seg.shape
    assert len(sz) == 3
    if uid is None:
        uid = seg
    uid_max = int(uid.max())
    out = np.zeros((1 + uid_max, 7 + do_count), dtype=np.int32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 2] = -1
    out[:, 3] = sz[1]
    out[:, 4] = -1
    out[:, 5] = sz[2]
    out[:, 6] = -1

    # for each slice
    zids = np.where((seg > 0).sum(axis=1).sum(axis=1) > 0)[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 1] = np.minimum(out[sid, 1], zid)
        out[sid, 2] = np.maximum(out[sid, 2], zid)

    # for each row
    rids = np.where((seg > 0).sum(axis=0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[:, rid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 3] = np.minimum(out[sid, 3], rid)
        out[sid, 4] = np.maximum(out[sid, 4], rid)

    # for each col
    cids = np.where((seg > 0).sum(axis=0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, :, cid])
        sid = sid[(sid > 0) * (sid <= uid_max)]
        out[sid, 5] = np.minimum(out[sid, 5], cid)
        out[sid, 6] = np.maximum(out[sid, 6], cid)

    if do_count:
        seg_ui, seg_uc = np.unique(seg, return_counts=True)
        out[seg_ui[seg_ui <= uid_max], -1] = seg_uc[seg_ui <= uid_max]

    return out[np.all(out != -1, axis=-1)].astype(np.uint32)


def merge_bbox(bbox_a, bbox_b):
    # [ymin,ymax,xmin,xmax,count(optional)]    
    num_element = len(bbox_a) // 2 * 2
    out = bbox_a
    out[: num_element: 2] = np.minimum(bbox_a[: num_element: 2], 
                                       bbox_b[: num_element: 2])
    out[1: num_element: 2] = np.maximum(bbox_a[1: num_element: 2], 
                                        bbox_b[1: num_element: 2])
    if num_element != len(bbox_a): 
        out[-1] = bbox_a[-1] + bbox_b[-1]
    
    return out


def merge_bbox_one_matrix(bbox_matrix):
    # input Nx2D
    # [ymin,ymax,xmin,xmax,count(optional)]    
    num_element = len(bbox_matrix) // 2 * 2
    out = np.zeros(bbox_matrix.shape[1], int)
    out[:num_element:2] = bbox_matrix[:, :num_element:2].min(axis=0)
    out[1:num_element:2] = bbox_matrix[:, 1:num_element:2].max(axis=0)
    if num_element != bbox_matrix.shape[1]:
        out[-1] = bbox_matrix[:, -1].sum()
    return out


def merge_bbox_two_matrices(bbox_matrix_a, bbox_matrix_b):
    # [index, ymin,ymax, xmin,xmax]
    bbox_a_id,  bbox_b_id = bbox_matrix_a[:, 0], bbox_matrix_b[:, 0]
    intersect_id = np.in1d(bbox_a_id, bbox_b_id)
    if intersect_id.sum() == 0 :
        # no intersection
        return np.vstack([bbox_matrix_a, bbox_matrix_b])
    else:
        for i in np.where(intersect_id)[0]:
            bbox_a = bbox_matrix_a[i, 1:]
            bbox_b_index = bbox_b_id == bbox_a_id[i]
            bbox_b = bbox_matrix_b[bbox_b_index, 1:][0]
            bbox_matrix_b[bbox_b_index, 1:] = merge_bbox(bbox_a, bbox_b)
        out = np.vstack([bbox_matrix_a[np.logical_not(intersect_id)], bbox_matrix_b])
        return out


def merge_bbox_chunk(load_bbox, chunk, chunk_size):
    num_dim = len(chunk)

    if num_dim == 3:
        # merge 3D chunks
        out = None
        for xi in range(chunk[2]):
            for yi in range(chunk[1]):
                for zi in range(chunk[0]):
                    bbox = load_bbox(zi, yi, xi)
                    if bbox is not None:
                        # update the local coordinates to global coordinates
                        if bbox.ndim == 1:
                            bbox = bbox.reshape(1, -1)
                        bbox[:, 1:7] += [
                            chunk_size[0] * zi,
                            chunk_size[0] * zi,
                            chunk_size[1] * yi,
                            chunk_size[1] * yi,
                            chunk_size[2] * xi,
                            chunk_size[2] * xi,
                        ]
                        if out is None:
                            out = bbox
                        else:  # merge bbox
                            out = merge_bbox_two_matrices(out, bbox)
    elif num_dim == 2:
        # merge 2D chunks
        out = None
        for xi in range(chunk[1]):
            for yi in range(chunk[0]):
                bbox = load_bbox(yi, xi)
                if bbox is not None:
                    # update the local coordinates to global coordinates
                    bbox[:, 1:5] += [
                        chunk_size[0] * yi,
                        chunk_size[0] * yi,
                        chunk_size[1] * xi,
                        chunk_size[1] * xi,
                    ]
                    if out is None:
                        out = bbox
                    else:  # merge bbox
                        out = merge_bbox_two_matrices(out, bbox)
    return out
