import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_opening, binary_closing, binary_fill_holes
from .image import im2col
from .arr import get_query_count

## image format
def get_seg_dtype(mid):
    """
    Get the appropriate data type for a segmentation based on the maximum ID.

    Args:
        mid (int): The maximum ID in the segmentation.

    Returns:
        numpy.dtype: The appropriate data type for the segmentation.

    Notes:
        - The function determines the appropriate data type based on the maximum ID in the segmentation.
        - The data type is selected to minimize memory usage while accommodating the maximum ID.
    """    
    m_type = np.uint64
    if mid < 2**8:
        m_type = np.uint8
    elif mid < 2**16:
        m_type = np.uint16
    elif mid < 2**32:
        m_type = np.uint32
    return m_type

def seg_to_rgb(seg):
    """
    Convert a segmentation map to an RGB image.

    Args:
        seg (numpy.ndarray): The input segmentation map.

    Returns:
        numpy.ndarray: The RGB image representation of the segmentation map.

    Notes:
        - The function converts a segmentation map to an RGB image, where each unique segment ID is assigned a unique color.
        - The RGB image is represented as a numpy array.
    """
    return np.stack([seg // 65536, seg // 256, seg % 256], axis=2).astype(
        np.uint8
    )


def rgb_to_seg(seg):
    """
    Convert an RGB image to a segmentation map.

    Args:
        seg (numpy.ndarray): The input RGB image.

    Returns:
        numpy.ndarray: The segmentation map.

    Notes:
        - The function converts an RGB image to a segmentation map, where each unique color is assigned a unique segment ID.
        - The segmentation map is represented as a numpy array.
    """
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

## seg statistics
def seg_to_count(seg, do_sort=True, rm_zero=False):
    """
    Convert a segmentation map to a count map.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        do_sort (bool, optional): Whether to sort the counts in descending order. Defaults to True.
        rm_zero (bool, optional): Whether to remove the count for the background class. Defaults to False.

    Returns:
        tuple: A tuple containing the unique segment IDs and their corresponding counts.

    Notes:
        - The function converts a segmentation map to a count map, where each unique segment ID is associated with its count.
        - The counts can be sorted in descending order if `do_sort` is set to True.
        - The count for the background class can be removed if `rm_zero` is set to True.
    """    
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

def seg_to_sphericity(seg):
    # compute the sphericity for all segments at the same time
    # https://en.wikipedia.org/wiki/Sphericity
    seg_erode = binary_erosion(seg > 0, iterations=1)
    sid, vol = np.unique(seg, return_counts=True)
    sid2, vol2 = np.unique(seg_erode * seg, return_counts=True)
    vol_erode = get_query_count(sid2, vol2, sid)
    vol_diff = vol - vol_erode
    vol_diff[sid == 0] = 0
    sphe = -np.ones(vol.shape)
    sphe[vol_diff > 0] = (
        np.pi ** (1.0 / 3) * ((6 * vol[vol_diff > 0]) ** (2.0 / 3))
    ) / vol_diff[vol_diff > 0]
    return sid, sphe, vol

## seg id manipulation

def seg_relabel(seg, uid=None, nid=None, do_sort=False, do_type=False):
    """
    Relabel the segments in a segmentation map.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        uid (numpy.ndarray, optional): The original segment IDs to be relabeled. Defaults to None.
        nid (numpy.ndarray, optional): The new segment IDs to assign. Defaults to None.
        do_sort (bool, optional): Whether to sort the segment IDs in ascending order. Defaults to False.
        do_type (bool, optional): Whether to adjust the data type of the segmentation map. Defaults to False.

    Returns:
        numpy.ndarray: The relabeled segmentation map.

    Notes:
        - The function relabels the segments in the segmentation map based on the provided segment IDs.
        - If `uid` is not provided, the unique segment IDs in the input segmentation map are used.
        - If `nid` is not provided, the segment IDs are relabeled in ascending order starting from 1.
        - If `do_sort` is set to True, the segment IDs are sorted in ascending order.
        - If `do_type` is set to True, the data type of the segmentation map is adjusted to accommodate the new segment IDs.
    """    
    if seg is None or seg.max() == 0:
        return seg
    if do_sort:
        uid, _ = seg_to_count(seg, do_sort=True)
    else:
        # get the unique labels
        uid = np.unique(seg) if uid is None else np.array(uid)
    uid = uid[uid > 0]  # leave 0 as 0, the background seg-id
    # get the maximum label for the segment
    mid = int(max(uid)) + 1

    # create an array from original segment id to reduced id
    # format opt
    m_type = seg.dtype
    if do_type:
        mid2 = len(uid) if nid is None else max(nid) + 1
        m_type = get_seg_dtype(mid2)

    mapping = np.zeros(mid, dtype=m_type)
    if nid is None:
        mapping[uid] = np.arange(1, 1 + len(uid), dtype=m_type)
    else:
        mapping[uid] = nid.astype(m_type)
    # if uid is given, need to remove bigger seg id
    seg[seg >= mid] = 0
    return mapping[seg]

def seg_remove_id(seg, bid=None, threshold=100):
    """
    Remove segments from a segmentation map based on their size.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        bid (numpy.ndarray, optional): The segment IDs to be removed. Defaults to None.
        thres (int, optional): The size threshold. Segments with a size below this threshold will be removed. Defaults to 100.

    Returns:
        numpy.ndarray: The updated segmentation map.

    Notes:
        - The function removes segments from the segmentation map based on their size.
        - Segments with a size below the specified threshold are removed.
        - If `bid` is provided, only the specified segment IDs are removed.
    """    
    rl = np.arange(seg.max() + 1).astype(seg.dtype)
    if bid is None:
        uid, uc = np.unique(seg, return_counts=True)
        bid = uid[uc < threshold]
    rl[bid] = 0
    return rl[seg]

def seg_remove_small(seg, threshold=100, invert=False):
    uid, uc = np.unique(seg, return_counts=True)
    bid = uid[uc < threshold]
    seg = seg_remove_id(seg, bid, invert)
    return seg
            
def seg_biggest(seg):
    """
    Keep only the largest-size non-zero seg in a multi-label segmentation map.

    Args:
        seg (numpy.ndarray): The input multi-label segmentation map.

    Returns:
        numpy.ndarray: The segmentation map with only the largest connected component.

    Notes:
        - The function identifies the connected components in the segmentation map using connected component labeling.
        - It keeps only the largest connected component and sets all other components to 0.
    """
    ui, uc = np.unique(seg, return_counts=True)
    uc[ui == 0] = 0
    mid = ui[np.argmax(uc)]
    seg[seg != mid] = 0
    return seg

## morphological processing
def seg_widen_border(seg, tsz_h=1):    
    """
    Widen the border of segments in a segmentation map.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        tsz_h (int, optional): The half size of the border to widen. Defaults to 1.

    Returns:
        numpy.ndarray: The segmentation map with widened segment borders.

    Notes:
        -  Kisuk Lee's thesis (A.1.4): we preprocessed the ground truth seg such that any voxel centered on a 3 x 3 x 1 window containing
        more than one positive segment ID (zero is reserved for background) is marked as background.    
        - The border is widened by marking any voxel centered on a (2*tsz_h+1) x (2*tsz_h+1) window containing more than one positive segment ID as background.
        - The background is represented by the segment ID 0.
    """
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

def seg_binary_fill_holes(seg):
    """
    Perform binary test_error_cases(test_id, filling of holes on a segmentation map.

    Args:
        seg iteration, (numpy.ndarray): The input segmentation map.

    Returns:
        numpy.ndarray: The segmentation map  with after applying binary filling.

    Notes:        
        - Binary filling is a morphological operation that fills holes in foreground regions.    
    """
    out = seg.copy() 
    for uid in np.unique(seg):
        seg_u = seg == uid
        out[seg_u] = uid * binary_fill_holes(seg_u)        
    return out

def seg_binary_opening(seg, iteration=4):
    """
    Perform binary opening on a segmentation map.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        iteration (int, optional): The number of iterations for the binary opening operation. Defaults to 4.

    Returns:
        numpy.ndarray: The segmentation map after applying binary opening.

    Notes:        
        - Binary opening is a morphological operation that removes small foreground regions and smooths the boundaries of larger regions.        
    """
    out = seg.copy() 
    for uid in np.unique(seg):
        seg_u = seg == uid
        out[seg_u] = uid * binary_opening(seg_u, iterations=iteration)        
    return out

def seg_binary_closing(seg, iteration=4):
    """
    Perform binary closing on a segmentation map.

    Args:
        seg (numpy.ndarray): The input segmentation map.
        iteration (int, optional): The number of iterations for the binary closing operation. Defaults to 4.

    Returns:
        numpy.ndarray: The segmentation map after applying binary closing.

    Notes:       
        - Binary closing is a morphological operation that fills small holes and smooths the boundaries of foreground regions.
    """    
    out = seg.copy() 
    for uid in np.unique(seg):
        seg_u = seg == uid
        out[seg_u] = uid * binary_closing(seg_u, iterations=iteration)        
    return out

## connected component
def seg_to_cc(seg, num_conn=None):
    import cc3d
    # https://github.com/seung-lab/connected-components-3d
    # more efficient than scipy/skimage "label"
    if num_conn is None:
        num_conn = 4 if seg.ndim==2 else 6
    return cc3d.connected_components(seg, connectivity=num_conn)

def seg3d_to_cc(seg3d, num_conn=None):
    for z in range(seg3d.shape[0]):
        seg3d[z] = seg_to_cc(seg3d[z], num_conn)
    return seg3d

def seg_remove_small_cc(seg, num_conn=None, threshold=100, invert=False):
    seg = seg_to_cc(seg, num_conn)
    return seg_remove_small(seg, threshold=100, invert=False)

def seg_biggest_cc(seg, num_conn=None):
    """
        Keep only the largest-size non-zero seg in a binary segmentation map.

    Args:
        seg (numpy.ndarray): The input binary segmentation map.

    Returns:
        numpy.ndarray: The segmentation map with only the largest connected component.

    Notes:
        - The function identifies the connected components in the segmentation map using connected component labeling.
        - It keeps only the largest connected component and sets all other components to 0.
    """
    seg = seg_to_cc(seg, num_conn)
    return seg_biggest(seg)    
