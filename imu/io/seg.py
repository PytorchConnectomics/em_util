import numpy as np
import glob

def getSegDtype(mid):
    m_type = np.uint64
    if mid<2**8:
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def segToCount(seg,do_sort=True,rm_zero=False):
    sm = seg.max()
    if sm==0:
        return None,None
    if sm>1:
        segIds,segCounts = np.unique(seg,return_counts=True)
        if rm_zero:
            segCounts = segCounts[segIds>0]
            segIds = segIds[segIds>0]
        if do_sort:
            sort_id = np.argsort(-segCounts)
            segIds=segIds[sort_id]
            segCounts=segCounts[sort_id]
    else:
        segIds=np.array([1])
        segCounts=np.array([np.count_nonzero(seg)])
    return segIds, segCounts

def segRelabelType(seg):
    m_type = getSegType(seg.max()+1)
    return seg.astype(m_type)

def segRemove(seg, bid=None, thres=100):
    rl = np.arange(seg.max()+1).astype(seg.dtype)
    if bid is None:
        uid, uc = np.unique(seg, return_counts=True)
        bid = uid[uc<thres]
    rl[bid] = 0
    return rl[seg]

def segRelabel(seg, uid=None,nid=None,do_sort=False,do_type=False):
    if seg is None or seg.max()==0:
        return seg
    if do_sort:
        uid,_ = segToCount(seg,do_sort=True)
    else:
        # get the unique labels
        if uid is None:
            uid = np.unique(seg)
        else:
            uid = np.array(uid)
    uid = uid[uid>0] # leave 0 as 0, the background seg-id
    # get the maximum label for the segment
    mid = int(max(uid)) + 1

    # create an array from original segment id to reduced id
    # format opt
    m_type = seg.dtype
    if do_type:
        mid2 = len(uid) if nid is None else max(nid)+1
        m_type = getSegType(mid2)

    mapping = np.zeros(mid, dtype=m_type)
    if nid is None:
        mapping[uid] = np.arange(1,1+len(uid), dtype=m_type)
    else:
        mapping[uid] = nid.astype(m_type)
    # if uid is given, need to remove bigger seg id 
    seg[seg>=mid] = 0
    return mapping[seg]

def segBiggest(seg):
    from skimage.measure import label
    mask = label(seg)
    ui, uc = np.unique(mask, return_counts=True)
    uc[ui == 0] = 0
    mid = ui[np.argmax(uc)]
    seg[mask != mid] = 0
    return seg


def segToRgb(seg):
    # convert to 24 bits
    return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)

def rgbToSeg(seg):
    # convert to 24 bits
    if seg.ndim==2 or seg.shape[-1]==1:
        return np.squeeze(seg)
    elif seg.ndim == 3: # 1 rgb image
        if (seg[:,:,1] != seg[:,:,2]).any() or (seg[:,:,0] != seg[:,:,2]).any(): 
            return seg[:,:,0].astype(np.uint32)*65536+seg[:,:,1].astype(np.uint32)*256+seg[:,:,2].astype(np.uint32)
        else: # gray image saved into 3-channel
            return seg[:,:,0]
    elif seg.ndim == 4: # n rgb image
        return seg[:,:,:,0].astype(np.uint32)*65536+seg[:,:,:,1].astype(np.uint32)*256+seg[:,:,:,2].astype(np.uint32)

def segWidenBorder(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4): 
    # we preprocessed the ground truth seg such that any voxel centered on a 3 x 3 x 1 window containing 
    # more than one positive segment ID (zero is reserved for background) is marked as background.
    # seg=0: background
    tsz = 2*tsz_h+1
    sz = seg.shape
    if len(sz)==3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(np.pad(seg[z],((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
            p0=patch.max(axis=1)
            patch[patch==0] = mm+1
            p1=patch.min(axis=1)
            seg[z] =seg[z]*((p0==p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(np.pad(seg,((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis = 1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg
