import numpy as np
import glob

def getSegType(mid):
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

def segRemove(seg, rid):
    rl = np.arange(seg.max()+1).astype(seg.dtype)
    rl[rid] = 0
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

def segToVast(seg):
    # convert to 24 bits
    return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)

def vastToSeg(seg):
    # convert to 24 bits
    if seg.ndim==2 or seg.shape[-1]==1:
        return np.squeeze(seg)
    elif seg.ndim == 3: # 1 rgb image
        return seg[:,:,0].astype(np.uint32)*65536+seg[:,:,1].astype(np.uint32)*256+seg[:,:,2].astype(np.uint32)
    elif seg.ndim == 4: # n rgb image
        return seg[:,:,:,0].astype(np.uint32)*65536+seg[:,:,:,1].astype(np.uint32)*256+seg[:,:,:,2].astype(np.uint32)

