import numpy as np
from ..io import get_bb_all2d, get_bb_all3d 

def seg2Count(seg,do_sort=True,rm_zero=False):
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

def seg_iou3d(seg1, seg2, ui0=None):
    ui,uc = np.unique(seg1,return_counts=True)
    uc=uc[ui>0]
    ui=ui[ui>0]
    ui2,uc2 = np.unique(seg2,return_counts=True)

    if ui0 is None:
        ui0=ui
    if len(ui0) == 0:
        return None

    out = np.zeros((len(ui0),5),int)
    bbs = get_bb_all3d(seg1,uid=ui0)[:,1:]
    out[:,0] = ui0
    out[:,2] = uc[np.in1d(ui,ui0)]

    for j,i in enumerate(ui0):
        bb= bbs[j]
        ui3,uc3=np.unique(seg2[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]*(seg1[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]==i), return_counts=True)
        uc3[ui3==0]=0
        out[j,1] = ui3[np.argmax(uc3)]
        out[j,3] = uc2[ui2==out[j,1]]
        out[j,4] = uc3.max()
    return out

def seg_iou2d(seg1, seg2, ui0=None, bb1=None, bb2=None):
    # bb1/bb2: first column of indexing, last column of size
    
    if bb1 is None:
        ui,uc = np.unique(seg1,return_counts=True)
        uc=uc[ui>0];ui=ui[ui>0]
    else:
        # should contain the seg count
        assert bb1.shape[1]==6
        ui = bb1[:,0]
        uc = bb1[:,-1]

    if bb2 is None:
        ui2, uc2 = np.unique(seg2,return_counts=True)
    else:
        assert bb2.shape[1]==6
        ui2 = bb2[:,0]
        uc2 = bb2[:,-1]

    if bb1 is None:
        if ui0 is None:
            bb1 = get_bb_all2d(seg1, uid=ui)
            ui0 = ui
        else:
            bb1 = get_bb_all2d(seg1, uid=ui0)
    else:
        if ui0 is None:
            ui0 = ui
        else:
            # make sure the order matches..
            bb1 = bb1[np.in1d(bb1[:,0], ui0)]
            ui0 = bb1[:,0] 

    if len(ui0) == 0:
        return None

    out = np.zeros((len(ui0),5),int)
    out[:,0] = ui0
    out[:,2] = uc[np.in1d(ui,ui0)]

    for j,i in enumerate(ui0):
        bb= bb1[j, 1:]
        ui3,uc3 = np.unique(seg2[bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(seg1[bb[0]:bb[1]+1,bb[2]:bb[3]+1]==i),return_counts=True)
        uc3[ui3==0] = 0
        if (ui3>0).any():
            out[j,1] = ui3[np.argmax(uc3)]
            out[j,3] = uc2[ui2==out[j,1]]
            out[j,4] = uc3.max()
    return out

def relabel(seg, do_dtype = False):
    if seg is None or seg.max()==0:
        return seg
    uid = np.unique(seg)
    uid = uid[uid > 0]
    max_id = int(max(uid))
    mapping = np.zeros(max_id + 1, dtype = seg.dtype)
    mapping[uid] = np.arange(1, len(uid) + 1)
    if do_dtype:
        return relabelDtype(mapping[seg])
    else:
        return mapping[seg]

def relabelDtype(seg):
    max_id = seg.max()
    m_type = np.uint64
    if max_id<2**8:
        m_type = np.uint8
    elif max_id<2**16:
        m_type = np.uint16
    elif max_id<2**32:
        m_type = np.uint32
    return seg.astype(m_type)


def seg_postprocess(seg, sids=[]):
    # watershed fill the unlabeled part
    if seg.ndim == 3:
        for z in range(seg.shape[0]):
            seg[z] = mahotas.cwatershed(seg[z]==0, seg[z])
            for sid in sids:
                tmp = binary_fill_holes(seg[z]==sid)
                seg[z][tmp>0] = sid
    elif seg.ndim == 2:
        seg = mahotas.cwatershed(seg==0, seg)
    return seg


