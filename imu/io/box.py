import numpy as np


def get_bb(seg, do_count=False):
    dim = len(seg.shape)
    a=np.where(seg>0)
    if len(a[0])==0:
        return [-1]*dim*2
    out=[]
    for i in range(dim):
        out+=[a[i].min(), a[i].max()]
    if do_count:
        out+=[len(a[0])]
    return out

def get_bb_all2d(seg, do_count=False, uid=None):
    sz = seg.shape
    assert len(sz)==2
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid>0]
    if len(uid) == 0:
        return np.zeros((1,5+do_count),dtype=np.uint32)

    um = uid.max()
    out = np.zeros((1+int(um),5+do_count),dtype=np.uint32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1] = sz[0]
    out[:,3] = sz[1]
    # for each row
    rids = np.where((seg>0).sum(axis=1)>0)[0]
    for rid in rids:
        sid = np.unique(seg[rid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,1] = np.minimum(out[sid,1],rid)
        out[sid,2] = np.maximum(out[sid,2],rid)
    cids = np.where((seg>0).sum(axis=0)>0)[0]
    for cid in cids:
        sid = np.unique(seg[:,cid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,3] = np.minimum(out[sid,3],cid)
        out[sid,4] = np.maximum(out[sid,4],cid)

    if do_count:
        ui,uc = np.unique(seg,return_counts=True)
        out[ui,-1]=uc
    return out[uid]

def get_bb_all3d(seg,do_count=False, uid=None):
    sz = seg.shape
    assert len(sz)==3
    if uid is None:
        uid = seg
    um = int(uid.max())
    out = np.zeros((1+um,7+do_count),dtype=np.int32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1] = sz[0]
    out[:,2] = -1
    out[:,3] = sz[1]
    out[:,4] = -1
    out[:,5] = sz[2]
    out[:,6] = -1

    # for each slice
    zids = np.where((seg>0).sum(axis=1).sum(axis=1)>0)[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,1] = np.minimum(out[sid,1],zid)
        out[sid,2] = np.maximum(out[sid,2],zid)

    # for each row
    rids = np.where((seg>0).sum(axis=0).sum(axis=1)>0)[0]
    for rid in rids:
        sid = np.unique(seg[:,rid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,3] = np.minimum(out[sid,3],rid)
        out[sid,4] = np.maximum(out[sid,4],rid)
    
    # for each col
    cids = np.where((seg>0).sum(axis=0).sum(axis=0)>0)[0]
    for cid in cids:
        sid = np.unique(seg[:,:,cid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,5] = np.minimum(out[sid,5],cid)
        out[sid,6] = np.maximum(out[sid,6],cid)

    if do_count:
        ui,uc = np.unique(seg,return_counts=True)
        out[ui[ui<=um],-1]=uc[ui<=um]

    return out[np.all(out!=-1, axis=-1)].astype(np.uint32)



