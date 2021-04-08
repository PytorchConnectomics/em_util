import os,sys
from scipy.ndimage import zoom
import json
from .io import readImage
from .seg import rgbToSeg
import numpy as np
import json

def readTileVolume(fns, z0p, z1p, y0p, y1p, x0p, x1p, tile_sz, tile_type = np.uint8,\
             tile_st = [0, 0], tile_ratio = 1, tile_resize_mode = 1, tile_seg = False, tile_blank = 'reflect', volume_sz = None):
    if not isinstance(tile_sz, (list,)):
        tile_sz = [tile_sz, tile_sz]
    if not isinstance(tile_ratio, (list,)):
        tile_ratio = [tile_ratio, tile_ratio]
    # [row,col]
    # no padding at the boundary
    # st: starting index 0 or 1

    bd = None
    if volume_sz is not None:
        bd = [max(-z0p,0), max(0,z1p-volume_sz[0]),\
             max(-y0p,0), max(0,y1p-volume_sz[1]),\
             max(-x0p,0), max(0,x1p-volume_sz[2])]
        z0, y0, x0 = max(z0p,0), max(y0p,0), max(x0p,0)
        z1, y1, x1 = min(z1p,volume_sz[0]), min(y1p,volume_sz[1]), min(x1p,volume_sz[2])
    else:
        z0, y0, x0, z1, y1, x1 = z0p, y0p, x0p, z1p, y1p, x1p

    result = np.zeros((z1 - z0, (y1 - y0) * tile_ratio[0], (x1 - x0) * tile_ratio[1]), tile_type)
    c0 = x0 // tile_sz[1] # floor
    c1 = (x1 + tile_sz[1]-1) // tile_sz[1] # ceil
    r0 = y0 // tile_sz[0]
    r1 = (y1 + tile_sz[0]-1) // tile_sz[0]
    z1 = min(len(fns)-1, z1)
    for z in range(z0, z1):
        pattern = fns[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                if '%' in pattern:
                    filename = pattern % (row + tile_st[0], column + tile_st[1])
                elif '{' in pattern:
                    filename = pattern.format(row=row + tile_st[0], column=column + tile_st[1])
                else:
                    filename = pattern
                if os.path.exists(filename):
                    patch = readImage(filename)
                    if tile_seg:
                        patch = rgbToSeg(patch)
                    if tile_ratio[0] != 1:
                        patch = zoom(patch, tile_ratio, order = tile_resize_mode)
                    # exception: last tile may not have the right size
                    psz = patch.shape
                    xp0 = column * tile_sz[1]
                    xp1 = min(xp0+psz[1], (column+1)*tile_sz[1])
                    yp0 = row * tile_sz[0]
                    yp1 = min(yp0+psz[0], (row+1)*tile_sz[0])

                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    try:
                        result[z-z0, (y0a-y0)*tile_ratio[0] : (y1a-y0)*tile_ratio[0], (x0a-x0)*tile_ratio[1] : (x1a-x0)*tile_ratio[1]] = patch[(y0a-yp0)*tile_ratio[0] : (y1a-yp0)*tile_ratio[0], (x0a-xp0)*tile_ratio[1] : (x1a-xp0)*tile_ratio[1]]
                    except:
                        import pdb; pdb.set_trace()
                else:
                    print('Non-exist: %s'%filename)
    if bd is not None and max(bd)>0:
        result = np.pad(result,
                ((bd[0], bd[1]),
                 (bd[2] * tile_ratio[0], bd[3] * tile_ratio[0]),
                 (bd[4] * tile_ratio[1], bd[5] * tile_ratio[1])),'reflect')
    return result

def writeTileInfo(sz, numT, imN, tsz=1024, tile_st=[0,0],zPad=[0,0], im_id=None, outName=None,st=0,ndim=1,rsz=1,dt='uint8'):
    # one tile for each section
    # st: starting index
    if im_id is None:
        im_id = range(zPad[0]+st,st,-1)+range(st,sz[0]+st)+range(sz[0]-2+st,sz[0]-zPad[1]-2+st,-1)
    else: # st=0
        if zPad[0]>0:
            im_id = [im_id[x] for x in range(zPad[0],0,-1)]+im_id
        if zPad[1]>0:
            im_id += [im_id[x] for x in range(sz[0]-2,sz[0]-zPad[1]-2,-1)]
    sec=[imN(x) for x in im_id]
    out={'image':sec, 'depth':sz[0]+sum(zPad), 'height':sz[1], 'width':sz[2], "tile_st":tile_st,
         'dtype':dt, 'n_columns':numT[1], 'n_rows':numT[0], "tile_size":tsz, 'ndim':ndim, 'tile_ratio':rsz}
    if outName is None:
        return out
    else:
        with open(outName,'w') as fid:
            json.dump(out, fid)
