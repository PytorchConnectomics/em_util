import os,sys
from emu.seg import *
from emu.io import writeH5
import numpy as np

def test_seg_track(seg, iou_thres=0.2):
    from emu.seg import predToSeg2d,seg2dToIoU
    seg3d_naive = predToSeg2d(seg, [-1])
    del seg
    iou = seg2dToIoU(seg3d_naive, iou_thres)
    seg3d = seg2dTo3d(seg3d_naive, iou)
    return seg3d


if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == '0':
        Di = '/n/boslfs02/LABS/lichtman_lab/zudi/zf_exM/ventral_10_histeq_cp.npy'
        seg = np.load(Di)
        out = test_seg_track(seg)
        writeH5('test.h5', out)
