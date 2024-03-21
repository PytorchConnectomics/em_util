import os,sys
from em_util.seg import *
from em_util.io import writeH5
import numpy as np

def test_seg_track(seg, iou_thres=0.2):
    from em_util.seg import predToSeg2d,seg2dToIoU
    seg3d_naive = predToSeg2d(seg, [-1])
    del seg
    matches = seg2dToIoU(seg3d_naive, iou_thres)
    seg3d = seg2dTo3d(seg3d_naive, matches)
    return seg3d


if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == '0':
        Di = '/n/boslfs02/LABS/lichtman_lab/zudi/zf_exM/ventral_10_histeq_cp.npy'
        seg = np.load(Di)
        out = test_seg_track(seg)
        writeH5('test.h5', out)
