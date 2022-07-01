import os,sys
import numpy as np
import h5py
from imageio import imsave,imread
import imu
from imu.seg import seg_iou2d, merge_id, segChunkStitcher
from imu.io import readH5, writeH5, mkdir, segToRgb, rgbToSeg

Dv = '/n/boslfs02/LABS/lichtman_lab/zudi/mitochondria/p14/'
opt = sys.argv[1]
job_id=0;job_num=1
if len(sys.argv) > 3:
    job_id = int(sys.argv[2])
    job_num = int(sys.argv[3])

zran = range(1000,2001,100)
yran = range(2900,9151,3125)
xran = range(7590,13841,3125)
Do = 'db/p14_mito/'
fn_count = Do + 'count_%d-%d-%d.h5'
fn_count_cum = Do + 'count.h5'
fn_count_max = Do + 'count_max.h5'
fn_mid_xy = Do + 'mid_%d-%d-%d.h5'
fn_mid_xy_all = Do + 'mid_xy'
fn_mid_z = Do + 'mid_z_%d.h5'
fn_mid_z_all = Do + 'mid_z.h5'
fn_png = 'db/p14_mito/'
do_offset = False # final h5 has augmented the seg id to avoid overlap
fn_out = '/n/boslfs02/LABS/lichtman_lab/donglai/nagP14/mito/vol2_seg/%04d.png'

mkdir(Do)

h5 = h5py.File(Dv+'final_1000-2000-2900-9150-7590-13840.h5','r')['main']

def getVol(z0,z1,y0,y1,x0,x1):
    return np.array(h5[z0-zran[0]: z1-zran[0],\
        y0-yran[0]: y1-yran[0],\
        x0-xran[0]: x1-xran[0]])

if __name__ == '__main__':
    stitcher = segChunkStitcher(zran, yran, xran, getVol, job_id, job_num)
    if opt == '0': 
        stitcher.chunkCount(fn_count)
    elif opt == '1': 
        if do_offset:
            stitcher.chunkCountCum(fn_count, fn_count_cum)
        else:
            stitcher.chunkCountMax(fn_count, fn_count_max)
    elif opt == '2': # merge xy by connected component
        if do_offset:
            offset = readH5(fn_offset, 'offset')
            stitcher.chunkMergeXY(fn_mid_xy, fn_count, offset)
        else:
            stitcher.chunkMergeXY(fn_mid_xy)
    elif opt == '3': # gather all merge-xy
        if do_offset:
            max_id = readH5(fn_count_cum, 'max_id')
        else:
            max_id = readH5(fn_count_max)
        stitcher.chunkMergeXYAll(fn_mid_xy, fn_mid_xy_all, max_id)
    elif opt == '4': # merge z by IoU
        stitcher.sectionMergeZ(fn_mid_xy_all, fn_mid_z, iou_thres=0.2)
    elif opt == '5': # gather all merge-z
        if do_offset:
            max_id = readH5(fn_count_cum, 'max_id')
        else:
            max_id = readH5(fn_count_max)
        stitcher.sectionMergeZAll(fn_mid_z, fn_mid_z_all, max_id)
    elif opt == '6': # output by z
        stitcher.sectionOutput(fn_mid_xy_all, fn_mid_z_all, fn_out)
    elif opt == '7': # generate low-res
        # 120x120x120 nm
        num_z = 250
        out =np.zeros([num_z, 417, 417], np.uint32)
        for z in range(num_z):
            out[z] = rgbToSeg(imread(fn_out % (1000+z*4)))[::15,::15]
        writeH5(Do + 'ng_120nm.h5', out)
