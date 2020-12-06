import os,sys
import json
from emu.ng import *
from emu.io import readH5, readBfly


DD='/n/pfister_lab2/Lab/public/ng/'
Do = 'file://' + '/n/pfister_lab2/Lab/public/ng/'

def test_snemi(option = 0):
    print('snemi example')
    # create image and seg info
    Di = '/n/pfister_lab2/Lab/vcg_connectomics/EM/snemi/'
    # xyz
    volume_size = [1024,1024,100]
    resolution = [6,6,30]
    mip_ratio = [[1,1,1],[2,2,1],[4,4,1],[8,8,1],[16,16,2],[32,32,4]]
    chunk_size = [64,64,50]
    dst = ngDataset(volume_size = volume_size, resolution = resolution,\
                 chunk_size=chunk_size, mip_ratio = mip_ratio)
    if option[0] == '0':
        output_im = Do + 'snemi_im'
        output_seg = Do + 'snemi_seg'
        do_subdir = False
    elif option[0] == '1':
        output_im = Do + 'snemi_im_subdir'
        output_seg = Do + 'snemi_seg_subdir'
        do_subdir = True

    if option in ['0', '1']: # cloud-volume env
        dst.createInfo(output_im, 'im')
        im = readH5(Di + 'img/train-input_df_150.h5')
        def get_im(z0, z1, y0, y1, x0, x1):
            return im[z0 : z1, y0 : y1, x0 : x1]
        dst.createTile(get_im, output_im, 'im', range(len(mip_ratio)), do_subdir = do_subdir)
        
    elif option in ['0.1', '1.1']: # cloud-volume env
        dst.createInfo(output_seg, 'seg')
        seg = readH5(Di + 'label/train-labels.h5')
        def get_seg(z0, z1, y0, y1, x0, x1):
            return seg[z0 : z1, y0 : y1, x0 : x1]
        dst.createTile(get_seg, output_seg, 'seg', range(len(mip_ratio)), do_subdir = do_subdir)
        # for _gz
        # dst.removeGz(output_seg, '_', 'remove_orig')
        # for _subdir
        # dst.removeGz(output_seg, '_', 'copy_subdir')
    elif option in ['0.1', '1.1']: # igneous env
        dst.createMesh(output_seg, 2, [256,256,100], 1)
        dst.removeGz(output_seg, 'mesh', True)

if __name__ == "__main__":
    opt = sys.argv[1]
    if opt[0] in ['0', '1']:
        test_snemi(opt)
