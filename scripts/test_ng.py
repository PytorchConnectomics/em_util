import os,sys
import json
from emu.ng import *
from emu.io import readH5, readBfly


DD='/n/pfister_lab2/Lab/public/ng/'
DD='/n/boslfs02/LABS/lichtman_lab/glichtman/public/ng/'
sz=[100,1024,1024]

Do = 'file://' + '/n/pfister_lab2/Lab/public/ng/'

def test_snemi(step = 0):
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
    output_im = Do + 'snemi_im'
    output_seg = Do + 'snemi_seg'
    if step == 0: # cloud-volume env
        dst.createInfo(output_im, 'im')
        im = readh5(Di + 'img/train-input_df_150.h5')
        def get_im(z0, z1, y0, y1, x0, x1):
            return im[z0 : z1, y0 : y1, x0 : x1]
        dst.createTile(get_im, output_im, 'im', range(len(mip_ratio)))
        
        dst.createInfo(output_seg, 'seg')
        seg = readh5(Di + 'label/train-labels.h5')
        def get_seg(z0, z1, y0, y1, x0, x1):
            return seg[z0 : z1, y0 : y1, x0 : x1]
        dst.createTile(get_seg, output_seg, 'seg', range(len(mip_ratio)))
        dst.removeGz(output_seg, '_', True)
    elif step == 1: # igneous env
        dst.createMesh(output_seg, 2, [256,256,100], 1)
        dst.removeGz(output_seg, 'mesh', True)

def test_l4dense(step = 0):
    print('l4dense example')
    # create image and seg info
    Di = '//n/boslfs/LABS/lichtman_lab/aligned_datasets/Moritz_L4_2019/'
    # xyz
    volume_size = [1024*6, 1024*9, 3306]
    resolution = [11,11,28]
    mip_ratio = [[1,1,1],[2,2,1],[4,4,2],[8,8,4],[16,16,8],[32,32,16]]
    chunk_size = [64,64,64]
    volume_size[2] = ((volume_size[2] + chunk_size[2] - 1) // chunk_size[2]) *  chunk_size[2] 
    dst = ngDataset(volume_size = volume_size, resolution = resolution,\
                 chunk_size=chunk_size, mip_ratio = mip_ratio)
    output_im = Do + 'l4dense_im'
    output_im = Do + 'l4dense_im'
    if step == 0: # cloud-volume env
        dst.createInfo(output_im, 'im')
        dst_info= json.load(open(Di + 'em/im.json'))
        dst_fns = dst_info['sections']
        dst_tile_size = dst_info['dimensions']['tile_size']
        
        def get_im(z0, z1, y0, y1, x0, x1):
            return readBfly(dst_fns, z0, z1, y0, y1,  x0, x1, tile_sz=dst_tile_size)
        dst.createTile(get_im, output_im, 'im', range(len(mip_ratio)))
        """ 
        dst.createInfo(Do + 'snemi_seg', 'seg')
        seg = readh5(Di + 'label/train-labels.h5')
        def get_seg(z0, z1, y0, y1, x0, x1):
            return seg[z0 : z1, y0 : y1, x0 : x1]
        dst.createTile(get_seg, Do + 'snemi_seg', 'seg', range(len(mip_ratio)))
        dst.removeGz(Do + 'snemi_seg/', '_', True)
        """ 
    elif step == 1: # igneous env
        dst.createMesh(Do + 'snemi_seg', 2, [256,256,100], 1)
        dst.removeGz(Do + 'snemi_seg/', 'mesh', True)


if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == '0':
        test_snemi()
    elif opt == '0.1':
        test_snemi(1)
    elif opt == '1':
        test_l4dense()
