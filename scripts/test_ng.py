import os,sys
import json
from emu.ng import *
from emu.io import readH5, readTileVolume
# https://neuroglancer-demo.appspot.com/#!%7B"dimensions":%7B"x":%5B6e-9%2C"m"%5D%2C"y":%5B6e-9%2C"m"%5D%2C"z":%5B3e-8%2C"m"%5D%7D%2C"position":%5B366.92974853515625%2C524.85791015625%2C0.5%5D%2C"crossSectionScale":1.4161533536984332%2C"projectionOrientation":%5B-0.16182157397270203%2C0.2875177562236786%2C0.1989087462425232%2C0.9228123426437378%5D%2C"projectionScale":1024%2C"layers":%5B%7B"type":"image"%2C"source":"precomputed://https://rhoana.rc.fas.harvard.edu/ng/snemi_im_subdir"%2C"tab":"source"%2C"name":"snemi_im_subdir"%7D%2C%7B"type":"segmentation"%2C"source":"precomputed://https://rhoana.rc.fas.harvard.edu/ng/snemi_seg_subdir"%2C"tab":"source"%2C"name":"snemi_seg_subdir"%7D%5D%2C"selectedLayer":%7B"layer":"snemi_seg_subdir"%2C"visible":true%7D%2C"layout":"xy"%2C"partialViewport":%5B0%2C0%2C1%2C1%5D%7D

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
    output_im = Do + 'snemi_im_subdir'
    output_seg = Do + 'snemi_seg_subdir'
    # do subdir folder structure to speed up file access
    do_subdir = True

    if option == '0': # cloudvolume env: make image tiles
        dst.createInfo(output_im, 'im')
        im = readH5(Di + 'img/train-input_df_150.h5')
        def get_im(z0, z1, y0, y1, x0, x1):
            return im[z0 : z1, y0 : y1, x0 : x1]
        dst.createTile(get_im, output_im, 'im', range(len(mip_ratio)), do_subdir = do_subdir)
        
    elif option == '1': # cloudvolume env: make segmentation tiles
        dst.createInfo(output_seg, 'seg')
        seg = readH5(Di + 'label/train-labels.h5')
        def get_seg(z0, z1, y0, y1, x0, x1):
            return seg[z0 : z1, y0 : y1, x0 : x1]
        dst.createTile(get_seg, output_seg, 'seg', range(len(mip_ratio)), do_subdir = do_subdir)
    elif option == '2': # igneous env: make 3D meshes
        dst.createMesh(output_seg, 2, [256,256,100], 1)
        #dst.removeGz(output_seg, 'mesh', True)

if __name__ == "__main__":
    opt = sys.argv[1]
    test_snemi(opt)
