import os,sys
import json
from em_util.ng import *
from em_util.io import readH5, readTileVolume

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
        
    elif option == '0.1': # cloudvolume env: make segmentation tiles
        dst.createInfo(output_seg, 'seg')
        seg = readH5(Di + 'label/train-labels.h5')
        def get_seg(z0, z1, y0, y1, x0, x1):
            return seg[z0 : z1, y0 : y1, x0 : x1]
        dst.createTile(get_seg, output_seg, 'seg', range(len(mip_ratio)), do_subdir = do_subdir)
    elif option == '0.2': # igneous env: make 3D meshes
        dst.createMesh(output_seg, 2, [256,256,100], 1, do_subdir = do_subdir)

def test_display():
    Dd = 'https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6e-9%2C%22m%22%5D%2C%22y%22:%5B6e-9%2C%22m%22%5D%2C%22z%22:%5B3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B520.8900756835938%2C534.8792114257812%2C99.3448257446289%5D%2C%22crossSectionScale%22:2.4102014483296315%2C%22projectionOrientation%22:%5B-0.0809287503361702%2C-0.9027095437049866%2C0.42252880334854126%2C-0.005954462569206953%5D%2C%22projectionScale%22:2325.6896087769196%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://rhoana.rc.fas.harvard.edu/ng/snemi_im%22%2C%22tab%22:%22source%22%2C%22name%22:%22snemi%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://https://rhoana.rc.fas.harvard.edu/ng/snemi_seg%22%2C%22tab%22:%22segments%22%2C%22colorSeed%22:2223730652%2C%22segmentQuery%22:%222%22%2C%22name%22:%22snemi_seg%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22snemi_seg%22%2C%22visible%22:true%7D%2C%22layout%22:%224panel%22%7D'
    with viewer.txn() as s:
        s.layers['image'] = neuroglancer.ImageLayer(source = 'precomputed://' + Dd)


if __name__ == "__main__":
    opt = sys.argv[1]
    if opt[0] == '0':
        # create precomputed format
        test_snemi(opt)
    elif opt[0] == '1':
        # display precomputed format
        test_display(opt)
