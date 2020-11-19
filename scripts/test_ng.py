import os,sys
from emu.ng import *
from emu.io import readh5


DD='/n/pfister_lab2/Lab/public/ng/'
DD='/n/boslfs02/LABS/lichtman_lab/glichtman/public/ng/'
sz=[100,1024,1024]

Do = '/n/pfister_lab2/Lab/public/ng/'

def test_snemi(step = 0):
    print('snemi example')
    # create image and seg info
    Di = '/n/pfister_lab2/Lab/vcg_connectomics/EM/snemi/'
    Do2 = 'file://' + Do
    # xyz
    volume_size = [1024,1024,100]
    resolution = [6,6,30]
    mip_ratio = [[1,1,1],[2,2,1],[4,4,1],[8,8,1],[16,16,2],[32,32,4]]
    chunk_size = [64,64,50]
    dst = ngDataset(volume_size = volume_size, resolution = resolution,\
                 chunk_size=chunk_size, mip_ratio = mip_ratio)
    if step == 0: # cloud-volume env
        dst.createInfo(Do2 + 'snemi_im', 'im')
        im = readh5(Di + 'img/train-input_df_150.h5')
        def get_im(z0, z1, y0, y1, x0, x1):
            return im[z0 : z1, y0 : y1, x0 : x1]

        seg = readh5(Di + 'label/train-labels.h5')
        def get_seg(z0, z1, y0, y1, x0, x1):
            return seg[z0 : z1, y0 : y1, x0 : x1]
        dst.createInfo(Do2 + 'snemi_seg', 'seg')
        dst.createTile(get_seg, Do + 'snemi_seg', 'seg', range(5))
    elif step == 1: # igneous env
        dst.createMesh(Do2 + 'snemi_seg', 2, [256,256,100], 1)
    elif step == 2:
        dst.removeGz(Do + 'snemi_seg/', '_')
        dst.removeGz(Do + 'snemi_seg/', 'mesh')


# example code
if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == '0':
        test_snemi()
    elif opt == '0.1':
        test_snemi(1)
    elif opt == '0.2':
        test_snemi(2)
