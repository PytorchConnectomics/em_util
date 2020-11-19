import os
import numpy as np
from ..io import mkdir
from cloudvolume import CloudVolume
from scipy.ndimage import zoom


def createFolder(output_folder = '', volume_size = [100,1024,1024], resolution = [6,6,30],\
                 chunk_size=[64,64,64], image_type = 'im-seg',\
                mip_ratio = [[1,1,1],[2,2,1],[4,4,1],[8,8,1],[16,16,2],[32,32,4]]):
    U_mkdir(output_folder)
    num_mip_level = len(ratio)
    optionss = image_type.split('-')
    for option in options:
        if option == 'im':
            m_enc = 'jpeg'
            m_type = 'image'
            m_dtype = 'uint8'
        elif optio == 'seg':
            m_enc = 'compressed_segmentation'
            m_type = 'segmentation'
            m_dtype = 'uint32'

            # CloudVolume.create_new_info: only do 1 scale ...
            scales = [None for x in range(num_mip_level)]
            for i in range(num_mip_level):
                m_raito = mip_ratio[i]
                m_res = [resolution[j]*rr[j] for j in range(len(resolution))]
                scales[i] = {
                    "encoding"      : m_enc, # raw, jpeg, compressed_segmentation, fpzip, kempressed
                    "chunk_sizes"   : [tuple(chunk_size)], # units are voxels
                    "key"           : "_".join(map(str, m_res)),
                    "resolution"    : m_res, # units are voxels
                    "voxel_offset"  : [0,0,0], # units are voxels
                    "mesh"          : 'mesh', # compute mesh
                    "compressed_segmentation_block_size" : (8,8,8),
                    "size"          : [(volume_size[2] + m_ratio[0] - 1) // m_ratio[0], 
                                       (volume_size[1] + m_ratio[1] - 1) // m_ratio[1], 
                                       (volume_size[0] + m_ratio[2] - 1) // m_ratio[2] ]
                } 
            info = {
                "num_channels"  : 1,
                "type"          : m_type,
                "data_type"     : m_dtype, # Channel images might be 'uint8'
                "scales"        : scales,
            }
            vol = CloudVolume('file://' + output_folder, info=info)
            vol.commit_info()

def createChunk(output_folder = '', getVolume, volume_size = [100,1024,1024], resolution = [6,6,30],\
                image_type = 'image', tile_size = [512,512], offset = [0,0], \
                mip_ratio = [[1,1,1],[2,2,1],[4,4,1],[8,8,1],[16,16,2],[32,32,4]]):
            # x,y,z

            if image_type == 'im':
                m_resize = 1
            elif image_type == 'seg':
                m_resize = 0
            else:
                raise ValueError('Unrecognized image-type: ', image_type)

            # setup cloudvolume writer
            num_mip_level = len(mip_ratio)

            # for axis alignment
            # < mipI: save each tile
            # >= mipI: save each slice
            mip_id_slice  = np.where([(volume_size[:2].astype(float)/ratio[:2]).min()<1 for ratio in mip_ratio])[0] 

            m_vols  = [None for i in range(num_mip_level)]
            m_tszA  = [None for i in range(num_mip_level)]
            m_szA   = [None for i in range(num_mip_level)]
            m_osA   = [None for i in range(num_mip_level)]
            m_tiles = [None for i in range(num_mip_level)]
            zres  = [0 for i in range(num_mip_level)] # number of slices to skip
            for i in ran:
                vols[i] = CloudVolume('file://'+Do, mip=i, parallel= True)
                tszA[i] = [tsz[j]//ratio[i][j] for j in range(2)]
                szA[i]  = vols[i].info['scales'][i]['size']
                osA[i]  = [offset[j]//ratio[i][j] for j in range(2)]
                zres[i] = ratio[i][-1]//ratio[ran[0]][-1]
                if i >=mipI: 
                    # chunk by slice
                    tiles[i] = np.zeros((szA[i][0],szA[i][1],cz[2],1), dtype=m_dtype)
                else: 
                    # chunk by tile
                    # first level or different z-level
                    tiles[i] = np.zeros((tszA[i][0],tszA[i][1],cz[2],1), dtype=m_dtype)

            # tile for the finest level
            x0   = [None for i in range(num_mip_level)]
            x1   = [None for i in range(num_mip_level)]
            y0   = [None for i in range(num_mip_level)]
            y1   = [None for i in range(num_mip_level)]
           
            numZ = data.shape[0]
            numChunk = (numZ+cz[2]-1)//cz[2]
            for z in range(numChunk):
                z0 = z*cz[2]
                z1 = np.min([numZ,(z+1)*cz[2]])
                print('do z-chunk: %d/%d'%(z,numChunk))
                for y in range(2):
                    for x in range(2):
                        # generate global coord
                        for i in ran:
                            # add offset for axis-aligned write
                            x0[i] = osA[i][0]+x*tszA[i][0]
                            x1[i] = min(x0[i]+tszA[i][0], szA[i][0])
                            y0[i] = osA[i][1]+y*tszA[i][1]
                            y1[i] = min(y0[i]+tszA[i][1], szA[i][1])
                        if True: #not os.path.exists(outN[7:]):
                            # read tiles
                            ims = data[z0:z1, y0[ran[0]]:y1[ran[0]], x0[ran[0]]:x1[ran[0]]]
                            for zz in range(z0,z1):
                                im = ims[zz-z0].transpose((1,0))
                                sz0 = im.shape
                                do_full_size = (sz0[0]==tszA[0][0])*(sz0[1]==tszA[0][1]) 
                                for i in ran:
                                    # iterative bilinear downsample
                                    # bug: last border tiles, not full size ...
                                    sz0 = im.shape
                                    if do_full_size:
                                        #im = cv2.resize(im.astype(np.float32), tuple(tszA[i]), m_resize).astype(m_dtype)
                                        sz_r = tszA[i] / np.array(sz0)
                                        im = zoom(im, sz_r, order=m_resize)
                                    else:
                                        tszA_t = [x1[i]-x0[i],y1[i]-y0[i]]
                                        sz_r = tszA_t / np.array(sz0)
                                        #im = cv2.resize(im.astype(np.float32), tuple(tszA_t), m_resize).astype(m_dtype)
                                        im = zoom(im, sz_r, order=m_resize)
                                    if (zz) % zres[i] == 0: # read image
                                        zzl = (zz//zres[i])%(cz[2])
                                        if i<mipI: # whole tile 
                                            tiles[i][:im.shape[0],:im.shape[1],zzl] = im[:,:,None]
                                        else: # piece into one slice
                                            # print(x,y,zz,x0[i],x1[i], y0[i],y1[i],ims[0].max())
                                            tiles[i][x0[i]:x1[i], y0[i]:y1[i], zzl] = im[:x1[i]-x0[i], :y1[i]-y0[i],None]
                            # < mipI: write for each tile 
                            for i in [ii for ii in ran if ii<mipI]:
                                if z1%(zres[i]*cz[2])==0 or z==numChunk-1:
                                    z1g = (z1+zres[i]-1)//zres[i]
                                    z0g = z1g-cz[2]//zres[i]
                                    if z1%(zres[i]*cz[2])!=0: # last unfilled chunk
                                        z0g = (z1g//cz[2])*cz[2]
                                    vols[i][x0[i]:x1[i], \
                                            y0[i]:y1[i],z0g:z1g,:] = tiles[i][:x1[i]-x0[i], :y1[i]-y0[i], :z1g-z0g,:]
                                    tiles[i][:] = 0
                # >= mipI: write for each slice 
                for i in [ii for ii in ran if ii>=mipI]:
                    if z1%(zres[i]*cz[2])==0 or z==numChunk-1:
                        z1g = (z1+zres[i]-1)//zres[i]
                        z0g = z1g-cz[2]
                        if z1%(zres[i]*cz[2])!=0: # last unfilled chunk
                            z0g = (z1g//cz[2])*cz[2]
                        vols[i][:,:,z0g:z1g,:] = tiles[i][:,:,:z1g-z0g,:]
                        tiles[i][:] = 0

    elif opt[0] == '3':
        Do = DD + 'snemi_seg/'
        if opt == '3':
            from taskqueue import LocalTaskQueue
            import igneous.task_creation as tc

            # Mesh on 8 cores, use True to use all cores
            sz2 = (256,256,100)
            cloudpath = 'file://'+Do
            tq = LocalTaskQueue(parallel=4)
            tasks = tc.create_mesh_manifest_tasks(cloudpath)
            tq.insert(tasks)
            tq.execute()
            tasks = tc.create_meshing_tasks(cloudpath, mip=2, shape=sz2, mesh_dir='mesh',dust_threshold=20,max_simplification_error=40)
            tq.insert(tasks)
            tq.execute()
            print("Done!")
        elif opt == '3.1':
            from glob import glob
            import shutil
            gzs = glob(Do + 'mesh/*.gz')
            for gz in gzs:
                shutil.move(gz, gz[:-3])
        elif opt == '3.11':
            from glob import glob
            import shutil
            fns = [x for x in glob(Do + '*') if '_' in x[x.rfind('/'):]]
            for fn in fns:
                print(fn)
                gzs = glob(fn + '/*.gz')
                for gz in gzs:
                    shutil.move(gz, gz[:-3])
        elif opt == '3.2':
            # debug noisy seg
            # uint32: cv2-float32 -> scipy.zoom
            from cloudvolume import CloudVolume
            vol = CloudVolume('file://'+Do, mip=2)
            seg = vol[:] 
            writeh5('db/ng/test.h5',seg.squeeze().transpose([2,1,0]))
        elif opt == '3.21':
            from skimage.measure import label
            seg = readh5('db/ng/test.h5')
            ll = label(seg==2)
            print(ll.max())
            import pdb; pdb.set_trace()
