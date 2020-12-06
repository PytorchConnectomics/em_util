import os
from glob import glob
import shutil

import numpy as np
from scipy.ndimage import zoom

from ..io import mkdir

class ngDataset(object):
    def __init__(self, volume_size = [1024,1024,100], \
                 resolution = [6,6,30], chunk_size=[64,64,64], offset = [0,0,0], \
                mip_ratio = [[1,1,1],[2,2,1],[4,4,1],[8,8,1],[16,16,2],[32,32,4]]):
        # dimension order: x,y,z
        self.volume_size = volume_size
        self.resolution = resolution
        self.chunk_size = chunk_size
        self.mip_ratio = mip_ratio
        self.offset = offset

    def createInfo(self, cloudpath = '', data_type = 'im'):
        from cloudvolume import CloudVolume
        if 'file' == cloudpath[:4]:
            mkdir(cloudpath[7:])
        num_mip_level = len(self.mip_ratio)
        if data_type == 'im':
            m_enc = 'jpeg'
            m_type = 'image'
            m_dtype = 'uint8'
        elif data_type == 'seg':
            m_enc = 'compressed_segmentation'
            m_type = 'segmentation'
            m_dtype = 'uint32'

        # CloudVolume.create_new_info: only do 1 scale ...
        scales = [None for x in range(num_mip_level)]
        for i in range(num_mip_level):
            m_ratio = self.mip_ratio[i]
            m_res = [self.resolution[j] * m_ratio[j] for j in range(len(self.resolution))]
            scales[i] = {
                "encoding"      : m_enc, # raw, jpeg, compressed_segmentation, fpzip, kempressed
                "chunk_sizes"   : [tuple(self.chunk_size)], # units are voxels
                "key"           : "_".join(map(str, m_res)),
                "resolution"    : m_res, # units are voxels
                "voxel_offset"  : [(self.offset[x] + m_ratio[x] - 1) // m_ratio[x] for x in range(3)], 
                "mesh"          : 'mesh', # compute mesh
                "compressed_segmentation_block_size" : (8,8,8),
                "size"          : [(self.volume_size[x] + m_ratio[x] - 1) // m_ratio[x] for x in range(3)], 
            } 
        info = {
            "num_channels"  : 1,
            "type"          : m_type,
            "data_type"     : m_dtype, # Channel images might be 'uint8'
            "scales"        : scales,
        }
        vol = CloudVolume(cloudpath, info=info)
        vol.commit_info()

    def createTile(self, getVolume, cloudpath = '', data_type = 'image', \
                   mip_levels = None, tile_size = [512,512], offset = [0,0,0], num_thread = 1, do_subdir = False):
        from cloudvolume import CloudVolume
        if data_type == 'im':
            m_resize = 1
            m_dtype = 'uint8'
        elif data_type == 'seg':
            m_resize = 0
            m_dtype = 'uint32'
        else:
            raise ValueError('Unrecognized data type: ', data_type)

        # setup cloudvolume writer
        num_mip_level = len(self.mip_ratio)
        if mip_levels is None:
            # if not specified, do all mip levels
            mip_levels = range(num_mip_level)
        mip_levels = [x for x in mip_levels if x < num_mip_level]
        # < mipI: save each tile
        # >= mipI: save each slice
        # find the mip-level that need to be tiled vs. whole section
        m_mip_id  = [i for i, ratio in enumerate(self.mip_ratio) if (self.volume_size[:2]/(tile_size*np.array(ratio[:2]))).min() <= 1]
        m_mip_id = 0 if len(m_mip_id) == 0 else m_mip_id[0]

        m_vols  = [None] * num_mip_level
        m_tszA  = [None] * num_mip_level
        m_szA   = [None] * num_mip_level
        m_osA   = [None] * num_mip_level
        m_tiles = [None] * num_mip_level
        m_zres  = [0   ] * num_mip_level # number of slices to skip
        for i in mip_levels:
            m_vols[i] = CloudVolume(cloudpath, mip=i, parallel= num_thread)
            if do_subdir:
                m_vols[i].meta.name_sep = '/'
            m_tszA[i] = [tile_size[j]//self.mip_ratio[i][j] for j in range(2)]
            m_szA[i]  = m_vols[i].info['scales'][i]['size']
            m_osA[i]  = [offset[j]//self.mip_ratio[i][j] for j in range(3)]
            m_zres[i] = self.mip_ratio[i][-1]//self.mip_ratio[mip_levels[0]][-1]
            if i >= m_mip_id: 
                # output whole section
                m_tiles[i] = np.zeros((m_szA[i][0], m_szA[i][1], self.chunk_size[2],1), dtype=m_dtype)
            else: 
                # output in tiles
                m_tiles[i] = np.zeros((m_tszA[i][0], m_tszA[i][1], self.chunk_size[2],1), dtype=m_dtype)

        # tile for the finest level
        x0   = [None] * num_mip_level
        x1   = [None] * num_mip_level
        y0   = [None] * num_mip_level
        y1   = [None] * num_mip_level
      
        num_chunk = [(self.volume_size[x] + m_tszA[mip_levels[0]][x]-1) // m_tszA[mip_levels[0]][x] for x in range(2)]
        num_chunk += [(self.volume_size[2] + self.chunk_size[2]-1) // self.chunk_size[2]]

        for z in range(num_chunk[2]):
            z0 = z * self.chunk_size[2]
            z1 = np.min([self.volume_size[2], (z+1) * self.chunk_size[2]])
            for y in range(num_chunk[1]):
                for x in range(num_chunk[0]):
                    print('do chunk: %d/%d, %d/%d, %d/%d' % (z, num_chunk[2], y, num_chunk[1], x, num_chunk[0]))
                    # generate global coord
                    for i in mip_levels:
                        # add offset for axis-aligned write
                        x0[i] = m_osA[i][0] + x * m_tszA[i][0]
                        x1[i] = min(x0[i] + m_tszA[i][0], m_szA[i][0])
                        y0[i] = m_osA[i][1] + y * m_tszA[i][1]
                        y1[i] = min(y0[i] + m_tszA[i][1], m_szA[i][1])
                    # read tiles
                    # input/output dimension order: z,y,x 
                    ims = getVolume(z0, z1, \
                                    y0[mip_levels[0]], y1[mip_levels[0]], \
                                    x0[mip_levels[0]], x1[mip_levels[0]])
                    
                    for zz in range(z0,z1):
                        im = ims[zz-z0].transpose((1,0))
                        sz0 = im.shape
                        # in case the output is not padded for invalid regions
                        full_size_tile = (sz0[0] == m_tszA[0][0])*(sz0[1] == m_tszA[0][1]) 
                        for i in mip_levels:
                            # iterative bilinear downsample
                            # bug: last border tiles, not full size ...
                            sz0 = im.shape
                            if full_size_tile:
                                #im = cv2.resize(im.astype(np.float32), tuple(tszA[i]), m_resize).astype(m_dtype)
                                sz_r = m_tszA[i] / np.array(sz0)
                                im = zoom(im, sz_r, order=m_resize)
                            else:
                                tszA_t = [x1[i]-x0[i], y1[i]-y0[i]]
                                sz_r = tszA_t / np.array(sz0)
                                im = zoom(im, sz_r, order=m_resize)

                            if (zz) % m_zres[i] == 0: # read image
                                zzl = (zz // m_zres[i]) % (self.chunk_size[2])
                                if i < m_mip_id: # whole tile 
                                    m_tiles[i][:im.shape[0], :im.shape[1], zzl] = im[:, :, None]
                                else: # piece into one slice
                                    m_tiles[i][x0[i]: x1[i], y0[i]: y1[i], zzl] = im[:x1[i]- x0[i], :y1[i]-y0[i], None]
                    # < mipI: write for each tile 
                    # x/y each chunk
                    for i in [ii for ii in mip_levels if ii < m_mip_id]:
                        if z1 % (m_zres[i] * self.chunk_size[2]) == 0 or z == num_chunk[2] - 1:
                            z1g = (z1 + m_zres[i] - 1) // m_zres[i]
                            z0g = z1g - self.chunk_size[2]
                            if z1 % (m_zres[i] * self.chunk_size[2]) != 0: # last unfilled chunk
                                z0g = (z1g // self.chunk_size[2]) * self.chunk_size[2]
                            # check volume align
                            # in z: has to be 
                            m_vols[i][x0[i] : x1[i], \
                                y0[i] : y1[i], z0g + m_osA[i][2] : z1g + m_osA[i][2], :] = \
                                m_tiles[i][: x1[i] - x0[i], : y1[i] - y0[i], : z1g - z0g, :]
                            m_tiles[i][:] = 0
            # >= mipI: write for each secion
            for i in [ii for ii in mip_levels if ii >= m_mip_id]:
                if z1 % (m_zres[i] * self.chunk_size[2]) == 0 or z == num_chunk[2] - 1:
                    z1g = (z1 + m_zres[i] - 1) // m_zres[i]
                    z0g = z1g - self.chunk_size[2]
                    if z1 % (m_zres[i] * self.chunk_size[2]) != 0: # last unfilled chunk
                        z0g = (z1g // self.chunk_size[2]) * self.chunk_size[2]
                    try:
                        m_vols[i][:, :, z0g + m_osA[i][2] : z1g + m_osA[i][2], :] = m_tiles[i][:, :, : z1g - z0g, :]
                    except:
                        import pdb; pdb.set_trace()
                    m_tiles[i][:] = 0

    def createMesh(self, cloudpath='', mip_level=0, volume_size=[256,256,100], num_thread = 1):
        from taskqueue import LocalTaskQueue
        import igneous.task_creation as tc
        
        tq = LocalTaskQueue(parallel = num_thread)
        tasks = tc.create_meshing_tasks(cloudpath, mip = mip_level, \
                                        shape = volume_size, mesh_dir='mesh',\
                                        dust_threshold=20,max_simplification_error=40)
        tq.insert(tasks)
        tq.execute()

        tq = LocalTaskQueue(parallel=num_thread)
        tasks = tc.create_mesh_manifest_tasks(cloudpath)
        tq.insert(tasks)
        tq.execute()

    def removeGz(self, cloudpath='', folder_key='_', option = 'copy'):
        if 'file' == cloudpath[:4]:
            cloudpath = cloudpath[7:]
        fns = [x for x in glob(cloudpath + '/*') if folder_key in x[x.rfind('/'):]]
        for fn in fns:
            print(fn)
            gzs = glob(fn + '/*.gz')
            if option == 'copy':
                for gz in gzs:
                    if not os.path.exists(gz[:-3]):
                        shutil.copy(gz, gz[:-3])
            elif option == 'move':
                for gz in gzs:
                    shutil.move(gz, gz[:-3])
            elif option == 'remove_orig':
                for gz in gzs:
                    os.remove(gz[:-3])
            elif option == 'copy_subdir':
                for gz in gzs:
                    gz2 = gz[gz.rfind('/'):-3].split('_')
                    mkdir(fn + gz2[0] + '/' + gz2[1], 2)
                    shutil.copy(gz, fn + '/'.join(gz2))
