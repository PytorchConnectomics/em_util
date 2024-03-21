import os
from glob import glob
import shutil
import struct
import itertools
import numpy as np
from cloudvolume import CloudVolume
from scipy.ndimage import zoom
from ..io import mkdir, write_txt


class NgDataset(object):
    def __init__(
        self,
        volume_size=[1024, 1024, 100],
        resolution=[6, 6, 30],
        mip_ratio=[
            [1, 1, 1],
            [2, 2, 1],
            [4, 4, 1],
            [8, 8, 1],
            [16, 16, 2],
            [32, 32, 4],
        ],
        chunk_size=[64, 64, 64],
        offset=[0, 0, 0],
        cloudpath="./",
    ):
        # dimension order: x,y,z
        self.volume_size = volume_size
        self.resolution = resolution
        self.chunk_size = chunk_size
        self.mip_ratio = mip_ratio
        self.offset = offset

        self.cloudpath = cloudpath
        if cloudpath != "" and "file" == cloudpath[:4]:
            mkdir(cloudpath[7:])

    def create_info(self, cloudpath="", data_type="im", num_channel=1):
        if cloudpath == "":
            cloudpath = self.cloudpath + "/seg/"

        if "file" == cloudpath[:4]:
            mkdir(cloudpath[7:])
        num_mip_level = len(self.mip_ratio)
        if data_type == "im":
            m_enc = "jpeg"
            m_type = "image"
            m_dtype = "uint8"
        elif data_type == "seg":
            m_enc = "compressed_segmentation"
            m_type = "segmentation"
            m_dtype = "uint32"
        elif data_type == "skel":
            m_enc = "raw"
            m_type = "segmentation"
            m_dtype = "uint32"

        # CloudVolume.create_new_info: only do 1 scale ...
        scales = [None for x in range(num_mip_level)]
        for i in range(num_mip_level):
            m_ratio = self.mip_ratio[i]
            m_res = [
                self.resolution[j] * m_ratio[j]
                for j in range(len(self.resolution))
            ]
            scales[i] = {
                "encoding": (
                    m_enc
                ),  # raw, jpeg, compressed_segmentation, fpzip, kempressed
                "chunk_sizes": [tuple(self.chunk_size)],  # units are voxels
                "key": "_".join(map(str, m_res)),
                "resolution": m_res,  # units are voxels
                "voxel_offset": [
                    (self.offset[x] + m_ratio[x] - 1) // m_ratio[x]
                    for x in range(3)
                ],
                "mesh": "mesh",  # compute mesh
                "skeletons": "skeletons",  # compute mesh
                "compressed_segmentation_block_size": (8, 8, 8),
                "size": [
                    (self.volume_size[x] + m_ratio[x] - 1) // m_ratio[x]
                    for x in range(3)
                ],
            }
        info = {
            "num_channels": num_channel,
            "type": m_type,
            "data_type": m_dtype,  # Channel images might be 'uint8'
            "scales": scales,
        }
        vol = CloudVolume(cloudpath, info=info)
        vol.commit_info()

    def create_setup(self, data_type):
        pass

    def create_tile_setup(
        self,
        cloudpath="",
        data_type="im",
        mip_levels=None,
        tile_size=None,
        num_thread=1,
        do_subdir=False,
        num_channel=1,
    ):
        if tile_size is None:
            tile_size = [512, 512]
        if data_type == "im":
            m_resize = 1
            m_dtype = "uint8"
        elif data_type == "seg":
            m_resize = 0
            m_dtype = "uint32"
        else:
            raise ValueError("Unrecognized data type: ", data_type)

        if cloudpath == "":
            cloudpath = f"{self.cloudpath}/{data_type}/"

        # write .htaccess
        if cloudpath[:4] == "file":
            self.write_htaccess(f"{cloudpath[7:]}/.htaccess", data_type, do_subdir)

        # setup cloudvolume writer
        num_mip_level = len(self.mip_ratio)
        if mip_levels is None:
            # if not specified, do all mip levels
            m_mip_levels = range(num_mip_level)
        else:
            # remove extra mip levels without the raio
            m_mip_levels = [x for x in mip_levels if x < num_mip_level]
        # < mipI: save each tile
        # >= mipI: save each slice
        # find the mip-level that need to be tiled vs. whole section
        m_mip_id = [
            i
            for i, ratio in enumerate(self.mip_ratio)
            if (self.volume_size[:2] / (tile_size * np.array(ratio[:2]))).max()
            <= 1
        ]
        m_mip_id = len(m_mip_id) if not m_mip_id else m_mip_id[0]
        # invalid chunk_size
        m_mip_id = min(
            m_mip_id,
            np.log2(np.array(tile_size) / self.chunk_size[:2])
            .astype(int)
            .min(),
        )
        m_vols = [None] * (m_mip_levels[-1] + 1)
        m_tile_size = [None] * (m_mip_levels[-1] + 1)
        m_size = [None] * (m_mip_levels[-1] + 1)
        m_oset = [None] * (m_mip_levels[-1] + 1)
        m_tiles = [None] * (m_mip_levels[-1] + 1)
        m_zres = [0] * (m_mip_levels[-1] + 1)
        # m_size: start from mip_level[0], relative coord
        for i in m_mip_levels:
            m_vols[i] = CloudVolume(cloudpath, mip=i, parallel=num_thread)
            if do_subdir:
                m_vols[i].meta.name_sep = "/"
            ratio_i = [
                self.mip_ratio[i][j] // self.mip_ratio[m_mip_levels[0]][j] for j in range(2)
            ]
            m_tile_size[i] = [tile_size[j] // ratio_i[j] for j in range(2)] + [
                num_channel
            ]
            m_size[i] = m_vols[i].shape[:3]
            # m_vols[i].info['scales'][i]['size'] # not updated
            m_oset[i] = [
                (self.offset[j] + self.mip_ratio[i][j] - 1)
                // self.mip_ratio[i][j]
                for j in range(3)
            ]
            m_zres[i] = self.mip_ratio[i][2]
            if i >= m_mip_id:
                # output whole section: sometimes #z is smaller than chunk size                
                m_tiles[i] = np.zeros(
                    (m_size[i][0], m_size[i][1], min(m_size[i][2], self.chunk_size[2]), num_channel),
                    dtype=m_dtype,
                )
            else:
                # output in tiles
                m_tiles[i] = np.zeros(
                    (
                        m_tile_size[i][0],
                        m_tile_size[i][1],
                        self.chunk_size[2],
                        num_channel,
                    ),
                    dtype=m_dtype,
                )

        return (
            m_resize,
            m_mip_levels,
            m_mip_id,
            m_vols,
            m_tile_size,
            m_size,
            m_oset,
            m_tiles,
            m_zres,
        )

    def create_tile_parallel(
        self,
        getVolume,
        cloudpath="",
        data_type="im",
        mip_levels=None,
        tile_size=[512, 512],
        num_thread=1,
        do_subdir=False,
        num_channel=1,
        start_chunk=0,
        step_chunk=1,
    ):
        (
            m_resize,
            m_mip_levels,
            m_mip_id,
            m_vols,
            m_tile_size,
            m_size,
            m_oset,
            m_tiles,
            m_zres,
        ) = self.create_tile_setup(
            cloudpath,
            data_type,
            mip_levels,
            tile_size,
            num_thread,
            do_subdir,
            num_channel,
        )        
        if m_mip_levels[0] < m_mip_id:
            # step 2: do 0-m_mip_id
            # z0,z1,y0,y1,x0,x1: relative coord
            # tile for the finest level
            x0 = [None] * m_mip_id
            x1 = [None] * m_mip_id
            y0 = [None] * m_mip_id
            y1 = [None] * m_mip_id
            # num of chunk: x and y (offset)
            # keep the size
            # num_chunk = [(m_size[mip_levels[0]][x] + m_tile_size[mip_levels[0]][x]-1 - m_oset[mip_levels[0]][x]) // m_tile_size[mip_levels[0]][x] for x in range(2)]
            num_chunk = [
                (m_size[m_mip_levels[0]][x] + m_tile_size[m_mip_levels[0]][x] - 1) // m_tile_size[m_mip_levels[0]][x]
                for x in range(2)
            ]

            # num of chunk: z
            # so that the tile-based mip-levels can output tiles
            num_ztile = m_zres[max(0, m_mip_id - 1)] * self.chunk_size[2]
            num_zchunk = (self.volume_size[2] + num_ztile - 1) // num_ztile
            zstep = self.mip_ratio[m_mip_levels[0]][2]
            for z in range(start_chunk, num_zchunk, step_chunk):
                z0 = (z * num_ztile) // zstep
                z1 = min(self.volume_size[2], (z + 1) * num_ztile) // zstep
                for y, x in itertools.product(range(num_chunk[1]), range(num_chunk[0])):            
                    print(
                        f"do tile-chunk: {z}/{num_zchunk}, {y}/{num_chunk[1]},"
                        "                                 "
                        f" {x}/{num_chunk[0]}"
                    )
                    # generate global coord
                    for i in range(m_mip_levels[0], m_mip_id):
                        # add offset for axis-aligned write
                        x0[i] = x * m_tile_size[i][0]
                        x1[i] = min(x0[i] + m_tile_size[i][0], m_size[i][0])
                        y0[i] = y * m_tile_size[i][1]
                        y1[i] = min(y0[i] + m_tile_size[i][1], m_size[i][1])
                    # read tiles
                    # input/output dimension order: z,y,x
                    # convert to global coord if mip_ratio[0]!=[0,0,0]
                    ims = getVolume(
                        z0,
                        z1,
                        y0[m_mip_levels[0]],
                        y1[m_mip_levels[0]],
                        x0[m_mip_levels[0]],
                        x1[m_mip_levels[0]],
                        self.mip_ratio[m_mip_levels[0]],
                    )

                    for zz in range(z0, z1):
                        if zz - z0 >= ims.shape[0]:
                            im = np.zeros(ims[0].shape, ims.dtype)
                        else:
                            im = ims[zz - z0]
                        im = im.transpose((1, 0)) if im.ndim == 2 else im.transpose((1, 0, 2))
                        sz0 = im.shape
                        # in case the output is not padded for invalid regions
                        full_size_tile = (sz0[0] == m_tile_size[m_mip_levels[0]][0]) * (
                            sz0[1] == m_tile_size[m_mip_levels[0]][1]
                        )
                        for i in range(m_mip_levels[0], m_mip_id):
                            # iterative bilinear downsample
                            # bug: last border tiles, not full size ...
                            sz0 = np.array(im.shape)
                            if full_size_tile:
                                # im = cv2.resize(im.astype(np.float32), tuple(tszA[i]), m_resize).astype(m_dtype)
                                sz_r = m_tile_size[i][: len(sz0)] / sz0
                                sz_im = m_tile_size[i]
                            else:
                                tszA_t = [
                                    x1[i] - x0[i],
                                    y1[i] - y0[i],
                                    num_channel,
                                ]
                                sz_r = tszA_t[: len(sz0)] / sz0
                                sz_im = tszA_t
                            if im.ndim == 2:
                                im = zoom(im, sz_r, order=m_resize)
                            else:
                                im0 = im.copy()
                                im = np.zeros(sz_im, im.dtype)
                                for c in range(im.shape[-1]):
                                    im[:, :, c] = zoom(
                                        im0[:, :, c],
                                        sz_r[:2],
                                        order=m_resize,
                                    )

                            # save image into tiles
                            zstep_mip = m_zres[i] // m_zres[m_mip_levels[0]]
                            if zz % zstep_mip == 0:
                                zzl = (zz // zstep_mip) % (
                                    self.chunk_size[2]
                                )
                                if i < m_mip_id:  # write the whole tile
                                    m_tiles[i][
                                        : im.shape[0], : im.shape[1], zzl
                                    ] = im.reshape(
                                        m_tiles[i][
                                            : im.shape[0],
                                            : im.shape[1],
                                            zzl,
                                        ].shape
                                    )
                                else:  # write into the whole section
                                    if (
                                        im[
                                            : (x1[i] - x0[i]),
                                            : (y1[i] - y0[i]),
                                        ].max()
                                        > 0
                                    ):
                                        tmp = m_tiles[i][
                                            x0[i] : x1[i],
                                            y0[i] : y1[i],
                                            zzl,
                                        ]
                                        tmp[:] = im[
                                            : (x1[i] - x0[i]),
                                            : (y1[i] - y0[i]),
                                        ].reshape(tmp.shape)

                            # < mipI: write for each tile
                            # chunk filled or last image
                            if (zz + 1) % (
                                zstep_mip * self.chunk_size[2]
                            ) == 0 or (z == num_zchunk - 1) * (
                                zz == z1 - 1
                            ):
                                # take the ceil for the last chunk
                                z1g = (zz + 1 + zstep_mip - 1) // zstep_mip
                                z0g = (
                                    (z1g - 1) // self.chunk_size[2]
                                ) * self.chunk_size[2]
                                # check volume align
                                if (
                                    m_tiles[i][
                                        : x1[i] - x0[i],
                                        : y1[i] - y0[i],
                                        : z1g - z0g,
                                        :,
                                    ].max()
                                    > 0
                                ):
                                    m_vols[i][
                                        x0[i]
                                        + m_oset[i][0] : x1[i]
                                        + m_oset[i][0],
                                        y0[i]
                                        + m_oset[i][1] : y1[i]
                                        + m_oset[i][1],
                                        z0g
                                        + m_oset[i][2] : z1g
                                        + m_oset[i][2],
                                        :,
                                    ] = m_tiles[i][
                                        : x1[i] - x0[i],
                                        : y1[i] - y0[i],
                                        : z1g - z0g,
                                        :,
                                    ]
                                    # print(i, m_oset[i][2] + z0g, m_oset[i][2] + z1g, m_tiles[i][: x1[i] - x0[i], : y1[i] - y0[i], : z1g - z0g, :].max())
                                    m_tiles[i][:] = 0
        if m_mip_levels[-1] >= m_mip_id:
            # step 3: do m_mip_id-all
            # num of chunk: z
            # so that the tile-based mip-levels can output tiles
            num_ztile = m_zres[-1] * self.chunk_size[2]
            num_zchunk = (self.volume_size[2] + num_ztile - 1) // num_ztile
            mip_start = max(m_mip_levels[0], m_mip_id)
            zstep = self.mip_ratio[mip_start][2]
            for z in range(start_chunk, num_zchunk, step_chunk):
                z0 = (z * num_ztile) // zstep
                z1 = min(self.volume_size[2], (z + 1) * num_ztile) // zstep
                print("do section-chunk: %d/%d" % (z, num_zchunk))
                ims = getVolume(
                    z0,
                    z1,
                    0,
                    m_size[mip_start][1],
                    0,
                    m_size[mip_start][0],
                    self.mip_ratio[mip_start],
                )

                for zz in range(z0, z1):
                    if zz - z0 >= ims.shape[0]:
                        im = np.zeros(ims[0].shape, ims.dtype)
                    else:
                        im = ims[zz - z0]
                    im = im.transpose((1, 0)) if im.ndim == 2 else im.transpose((1, 0, 2))
                    # in case the output is not padded for invalid regions
                    i_prev = mip_start
                    for i in range(mip_start, m_mip_levels[-1] + 1):
                        # iterative bilinear downsample
                        # bug: last border tiles, not full size ...
                        sz_r = [
                            float(self.mip_ratio[i_prev][j]
                            / self.mip_ratio[i][j])
                            for j in range(2)
                        ]
                        i_prev = i
                        if im.ndim == 2:  # single-channel
                            im = zoom(im, sz_r, order=m_resize)
                        else:  # multi-channel
                            im0 = im.copy()
                            im = np.zeros(sz_im, im.dtype)
                            for c in range(im.shape[-1]):
                                im[:, :, c] = zoom(
                                    im0[:, :, c], sz_r, order=m_resize
                                )

                        # save image into tiles
                        zstep_mip = m_zres[i] // m_zres[mip_start]
                        if zz % zstep_mip == 0:
                            zzl = (zz // zstep_mip) % (self.chunk_size[2])
                            try:
                                m_tiles[i][: im.shape[0], : im.shape[1], zzl] = (
                                    im.reshape(
                                        m_tiles[i][
                                            : im.shape[0], : im.shape[1], zzl
                                        ].shape
                                    )
                                )
                            except:
                                import pdb;pdb.set_trace()

                        # < mipI: write for each tile
                        # chunk filled or last image
                        if (zz + 1) % (zstep_mip * self.chunk_size[2]) == 0 or (
                            z == num_zchunk - 1
                        ) * (zz == z1 - 1):
                            # take the ceil for the last chunk
                            z1g = (zz + 1 + zstep_mip - 1) // zstep_mip
                            # z0g = z1g - self.chunk_size[2]
                            # outlier: last unfilled chunk
                            z0g = ((z1g - 1) // self.chunk_size[2]) * self.chunk_size[2]
                            if m_tiles[i][:, :, : z1g - z0g, :].max() > 0:
                                try:
                                    m_vols[i][
                                        m_oset[i][0] :,
                                        m_oset[i][1] :,
                                        z0g + m_oset[i][2] : z1g + m_oset[i][2],
                                        :,
                                    ] = m_tiles[i][:, :, : z1g - z0g, :]
                                    m_tiles[i][:] = 0
                                except:
                                    import pdb;pdb.set_trace()

    def create_tile(
        self,
        getVolume,
        cloudpath="",
        data_type="im",
        mip_levels=None,
        tile_size=None,
        num_thread=1,
        do_subdir=True,
        num_channel=1,
        start_chunk=0,
        step_chunk=1,
    ):
        (
            m_resize,
            m_mip_levels,
            m_mip_id,
            m_vols,
            m_tile_size,
            m_size,
            m_oset,
            m_tiles,
            m_zres,
        ) = self.create_tile_setup(
            cloudpath,
            data_type,
            mip_levels,
            tile_size,
            num_thread,
            do_subdir,
            num_channel,
        )

        # num of chunk: x and y (offset)
        # keep the size
        # num_chunk = [(m_size[mip_levels[0]][x] + m_tile_size[mip_levels[0]][x]-1 - m_oset[mip_levels[0]][x]) // m_tile_size[mip_levels[0]][x] for x in range(2)]
        if tile_size is None:
            tile_size = [self.mip_ratio[-1][0]//self.mip_ratio[0][0]*self.chunk_size[x] for x in range(2)]
        num_chunk = [
            (m_size[0][x] + m_tile_size[0][x] - 1) // m_tile_size[0][x] for x in range(2)
        ]
        # num of chunk: z
        # so that the tile-based mip-levels can output tiles
        num_ztile = (
            m_zres[max(0, m_mip_id - 1)] // m_zres[0] * self.chunk_size[2]
        )
        num_chunk += [(m_size[0][2] + num_ztile - 1) // num_ztile]
        # num_chunk += [(m_size[mip_levels[0]][2] - m_oset[mip_levels[0]][2] + num_ztile - 1) // num_ztile]
        # z0,z1,y0,y1,x0,x1: relative coord
        # tile for the finest level
        x0 = [None] * len(mip_levels)
        x1 = [None] * len(mip_levels)
        y0 = [None] * len(mip_levels)
        y1 = [None] * len(mip_levels)
        for z in range(start_chunk, num_chunk[2], step_chunk):
            z0 = z * num_ztile
            z1 = min(self.volume_size[2], (z + 1) * num_ztile)
            for y in range(num_chunk[1]):
                for x in range(num_chunk[0]):
                    print(
                        "do chunk: %d/%d, %d/%d, %d/%d"
                        % (z, num_chunk[2], y, num_chunk[1], x, num_chunk[0])
                    )
                    # generate global coord
                    for i in range(len(mip_levels)):
                        # add offset for axis-aligned write
                        x0[i] = x * m_tile_size[i][0]
                        x1[i] = min(x0[i] + m_tile_size[i][0], m_size[i][0])
                        y0[i] = y * m_tile_size[i][1]
                        y1[i] = min(y0[i] + m_tile_size[i][1], m_size[i][1])
                    # read tiles
                    # input/output dimension order: z,y,x
                    # convert to global coord if mip_ratio[0]!=[0,0,0]
                    ims = getVolume(
                        z0, z1, y0[0], y1[0], x0[0], x1[0], self.mip_ratio[0]
                    )

                    for zz in range(z0, z1):
                        if zz - z0 >= ims.shape[0]:
                            im = np.zeros(ims[0].shape, ims.dtype)
                        else:
                            im = ims[zz - z0]
                        im = im.transpose((1, 0)) if im.ndim == 2 else im.transpose((1, 0, 2))
                        sz0 = im.shape
                        # in case the output is not padded for invalid regions
                        full_size_tile = (sz0[0] == m_tile_size[0][0]) * (
                            sz0[1] == m_tile_size[0][1]
                        )
                        for i in range(len(mip_levels)):
                            # iterative bilinear downsample
                            # bug: last border tiles, not full size ...
                            sz0 = im.shape
                            if full_size_tile:
                                # im = cv2.resize(im.astype(np.float32), tuple(tszA[i]), m_resize).astype(m_dtype)
                                sz_r = m_tile_size[i][: len(sz0)] / np.array(sz0)
                                sz_im = m_tile_size[i]
                            else:
                                tszA_t = [
                                    x1[i] - x0[i],
                                    y1[i] - y0[i],
                                    num_channel,
                                ]
                                sz_r = tszA_t[: len(sz0)] / np.array(sz0)
                                sz_im = tszA_t
                            if im.ndim == 2:
                                im = zoom(im, sz_r, order=m_resize)
                            else:
                                im0 = im.copy()
                                im = np.zeros(sz_im, im.dtype)
                                for c in range(im.shape[-1]):
                                    im[:, :, c] = zoom(
                                        im0[:, :, c], sz_r[:2], order=m_resize
                                    )

                            # save image into tiles
                            if zz % m_zres[i] == 0:
                                zzl = (zz // m_zres[i]) % (self.chunk_size[2])
                                if i < m_mip_id:  # write the whole tile
                                    m_tiles[i][
                                        : im.shape[0], : im.shape[1], zzl
                                    ] = im.reshape(
                                        m_tiles[i][
                                            : im.shape[0], : im.shape[1], zzl
                                        ].shape
                                    )
                                elif (
                                        im[
                                            : (x1[i] - x0[i]),
                                            : (y1[i] - y0[i]),
                                        ].max()
                                        > 0):
                                    # write into the whole section
                                    tmp = m_tiles[i][
                                        x0[i] : x1[i], y0[i] : y1[i], zzl
                                    ]
                                    tmp[:] = im[
                                        : (x1[i] - x0[i]),
                                        : (y1[i] - y0[i]),
                                    ].reshape(tmp.shape)

                        # < mipI: write for each tile
                        for i in [ii for ii in mip_levels if ii < m_mip_id]:
                            # chunk filled or last image
                            if (zz + 1) % (
                                m_zres[i] * self.chunk_size[2]
                            ) == 0 or (z == num_chunk[2] - 1) * (zz == z1 - 1):
                                # take the ceil for the last chunk
                                z1g = (zz + m_zres[i]) // m_zres[i]
                                z0g = (
                                    (z1g - 1) // self.chunk_size[2]
                                ) * self.chunk_size[2]
                                # check volume align
                                if (
                                    m_tiles[i][
                                        : x1[i] - x0[i],
                                        : y1[i] - y0[i],
                                        : z1g - z0g,
                                        :,
                                    ].max()
                                    > 0
                                ):
                                    m_vols[i][
                                        x0[i]
                                        + m_oset[i][0] : x1[i]
                                        + m_oset[i][0],
                                        y0[i]
                                        + m_oset[i][1] : y1[i]
                                        + m_oset[i][1],
                                        z0g + m_oset[i][2] : z1g + m_oset[i][2],
                                        :,
                                    ] = m_tiles[i][
                                        : x1[i] - x0[i],
                                        : y1[i] - y0[i],
                                        : z1g - z0g,
                                        :,
                                    ]
                                    # print(i, z0g, z1g)
                                    # print(i, m_oset[i][2] + z0g, m_oset[i][2] + z1g, m_tiles[i][: x1[i] - x0[i], : y1[i] - y0[i], : z1g - z0g, :].max())
                                    m_tiles[i][:] = 0
                        # >= mipI: write for each section
                        # in one zchunk, there can be multiple chunk in z
                        # last xy chunk
                        # reuse compute, but can't be paralelled...
                        if y == num_chunk[1] - 1 and x == num_chunk[0] - 1:
                            for i in [
                                ii for ii in mip_levels if ii >= m_mip_id
                            ]:
                                if (zz + 1) % (
                                    m_zres[i] * self.chunk_size[2]
                                ) == 0 or zz == m_size[0][2] - 1:
                                    z1g = (zz + 1 + m_zres[i] - 1) // m_zres[i]
                                    z0g = z1g - self.chunk_size[2]
                                    if (zz + 1) % (
                                        m_zres[i] * self.chunk_size[2]
                                    ) != 0:  # last unfilled chunk
                                        z0g = (
                                            z1g // self.chunk_size[2]
                                        ) * self.chunk_size[2]
                                    if (
                                        m_tiles[i][:, :, : z1g - z0g, :].max()
                                        > 0
                                    ):
                                        m_vols[i][
                                            m_oset[i][0] :,
                                            m_oset[i][1] :,
                                            z0g
                                            + m_oset[i][2] : z1g
                                            + m_oset[i][2],
                                            :,
                                        ] = m_tiles[i][:, :, : z1g - z0g, :]
                                        m_tiles[i][:] = 0

    def create_mesh(
        self,
        cloudpath="",
        mip_level=0,
        volume_size=[256, 256, 100],
        num_thread=1,
        dust_threshold=None,
        do_subdir=False,
        object_ids=None,
    ):
        from taskqueue import LocalTaskQueue
        import igneous.task_creation as tc

        if cloudpath == "":
            cloudpath = f"{self.cloudpath}/seg/"
        tq = LocalTaskQueue(parallel=num_thread)
        tasks = tc.create_meshing_tasks(
            cloudpath,
            mip=mip_level,
            shape=volume_size,
            mesh_dir="mesh",
            object_ids=object_ids,
            dust_threshold=20,
            max_simplification_error=40,
            do_subdir=do_subdir,
        )
        tq.insert(tasks)
        tq.execute()

        tq = LocalTaskQueue(parallel=num_thread)
        tasks = tc.create_mesh_manifest_tasks(cloudpath)
        tq.insert(tasks)
        tq.execute()

    def create_skeleton(
        self, coordinates, cloudpath="", volume_size=None, resolution=None
    ):
        # coordinates is a list of tuples (x,y,z)
        if cloudpath == "":
            cloudpath = os.path.join(self.cloudpath, "skeletons")
        if volume_size is None:
            volume_size = self.volume_size
        if resolution is None:
            resolution = self.resolution

        foldername = cloudpath
        if cloudpath[:4] == "file":
            # convert from cloudvolume path to local path
            foldername = cloudpath[7:]
        mkdir(foldername)
        mkdir(os.path.join(foldername, "spatial0"))

        self.write_skeleton_info(
            os.path.join(foldername, "info"), volume_size, resolution
        )

        with open(
                os.path.join(foldername, "spatial0", "0_0_0"), "wb"
            ) as outfile:
            self._extracted_from_create_skeleton(coordinates, outfile)

    # TODO Rename this here and in `create_skeleton`
    def _extracted_from_create_skeleton(self, coordinates, outfile):
        total_count = len(
            coordinates
        )  # coordinates is a list of tuples (x,y,z)
        buf = struct.pack("<Q", total_count)
        for x, y, z in coordinates:
            pt_buf = struct.pack("<3f", x, y, z)
            buf += pt_buf
            # write the ids at the end of the buffer as increasing integers
        id_buf = struct.pack(f"<{len(coordinates)}Q", *range(len(coordinates)))
        buf += id_buf
        outfile.write(buf)

    def write_skeleton_info(
        self, output_file="", volume_size=None, resolution=None
    ):
        if output_file is None:
            output_file = f"{self.cloudpath}skeletons/info"
        if volume_size is None:
            volume_size = self.volume_size
        if resolution is None:
            resolution = self.resolution

        out = """{
           "@type" : "neuroglancer_annotations_v1",
           "annotation_type" : "POINT",
           "by_id" : {
              "key" : "by_id"
           },
           "dimensions" : {
              "x" : [ %.e, "m" ],
              "y" : [ %.e, "m" ],
              "z" : [ %.e, "m" ]
           },
           "lower_bound" : [ 0, 0, 0 ],
           "properties" : [],
           "relationships" : [],
           "spatial" : [
              {
                 "chunk_size" : [ %d, %d, %d ],
                 "grid_shape" : [ 1, 1, 1 ],
                 "key" : "spatial0",
                 "limit" : 1
              }
           ],
           "upper_bound" : [ %d, %d, %d]
        }""" % (
            resolution[0],
            resolution[1],
            resolution[2],
            volume_size[0],
            volume_size[1],
            volume_size[2],
            volume_size[0],
            volume_size[1],
            volume_size[2],
        )
        write_txt(output_file, out)

    def write_htaccess(self, output_file, data_type="im", do_subdir=False):
        out = """# If you get a 403 Forbidden error, try to comment out the Options directives
        # below (they may be disallowed by your server's AllowOverride setting).

        #<IfModule headers_module>
            # Needed to use the data from a Neuroglancer instance served from a
            # different server (see http://enable-cors.org/server_apache.html).
        #    Header set Access-Control-Allow-Origin "*"
        #</IfModule>

        # Data chunks are stored in sub-directories, in order to avoid having
        # directories with millions of entries. Therefore we need to rewrite URLs
        # because Neuroglancer expects a flat layout.
        #Options FollowSymLinks
        """
        if do_subdir:
            out += """RewriteEngine On
        RewriteRule "^(.*)/([0-9]+-[0-9]+)_([0-9]+-[0-9]+)_([0-9]+-[0-9]+)$" "$1/$2/$3/$4"
        """

        """
        # Microsoft filesystems do not support colons in file names, but pre-computed
        # meshes use a colon in the URI (e.g. 100:0). As :0 is the most common (only?)
        # suffix in use, we will serve a file that has this suffix stripped.
        # RewriteCond "%{REQUEST_FILENAME}" !-f
        # RewriteRule "^(.*):0$" "$1"
        """

        if data_type == "seg":
            out += """<IfModule mime_module>
            # Allow serving pre-compressed files, which can save a lot of space for raw
            # chunks, compressed segmentation chunks, and mesh chunks.
            #
            # The AddType directive should in theory be replaced by a "RemoveType .gz"
            # directive, but with that configuration Apache fails to serve the
            # pre-compressed chunks (confirmed with Debian version 2.2.22-13+deb7u6).
            # Fixes welcome.
            # Options Multiviews
            AddEncoding x-gzip .gz
            AddType application/octet-stream .gz
        </IfModule>
        """
        write_txt(output_file, out)

    def remove_gz(self, cloudpath="", folder_key="_", option="copy"):
        # utility function
        if "file" == cloudpath[:4]:
            cloudpath = cloudpath[7:]
        fns = [
            x
            for x in glob(cloudpath + "/*")
            if folder_key in x[x.rfind("/"):]
        ]
        for fn in fns:
            print(fn)
            gzs = glob(fn + "/*.gz")
            if option == "copy":
                for gz in gzs:
                    if not os.path.exists(gz[:-3]):
                        shutil.copy(gz, gz[:-3])
            elif option == "move":
                for gz in gzs:
                    shutil.move(gz, gz[:-3])
            elif option == "remove_orig":
                for gz in gzs:
                    os.remove(gz[:-3])
            elif option == "copy_subdir":
                for gz in gzs:
                    gz2 = gz[gz.rfind("/") : -3].split("_")
                    mkdir(fn + gz2[0] + "/" + gz2[1], 2)
                    shutil.copy(gz, fn + "/".join(gz2))
