import numpy as np
import h5py
from tqdm import tqdm
from .io import read_h5, write_h5, get_h5_chunk2d
from .box import merge_bbox_two_matrices

def split_arr_by_chunk(index, chunk_id, chunk_num, overlap=0):
    num = np.ceil(len(index) / float(chunk_num)).astype(int)
    return index[num * chunk_id : num * (chunk_id + 1) + overlap]

def vol_func_chunk(input_file, vol_func, output_file=None, output_chunk=8192, chunk_num=1, no_tqdm=False, dtype=None):
    if output_file is None or chunk_num==1:
        vol = read_h5(input_file)
        vol = vol_func(vol)
        if output_file is None:
            return vol
        else:
            write_h5(output_file, vol)
    else:
        fid_in = h5py.File(input_file, 'r')
        fid_in_data = fid_in[list(fid_in)[0]]
        fid_out = h5py.File(output_file, "w")
        vol_sz = np.array(fid_in_data.shape)
        num_z = int(np.ceil(vol_sz[0] / float(chunk_num)))

        chunk_sz = get_h5_chunk2d(output_chunk/num_z, vol_sz[1:])
        dtype = fid_in_data.dtype if dtype is None else dtype
        result = fid_out.create_dataset('main', vol_sz, dtype=dtype, \
            compression="gzip", chunks=(num_z,chunk_sz[0],chunk_sz[1]))

        for z in tqdm(range(chunk_num), disable=no_tqdm):
            tmp = read_vol_chunk(fid_in_data, z, chunk_num, num_z)
            result[z*num_z:(z+1)*num_z] = vol_func(tmp)

        fid_in.close()
        fid_out.close()

def vol_downsample_chunk(input_file, ratio, output_file=None, output_chunk=8192, chunk_num=1, no_tqdm=False):
    if output_file is None or chunk_num==1:
        vol = read_h5(input_file)
        vol = vol[::ratio[0], ::ratio[1], ::ratio[2]]
        if output_file is None:
            return vol
        else:
            write_h5(output_file, vol)
    else:
        fid_in = h5py.File(input_file, 'r')
        fid_in_data = fid_in[list(fid_in)[0]]
        fid_out = h5py.File(output_file, "w")
        vol_sz = (np.array(fid_in_data.shape) + ratio-1) // ratio
        num_z = int(np.ceil(vol_sz[0] / float(chunk_num)))
        # round it to be multiple
        num_z = ((num_z + ratio[0] - 1) // ratio[0]) * ratio[0]

        chunk_sz = get_h5_chunk2d(output_chunk/num_z, vol_sz[1:])
        result = fid_out.create_dataset('main', vol_sz, dtype=fid_in_data.dtype, \
            compression="gzip", chunks=(num_z,chunk_sz[0],chunk_sz[1]))

        for z in tqdm(range(chunk_num), disable=no_tqdm):
            tmp = read_vol_chunk(fid_in_data, z, chunk_num, num_z*ratio[0], ratio[0])[:, ::ratio[1],::ratio[2]]
            result[z*num_z:(z+1)*num_z] = tmp

        fid_in.close()
        fid_out.close()


def read_vol_chunk(file_handler, chunk_id=0, chunk_num=1, num_z=-1, ratio=1):
    """
    Read a chunk of data from a file handler.

    Args:
        file_handler: The file handler object.
        chunk_id: The ID of the chunk to read. Defaults to 0.
        chunk_num: The total number of chunks. Defaults to 1.

    Returns:
        numpy.ndarray: The read chunk of data.
    """
    if chunk_num == 1:
        # read the whole chunk
        return np.array(file_handler)
    elif chunk_num == -1:
        # read a specific slice
        return np.array(file_handler[chunk_id])
    else:
        # read a chunk
        if num_z == -1:
            num_z = int(np.ceil(file_handler.shape[0] / float(chunk_num)))
        return np.array(file_handler[chunk_id * num_z : (chunk_id + 1) * num_z: ratio])

def seg_unique_id_chunk(input_file, chunk_num=1, no_tqdm=False):
    if chunk_num == 1:
        return np.unique(read_h5(input_file))
    else:
        uid = []
        if isinstance(input_file, str):
            fid = h5py.File(input_file, 'r')
            seg = fid[list(fid)[0]]
        else:
            seg = input_file
        num_z = int(np.ceil(seg.shape[0] / float(chunk_num)))
        for i in tqdm(range(chunk_num), disable=no_tqdm):
            if i == 0:
                uid = np.unique(np.array(seg[i*num_z:(i+1)*num_z]))
            else:
                uid = np.hstack([uid, np.unique(np.array(seg[i*num_z:(i+1)*num_z]))])
                uid = np.unique(uid)
        if isinstance(input_file, str):
            fid.close()
    return uid

def merge_bbox_chunk(load_bbox, chunk, chunk_size):
    """
    Merge bounding boxes from chunks.

    Args:
        load_bbox (function): A function to load the bounding box for a given chunk.
        chunk (tuple): The dimensions of the chunk in each dimension.
        chunk_size (tuple): The size of each chunk in each dimension.

    Returns:
        numpy.ndarray: The merged bounding boxes.

    Notes:
        - The `load_bbox` function should take the chunk coordinates as arguments and return the bounding box for that chunk.
        - The `chunk` argument specifies the number of chunks in each dimension.
        - The `chunk_size` argument specifies the size of each chunk in each dimension.
        - The function iterates over each chunk, loads the bounding box, and merges them using the `merge_bbox_two_matrices` function.
        - The merged bounding boxes are returned as a numpy array.
    """
    num_dim = len(chunk)
    out = None
    if num_dim == 2:
        # merge 2D chunks    
        for xi in range(chunk[1]):
            for yi in range(chunk[0]):
                bbox = load_bbox(yi, xi)
                if bbox is not None:
                    # update the local coordinates to global coordinates
                    bbox[:, 1:5] += [
                        chunk_size[0] * yi,
                        chunk_size[0] * yi,
                        chunk_size[1] * xi,
                        chunk_size[1] * xi,
                    ]                    
                    out = merge_bbox_two_matrices(out, bbox)
    elif num_dim == 3:
        # merge 3D chunks        
        for xi in range(chunk[2]):
            for yi in range(chunk[1]):
                for zi in range(chunk[0]):
                    bbox = load_bbox(zi, yi, xi)
                    if bbox is not None:
                        # update the local coordinates to global coordinates
                        if bbox.ndim == 1:
                            bbox = bbox.reshape(1, -1)
                        bbox[:, 1:7] += [
                            chunk_size[0] * zi,
                            chunk_size[0] * zi,
                            chunk_size[1] * yi,
                            chunk_size[1] * yi,
                            chunk_size[2] * xi,
                            chunk_size[2] * xi,
                        ]
                        out = merge_bbox_two_matrices(out, bbox)
    
    return out


