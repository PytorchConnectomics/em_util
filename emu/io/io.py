import os
import sys
import pickle
import numpy as np
import imageio
from scipy.ndimage import zoom
import h5py
import json

from .seg import rgb_to_seg


def mkdir(foldername, opt=""):
    """
    Create a directory.

    Args:
        fn (str): The path of the directory to create.
        opt (str, optional): The options for creating the directory. Defaults to "".

    Returns:
        None

    Raises:
        None
    """
    if opt == "parent":  # until the last /
        foldername = os.path.dirname(foldername)
    if not os.path.exists(foldername):
        if "all" in opt:
            os.makedirs(foldername)
        else:
            os.mkdir(foldername)


def read_image(filename, image_type="image", ratio=None, resize_mode=1, data_type="2d"):
    """
    Read an image from a file.

    Args:
        filename (str): The path to the image file.
        image_type (str, optional): The type of image to read. Defaults to "image".
        ratio (int or list, optional): The scaling ratio for the image. Defaults to None.
        resize_mode (int, optional): The order of interpolation for scaling. Defaults to 1.
        data_type (str, optional): The type of image data to read. Defaults to "2d".

    Returns:
        numpy.ndarray: The image data.

    Raises:
        AssertionError: If the ratio dimensions do not match the image dimensions.
    """
    if data_type == "2d":
        # assume the image of the size M x N x C
        image = imageio.imread(filename)
        if image_type == "seg":
            image = rgb_to_seg(image)
        if ratio is not None:
            if str(ratio).isnumeric():
                ratio = [ratio, ratio]
            if image_type == "seg":
                resize_mode = 0
            if image.ndim == 2:
                image = zoom(image, ratio, order=resize_mode)
            else:
                # do not zoom the color channel
                image = zoom(image, ratio + [1], order=resize_mode)
    else:
        # read in nd volume
        image = imageio.volread(filename)
        if ratio is not None:
            assert (
                str(ratio).isnumeric() or len(ratio) == image.ndim
            ), f"ratio's dim {len(ratio)} is not equal to image's dim {image.ndim}"
            image = zoom(image, ratio, order=resize_mode)
    return image

def write_image(filename, image, image_type="image"):
    if image_type=='seg':
        image = rgb_to_seg(image)
    imageio.imsave(filename, image)


def get_file_number(filename, index):
    return len(filename) if isinstance(filename, list) else len(index)
def get_filename(filename, index, x):
    return filename[x] if isinstance(filename, list) else filename % index[x]
def read_image_folder(
    filename, index=None, image_type="image", ratio=None, resize_mode=None
):
    """
    Read a folder of images.

    Args:
        filename (str or list): The path to the image folder or a list of image file paths.
        index (int or list, optional): The index or indices of the images to read. Defaults to None.
        image_type (str, optional): The type of image to read. Defaults to "image".
        ratio (list, optional): The downsampling ratio for the images. Defaults to None.
        resize_mode (int, optional): The order of interpolation for scaling. Defaults to 1.

    Returns:
        numpy.ndarray: The folder of images.

    Raises:
        None
    """
    if ratio is None:
        ratio = [1, 1]
    # either filename or index is a list
    num_image = get_file_number(filename, index)    
    im0 = read_image(get_filename(filename, index, 0), image_type, ratio, resize_mode)
    sz = list(im0.shape)
    out = np.zeros([num_image] + sz, im0.dtype)
    out[0] = im0
    for i in range(1, num_image):
        out[i] = read_image(get_filename(filename, index, i), image_type, ratio, resize_mode)
    return out

def write_image_folder(
    filename, data, index=None, image_type="image"):
    if index is None:
        index = range(data.shape[0]) 
    for i in index:
        write_image(get_filename(filename, index, i), data[i], image_type) 


def read_vol_chunk(file_handler, chunk_id=0, chunk_num=1):
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
        num_z = int(np.ceil(file_handler.shape[0] / float(chunk_num)))
        return np.array(file_handler[chunk_id * num_z : (chunk_id + 1) * num_z])
    
def read_vol(filename, dataset=None, chunk_id=0, chunk_num=1):   
    """
    Read data from various file formats.

    Args:
        filename (str): The path to the file.
        dataset (str, optional): The name of the dataset within the file. Defaults to None.
        chunk_id (int, optional): The ID of the chunk to read. Defaults to 0.
        chunk_num (int, optional): The total number of chunks. Defaults to 1.

    Returns:
        numpy.ndarray: The read data.

    Raises:
        Exception: If the file format is not supported or if the file cannot be read.
    """
    if filename[-3:] == "npy":
        out = np.load(filename)
    elif filename[-3:] in ["tif", "iff"]:
        out = read_image(filename, data_type="nd")
    elif filename[-2:] == "h5":
        out = read_h5(filename, dataset, chunk_id, chunk_num)
    elif filename[-3:] == "zip":
        out = read_zarr(filename, dataset, chunk_id, chunk_num)
    else:
        raise f"Can't read the file type of {filename}"
    return out

def read_zarr(filename, dataset=None, chunk_id=0, chunk_num=1):
    """
    Read data from a Zarr file.

    Args:
        filename (str): The path to the Zarr file.
        dataset (str, optional): The name of the dataset within the file. Defaults to None.
        chunk_id (int, optional): The ID of the chunk to read. Defaults to 0.
        chunk_num (int, optional): The total number of chunks. Defaults to 1.

    Returns:
        numpy.ndarray: The read data.

    Raises:
        Exception: If the dataset is not found or if there is an error reading the file.
    """
    import zarr

    fid = zarr.open_group(filename)
    if dataset is None:
        dataset = fid.info_items()[-1][1]
        if "," in dataset:
            dataset = dataset[: dataset.find(",")]
    
    return read_vol_chunk(fid[dataset], chunk_id, chunk_num)
    
def read_h5(filename, dataset=None, chunk_id=0, chunk_num=1):
    """
    Read data from an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        dataset (str or list, optional): The name or names of the dataset(s) to read. Defaults to None.
        chunk_id (int, optional): The ID of the chunk to read. Defaults to 0.
        chunk_num (int, optional): The total number of chunks. Defaults to 1.

    Returns:
        numpy.ndarray or list: The data from the HDF5 file.

    Raises:
        None
    """
    fid = h5py.File(filename, "r")
    if dataset is None:
        dataset = fid.keys() if sys.version[0] == "2" else list(fid)

    out = [None] * len(dataset)
    for di, d in enumerate(dataset):
        out[di] = read_vol_chunk(fid[d], chunk_id, chunk_num)

    return out[0] if len(out) == 1 else out


def write_h5(filename, data, dataset="main"):
    """
    Write data to an HDF5 file.

    Args:
        filename (str): The path to the HDF5 file.
        data (numpy.ndarray or list): The data to write.
        dataset (str or list, optional): The name or names of the dataset(s) to create. Defaults to "main".

    Returns:
        None

    Raises:
        None
    """
    fid = h5py.File(filename, "w")
    if isinstance(dataset, (list,)):
        for i, dd in enumerate(dataset):
            ds = fid.create_dataset(
                dd,
                data[i].shape,
                compression="gzip",
                dtype=data[i].dtype,
            )
            ds[:] = data[i]
    else:
        ds = fid.create_dataset(
            dataset, data.shape, compression="gzip", dtype=data.dtype
        )
        ds[:] = data
    fid.close()


def read_txt(filename):
    """
    Args:
    filename (str): The path to the text file.

    Returns:
    "main",  list: The lines of the text file as a list of strings.

    Raises:
        None
    """
    with open(filename, "r") as a:
        content = a.readlines()
    return content


def write_txt(filename, content):
    """
    Write content to a text file.

    Args:
        filename (str): The path to the text file.
        content (str or list): The content to write. If a list, each element will be written as a separate line.

    Returns:
        None

    Raises:
        None
    """
    with open(filename, "w") as a:
        if isinstance(content, (list,)):
            for ll in content:
                a.write(ll)
                if "\n" not in ll:
                    a.write("\n")
        else:
            a.write(content)


def write_gif(filename, data, duration=0.5):
    """
    Write a GIF animation to a file.

    Args:
        filename (str): The path to save the GIF animation.
        data (numpy.ndarray): The frames of the animation.
        duration (float, optional): The duration of each frame in seconds. Defaults to 0.5.

    Returns:
        None

    Raises:
        None
    """
    imageio.mimsave(filename, data, "GIF", duration=duration)


def read_pkl(filename):
    """
    The function `read_pkl` reads a pickle file and returns a list of the objects stored in the file.

    :param filename: The filename parameter is a string that represents the name of the file you want to
    read. It should include the file extension, such as ".pkl" for a pickle file
    :return: a list of objects that were read from the pickle file.
    """
    data = []
    with open(filename, "rb") as fid:
        while True:
            try:
                data.append(pickle.load(fid))
            except EOFError:
                break
    return data


def write_pkl(filename, content):
    """
    Write content to a pickle file.

    Args:
        filename (str): The path to the pickle file.
        content (object or list): The content to write. If a list, each element will be pickled separately.

    Returns:
        None

    """
    with open(filename, "wb") as f:
        if isinstance(content, (list,)):
            for val in content:
                pickle.dump(val, f)
        else:
            pickle.dump(content, f)
            
            
def write_json(filename, content):            
    """
    Write content to a JSON file.
    Args:
        filename (str): The name of the output file.
        ("HP02", content (Any): The content to be written to the file.

    Returns:
        None
    """ 
    with open(filename, "w") as fid:
        json.dump(filename, fid)            
