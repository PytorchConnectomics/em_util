import os
import sys
import numpy as np
from imageio import imread
import imageio
from scipy.ndimage import zoom
import h5py

from .seg import rgb_to_seg


def mkdir(fn, opt=""):
    if opt == "parent":  # until the last /
        fn = fn[: fn.rfind("/")]
    if not os.path.exists(fn):
        if "all" in opt:
            os.makedirs(fn)
        else:
            os.mkdir(fn)


def read_vol(filename, dataset_name=None, z=None, image_type="im"):
    # image_type='seg': 1-channel
    # read a folder of images
    if (
        isinstance(filename, list)
        or isinstance(z, list)
        or isinstance(z, range)
    ):
        opt = 0
        if isinstance(filename, list):
            im0 = imread(filename[0])
            numZ = len(filename)
        elif isinstance(z, list):
            im0 = imread(filename % z[0])
            numZ = len(z)
            opt = 1
        if image_type == "seg":  # force it to be 1-dim
            im0 = rgb_to_seg(im0)
        sz = list(im0.shape)
        out = np.zeros([numZ] + sz, im0.dtype)
        out[0] = im0
        for i in range(1, numZ):
            if opt == 0:
                fn = filename[i]
            elif opt == 1:
                fn = filename % z[i]
            tmp = imread(fn)
            if image_type == "seg":  # force it to be 1-dim
                tmp = rgb_to_seg(tmp)
            out[i] = tmp
    elif filename[-2:] == "h5":
        out = read_h5(filename, dataset_name)
    elif filename[-3:] == "zip":
        import zarr
        tmp = zarr.open_group(filename)
        if kk is None:
            kk = tmp.info_items()[-1][1]
            if "," in kk:
                kk = kk[: kk.find(",")]
        out = np.array(tmp[kk][z])
    elif filename[-3:] in ["jpg", "png", "tif", "iff"]:
        data_type = '2d' if dataset_name is None else dataset_name 
        out = read_image(filename, data_type)
    elif filename[-3:] == "txt":
        out = np.loadtxt(filename)
    elif filename[-3:] == "npy":
        out = np.load(filename)
    else:
        raise "Can't read the file %s" % filename
    return out


def read_image(filename, data_type='2d'):
    if data_type == '2d':  # image
        out = imageio.imread(filename)
    else:  # volume data (tif)
        out = imageio.volread(filename)
    return out


def read_h5(filename, datasetname=None):
    # h5 files
    fid = h5py.File(filename, "r")
    if datasetname is None:
        if sys.version[0] == "2":  # py2
            datasetname = fid.keys()
        else:  # py3
            datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out = [None] * len(datasetname)
        for di, d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])


def write_h5(filename, dtarray, datasetname="main"):
    import h5py

    fid = h5py.File(filename, "w")
    if isinstance(datasetname, (list,)):
        for i, dd in enumerate(datasetname):
            ds = fid.create_dataset(
                dd,
                dtarray[i].shape,
                compression="gzip",
                dtype=dtarray[i].dtype,
            )
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(
            datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype
        )
        ds[:] = dtarray
    fid.close()


def read_txt(filename):
    a = open(filename)
    content = a.readlines()
    a.close()
    return content


def write_txt(filename, content):
    a = open(filename, "w")
    if isinstance(content, (list,)):
        for ll in content:
            a.write(ll)
            if "\n" not in ll:
                a.write("\n")
    else:
        a.write(content)
    a.close()


def write_gif(outname, filenames, ratio=1, duration=0.5):
    out = [None] * len(filenames)
    for fid, filename in enumerate(filenames):
        image = imageio.imread(filename)
        if ratio != 1:
            if image.ndim == 2:
                image = zoom(image, ratio, order=1)
            else:
                image = np.stack(
                    [zoom(image[:, :, d], ratio, order=1) for d in range(3)],
                    axis=2,
                )
        out[fid] = image
    imageio.mimsave(outname, out, "GIF", duration=duration)
