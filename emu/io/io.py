import os, sys
import numpy as np
import imageio

def mkdir(fn,opt=0):
    if opt == 1 :# until the last /
        fn = fn[:fn.rfind('/')]
    if not os.path.exists(fn):
        if opt==2:
            os.makedirs(fn)
        else:
            os.mkdir(fn)

def readImage(filename):
    image = imageio.imread(filename)
    return image

# h5 files
def readH5(filename, datasetname=None):
    import h5py
    fid = h5py.File(filename,'r')
    if datasetname is None:
        if sys.version[0]=='2': # py2
            datasetname = fid.keys()
        else: # py3
            datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, (list,)):
        out=[None]*len(datasetname)
        for di,d in enumerate(datasetname):
            out[di] = np.array(fid[d])
        return out
    else:
        return np.array(fid[datasetname])

def writeH5(filename, dtarray, datasetname='main'):
    import h5py
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def readTxt(filename):
    a= open(filename)
    content = a.readlines()
    a.close()
    return content

def writeTxt(filename, content):
    a= open(filename,'w')
    if isinstance(content, (list,)):
        for ll in content:
            a.write(ll)
            if '\n' not in ll:
                a.write('\n')
    else:
        a.write(content)
    a.close()


