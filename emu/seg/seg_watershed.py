import numpy as np

from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import remove_small_objects, dilation, square
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage.morphology import binary_erosion

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage.morphology import binary_fill_holes

def segResize(seg, target_size):
    return resize(seg, target_size, order=0, anti_aliasing=False, preserve_range=True)

def probToInstanceSeg_cc(probability, thres_prob = 0.8, thres_small = 128):
    # connected component
    if probability.max() > 1:
        probability = probability / 255.
    foreground = (probability > thres_prob)
    segm = label(foreground)
    segm = remove_small_objects(segm, thres_small)
        
    return segm.astype(np.uint32)

def probToInstanceSeg_watershed(probability, thres1=0.98, thres2=0.85, do_resize=False):
    # watersehd
    seed_map = probability > 255*thres1
    foreground = probability > 255*thres2
    seed = label(seed_map)
    segm = watershed(-semantic, seed, mask=foreground)
    segm = remove_small_objects(segm, 128)
    if do_resize:
        target_size = (semantic.shape[0], semantic.shape[1] // 2, semantic.shape[2] // 2)
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)

def boundaryContourToSeg(volume, thres1=0.8, thres2=0.4, do_resize=False):
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255*thres1)) * (boundary < int(255*thres2))
    # foreground = (semantic > int(255*thres1))
    segm = label(foreground)
    struct = np.ones((1,5,5))
    segm = dilation(segm, struct)
    segm = remove_small_objects(segm, 128)
    if do_resize:
        target_size = (semantic.shape[0], semantic.shape[1] // 2, semantic.shape[2] // 2)
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)

def probBoundaryToSeg(volume, thres1=0.9, thres2=0.8, thres3=0.85, do_resize=False):
    semantic = volume[0]
    boundary = volume[1]
    seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2))
    foreground = (semantic > int(255*thres3))
    seed = label(seed_map)
    segm = watershed(-semantic, seed, mask=foreground)
    segm = remove_small_objects(segm, 128)
    if do_resize:
        target_size = (semantic.shape[0], semantic.shape[1] // 2, semantic.shape[2] // 2)
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)


def watershed_3d(volume):
    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    distance = ndi.distance_transform_edt(volume)
    local_maxi = peak_local_max(distance, indices=False, 
                                footprint=np.ones((3, 3, 3)), labels=volume)
    markers = ndi.label(local_maxi)[0]
    seg = watershed(-distance, markers, mask=volume)
    return seg

def watershed_3d_open(volume, res=[1,1,1], erosion_iter = 0, marker_size = 0, zcut = -1):
    markers = ndi.label(binary_erosion(volume, iterations = erosion_iter))[0]
    if marker_size > 0:
        ui, uc = np.unique(markers, return_counts = True)
        rl = np.arange(1+ui.max()).astype(volume.dtype)
        rl[ui[uc < marker_size]] = 0
        markers = rl[markers]
    marker_id = np.unique(markers)
    if (marker_id > 0).sum()==1:
        # try the simple z-2D cut
        if zcut > 0 and volume.shape[0] >= zcut*1.5:
            from scipy.signal import find_peaks
            zsum = (volume>0).sum(axis=1).sum(axis=1)
            peaks = find_peaks(zsum, zsum.max()*0.7, distance=zcut*0.8)[0]
            if len(peaks) > 1:
                markers[:] = 0
                for i,peak in enumerate(peaks):
                    markers[peak][volume[peak]>0] = 1 + i
                marker_id = np.arange(len(peaks)+1)

    if (marker_id > 0).sum()==1:
        seg = volume
    else:
        if min(res) == max(res):
            distance = ndi.distance_transform_edt(volume)
        else:
            import edt
            # zyx: order='C'
            distance = edt.edt(volume, anisotropy=res)
        seg = watershed(-distance, markers, mask=volume)
    return seg

def imToSeg_2d(im, th_hole=512, th_small=10000, seed_footprint=[71,71]):
    if len(np.unique(im)) == 1:
        return (im<0).astype(np.uint8)
    thresh = threshold_otsu(im)
    seg = remove_small_objects(binary_fill_holes(remove_small_holes(im>thresh, th_hole)), th_small)
    if seed_footprint[0] > 0:
        seg = watershed_2d(seg, seed_footprint)
    return seg


def watershed_2d(volume, seed_footprint=[71,71]):
    distance = ndi.distance_transform_edt(volume)
    local_maxi = peak_local_max(distance, indices=False,footprint=np.ones(seed_footprint), labels=volume)
    seg_cc = label(volume)
    bid = list(set(range(1,seg_cc.max()+1))-set(np.unique(seg_cc[local_maxi])))
    # fill missing seg
    for i in bid:
        tmp = ndi.binary_erosion(seg_cc==i, iterations=10)
        local_maxi[tmp>0] = 1

    markers = ndi.label(local_maxi)[0]
    seg = watershed(-distance, markers, mask=volume)
    return seg
