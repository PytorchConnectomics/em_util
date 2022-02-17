import numpy as np

def imTrimBlack(I, return_ind=False):
    # trim the black pixels on the border
    ind = np.zeros(6, int)
    for cid in range(3):
        if cid == 0:
            tmp_max = I.max(axis=1).max(axis=1)
        elif cid == 1:
            tmp_max = I.max(axis=0).max(axis=1)
        elif cid == 2:
            tmp_max = I.max(axis=0).max(axis=0)
        tmp_ind = np.where(tmp_max>0)[0]
        ind[cid * 2] = tmp_ind[0]
        ind[cid * 2 + 1] = tmp_ind[-1] +1
    if return_ind:
        return I[ind[0]:ind[1], ind[2]:ind[3], ind[4]:ind[5]], ind
    else:
        return I[ind[0]:ind[1], ind[2]:ind[3], ind[4]:ind[5]]

def imAdjust(I, thres=[1,99,True], autoscale=None):
    # compute percentile: remove too big or too small values
    # thres: [thres_low, thres_high, percentile]
    if thres[2]:
        I_low, I_high = np.percentile(I.reshape(-1), thres[:2])
    else:
        I_low, I_high = thres[0], thres[1]
    # thresholding
    I[I > I_high] = I_high
    I[I < I_low] = I_low
    if autoscale is not None:
        # scale to 0-1
        I = (I.astype(float)- I_low)/ (I_high-I_low)
        if autoscale =='uint8':
            # convert it to uint8
            I = (I * 255).astype(np.uint8)    
    return I
