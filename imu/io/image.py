import numpy as np

def imAdjust(I, thres=[1,99], do_percentile=True, autoscale=None):
    # compute percentile: remove too big or too small values
    if do_percentile:
        I_low, I_high = np.percentile(I.reshape(-1), thres)
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
