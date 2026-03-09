import numpy as np

def im2col(A, BSZ, stepsize=1):
    """
    Convert an input array to a matrix of overlapping blocks.

    Args:
        A (numpy.ndarray): The input array.
        BSZ (tuple): The size of each block.
        stepsize (int, optional): The step size between blocks. Defaults to 1.

    Returns:
        numpy.ndarray: The matrix of overlapping blocks.

    Notes:
        - The function converts the input array into a matrix of overlapping blocks.
        - Each block has a size specified by `BSZ`.
        - The step size between blocks can be adjusted using `stepsize`.
    """
    M,N = A.shape
    # Get Starting block indices
    start_idx = np.arange(0,M-BSZ[0]+1,stepsize)[:,None]*N + np.arange(0,N-BSZ[1]+1,stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A,start_idx.ravel()[:,None] + offset_idx.ravel())

def im_trim_by_intensity(I, threshold, return_ind=False):
    """
    Trim the pixels below a certain intensity threshold from an image (2d or 3d).

    Args:
        I (numpy.ndarray): The input image.
        threshold (float): The intensity threshold.
        return_ind (bool, optional): Whether to return the indices of the trimmed region. Defaults to False.

    Returns:
        numpy.ndarray: The trimmed image.
        tuple, optional: The trimmed image and the indices of the trimmed region.

    Notes:
        - The function trims the pixels below the specified intensity threshold from the image.
        - The indices of the trimmed region can be returned if `return_ind` is set to True.
    """
    ind = np.zeros(I.ndim * 2, int)
    for cid in range(I.ndim):
        if cid == 0:
            tmp_max = I.max(axis=1).max(axis=1)
        elif cid == 1:
            tmp_max = I.max(axis=0).max(axis=1)
        elif cid == 2:
            tmp_max = I.max(axis=0).max(axis=0)
        tmp_ind = np.where(tmp_max > threshold)[0]
        ind[cid * 2] = tmp_ind[0]
        ind[cid * 2 + 1] = tmp_ind[-1] + 1
    if I.ndim == 2:
        out = I[ind[0] : ind[1], ind[2] : ind[3]]
    else:
        out = I[ind[0] : ind[1], ind[2] : ind[3], ind[4] : ind[5]]
    return (out, ind) if return_ind else out


def im_adjust(I, threshold=None, auto_scale=None):
    """
    Adjust the intensity values of an image.

    Args:
        I (numpy.ndarray): The input image.
        threshold (list, optional): The lower and upper intensity thresholds. The third element indicates whether the thresholds are percentiles. Defaults to [1, 99, True].
        autoscale (str, optional): The type of autoscaling to apply. Defaults to None.

    Returns:
        numpy.ndarray: The adjusted image.

    Notes:
        - The function adjusts the intensity values of the image based on the specified thresholds.
        - If `autoscale` is set to "uint8", the image is scaled to the range [0, 255] and converted to uint8.
        - If `autoscale` is set to None, the image is thresholded based on the specified thresholds.
        - If the third element of `threshold` is True, the thresholds are interpreted as percentiles.
    """
    if threshold is None:
        threshold = [1, 99, True]

    if threshold[2]:
        # threshold by percentile
        I_low, I_high = np.percentile(I.reshape(-1), threshold[:2])
    else:
        # threshold by intensity value
        I_low, I_high = threshold[0], threshold[1]
    I  = np.clip(I, I_low, I_high)    
    if auto_scale is not None:
        # scale to 0-1
        I = (I.astype(float) - I_low) / (I_high - I_low)
        if auto_scale == "uint8":
            # convert it to uint8
            I = (I * 255).astype(np.uint8)
    return I


class CLAHE:
    """Reusable CLAHE (Contrast Limited Adaptive Histogram Equalization) wrapper.

    Create once with desired parameters, then apply to multiple images.

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 2.0.
        tile_grid_size (tuple): Size of grid for histogram equalization. Default: (8, 8).

    Example:
        clahe = CLAHE(clip_limit=3.0, tile_grid_size=(8, 8))
        result1 = clahe(image1)
        result2 = clahe(image2)
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        import cv2
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def __call__(self, image):
        """Apply CLAHE to an image.

        Args:
            image (numpy.ndarray): 2D grayscale or 3D volume (applied per-slice along axis 0).

        Returns:
            numpy.ndarray: CLAHE-enhanced image with same shape and dtype.
        """
        if image.ndim == 2:
            return self._apply_2d(image)
        elif image.ndim == 3:
            return np.stack([self._apply_2d(image[z]) for z in range(image.shape[0])])
        else:
            raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

    def _apply_2d(self, image):
        orig_dtype = image.dtype
        if image.dtype != np.uint8:
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image_u8 = ((image.astype(np.float64) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image_u8 = np.zeros_like(image, dtype=np.uint8)
            result_u8 = self.clahe.apply(image_u8)
            if orig_dtype != np.uint8:
                result = (result_u8.astype(np.float64) / 255 * (img_max - img_min) + img_min).astype(orig_dtype)
                return result
            return result_u8
        return self.clahe.apply(image)
