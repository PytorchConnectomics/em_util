import numpy as np
import cv2

def remove_segment_gaps(image, iteration = 4):
    """Clean image using dilation and erosion

     Parameters
    ----------
    image : np.ndarray
        original image
    iteration : int
        number of interation to run erosion and dilation

    Returns
    -------
    image_cleaned : np.ndarray

    References
    ----------
    [1]: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    """
    height = len(image)
    width = len(image[0])
    layers = np.zeros((image.max()+1,height,width))
    kernel = np.ones((3,3),np.uint8)
    image = np.float32(image)
    #split each segment into a layer
    for i in range(height):
        for j in range(width):
            color = image[i][j]
            layers[int(color),i,j] = color
    #clean each layer
    for i in range(len(layers)):
        image=layers[i]
        dilation = cv2.dilate(image,kernel,iterations = iteration)
        erosion = cv2.erode(dilation,kernel,iterations = iteration)
        layers[i] = erosion
    image_cleaned = np.int32(np.max(layers,axis=0))
    return image_cleaned