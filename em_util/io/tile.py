import os
from .io import read_image, write_json
import numpy as np


def get_tile_name(pattern, row=None, column=None):
    """
    Generate the tile name based on the pattern and indices.

    Args:
        pattern (str): The pattern for the tile name.
        row (int, optional): The row index. Defaults to None.
        column (int, optional): The column index. Defaults to None.

    Returns:
        str: The generated tile name.
    """    
    
    if "%" in pattern:
        return pattern % (row, column)
    elif "{" in pattern:
        return pattern.format(row, column)
    else:
        return pattern
    
def read_tile_volume(filenames, z0p, z1p, y0p, y1p, x0p, x1p, tile_sz, tile_st=None, tile_dtype=np.uint8, tile_type="image", tile_ratio=1, tile_resize_mode=1, tile_border_padding="reflect", tile_blank="", volume_sz=None, zstep=1):
    """
    Read and assemble a volume from a set of tiled images.

    Args:
        filenames (list): The list of file names or patterns for the tiled images.
        z0p (int): The starting index of the z-axis.
        z1p (int): The ending index of the z-axis.
        y0p (int): The starting index of the y-axis.
        y1p (int): The ending index of the y-axis.
        x0p (int): The starting index of the x-axis.
        x1p (int): The ending index of the x-axis.
        tile_sz (int or list): The size of each tile in pixels. If an integer is provided, the same size is used for both dimensions.
        tile_st (list, optional): The starting index of the tiles. Defaults to [0, 0].
        tile_dtype (numpy.dtype, optional): The data type of the tiles. Defaults to np.uint8.
        tile_type (str, optional): "image" or "seg"
        tile_ratio (float or list, optional): The scaling factor for resizing the tiles. If a float is provided, the same ratio is used for both dimensions. Defaults to 1.
        tile_resize_mode (int, optional): The interpolation mode for resizing the tiles. Defaults to 1.
        tile_seg (bool, optional): Whether the tiles represent segmentation maps. Defaults to False.
        tile_border_padding (str, optional): The padding mode for tiles at the boundary. Defaults to "reflect".
        tile_blank (str, optional): The value or pattern to fill empty tiles. Defaults to "".
        volume_sz (list, optional): The size of the volume in each dimension. Defaults to None.
        zstep (int, optional): The step size for the z-axis. Defaults to 1.

    Returns:
        numpy.ndarray: The assembled volume.

    Notes:        
        - The tiles are specified by file names or patterns in the `filenames` parameter.
        - The volume is constructed by arranging the tiles according to their indices.
        - The size of each tile is specified by the `tile_sz` parameter.
        - The tiles can be resized using the `tile_ratio` parameter.
        - The tiles can be interpolated using the `tile_resize_mode` parameter.
        - The tiles can represent either grayscale images or segmentation maps.
        - The volume can be padded at the boundary using the `tile_bd` parameter.
        - Empty tiles can be filled with a value or pattern using the `tile_blank` parameter.
        - The size of the volume can be specified using the `volume_sz` parameter.
        - The step size for the z-axis can be adjusted using the `zstep` parameter.
    """
    if tile_st is None:
        tile_st = [0, 0]
    if not isinstance(tile_sz, (list,)):
        tile_sz = [tile_sz, tile_sz]
    if not isinstance(tile_ratio, (list,)):
        tile_ratio = [tile_ratio, tile_ratio]
    # [row,col]
    # no padding at the boundary
    # st: starting index 0 or 1

    bd = None
    if volume_sz is not None:
        bd = [
            max(-z0p, 0),
            max(0, z1p - volume_sz[0]),
            max(-y0p, 0),
            max(0, y1p - volume_sz[1]),
            max(-x0p, 0),
            max(0, x1p - volume_sz[2]),
        ]
        z0, y0, x0 = max(z0p, 0), max(y0p, 0), max(x0p, 0)
        z1, y1, x1 = (
            min(z1p, volume_sz[0]),
            min(y1p, volume_sz[1]),
            min(x1p, volume_sz[2]),
        )
    else:
        z0, y0, x0, z1, y1, x1 = z0p, y0p, x0p, z1p, y1p, x1p

    result = np.zeros(
        ((z1 - z0 + zstep - 1) // zstep, y1 - y0, x1 - x0), tile_dtype
    )
    c0 = x0 // tile_sz[1]  # floor
    c1 = (x1 + tile_sz[1] - 1) // tile_sz[1]  # ceil
    r0 = y0 // tile_sz[0]
    r1 = (y1 + tile_sz[0] - 1) // tile_sz[0]
    z1 = min(len(filenames) - 1, z1)
    for i, z in enumerate(range(z0, z1, zstep)):
        pattern = filenames[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                filename = get_tile_name(pattern, row + tile_st[0], column + tile_st[1])
                if os.path.exists(filename):
                    patch = read_image(filename, tile_type, tile_ratio, tile_resize_mode)
                    # exception: last tile may not have the right size
                    psz = patch.shape
                    xp0 = column * tile_sz[1]
                    xp1 = min(xp0 + psz[1], (column + 1) * tile_sz[1])
                    yp0 = row * tile_sz[0]
                    yp1 = min(yp0 + psz[0], (row + 1) * tile_sz[0])
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[i, y0a - y0 : y1a - y0, x0a - x0 : x1a - x0] = (
                            patch[y0a - yp0 : y1a - yp0, x0a - xp0 : x1a - xp0]
                    )                    
                else:
                    print(f"Non-exist: {filename}")
    # blank case
    if tile_blank != "":
        blank_st = 0
        blank_lt = result.shape[0] - 1
        while blank_st <= blank_lt and not np.any(result[blank_st] > 0):
            blank_st += 1
        if blank_st == blank_lt + 1:
            print("!! This volume is all 0 !!")
        else:
            result[:blank_st] = result[blank_st : blank_st + 1]
            while blank_lt >= blank_st and not np.any(result[blank_lt] > 0):
                blank_lt -= 1
            result[blank_lt:] = result[blank_lt - 1 : blank_lt]
            for z in range(blank_st + 1, blank_lt):
                if not np.any(result[z] > 0):
                    result[z] = result[z - 1]

    # boundary case
    if bd is not None and max(bd) > 0:
        result = np.pad(
            result, ((bd[0], bd[1]), (bd[2], bd[3]), (bd[4], bd[5])), tile_border_padding
        )
    return result


def write_tile_info(sz, numT, imN, tsz=1024, tile_st=None, zPad=None, im_id=None, outName=None, st=0, ndim=1, rsz=1, dt="uint8"):
    if tile_st is None:
        tile_st = [0, 0]
    if zPad is None:
        zPad = [0, 0]
    # one tile for each section
    # st: starting index
    if im_id is None:
        im_id = (
            range(zPad[0] + st, st, -1)
            + range(st, sz[0] + st)
            + range(sz[0] - 2 + st, sz[0] - zPad[1] - 2 + st, -1)
        )
    else:  # st=0
        if zPad[0] > 0:
            im_id = [im_id[x] for x in range(zPad[0], 0, -1)] + im_id
        if zPad[1] > 0:
            im_id += [
                im_id[x] for x in range(sz[0] - 2, sz[0] - zPad[1] - 2, -1)
            ]
    sec = [imN(x) for x in im_id]
    out = {
        "image": sec,
        "depth": sz[0] + sum(zPad),
        "height": sz[1],
        "width": sz[2],
        "tile_st": tile_st,
        "dtype": dt,
        "n_columns": numT[1],
        "n_rows": numT[0],
        "tile_size": tsz,
        "ndim": ndim,
        "tile_ratio": rsz,
    }
    if outName is None:
        return out
    else:
        write_json(outName, out)
