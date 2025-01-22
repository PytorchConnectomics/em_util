import numpy as np

   

def arr_dim_convertor(arr, factor=10000):
    if arr.shape[1] == 3:
        # Nx3 -> N
        return arr[:, 0] * factor * factor + arr[:, 1] * factor + arr[:, 2]
    elif arr.ndim == 1:
        # N -> Nx3
        return np.vstack(
            [arr // (factor * factor), (arr // factor) % factor, arr % factor]
        ).T
   
def arr_to_str(arr):
    """
    Convert an array to a string representation.

    Args:
        arr (numpy.ndarray): The array to be converted.

    Returns:
        str: The string representation of the array.
    """    
    return ",".join([str(int(x)) for x in arr])

def print_arr(arr, num=10):
    """
    Print an array in a formatted manner.

    Args:
        arr (numpy.ndarray): The array to be printed.
        num (int, optional): The number of elements to print per row. Defaults to 10.
    """    
    num_row = (len(arr) + num -1) // num
    for rid in range(num_row):
        print(arr_to_str(arr[rid*num: (rid+1)*num]))

def get_query_in(uu, uf, qid):
    dd = [uf(x) for x in uu]
    out = np.array([uf(x) in dd for x in qid])
    return out


def get_query_count_dict(uu, uf, uc, qid):
    dd = dict(zip([uf(x) for x in uu], uc))
    out = np.array([dd[uf(x)] for x in qid])
    return out

def get_query_count(ui, uc, qid, mm=0):    
    """
    Get the query count for each query ID.

    Args:
        ui (numpy.ndarray): The unique IDs.
        uc (numpy.ndarray): The unique counts.
        qid (numpy.ndarray): The query IDs.
        mm (int, optional): The value to ignore. Defaults to 0.

    Returns:
        numpy.ndarray: The query count for each query ID.

    Notes:
        - The function calculates the query count for each query ID based on the unique IDs and counts.
        - The query count is determined by matching the query IDs with the unique IDs and retrieving the corresponding counts.
        - The value specified by `mm` is used to ignore certain values in the calculation.
    """
    if len(qid) == 0:
        return []
    ui_r = [ui[ui > mm].min(), max(ui.max(), qid.max())]
    rl = mm * np.ones(1 + int(ui_r[1] - ui_r[0]), uc.dtype)
    rl[ui[ui > mm] - ui_r[0]] = uc[ui > mm]

    cc = mm * np.ones(qid.shape, uc.dtype)
    gid = np.logical_and(qid >= ui_r[0], qid <= ui_r[1])
    cc[gid] = rl[qid[gid] - ui_r[0]]
    return cc

def get_kwarg(kwargs, query):
    return kwargs[query] if query in kwargs else None        
