import neuroglancer

def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
    # input zyx -> display xyz
    dim = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'],
                                              units='nm',
                                              scales=res)
    return neuroglancer.LocalVolume(data.transpose([2,1,0]),volume_type=tt,dimensions=dim,voxel_offset=oo)
