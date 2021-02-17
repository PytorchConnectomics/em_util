Neuroglancer script (`test_ng.py`)
---
1. Installation
    - for image/segmentation: install our fork of [cloudvolume](https://github.com/donglaiw/cloud-volume)
    - for 3D mesh: install our fork of [igneous](https://github.com/donglaiw/igneous)

2. Steps
    - generate image tiles: `python test_ng.py 0`
    - generate segmentation tiles: `python test_ng.py 1`
    - generate 3D meshes: `python test_ng.py 2`
    - open an [neuroglancer instance](https://neuroglancer-demo.appspot.com/)
    - add a tab and enter: `precomputed://https://SERVER_NAME/FOLDER_NAME/snemi_im/` in the data source for images
    - add a tab and enter: `precomputed://https://SERVER_NAME/FOLDER_NAME/snemi_seg/` in the data source for segmentation

Skeleton script (`test_skel.py`)
---
## Skelton Length Computation
- `python test_skel.py 0 PATH_SKEL_PICKLE_FILE

## ERL Evaluation
- install [funlib.evaluate](https://github.com/funkelab/funlib.evaluate)
```
conda env create -n erl-eval
source activate erl-eval
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
pip install -r requirements.txt
pip install --editable .
```
- `python test_skel.py 1 PATH_SKEL_PICKLE_FILE PATH_SEGMENT_H5_FILE`
