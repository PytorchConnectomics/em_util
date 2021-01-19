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

