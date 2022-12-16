# SegMask for Frustum PointPillar: A Multi-Sensor Approach for 3D Object Detection


**Collaborators: Vignesh Ravikumar, Keshav Bharadwaj Vaidyanathan, Vanshika Jain**

<img src="https://github.com/vigneshr2306/SegMask-Frustum-Pointpillars/images/arch.png" alt="drawing" width="400"/><img src="https://github.com/vigneshr2306/SegMask-Frustum-Pointpillars/images/arch.png" alt="drawing" width="300"/>


## Getting Started

### Code Support

### Install

#### 1. Clone code

```bash
git clone <repo_name>
```

#### 2. Install Python packages

You can use pip or Anaconda package manager to install following packages.
```bash
pip install --upgrade pip
pip install fire tensorboardX shapely pybind11 protobuf scikit-image numba pillow sparsehash
```

Finally, install SparseConvNet. This is not required for PointPillars, but the general SECOND code base expects this
to be correctly configured. 
```bash
pip install spconv
```

Additionally, you may need to install Boost geometry:

```bash
sudo apt-get install libboost-all-dev
```

#### 4. PYTHONPATH

Add SegMask-Frustum-PointPillars/ to your PYTHONPATH.

### Prepare dataset

#### 1. Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

Note: PointPillar's protos use ```KITTI_DATASET_ROOT=/data/sets/kitti_second/```.

#### 2. Create kitti infos:

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

#### 3. Create reduced point cloud:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

#### 4. Create groundtruth-database infos:

```bash
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

#### 5. Modify config file 
The config file is at ```second/configs/pointpillar/xyres_16.proto```
The config file needs to be edited to point to the above datasets:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```
6. Download the YOLOv7 pre-trained model from this link: [Link]{https://drive.google.com/file/d/1X1bU06Zj-9-Tl31LQ-Y1FFdYLeuqeXoz/view?usp=sharing}

7. Download the PSPNet pre-trained model from this link: [Link]{https://drive.google.com/file/d/1IbJnD3yDX9ckMXJK3idwE9sX_fIoDnfD/view?usp=sharing}

### Train

```bash
python second/pytorch/train.py train --config_path=second/configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* If you want to train a new model, make sure "/path/to/model_dir" doesn't exist.
* If "/path/to/model_dir" does exist, training will be resumed from the last checkpoint.
* Training only supports a single GPU. 
* Training uses a batchsize=2 which should fit in memory on most standard GPUs.
* On a single 1080Ti, training xyres_16 requires approximately 20 hours for 160 epochs.


### Evaluate


```bash
python second/pytorch/train.py evaluate --config_path= second/configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* Detection result will saved in model_dir/eval_results/step_xxx.
* The evaluated labels cacn be visualized using Kitti Visualizer [Link]{https://github.com/HengLan/Visualize-KITTI-Objects-in-Videos}

## Results
<img src="https://github.com/vigneshr2306/SegMask-Frustum-Pointpillars/images/img_bbox.png" alt="drawing" width="800"/>
<img src="https://github.com/vigneshr2306/SegMask-Frustum-Pointpillars/images/3d.png" alt="drawing" width="800"/>


## References

[1] A. Paigwar, D. Sierra-Gonzalez, Ö. Erkent and C. Laugier, "Frustum-PointPillars: A Multi-Stage Approach for 3D Object Detection using RGB Camera and LiDAR," 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021, pp. 2926-2933, doi: 10.1109/ICCVW54120.2021.00327.

[2] Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2019). Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 12697-12705).

[3] Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

