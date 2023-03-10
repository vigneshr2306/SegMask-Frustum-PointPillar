import cv2
import pathlib
import pickle
import time
from collections import defaultdict
import snoop
import numpy as np
from skimage import io as imgio
# import segmentation_models_pytorch as smp
from second.core import box_np_ops
from second.core import preprocess as prep
from second.core.geometry import points_in_convex_polygon_3d_jit
from second.core.point_cloud.bev_ops import points_to_bev
from second.data import kitti_common as kitti
import numba
from numba import jit, types
from PIL import Image
import torch
from second.data.segmentation import segmentation_full, segmentation_det
from second.data.kitti_bbox_sanity_checker import bbox_sanity_check
from yolov7.car_detector import car_detector


def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'gt_boxes', 'voxel_labels',
                'match_indices'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


# @snoop
def prep_pointcloud(input_dict,
                    root_path,
                    voxel_generator,
                    target_assigner,
                    db_sampler=None,
                    max_voxels=20000,
                    class_names=['Car'],
                    remove_outside_points=False,
                    training=True,
                    create_targets=True,
                    shuffle_points=False,
                    reduce_valid_area=False,
                    remove_unknown=False,
                    gt_rotation_noise=[-np.pi / 3, np.pi / 3],
                    gt_loc_noise_std=[1.0, 1.0, 1.0],
                    global_rotation_noise=[-np.pi / 4, np.pi / 4],
                    global_scaling_noise=[0.95, 1.05],
                    global_loc_noise_std=(0.2, 0.2, 0.2),
                    global_random_rot_range=[0.78, 2.35],
                    generate_bev=False,
                    without_reflectivity=False,
                    num_point_features=4,
                    anchor_area_threshold=1,
                    gt_points_drop=0.0,
                    gt_drop_max_keep=10,
                    remove_points_after_sample=True,
                    anchor_cache=None,
                    remove_environment=False,
                    random_crop=False,
                    reference_detections=None,
                    lidar_input=False,
                    unlabeled_db_sampler=None,
                    out_size_factor=2,
                    min_gt_point_dict=None,
                    bev_only=False,
                    use_group_id=False,
                    out_dtype=np.float32,
                    frustum_pp=True,
                    add_points_to_example=False):
    """convert point cloud to voxels, create targets if ground truths 
    exists.
    """
    points = input_dict["points"]
    # print("input dict========", input_dict)
    # exit()
    local_path = "/media/vicky/Office1/kitti/data/"
    img_path = local_path + input_dict["image_path"]
    # print("IMAGE PATH in dict===", input_dict["image_path"])
    # img_path = "/home/vicky/output.png"
    # img_path = local_path + img_path
    # image_path_temp = '/media/vicky/Office1/kitti/data/' + image_path_temp
    # print("IMAGE PATH", image_path_temp)
    # img = cv2.imread(image_path_temp)
    # # cv2.imshow("kitti image", img)
    # temp_bbox = input_dict["bboxes"][0]
    # point1 = (int(temp_bbox[0]), int(temp_bbox[1]))
    # point2 = (int(temp_bbox[2]), int(temp_bbox[3]))
    # # print("point1=====", point1)
    # print("point1=====", point2)
    # print("IMAGE============")
    # print(img)
    # cv2.imshow("img", img)

    # img = cv2.circle(img, point1, 1, (255, 0, 0), 2)
    # img = cv2.rectangle(img, point1, point2, (255, 0, 0), 2)
    # cv2.imshow("img2", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite('/home/vicky/output.png', img)
    # seg_mask = seg_model(torch.from_numpy(img))
    # print(seg_mask)
    # seg_mask2 = seg_mask.numpy()
    # cv2.imwrite('/home/vicky/output_seg.png', seg_mask2)
    # print("done saving")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()
    # print(input_dict)
    if training:
        gt_boxes = input_dict["gt_boxes"]
        gt_names = input_dict["gt_names"]
        difficulty = input_dict["difficulty"]
    image_idx = input_dict["image_idx"]
    group_ids = None
    if use_group_id and "group_ids" in input_dict:
        group_ids = input_dict["group_ids"]
    rect = input_dict["rect"]
    Trv2c = input_dict["Trv2c"]
    P2 = input_dict["P2"]
    unlabeled_training = unlabeled_db_sampler is not None

    if remove_outside_points and not lidar_input:
        image_shape = input_dict["image_shape"]
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,
                                                  image_shape)
    if remove_environment is True and training:
        selected = kitti.keep_arrays_by_name(gt_names, class_names)
        gt_boxes = gt_boxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        if group_ids is not None:
            group_ids = group_ids[selected]
        points = prep.remove_points_outside_boxes(points, gt_boxes)

    if frustum_pp and not training:
        m = np.zeros((points.shape[0], 1))  # add extra feature to point cloud
        points = np.concatenate((points, m), axis=1)
        if reference_detections:
            frame_det_dict = reference_detections.get(image_idx, None)
            if frame_det_dict:
                ref_names = frame_det_dict["names"]
                ref_bboxes = frame_det_dict["bboxes"]
                ref_bboxes = np.array(ref_bboxes)
                ref_boxes_mask = np.array(
                    [n in class_names for n in ref_names], dtype=np.bool_)
                ref_bboxes = ref_bboxes[ref_boxes_mask]
            else:
                ref_bboxes = []
                print("ref_bboxes is none for image_idx:", image_idx)
        else:  # use input detections
            ref_bboxes = input_dict["bboxes"]
            gt_names = input_dict["gt_names"]
            gt_boxes_mask = np.array(
                [n in class_names for n in gt_names], dtype=np.bool_)
            ref_bboxes = ref_bboxes[gt_boxes_mask]

        image_shape = input_dict["image_shape"]
        img_height, img_width = image_shape

        pc_image_coord, img_fov_inds = box_np_ops.get_lidar_in_image_fov(points[:, 0:3], rect, Trv2c, P2,
                                                                         0, 0, img_width, img_height, 2.5)

        # gt_box3d_camera = box_np_ops.box_lidar_to_camera(ref_bboxes, rect, Trv2c)
        # detections = box_np_ops.box3d_to_bbox(gt_box3d_camera, rect, Trv2c, P2)

        mask = np.zeros(points.shape[0], dtype=bool)
        points = get_masked_points(img_path,
                                   points, mask, ref_bboxes, pc_image_coord, img_fov_inds)

        if len(points) == 0:  # Change this hack!!
            points = np.zeros((100, 5))
        elif len(points) < 50:  # Change this hack!!
            temp = points
            points = np.zeros((100, 5))
            points[:len(temp)] = temp
            # points = np.tile(points,(int(50/len(points)) + 1, 1))

        # # detections = bboxes[gt_boxes_mask]

        # for box2d in detections:
        #     # print("here")
        #     xmin, ymin, xmax, ymax = box2d
        #     w = xmax-xmin
        #     h = ymax-ymin
        #     x0 =(xmax+xmin)/2
        #     y0 =(ymax+ymin)/2

        #     box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
        #                    (pc_image_coord[:, 0] >= xmin) & \
        #                    (pc_image_coord[:, 1] < ymax) & \
        #                    (pc_image_coord[:, 1] >= ymin)
        #     box_fov_inds = box_fov_inds & img_fov_inds
        #     box_pc_img_coord = pc_image_coord[box_fov_inds]
        #     prob = gauss_func(x0, y0, h, w, box_pc_img_coord)
        #     # print(prob)
        #     points[box_fov_inds, -1] = prob
        #     mask |= box_fov_inds
        # masked_points = points[mask]
        # if masked_points.shape[0] > 50 : #Change this hack!!
        #     points = masked_points

    # if frustum_pp and not training:
    #     if reference_detections:
    #         frame_det_dict = reference_detections.get(image_idx, None)
    #         if frame_det_dict:
    #             ref_names = frame_det_dict["names"]
    #             ref_bboxes = frame_det_dict["bboxes"]
    #             ref_bboxes = np.array(ref_bboxes)
    #             ref_boxes_mask = np.array(
    #                 [n in class_names for n in ref_names], dtype=np.bool_)
    #             ref_bboxes = ref_bboxes[ref_boxes_mask]
    #         else:
    #             ref_bboxes = []
    #             print("ref_bboxes is none for image_idx:", image_idx)
    #     else: #use input detections
    #         ref_bboxes = input_dict["bboxes"]
    #         gt_names = input_dict["gt_names"]
    #         gt_boxes_mask = np.array([n in class_names for n in gt_names], dtype=np.bool_)
    #         ref_bboxes = ref_bboxes[gt_boxes_mask]

    #     if len(ref_bboxes)!=0 :
    #         C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
    #         frustums = box_np_ops.get_frustum_v2(ref_bboxes, C)
    #         frustums -= T
    #         # frustums = np.linalg.inv(R) @ frustums.T
    #         frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
    #         frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
    #         surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
    #         masks = points_in_convex_polygon_3d_jit(points, surfaces)
    #         masked_points = points[masks.any(-1)]
    #         # print("before", points.shape)
    #         if masked_points.shape[0] > 50 : #Change this hack!!
    #             points = masked_points
    #         # print(points.shape)

    if training:
        # print(gt_names)
        selected = kitti.drop_arrays_by_name(gt_names, ["DontCare"])
        gt_boxes = gt_boxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        if group_ids is not None:
            group_ids = group_ids[selected]

        gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
        if remove_unknown:
            remove_mask = difficulty == -1
            """
            gt_boxes_remove = gt_boxes[remove_mask]
            gt_boxes_remove[:, 3:6] += 0.25
            points = prep.remove_points_in_boxes(points, gt_boxes_remove)
            """
            keep_mask = np.logical_not(remove_mask)
            gt_boxes = gt_boxes[keep_mask]
            gt_names = gt_names[keep_mask]
            difficulty = difficulty[keep_mask]
            if group_ids is not None:
                group_ids = group_ids[keep_mask]
        gt_boxes_mask = np.array(
            [n in class_names for n in gt_names], dtype=np.bool_)

        if db_sampler is not None:
            sampled_dict = db_sampler.sample_all(
                root_path,
                gt_boxes,
                gt_names,
                num_point_features,
                random_crop,
                gt_group_ids=group_ids,
                rect=rect,
                Trv2c=Trv2c,
                P2=P2)

            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                # gt_names = gt_names[gt_boxes_mask].tolist()
                gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                # gt_names += [s["name"] for s in sampled]
                gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0)
                if group_ids is not None:
                    sampled_group_ids = sampled_dict["group_ids"]
                    group_ids = np.concatenate([group_ids, sampled_group_ids])

                if remove_points_after_sample:
                    points = prep.remove_points_in_boxes(
                        points, sampled_gt_boxes)

                points = np.concatenate([sampled_points, points], axis=0)

        # unlabeled_mask = np.zeros((gt_boxes.shape[0], ), dtype=np.bool_)
        if without_reflectivity:
            used_point_axes = list(range(num_point_features))
            used_point_axes.pop(3)
            points = points[:, used_point_axes]
        pc_range = voxel_generator.point_cloud_range
        if bev_only:  # set z and h to limits
            gt_boxes[:, 2] = pc_range[2]
            gt_boxes[:, 5] = pc_range[5] - pc_range[2]
        prep.noise_per_object_v3_(
            gt_boxes,
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_loc_noise_std,
            global_random_rot_range=global_random_rot_range,
            group_ids=group_ids,
            num_try=100)
        # should remove unrelated objects after noise per object
        gt_boxes = gt_boxes[gt_boxes_mask]
        gt_names = gt_names[gt_boxes_mask]

        m = np.zeros((points.shape[0], 1))  # add extra feature to point cloud
        points = np.concatenate((points, m), axis=1)
        # print(points.shape)

        if frustum_pp:
            image_shape = input_dict["image_shape"]
            img_height, img_width = image_shape

            pc_image_coord, img_fov_inds = box_np_ops.get_lidar_in_image_fov(points[:, 0:3], rect, Trv2c, P2,
                                                                             0, 0, img_width, img_height, 2.5)

            gt_box3d_camera = box_np_ops.box_lidar_to_camera(
                gt_boxes, rect, Trv2c)
            detections = box_np_ops.box3d_to_bbox(
                gt_box3d_camera, rect, Trv2c, P2)
            # agument 2d bounding box to get better generalisation
            # The object are placed even outside the camera fov in pp augumentation code so random shift 2d box code cliping removed.

            # image_shape = input_dict["image_shape"]
            # img_height, img_width = image_shape
            # detections = [box_np_ops.random_shift_box2d(bbox, img_height, img_width, 0.1) for bbox in detections]
            # detections = np.atleast_2d(np.array(detections))

            mask = np.zeros(points.shape[0], dtype=bool)

            points = get_masked_points(
                img_path, points, mask, detections, pc_image_coord, img_fov_inds)

            # if len(points) == 0 : #Change this hack!!
            #     points = np.zeros((100,5))
            # elif len(points) < 50 : #Change this hack!!
            #     temp = points
            #     points = np.zeros((100,5))
            #     points[:len(temp)] = temp

            # grid_size = np.array([0, -39.68, 69.12, 39.68])
            # limg = lidar_to_img(points.copy(), grid_size, voxel_size = 0.16, fill= 1)
            # Image.fromarray(limg).show()
            # pts2 = points[masks.any(-1)]
            # limg = lidar_to_img(pts2, grid_size, voxel_size = 0.16, fill= 1)
            # Image.fromarray(limg*255).show()
            # pdb.set_trace()

        # if frustum_pp:
        #     # print("HERE")
        #     gt_box3d_camera = box_np_ops.box_lidar_to_camera(gt_boxes, rect, Trv2c)
        #     detections = box_np_ops.box3d_to_bbox(gt_box3d_camera, rect, Trv2c, P2)

        #     if len(detections) != 0:
        #         C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        #         frustums = box_np_ops.get_frustum_v2(detections, C)
        #         frustums -= T
        #         # frustums = np.linalg.inv(R) @ frustums.T
        #         frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
        #         frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        #         surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
        #         masks = points_in_convex_polygon_3d_jit(points, surfaces)
        #         points = points[masks.any(-1)]

        if group_ids is not None:
            group_ids = group_ids[gt_boxes_mask]
        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

        gt_boxes, points = prep.random_flip(gt_boxes, points)
        gt_boxes, points = prep.global_rotation(
            gt_boxes, points, rotation=global_rotation_noise)
        gt_boxes, points = prep.global_scaling_v2(gt_boxes, points,
                                                  *global_scaling_noise)

        # Global translation
        gt_boxes, points = prep.global_translate(
            gt_boxes, points, global_loc_noise_std)

        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        mask = prep.filter_gt_box_outside_range(gt_boxes, bv_range)
        gt_boxes = gt_boxes[mask]
        gt_classes = gt_classes[mask]
        if group_ids is not None:
            group_ids = group_ids[mask]

        # limit rad to [-pi, pi]
        gt_boxes[:, 6] = box_np_ops.limit_period(
            gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)

    # [0, -40, -3, 70.4, 40, 1]
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    # [352, 400]

    voxels, coordinates, num_points = voxel_generator.generate(
        points, max_voxels)

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64)
    }
    example.update({
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
    })

    if add_points_to_example:
        example.update({
            'points': points
        })
    # if not lidar_input:
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    if anchor_cache is not None:
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]
    else:
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
    example["anchors"] = anchors
    # print("debug", anchors.shape, matched_thresholds.shape)
    # anchors_bv = anchors_bv.reshape([-1, 4])
    anchors_mask = None
    if anchor_area_threshold >= 0:
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
            coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold
        # example['anchors_mask'] = anchors_mask.astype(np.uint8)
        # print(anchors_mask.any(-1))
        example['anchors_mask'] = anchors_mask
    if generate_bev:
        bev_vxsize = voxel_size.copy()
        bev_vxsize[:2] /= 2
        bev_vxsize[2] *= 2
        bev_map = points_to_bev(points, bev_vxsize, pc_range,
                                without_reflectivity)
        example["bev_map"] = bev_map
    if not training:
        return example
    if create_targets:
        targets_dict = target_assigner.assign(
            anchors,
            gt_boxes,
            anchors_mask,
            gt_classes=gt_classes,
            matched_thresholds=matched_thresholds,
            unmatched_thresholds=unmatched_thresholds)
        example.update({
            'labels': targets_dict['labels'],
            'reg_targets': targets_dict['bbox_targets'],
            'reg_weights': targets_dict['bbox_outside_weights'],
        })
    return example


def _read_and_prep_v9(info, root_path, num_point_features, prep_func):
    """read data from KITTI-format infos, then call prep function.
    """
    # velodyne_path = str(pathlib.Path(root_path) / info['velodyne_path'])
    # velodyne_path += '_reduced'
    v_path = pathlib.Path(root_path) / info['velodyne_path']
    v_path = v_path.parent.parent / (
        v_path.parent.stem + "_reduced") / v_path.name

    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_point_features])
    image_idx = info['image_idx']
    rect = info['calib/R0_rect'].astype(np.float32)
    Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = info['calib/P2'].astype(np.float32)

    input_dict = {
        'points': points,
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
        'image_shape': np.array(info["img_shape"], dtype=np.int32),
        'image_idx': image_idx,
        'image_path': info['img_path'],
        # 'pointcloud_num_features': num_point_features,
    }

    if 'annos' in info:
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = kitti.remove_dontcare(annos)
        loc = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        bboxes = annos["bbox"]
        # print(gt_names, len(loc))
        gt_boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        # gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
        difficulty = annos["difficulty"]
        input_dict.update({
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'difficulty': difficulty,
            'bboxes': bboxes,
        })
        if 'group_ids' in annos:
            input_dict['group_ids'] = annos["group_ids"]
    example = prep_func(input_dict=input_dict)
    example["image_idx"] = image_idx
    example["image_shape"] = input_dict["image_shape"]
    if "anchors_mask" in example:
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    return example


@jit(nopython=True)
def lidar_to_img(points, grid_size, voxel_size, fill):
    # pdb.set_trace()
    lidar_data = points[:, :2]  # neglecting the z co-ordinate
    # height_data = points[:, 2] + 1.732
    # pdb.set_trace()
    lidar_data -= np.array([grid_size[0], grid_size[1]])
    lidar_data = lidar_data / voxel_size  # multiplying by the resolution
    lidar_data = np.floor(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    # lidar_data = np.reshape(lidar_data, (-1, 2))
    voxelmap_shape = (grid_size[2:]-grid_size[:2])/voxel_size
    lidar_img = np.zeros((int(voxelmap_shape[0]), int(voxelmap_shape[1])))
    N = lidar_data.shape[0]
    for i in range(N):
        if (0 < lidar_data[i, 0] < lidar_img.shape[0]) and (0 < lidar_data[i, 1] < lidar_img.shape[1]):
            if lidar_img[lidar_data[i, 0], lidar_data[i, 1]] < lidar_data[i, 4]:
                lidar_img[lidar_data[i, 0],
                          lidar_data[i, 1]] = lidar_data[i, 4]

    return lidar_img


# def gauss_func(x0, y0, h, w, xy):
#     prob = np.exp(-((xy[:,0] - x0)**2/(0.5*w**2)) - ((xy[:,1] - y0)**2/(0.5*h**2) ))
#     return prob

# @jit(nopython=True)


# @snoop
def get_masked_points(img_path, points, mask, detections, pc_image_coord, img_fov_inds=None):
    # mask = np.zeros(points.shape[0], dtype=bool)
    image = cv2.imread(img_path)
    detections = car_detector(image)
    # print("img_path", img_path)
    # cv2.imwrite("/home/vicky/out.png", image)
    segmentation_output_full, prob_per_pixel_full = segmentation_full(
        image)
    # detections = bbox_sanity_check(img_path)
    # print("detections", detections)

    # print("detections shape", len(detections))
    for box2d in detections:
        # print("box2d====", box2d)
        box2d = [int(i) for i in box2d]

        xmin, ymin, xmax, ymax = box2d
        # xmin, ymin, xmax, ymax = min(abs(xmin), image.shape[1]), min(
        #     abs(ymin), image.shape[0]), min(abs(xmax), image.shape[1]), min(abs(ymax), image.shape[0])

        # print(xmin, ymin, xmax, ymax)
        # exit()
        w = xmax-xmin
        h = ymax-ymin
        x0 = (xmax+xmin)/2
        y0 = (ymax+ymin)/2

        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        xy = pc_image_coord[box_fov_inds]
        # new_prob = np.exp(-((((xy[:,0] - x0)/w)**4) + ((xy[:,1] - y0)/h)**4 ))
       # print("IMAGE PATH===", img_path)
        # exit()
        # new_prob = segmentation_frustum(img_path, box2d, xy, show=False)
        if xmin < 0 or ymin < 0 or xmax > image.shape[1] or ymax > image.shape[0]:
            print("bad bbox", box2d)
        try:
            # print("negative_bbox\n")
            # print("detections", box2d)
            new_prob = segmentation_det(image, xy,
                                        box2d, segmentation_output_full, prob_per_pixel_full, show=False)
        # except Exception:
        # print(Exception.__name__)
        # new_prob = np.exp(-((xy[:, 0] - x0)**2/(0.5*w**2)) -
        #                   ((xy[:, 1] - y0)**2/(0.5*h**2)))  # original equation
        # # negative_bbox_count += 1
        # print("NEW PROB=====================", new_prob.shape)

        # exit()
            cur_prob = points[box_fov_inds, -1]
            # print("current probability", cur_prob, cur_prob.shape)
            # print("new probability", new_prob, new_prob.shape)
            prob_mask = new_prob < cur_prob
            new_prob[prob_mask] = cur_prob[prob_mask]
        except:
            new_prob = np.exp(-((xy[:, 0] - x0)**2/(0.5*w**2)) -
                              ((xy[:, 1] - y0)**2/(0.5*h**2)))  # original equation
            cur_prob = points[box_fov_inds, -1]
            prob_mask = new_prob < cur_prob
            new_prob[prob_mask] = cur_prob[prob_mask]

        points[box_fov_inds, -1] = new_prob
        mask |= box_fov_inds

    if np.sum(mask) > 50:  # Change this hack!!
        points = points[mask]

    # points = points[mask]

    return points


# @jit(nopython=True)
# def get_points_mask_detections(mask, detections, pc_image_coord, img_fov_inds = None):

#     # for det_id, box2d in enumerate(gt_boxes2d):
#     for box2d in detections:
#         xmin, ymin, xmax, ymax = box2d

#         box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
#                        (pc_image_coord[:, 0] >= xmin) & \
#                        (pc_image_coord[:, 1] < ymax) & \
#                        (pc_image_coord[:, 1] >= ymin)
#         box_fov_inds = box_fov_inds & img_fov_inds
#         mask |= box_fov_inds
#     return mask

    # if frustum_pp:
    #     image_shape = input_dict["image_shape"]
    #     img_height, img_width = image_shape

    #     pc_image_coord, img_fov_inds = box_np_ops.get_lidar_in_image_fov(points[:, 0:3], rect, Trv2c, P2,
    #                                                             0, 0, img_width, img_height, 2.5)

    #     gt_box3d_camera = box_np_ops.box_lidar_to_camera(gt_boxes[gt_boxes_mask], rect, Trv2c)
    #     detections = box_np_ops.box3d_to_bbox(gt_box3d_camera, rect, Trv2c, P2)
    #     # detections = bboxes[gt_boxes_mask]
    #     mask = np.zeros(points.shape[0], dtype=bool)

    #     mask = get_points_mask_detections(mask, detections, pc_image_coord, img_fov_inds)

    #     grid_size = np.array([0, -39.68, 69.12, 39.68])

    #     limg = lidar_to_img(points.copy(), grid_size, voxel_size = 0.16, fill= 1)
    #     Image.fromarray(limg*255).show()

    #     masked_points = points[mask]
    #     limg = lidar_to_img(masked_points, grid_size, voxel_size = 0.16, fill= 1)
    #     Image.fromarray(limg*255).show()


if __name__ == '__main__':
    car_detector("abc")
