import cv2
import sys
import csv
# import tensorflow
import numpy as np
# from second.data.mmsegmentation.mmseg.core.evaluation import get_palette
# from second.data.mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import snoop
# import mmseg
import torch
import torchvision
import pickle

PATH = "/home/vicky/Coding/Projects/Frustum-Pointpillars/second/data/"
#PATH = "/home/jain.van/updated_fp/Frustum-Pointpillars/second/data/"
config_file = PATH + "mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
checkpoint_file = PATH + \
    "mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"

model = init_segmentor(config_file, checkpoint_file, device="cuda:0")


def bbox_extract(img, bbox, segmentation_output_full, prob_per_pixel_full):
    # print(bbox)
    # ymax, ymin, xmax, xmin = bbox
    xmin, ymin, xmax, ymax = bbox
    # xmin, ymin, xmax, ymax = max(xmin, 0), max(
    #     ymin, 0), max(xmax, 0), max(ymax, 0)

    # image = cv2.imread(image_path)
    # print("img shape before ", image.shape)
    # xmin, ymin, xmax, ymax = min(abs(xmin), image.shape[1]), min(
    #     abs(ymin), image.shape[0]), min(abs(xmax), image.shape[1]), min(abs(ymax), image.shape[0])
    # image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    # print("img shape after ", image.shape)
    segmentation_output = segmentation_output_full[ymin:ymax, xmin:xmax]
    prob_per_pixel = prob_per_pixel_full[ymin:ymax, xmin:xmax]
    img = img[ymin:ymax, xmin:xmax]
    # cv2.imwrite("/home/vicky/det1.png", img)
    return segmentation_output, prob_per_pixel


# @snoop
def segmentation_full(img):
    # print("bbox", bbox)
    # img = bbox_extract(image_path, bbox)
    # xmin, ymin, xmax, ymax = bbox
    # if (xmin < 0):
    #     print("negative bbox image", image_path)
    #     exit()

    # print("img shape", img.shape)
    segmentation_output_full, prob_per_pixel_full = inference_segmentor(
        model, img)
    segmentation_output_full = np.array(segmentation_output_full).squeeze()
    # print("seg_full", segmentation_output_full.shape)
    prob_per_pixel_full = (
        prob_per_pixel_full.cpu().squeeze().transpose(0, 1).transpose(1, 2).numpy()
    )
    # print("prob full", prob_per_pixel_full.shape)

    # cv2.imwrite("/home/vicky/out_seg3.png", segmentation_output_full)
    # segmentation_output_full = np.array(segmentation_output_full).squeeze()
    return segmentation_output_full, prob_per_pixel_full


def segmentation_det(img, xy, bbox, segmentation_output_full, prob_per_pixel_full, show=False):
    # print("xy,bbox", xy, bbox)
    xmin, ymin, xmax, ymax = bbox
    # print("bbox", bbox)
    segmentation_output, prob_per_pixel = bbox_extract(img,
                                                       bbox, segmentation_output_full, prob_per_pixel_full)
    # print("after bbox extraction===",
    #       segmentation_output.shape, prob_per_pixel.shape)
    # seg_err_removed = segmentation_output(segmentation_output != 255)
    unique_class, count = np.unique(segmentation_output, return_counts=True)
    # print("segmentation output", segmentation_output)
    # print("unique class", unique_class)
    # print("count", count)
    count = count[unique_class != 255]
    if (len(unique_class) == 1 and unique_class[0] == 255):
        unique_class[0] = 0
    unique_class = unique_class[unique_class != 255]
    # if (len(unique_class) == 0):
    #     unique_class = np.append(unique_class, 0).astype(int)
    # print("unique class", unique_class)
    needed_class = unique_class[count == count.max()]

    # print("unique class", unique_class)
    # print("needed class", needed_class)
    segmentation_output[segmentation_output != needed_class] = 0
    segmentation_output[segmentation_output > 0] = 255
    output = np.empty(
        (segmentation_output.shape[0], segmentation_output.shape[1], 3))
    prob_output = np.empty(
        (segmentation_output.shape[0], segmentation_output.shape[1], 1)
    )
    for i in range(segmentation_output.shape[0]):
        for j in range(segmentation_output.shape[1]):
            if segmentation_output[i][j] == 0:
                output[i][j] = np.array([0, 0, 0])
                prob_output[i][j] = 0
            else:
                output[i][j] = np.array([255, 0, 0])
                prob_output[i][j] = prob_per_pixel[i][j][needed_class]
    # print("prob_output")
    # print(prob_output.shape, prob_output.dtype)
    # l = np.array([prob_output[[120, 130, 140, 150], [120, 120, 120, 120]]])
    # print(xy[:, 0], xy[:, 1], xy.shape)
    # astype(int) done as velo projected to camera coordinates are float and indices can't be float
    cv2.imwrite("/home/vicky/man_seg.png", output)

    l = np.array(
        [prob_output[xy[:, 1].astype(int)-ymin, xy[:, 0].astype(int)-xmin]]).squeeze()
    # print(l)

    if show:
        cv2.imshow("segmentation_output", segmentation_output)
        cv2.imwrite("/home/vicky/segmentation_output1.png",
                    segmentation_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # print("done")

    return l


if __name__ == "__main__":
    # print("here")
    # img_path = '/home/vicky/Coding/Projects/Visualize-KITTI-Objects-in-Videos/data/KITTI/image_2/0001/000000.png'
    img_path = "/home/vicky/man.jpeg"
    bbox = (716, 149, 818, 306)
    xy = np.array([[750, 180], [760, 200]])
    img = cv2.imread(img_path)
    # segmentation_full(img, show=True)
    # segmentation_output_full, prob_per_pixel_full = segmentation_full(img)
    segmentation_output_full, prob_per_pixel_full = segmentation_full(
        img)
    new_prob = segmentation_det(img, xy,
                                bbox, segmentation_output_full, prob_per_pixel_full, show=True)

    # segmentation_full(img_path2, bbox, xy, show=True)
