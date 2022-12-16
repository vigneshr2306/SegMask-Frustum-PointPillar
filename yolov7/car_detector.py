import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)

classes_to_filter = ["car"]
opt = {
    "weights": "/home/vicky/Coding/Projects/Frustum-Pointpillars/yolov7/yolov7-e6e.pt",
    "yaml": "data/coco.yaml",
    "img-size": 640,
    "conf-thres": 0.25,
    "iou-thres": 0.45,
    "device": "0",
    "classes": classes_to_filter,
}


weights, imgsz = opt["weights"], opt["img-size"]
set_logging()
device = select_device(opt["device"])
half = device.type != "cpu"
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def car_detector(img):
    # start = time.time()
    # img0 = img.copy()
    with torch.no_grad():
        if half:
            model.half()

        names = model.module.names if hasattr(model, "module") else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != "cpu":
            model(
                torch.zeros(1, 3, imgsz, imgsz)
                .to(device)
                .type_as(next(model.parameters()))
            )

        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        # Apply NMS
        classes = None
        if opt["classes"]:
            classes = []
            for class_name in opt["classes"]:

                classes.append(names.index(class_name))

        if classes:
            classes = [i for i in range(len(names)) if i in classes]
        # print(pred, classes)
        pred = non_max_suppression(
            pred, opt["conf-thres"], opt["iou-thres"], classes=classes, agnostic=False
        )
        t2 = time_synchronized()
        output = list()
        for i, det in enumerate(pred):
            s = ""
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    # plot_one_box(
                    #     xyxy,
                    #     img,
                    #     label=label,
                    #     color=colors[int(cls)],
                    #     line_thickness=3,
                    # )

                    xyxy = list(
                        [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    )

                    cv2.rectangle(
                        img0,
                        (xyxy[0], xyxy[1]),
                        (xyxy[2], xyxy[3]),
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    output.append(xyxy)
        # end = time.time()
        # print("inf_time:", end - start)

        return output


if __name__ == "__main__":
    source = "/media/vicky/Office1/kitti/data/training/image_2/000006.png"

    # source = "inference/images/horses.jpg"
    img0 = cv2.imread(source)
    box_plots = car_detector(img0)
    print(box_plots, img0.shape)
    cv2.rectangle(
        img0,
        (548, 171),
        (572, 194),
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        img0,
        (505, 168),
        (575, 209),
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        img0,
        (49, 185),
        (227, 247),
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        img0,
        (328, 170),
        (397, 204),
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("val", img0)
    cv2.imwrite("/home/vicky/yolov7.png", img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
