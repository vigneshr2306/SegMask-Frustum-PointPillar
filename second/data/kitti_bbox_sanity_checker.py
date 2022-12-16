import numpy as np


def bbox_sanity_check(img_path):
    filename = '/media/vicky/Office1/kitti/data/training/label_2/'
    img_idx = img_path[-10:-4]
    print(img_idx)
    filename = filename + img_idx + '.txt'
    f = open(filename, "r")
    lines = f.readlines()
    detections = []
    for line in lines:
        new_line = line.strip()
        arr = new_line.split(' ')
        if arr[0] == 'Car':

            detections1 = [int(eval(i)) for i in arr[4:8]]
            detections.append(detections1)
            # print(new_line)
    # print(detections)
    # detections = np.array(detections)
    return detections


if __name__ == '__main__':
    bbox_sanity_check(
        '/media/vicky/Office1/kitti/data/training/image_2/000004.png')
