import cv2
# img = cv2.imread('/media/vicky/Office1/kitti/data/training/image_2/000000.png')
img = cv2.imread(
    '/home/vicky/Coding/Projects/Visualize-KITTI-Objects-in-Videos/data/KITTI/image_2/0001/000000.png')

# img1 = img[160:358, 756:1097]
cv2.imshow("img", img)
print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
