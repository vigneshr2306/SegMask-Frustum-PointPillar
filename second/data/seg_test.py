import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
seg_model = smp.PSPNet(encoder_name="resnet101",
                       encoder_weights="imagenet", in_channels=3, classes=3)


def prepare_plot(origImage, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    # ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    # ax[1].set_title("Original Mask")
    ax[1].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()
        # find the filename and generate the path to ground truth
        # mask
        # filename = imagePath.split(os.path.sep)[-1]
        # groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
        #                                filename)
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        # gtMask = cv2.imread(groundTruthPath, 0)
        # gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
        #                              config.INPUT_IMAGE_HEIGHT))
        print("size before transpose", image.shape)
        image = np.transpose(image, (2, 0, 1))
        print("size after transpose", image.shape)
        image = np.expand_dims(image, 0)
        print("size after expand dims", image.shape)
        image = torch.from_numpy(image).to("cpu")
        print("size after becoming tensor", image.shape)
        # image = image.unsqueeze(dim=0)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image)
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        predMask = predMask.squeeze(axis=0)
        print("predmask shape", predMask.shape)

        # filter out the weak predictions and convert them to integers
        predMask = (predMask > 0.5) * 255
        predMask = predMask.astype(np.uint8)
        # prepare a plot for visualization
        # prepare_plot(orig, predMask)
        # cv2.imshow("output", predMask)
        # cv2.waitKey(1000000)
        # cv2.destroyAllWindows()
        # print(predMask.shape)
        # predMask = predMask.squeeze(axis=0)
        predMask = predMask.transpose(1, 2, 0)
        cv2.imwrite('/home/vicky/seg_output.png', predMask)


make_predictions(seg_model, '/home/vicky/cat.jpeg')

# img = cv2.imread('/home/vicky/dog.png')
# print("input shape", img.shape)
# rem_32_x, rem_32_y = 32 - (img.shape[0] % 32), 32 - (img.shape[1] % 32)

# img = cv2.resize(img, (img.shape[0]+rem_32_x, img.shape[1]+rem_32_y))
# img = torch.from_numpy(img).float()
# seg_mask = seg_model(img.transpose(2, 0).unsqueeze(dim=0))
# print("seg_mask output shape ", seg_mask.shape)
# # print(img2)
# # cv2.imshow('img', img2)
# seg_mask = softmax(seg_mask, dim=1)
# img2 = seg_mask.detach().squeeze(dim=0).transpose(1, 0).transpose(1, 2).numpy()
# print("img2pixel probability", img2[0][0])
# classes = [np.array([255, 0, 0]), np.array([0, 0, 255])]
# print(img2.shape, img2[0][0])
# output = np.empty((img2.shape[0], img2.shape[1], 3))

# for x in range(len(img2)):
#     for y in range(len(img2[0])):
#         if img2[x][y][0] > img2[x][y][1]:
#             output[x][y] = classes[0]
#         else:
#             output[x][y] = classes[1]
# print("output shape", output.shape)
# cv2.imshow("output", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
