import scipy.io
import scipy.misc
import numpy as np
import cv2
import PIL
import struct
from numpy import expand_dims
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Rectangle

import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def video_detect(f_name):

    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    photo_filename = f_name
    # load and prepare image
    image, image_w, image_h = load_image_pixels(photo_filename, (net_w, net_w))
    # print(image.shape, image_h, image_w)

    # make prediction
    yolos = yolov3.predict(image)
    # summarize the shape of the list of arrays
    # print([a.shape for a in yolos])

    # define the anchors
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    # define the probability threshold for detected objects
    class_threshold = 0.2
    boxes = list()

    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh,  net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        # get the details of the detected objects
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
        # summarize what we found
        for i in range(len(v_boxes)):
            print(v_labels[i], v_scores[i])

        # draw what we found
        draw_boxes(photo_filename, v_boxes, v_labels, v_scores)