# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:14:29 2018

@author: adrian
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

#import coco
import utils
import model as modellib
import visualize
import time
import cv2
#%%

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = "D://images_clean//"
#%%
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #BATCH_SIZE = 5
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    MAX_GT_INSTANCES = 10
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 32
    IMAGE_SHAPE = [128,128,3]
    MINI_MASK_SHAPE =  (32, 32)
    RPN_ANCHOR_SCALES =  ( 64,128, 256, 378, 512)
    DETECTION_MAX_INSTANCES = 15
    RPN_TRAIN_ANCHORS_PER_IMAGE = 60
    POST_NMS_ROIS_INFERENCE = 1000
    VALIDATION_STEPS = 25
    RPN_ANCHOR_STRIDE =1
config = InferenceConfig()
config.display()
#%%
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
#%%
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'sports ball']
#%%
# Load a random image from the images folder

#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
filenames = os.listdir(IMAGE_DIR)
plt.interactive(True)
_, ax = plt.subplots(1, figsize=(16,16))
for file_name in filenames:
    big_image = np.zeros((128,128,3), dtype= int)
    #big_image[:,:,1] += 255
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
    image = cv2.imread(os.path.join(IMAGE_DIR, file_name))
    image = image[:, :, (2, 1, 0)]
    zoom = min(big_image.shape[0]/image.shape[0],big_image.shape[1]/image.shape[1])
    image = cv2.resize(image,dsize=(int(image.shape[1]*zoom),int(image.shape[0]*zoom)), interpolation=cv2.INTER_LINEAR)
    x,y,ch = [int(el/2) for el in image.shape]
    xc,yc,ch = [int(el/2) for el in big_image.shape]
    # plt.imshow(image)
    # plt.show()
    # plt.pause(0.01)
    big_image += image[0, 0, :]
    big_image[xc-x:xc+x, yc-y:yc+y] = image[0:2*x, 0:2*y]
    # plt.imshow(big_image)
    # plt.show()
    # plt.pause(0.01)
    start_time = time.time()
    results = model.detect([big_image], verbose=1)
    end_time = time.time()
    print(end_time-start_time)
    r = results[0]
    print(r['class_ids'])
    r_class_ids_filtered = np.array([1 if element>1 else element for element in r['class_ids'] ])
    visualize.display_instances(big_image, r['rois'], r['masks'],r_class_ids_filtered, class_names, r['scores'], ax=ax)
    plt.savefig("D://images_out//" + file_name[:-4]+".png");
    plt.pause(0.2)
    plt.cla()
    # plt.pause(0.1)
# Visualize results

