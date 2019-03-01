"""
It plays a video, lets you click with the mouse to a point on the field, applies player_mrcnn at that point and save the images
"""

import datetime
from ml.tools.match import Match
import cv2
import numpy as np
import os

ROOT_DIR = os.path.abspath("../../")
import players
import mrcnn.model as modellib
import tensorflow as tf
from mrcnn import visualize
from matplotlib import pyplot as plt


def on_mouse_click(event, x, y, flags, param):
    global mouse_x, mouse_y, frame_num, sequence_number, sequence_frame_start
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x = x
        mouse_y = y
        sequence_number += 1
        sequence_frame_start = frame_num
        os.mkdir("D:\\images_movie\\" + str(sequence_number))
        os.mkdir("D:\\images_movie\\" + str(sequence_number) + "\\mrcnn")
        os.mkdir("D:\\images_movie\\" + str(sequence_number) + "\\img1")

    if event == cv2.EVENT_RBUTTONDOWN:
        frame_num += 50


MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class InferenceConfig(players.PlayersConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3


config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
PLAYER_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon_0030.h5")
model.load_weights(PLAYER_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'sports ball']

global mouse_x
global mouse_y
global frame_num
global sequence_number
global sequence_frame_start
sequence_number = 0
mouse_x = 500
mouse_y = 500

# video_file = "E://data//Videos//2018-10-28 - FC Barcelona - Real Madrid - Plan Large.mp4"
video_file = "E://data//Videos//Liga 1//W5//2017-09-09 - Nice - Monaco.mp4"
# video_file = "E://data//Videos//FIFA//2017-12-23-Test GPS 1 crf21.mp4"
cap = cv2.VideoCapture(video_file)
file_name = "maskout_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())

width = 256
height = 256
fps = 25
vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

cv2.namedWindow('image')
cv2.namedWindow('zoom')
ret, frame_img = cap.read()
cv2.imshow("image", frame_img)
cv2.setMouseCallback("image", on_mouse_click)

plt.interactive(True)
_, ax = plt.subplots(1, figsize=(5, 5))
frame_num = 4450
sequence_frame_start = frame_num

while cap.isOpened():

    frame_curr = cap.get(1) - 1
    if frame_curr + 1 < frame_num:
        cap.set(1, frame_num)

    frame_num = frame_curr
    ret, frame_buff = cap.read()
    frame_img[:, :, :] = frame_buff[:, :, :]

    if mouse_y - int(height / 2) < 0 or mouse_y + int(height / 2) > 1080 or mouse_x - int(
            width / 2) < 0 or mouse_x + int(width / 2) > 1920:
        continue
    img_fix = frame_img[mouse_y - int(height / 2):mouse_y + int(height / 2),
              mouse_x - int(width / 2):mouse_x + int(width / 2)]

    results = model.detect([img_fix], verbose=1)
    r = results[0]
    r_class_ids_filtered = np.array([1 if element > 1 else element for element in r['class_ids']])
    visualize.display_instances(img_fix, r['rois'], r['masks'], r_class_ids_filtered, class_names, r['scores'], ax=ax)
    plt.pause(0.01)

    cv2.imshow("zoom", img_fix)
    cv2.imwrite(
        "D:\\images_movie\\" + str(sequence_number) + "\\img1\\" + str(int(frame_num - sequence_frame_start)) + ".bmp",
        img_fix)
    try:
        plt.savefig("D:\\images_movie\\" + str(sequence_number) + "\\mrcnn\\" + str(
            int(frame_num - sequence_frame_start)) + ".png", bbox_inches='tight', pad_inches=0)
    except Exception:
        print("no mrccn folder")

    plt.cla()
    cv2.waitKey(25)
    cv2.imshow("image", frame_img)
