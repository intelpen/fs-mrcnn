import cv2
import datetime
from ml.tools.match import Match
import cv2
import numpy as np
import os

ROOT_DIR = os.path.abspath("../../")
from samples.players import players
import mrcnn.model as modellib
import tensorflow as tf
from mrcnn import visualize
from matplotlib import pyplot as plt


def create_mot_boxes(filename, rois, scores):
    mot_lines = []
    for roi, score in zip(rois, scores):
        frame_num = int(filename[:-4])
        x, y, w, h, = roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]
        mot_line = [frame_num, -1, y, x, h, w, score * 100, -1, -1, -1]
        mot_lines.append(mot_line)
    return mot_lines


class InferenceConfig(players.PlayersConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
PLAYER_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon_0030.h5")
model.load_weights(PLAYER_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'sports ball']

width = 256
height = 256
fps = 25

root_path = "D:\\images_movie\\4\\"
image_path = root_path + "img1\\"
det_path = root_path + "det\\"

try:
    os.mkdir(det_path)
except Exception:
    print("Except")

cv2.namedWindow('image')
cv2.namedWindow('zoom')

plt.interactive(True)
_, ax = plt.subplots(1, figsize=(5, 5))
frame_num = 4450
sequence_frame_start = frame_num

filenames = [os.listdir(image_path)][0]
bmp_filenames = [filename for filename in filenames if filename[-4:] == ".bmp"]
sorted_filenames = sorted(bmp_filenames, key=lambda x: int(x[:-4]))
print(sorted_filenames)
predicted_mot_lines = []
for filename in sorted_filenames:
    if filename[-4:] != ".bmp":
        continue
    img_fix = cv2.imread(image_path + filename)

    results = model.detect([img_fix], verbose=1)
    r = results[0]
    r_class_ids_filtered = np.array([1 if element > 1 else element for element in r['class_ids']])
    visualize.display_instances(img_fix, r['rois'], r['masks'], r_class_ids_filtered, class_names, r['scores'], ax=ax,
                                level=0.3)
    plt.pause(0.01)

    rois = r["rois"]
    scores = r["scores"]
    mot_lines = create_mot_boxes(filename, rois, scores)
    print(mot_lines)
    for mot_line in mot_lines:
        predicted_mot_lines.append(mot_line)

    cv2.imshow("zoom", img_fix)
    plt.cla()
    cv2.waitKey(25)

np.savetxt(det_path + "det.txt", predicted_mot_lines, fmt='%d,%d,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%d,%d,%d')
