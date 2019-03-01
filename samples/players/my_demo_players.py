import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
weights_path = os.path.join(ROOT_DIR, "mask_rcnn_players_0030.h5")
# Download COCO trained weights from Releases if needed

from samples.players import players

config = players.PlayersConfig()
config.NAME = "Player"
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

players_to_predict_DIR = os.path.join("F:\data\ml\mask_rcnn", "datasets\\256px_with_10_occlusions")

model.load_weights(weights_path, by_name=True)

dataset = players.PlayersDataset()
dataset.load_players(players_to_predict_DIR, "val")
dataset.class_names = ['BG', 'person']
dataset.prepare()


print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Load and display random samples

plt.interactive(True)
_, ax = plt.subplots(1, figsize=(8,8))
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))
visualize.display_top_masks(image, gt_mask, gt_class_id, dataset.class_names)

# Run object detection
results = model.detect([image], verbose=1)

# Display results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")

AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,  r['rois'], r['class_ids'], r['scores'], r['masks'])
visualize.plot_precision_recall(AP, precisions, recalls)
visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                        overlaps, dataset.class_names)



# Compute VOC-style Average Precision
def compute_batch_ap(image_ids):
    APs = []
    precs = []
    recs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=1)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        precs.append(precisions.tolist())
        recs.append(recalls.tolist())
        APs.append(AP)
    return APs,precs,recs

def avg_lst(precs):
    avg_prec = [float(sum(col)) / len(col) for col in zip(*precs)]
    return avg_prec


# Pick a set of random images
image_ids = np.random.choice(dataset.image_ids, 200)
APs, precs, recs = compute_batch_ap(image_ids)
precs = avg_lst(precs)
recs = avg_lst(recs)
print("mAP @ IoU=50: ", np.mean(APs))

visualize.plot_precision_recall( np.mean(APs), precs, recs)

plt.show()
plt.pause(0.01)


