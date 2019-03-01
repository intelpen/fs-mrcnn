import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
from ml.mask_rcnn.mrcnn import utils
import  ml.mask_rcnn.mrcnn.model as modellib
from  ml.mask_rcnn.mrcnn import visualize
from ml.mask_rcnn.samples.players import players
import time




# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
weights_path = os.path.join(ROOT_DIR, "logs/players_fast20190213T1737/mask_rcnn_players_fast_0150.h5")
# Download COCO trained weights from Releases if needed


config = players_fast.PlayersFast()
config.NAME = "Players Fast"
config.IMAGES_PER_GPU= 8
config.BATCH_SIZE = 8
config.NUM_CLASSES = 1 +80
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
players_to_predict_DIR = os.path.join("F:\data\ml\mask_rcnn", "datasets\\256px_with_10_occlusions - Copy")
model.load_weights(weights_path, by_name=True)

dataset = players_fast.PlayerDataset()
dataset.load_players(players_to_predict_DIR, "val")
#dataset.class_names = ['BG', 'person']
dataset.prepare()


print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Load and display random samples

plt.ion()

_, ax = plt.subplots(1, figsize=(8,8))
ax.plot([1,2],[1,2])
plt.pause(0.01)
plt.show()
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))
#visualize.display_top_masks(image, gt_mask, gt_class_id, dataset.class_names)
# Display results
results = model.detect([image]*config.BATCH_SIZE, verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")


# Run object detection
total_time = 0
for image_id in dataset.image_ids:
    print(image_id)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id,
        use_mini_mask=False)
    time_start = time.time()
    results = model.detect([image]*config.BATCH_SIZE, verbose=0)
    elapsed_time = time.time() - time_start
    total_time += elapsed_time
    print(f"Total time/image = {total_time/image_id}")
    # Display results
    r = results[0]
    plt.cla()
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax,
         title="Predictions")
    plt.savefig(f"D:/images_small/{image_id}.png")

print(f"Elapsed {total_time} for len{dataset.image_ids} images" )
#
# AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,  r['rois'], r['class_ids'], r['scores'], r['masks'])
# visualize.plot_precision_recall(AP, precisions, recalls)
# visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
#                         overlaps, dataset.class_names)
#
# plt.show()
#
#
# # Pick a set of random images
# image_ids = np.random.choice(dataset.image_ids, 200)
# APs, precs, recs = compute_batch_ap(image_ids)
# precs = avg_lst(precs)
# recs = avg_lst(recs)
# print("mAP @ IoU=50: ", np.mean(APs))
#
# visualize.plot_precision_recall( np.mean(APs), precs, recs)

plt.show()
plt.pause(0.01)


