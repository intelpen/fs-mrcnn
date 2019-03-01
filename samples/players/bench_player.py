import os
import random
import matplotlib.pyplot as plt


import  mrcnn.model as modellib
from  mrcnn import visualize
from ml.mask_rcnn.samples.players_fast import players_fast_transfer_learning_players_db
from ml.mask_rcnn.samples.players_fast.players_fast_dataset import PlayerDataset
import time


# Local path to trained weights file
weights_path = os.path.abspath("../../mask_rcnn_players_fast_0180.h5")
DEFAULT_LOGS_DIR = os.path.join("../../logs")

config = players_fast_transfer_learning_players_db.PlayersFastTransferLearning
config.IMAGES_PER_GPU= 8
config.BATCH_SIZE = 8
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
players_to_predict_DIR = os.path.join("F:\data\ml\mask_rcnn", "datasets\\256px_with_10_occlusions")
model.load_weights(weights_path, by_name=True)

dataset = PlayerDataset()
dataset.load_players(players_to_predict_DIR, "val")
dataset.class_names = ['BG', 'person']
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
plt.show()
plt.pause(0.01)


