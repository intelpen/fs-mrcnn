# To be able to use the model in C# we need to save 3 files:
# 1. the tensorflow graph and weights which we later we freeeze in one .pb file
# 2. image_metas_players.csv file
# 3. anchors_players.csv file
# This script creates them in: D:\model\tf_save_player
import os
import sys
import numpy as np
import keras.backend as K
import tensorflow as tf
import ml.mask_rcnn.mrcnn.model as modellib
from ml.mask_rcnn.samples.players_fast.players_fast_dataset import PlayerDataset


from ml.mask_rcnn.samples.players_fast import players_fast_params


tf_saved_model_dir = "D:/model/tf_save_player/"

# Create model object in inference mode.
config = players_fast_params.PlayersFastTransferLearningInferenceParams()
config.display()
weights_path = os.path.abspath("../../mask_rcnn_players_fast_0100.h5")
model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
model.load_weights(weights_path, by_name=True)
#load dataset
players_to_predict_DIR = os.path.join("D:\data\ml\mask_rcnn", "players_fast")
dataset = PlayerDataset()
dataset.load_players(players_to_predict_DIR, "val")
dataset.class_names = ['BG', 'person']
dataset.prepare()

image_id=0
image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
images = [image]*config.BATCH_SIZE
molded_images, image_metas, windows = model.mold_inputs(images)
image_shape = molded_images[0].shape
# Anchors
anchors = model.get_anchors(image_shape)
anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

# Run detection so that the keras model gets instantiated
results = model.detect(images, verbose=1)
sess = K.get_session()
graph_tf = sess.graph
saver = tf.train.Saver()
tf.train.write_graph(sess.graph_def, tf_saved_model_dir, "tf_s.pb", as_text=True)
saver.save(sess, tf_saved_model_dir)

np.savetxt(os.path.join(tf_saved_model_dir, "anchors_players.csv"), anchors[0], delimiter=",")
np.savetxt(os.path.join(tf_saved_model_dir, "image_metas_players.csv"), image_metas, fmt='%i', delimiter=",")
print("Model + anchors + image_metas saved in {0}".format(tf_saved_model_dir))
print('Now save the frozen model using the following command : ' +\
      r'python "C:\ProgramData\Anaconda3\Lib\site-packages\tensorflow\python\tools\freeze_graph.py" --input_graph="D:\model\tf_save_player\tf_s.pb"  --input_checkpoint="D:\model\tf_save_player\\" --output_node_names="mrcnn_detection/Reshape_1,mrcnn_mask/Reshape_1" --output_graph="D:\model\tf_save_player\OcclusionFrozzen.pb"')

# How to freeze the tensoflow model:
#     ```
#     python "C:\ProgramData\Anaconda3\Lib\site-packages\tensorflow\python\tools\freeze_graph.py" --input_graph="D:\model\tf_save_player\tf_s.pb"  --input_checkpoint="D:\model\tf_save_player\\" --output_node_names="mrcnn_detection/Reshape_1,mrcnn_mask/Reshape_1" --output_graph="D:\model\tf_save_player\OcclusionFrozzen.pb"
#     ```
