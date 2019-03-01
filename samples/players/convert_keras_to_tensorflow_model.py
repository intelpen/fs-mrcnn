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


tf_saved_model_dir = "D:/model/tf_save_player/"
# Root directory of the mrcnn project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

from samples.players import players
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
PLAYER_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_players_0030.h5")
class InferenceConfig(players.PlayersConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights
model.load_weights(PLAYER_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'sports_ball']


image = np.zeros(shape = (256,256,3), dtype=np.uint8)
images =[image]
molded_images, image_metas, windows = model.mold_inputs(images)
image_shape = molded_images[0].shape
# Anchors
anchors = model.get_anchors(image_shape)
anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

# Run detection so that the keras model gets instantiated
results = model.detect([image], verbose=1)
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
