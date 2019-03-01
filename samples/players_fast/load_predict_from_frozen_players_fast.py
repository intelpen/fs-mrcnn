import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import ml.mask_rcnn.mrcnn.model as modellib
import sys
import skimage.io
from ml.mask_rcnn.mrcnn import visualize
# Import COCO config
ROOT_DIR = os.path.abspath("../../")
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import numpy as np
from ml.mask_rcnn.samples.players_fast.players_fast_params import PlayersFastTransferLearningInferenceParams


class FrozenMrcnnPlayerModel:
    image_width = 80
    image_height = 44
    image_channels = 3
    no_classes = 2
    tf_model_load_path = "D:\\model\\tf_save_player\\"
    tf_model_save_path = "D://model//tf_save_player//"
    data_dir = "D://Data//ML//PlayersWithBall//TrainSet//"
    randomize = True

    def __init__(self):
        self.frozen_graph_filename = self.tf_model_save_path + "OcclusionFrozzenFast.pb"

    def load_frozen_model(self):
        with tf.gfile.GFile(self.frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
            self.graph = graph
        self.sess = tf.Session(graph=graph)
        # self.sess = sess
        return self.sess, self.graph

    def print_tensors(self):
        print([n.name for n in self.graph.as_graph_def().node])


MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = PlayersFastTransferLearningInferenceParams()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

player_model = FrozenMrcnnPlayerModel()
sess, graph = player_model.load_frozen_model()
player_model.print_tensors()


plt.interactive(True)
IMAGE_DIR = "F:\images_clean"
file_names = list(os.walk(IMAGE_DIR))[0][2]
plt.interactive(True)
_, ax = plt.subplots(1, figsize=(16,16))
for file_name in file_names[20:]:
    print(file_name)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
    image = cv2.resize(image,dsize=(128,128), interpolation=cv2.INTER_LINEAR)
    images =[image]*8
    molded_images, image_metas, windows = model.mold_inputs(images)

    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
    # Anchors
    anchors = model.get_anchors(image_shape)
    anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)
#    np.savetxt("anchors_players.csv", anchors[0], delimiter=",")

    images_in = graph.get_tensor_by_name("input_image:0")
    metas_in = graph.get_tensor_by_name("input_image_meta:0")
    anchors_in = graph.get_tensor_by_name("input_anchors:0")
    mrc_detection_out = graph.get_tensor_by_name("mrcnn_detection/packed:0")
    mrc_mask_out = graph.get_tensor_by_name("mrcnn_mask/Reshape_1:0")
    det_out, mask_out = sess.run(fetches=[mrc_detection_out,mrc_mask_out], feed_dict={images_in:molded_images, metas_in:image_metas, anchors_in:anchors})
    print(det_out,mask_out )
 #   np.savetxt("image_metas_players.csv", image_metas, fmt='%i', delimiter=",")
    results = []
    for i, image in enumerate(images):
        final_rois, final_class_ids, final_scores, final_masks = \
            model.unmold_detections(det_out[i], mask_out[i], image.shape, molded_images[i].shape, windows[i])
        results.append({
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        })
    print(results)

    r = results[0]
    class_names = ['BG', 'person', 'sports ball']
    r_class_ids_filtered = np.array([1 if element > 1 else element for element in r['class_ids']])
    visualize.display_instances(image, r['rois'], r['masks'], r_class_ids_filtered, class_names, r['scores'], ax=ax)
    break

plt.pause(0.1)
# Warning: \\ at the end of --input_checkpoint="D:\model\tf_save\\"
# Warning: do not put spaces around tensor names

# python "C:\ProgramData\Anaconda3\Lib\site-packages\tensorflow\python\tools\freeze_graph.py" --input_graph="D:\model\tf_save_player\tf_s.pb"  --input_checkpoint="D:\model\tf_save_player\\" --output_node_names="mrcnn_detection/Reshape_1,mrcnn_mask/Reshape_1" --output_graph="D:\model\tf_save_player\OcclusionFrozzen.pb"

