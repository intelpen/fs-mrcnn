import tensorflow as tf
import numpy as np
import cv2
import os
import skimage.io
from mrcnn import visualize
from matplotlib import pyplot as plt
import mrcnn.model as modellib
from mrcnn import utils
import time

class MrcnnFrozenPlayerPredictor(object):

    def __init__(self, frozen_model_filename, image_metas_filename, anchors_filename):
        batch_size = 1
        self.load_frozen_graph(frozen_model_filename)
        self.image_metas = np.loadtxt(image_metas_filename,  delimiter=",")
        self.image_metas = self.image_metas.reshape(1,-1)
        self.anchors = np.loadtxt(anchors_filename, delimiter=",")
        self.anchors = np.broadcast_to(self.anchors, (batch_size,) + self.anchors.shape)

    def load_frozen_graph(self, frozen_model_filename):
        with tf.gfile.GFile(frozen_model_filename, "rb") as graph_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_file.read())
        # Then, we import the graph_def into a new Graph and return it
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
            self.graph = graph
        self.sess = tf.Session(graph=self.graph)
        self.images_in_tensor = self.graph.get_tensor_by_name("input_image:0")
        self.metas_in_tensor = self.graph.get_tensor_by_name("input_image_meta:0")
        self.anchors_in_tensor = self.graph.get_tensor_by_name("input_anchors:0")
        self.mrc_detection_out_tensor = self.graph.get_tensor_by_name("mrcnn_detection/Reshape_1:0")
        self.mrc_mask_out_tensor = self.graph.get_tensor_by_name("mrcnn_mask/Reshape_1:0")

    def predict_single_image(self, image):
        if image.shape[0] != 256:
            image = cv2.resize(image,dsize=(256,256), interpolation=cv2.INTER_LINEAR)
        det_out, mask_out = self.sess.run(fetches=[self.mrc_detection_out_tensor, self.mrc_mask_out_tensor],
                                     feed_dict={self.images_in_tensor: [image],
                                                self.metas_in_tensor: self.image_metas,
                                                self.anchors_in_tensor: self.anchors})
        return det_out, mask_out

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural network output to a format suitable
        for use in the rest of the application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def convert_detections_for_visualization(self, image, det_out, mask_out):
        images = [image]
        results = []
        for i, image in enumerate(images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(det_out[i], mask_out[i], image.shape, image.shape, window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def scale_and_translate_box(self, original_image_box, mrcnn_predicted_box, model_image_size=(256, 256)):
        x_original, y_original, w_original, h_original = original_image_box
        w_model, h_model = model_image_size
        tl_y_mrcnn_predicted, tl_x_mrcnn_predicted, br_y_mrcnn_predicted, br_x_mrcnn_predicted, _, score = mrcnn_predicted_box
        x_scaled = x_original + tl_x_mrcnn_predicted * w_original
        y_scaled = y_original + tl_y_mrcnn_predicted * h_original
        w_scaled = (br_x_mrcnn_predicted - tl_x_mrcnn_predicted) * w_original
        h_scaled = (br_y_mrcnn_predicted - tl_y_mrcnn_predicted) * h_original
        return [x_scaled, y_scaled, w_scaled, h_scaled]

    def predict_images_list(self, merged_images_list, merged_boxes_list):
        split_boxes_list = []
        scores = []
        for image, original_image_box in zip(merged_images_list, merged_boxes_list):
            split_boxes = self.predict_single_image(image)[0][0]
            for predicted_box in split_boxes:
                if predicted_box[2] == 0:  # we have a list of boxes, and only the first x contain info, the rest are 0,
                    break
                scaled_box = self.scale_and_translate_box(original_image_box, predicted_box, model_image_size=(256, 256))
                split_boxes_list.append(scaled_box)
                score = predicted_box[5]
                scores.append(score)
        return split_boxes_list, scores


def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class SingletonFrozenPlayerPredictor(MrcnnFrozenPlayerPredictor):
    pass

def predict_and_display_image(image, frozen_model_filename, image_metas_filename, anchors_filename):
    predictor = SingletonFrozenPlayerPredictor(frozen_model_filename, image_metas_filename, anchors_filename)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

    # Run object detection
    start_time = time.time()
    det_out, mask_out = predictor.predict_single_image(image)
    end_time = time.time()
    print("Prediction in {0} s".format(end_time-start_time))
    results = predictor.convert_detections_for_visualization(image, det_out, mask_out)
    # Display results
    r = results[0]
    class_names = ["BG", "players"]
    _, ax = plt.subplots(1, figsize=(8, 8))
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax=ax,
                                title="Predictions")


if __name__ == "__main__":

    root_dir = "D:/model/tf_save_player/"
    frozen_model_filename = os.path.join(root_dir, "OcclusionFrozzen.pb")
    image_metas_filename = os.path.join(root_dir, "image_metas_players.csv")
    anchors_filename = os.path.join(root_dir, "anchors_players.csv")
    predictor = SingletonFrozenPlayerPredictor(frozen_model_filename, image_metas_filename, anchors_filename)

    image_dir = "E://imagesd//images_clean//"
    file_name = list(os.walk(image_dir))[0][2][0]
    image = cv2.imread(os.path.join(image_dir, file_name))
    predict_and_display_image(image, frozen_model_filename, image_metas_filename, anchors_filename)

    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to RGB
    for i in range(5):
        predict_and_display_image(new_img, frozen_model_filename, image_metas_filename, anchors_filename)

