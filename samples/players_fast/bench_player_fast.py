import os
import random
import matplotlib.pyplot as plt
import numpy as np

import ml.mask_rcnn.mrcnn.model as modellib
from ml.mask_rcnn.mrcnn import visualize
from ml.mask_rcnn.samples.players_fast.players_fast_params import PlayersFastTransferLearningInferenceParams
from ml.mask_rcnn.samples.players_fast.players_fast_dataset import PlayerDataset
import time
from ml.mask_rcnn.benchmarks.benchmark_utils import compute_batch_ap, avg_lst

class MRCNNVizualizer():
    def __init__(self):
        plt.ion()
        _, self.ax = plt.subplots(1, figsize=(8, 8))
        plt.show()
        plt.pause(0.01)


def load_and_display_image(dataset, image_id, ax, display = True, save_filename=None):
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id,
        use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))
    # Display results
    time_start = time.time()
    r = model.detect([image] * config.BATCH_SIZE, verbose=1)[0]
    elapsed_time = time.time() - time_start
    plt.cla()
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")
    if display:
        plt.show()
        plt.pause(0.01)
    if save_filename:
        plt.savefig(save_filename)
    return elapsed_time




if __name__ == "__main__":
    # Local path to trained weights file
    config = PlayersFastTransferLearningInferenceParams()
    config.display()

    weights_path = os.path.abspath("../../mask_rcnn_players_fast_0100.h5")
    model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
    model.load_weights(weights_path, by_name=True)

    players_to_predict_DIR = os.path.join("D:\data\ml\mask_rcnn", "players_fast")
    dataset = PlayerDataset()
    dataset.load_players(players_to_predict_DIR, "val")
    dataset.class_names = ['BG', 'person']
    dataset.prepare()

    # Load and display a random sample
    vizualizer = MRCNNVizualizer()
    image_id = random.choice(dataset.image_ids)
    load_and_display_image(dataset, image_id, vizualizer.ax, display=True)

    # Run object detection and benchmark
    total_time = 0
    for image_id in dataset.image_ids[0:16]:
        total_time += load_and_display_image(dataset, image_id, vizualizer.ax, display=False,
            save_filename=f"D:\images_out\{image_id}.png")



    print(f"Average {total_time/len(dataset.image_ids)}s/image")

    #calculate accuracies

    APs, precs, recs = compute_batch_ap(dataset.image_ids[0:25], dataset, config, model)

    dic_prec_vs_rec = {}

    for prec,rec in zip(precs, recs):
        for p, r in zip(prec,rec):
            if r in dic_prec_vs_rec.keys():
                dic_prec_vs_rec[r].append(p)
            else:
                dic_prec_vs_rec[r] = []

    rec_list = []
    prec_list =[]
    for recall, precisions in dic_prec_vs_rec.items():
        rec_list.append(recall)
        prec_list.append(np.mean(prec_list))


    print(prec_list)
    print(rec_list)
  #  precs = avg_lst(precs)
  #  recs = avg_lst(recs)

    print("mAP @ IoU=50: ", np.mean(APs))
    visualize.plot_precision_recall(np.mean(APs), prec_list, rec_list)
    plt.pause(0.01)
    plt.show()
    plt.ioff()

    r = input()


