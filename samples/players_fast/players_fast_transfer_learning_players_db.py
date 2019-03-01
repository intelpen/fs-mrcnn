"""
Mask R-CNN
Fast transfer on Players Database
"""

import os

from ml.mask_rcnn.mrcnn import model as modellib
from ml.mask_rcnn.samples.players.players import PlayersDataset

from ml.mask_rcnn.samples.players_fast.players_fast_params import PlayersFastTransferLearningParams

from ml.mask_rcnn.samples.coco.coco import CocoDataset
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Directory to save logs and model checkpoints, if not provided  through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
#  Training by transfer learning
############################################################
def train(model, layers="heads"):
    """Train the model."""
    # Training dataset.
    dataset_train = PlayersDataset()
    dataset_train.load_players(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PlayersDataset()
    dataset_val.load_players(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network " + layers)
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=100,
        layers=layers)




############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Transfer-learnin train Mask R-CNN on players database')
    parser.add_argument("--command",
        metavar="command", required=False,
        help="'train'", default='train')
    parser.add_argument('--dataset', required=False,
        metavar="/path/to/player/dataset/",
        help='Directory of the Player dataset',
        default=os.path.join("F:\data", "players_fast"))
    parser.add_argument('--weights', required=False,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file ", default="last")
    parser.add_argument('--layers', required=False,
        help="Path to weights .h5 file or 'coco'", default="heads")  # default="all"

    parser.add_argument('--logs', required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
        metavar="path or URL to image",
        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
        metavar="path or URL to video",
        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    assert args.dataset, "Argument --dataset is required for training"
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config = PlayersFastTransferLearningParams()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "last":
         # Find last trained weights
         weights_path = model.find_last()
    else:
         weights_path = args.weights
    weights_path = "D:\Dev\python-prototypes\ml\mask_rcnn\mask_rcnn_players_fast_0160.h5"
    # Load weights
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    train(model, args.layers)
