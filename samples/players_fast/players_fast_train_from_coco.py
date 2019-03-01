"""
Mask R-CNN
Player Fast Train from Coco

"""

import os

import imgaug

from ml.mask_rcnn.mrcnn import model as modellib
from ml.mask_rcnn.samples.coco.coco import CocoDataset
from ml.mask_rcnn.samples.players_fast.players_fast_params import PlayersFastFromCocoParams

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Directory to save logs and model checkpoints, if not provided through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
#  Training
############################################################

def train_on_coco_db(model, layers):
    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    args.year = "2014"
    args.download = True
    dataset_train = CocoDataset()
    dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
    if args.year in '2014':
        dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    val_type = "val" if args.year in '2017' else "minival"
    dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training all ")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=10,
        layers='all',
        augmentation=augmentation)

    print("Training heads ")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=40,
        layers='heads',
        augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=120,
        layers='4+',
        augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=160,
        layers='all',
        augmentation=augmentation)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect players.')
    parser.add_argument("--command",
        metavar="command", required=False,
        help="'train on coco'", default='train_on_coco')
    parser.add_argument('--dataset', required=False,
        metavar="/path/to/player/dataset/",
        help='Directory of the Player dataset',
        default=os.path.join("D:\data\ml\mask_rcnn", "datasets\coco"))
    parser.add_argument('--layers', required=False,
        help="Layers to train, default all", default="all")
    parser.add_argument('--logs', required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    if args.command != "train_on_coco":
        print("This script only trains on coco, exiting")
        exit(-1)

    # Configurations
    config = PlayersFastFromCocoParams()
    config.display()
    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    # train model
    train_on_coco_db(model, args.layers)
