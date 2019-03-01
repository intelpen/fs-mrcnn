# Mask R-CNN - Object Detection and Segmentation

This package is based on [Matterport's](https://github.com/matterport/Mask_RCNN) implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) using Python 3, Keras, and TensorFlow.

The main additions are:

* `/samples/players` - contains an implementation of mask_rcnn retrained for players from 2 matches, using Resnet50 and 256*256 images (170MB model)

* `/occlusions` - contains tools to extract occlusion images, create a new player database, freeze the obtained models    
 
The original readme is found in [README_orig.md](README_orig.md)


# Training on Your Own Dataset

You can use [players project](/samples/players/README.md) to retrain on your own database. 

In summary, to train the model on your own dataset you'll need to extend two classes:

`Config`
This class contains the default configuration. Subclass it and modify the attributes you need to change.

`Dataset`
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

See examples in `samples/players/players.py`, `samples/baloon`

## Requirements

Python 3.6, TensorFlow 1.9, Keras 2.0.8 and other common packages listed in `requirements.txt`.

### MS COCO Requirements:

To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)
* [MS COCO Dataset](http://cocodataset.org/#home)
* Download the 5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)
  and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
  subsets. More details in the original [Faster R-CNN implementation](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).


## Installation (if you want to separate the package from the main package - ml)

1. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

