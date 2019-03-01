from ml.mask_rcnn.mrcnn.config import Config
import numpy as np
############################################################
#  Configurations : From Coco, With Transfer, Inference
############################################################
class PlayersFastFromCocoParams(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "players_fast"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    BACKBONE = "resnet_custom"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # 80  # Background + player

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    DETECTION_MIN_CONFIDENCE = 0.8
    RPN_ANCHOR_SCALES = (50, 64, 80, 100, 128 )
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 0.8, 1.2]

    # Size of the fully-connected layers in the classification graph
    # FPN_CLASSIF_FC_LAYERS_SIZE = 512
    # TOP_DOWN_PYRAMID_SIZE = 128

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 128

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 64

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 50

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 25

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 25

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 2000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 250

############################################################
#  Transfer Learning Configuration
############################################################
class PlayersFastTransferLearningParams(PlayersFastFromCocoParams):
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # 80  # Background + player
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 2
    DETECTION_MIN_CONFIDENCE = 0.8
    RPN_ANCHOR_SCALES = (64, 100, 128)
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.8,1.1, 1.2]

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Inference Configuration
############################################################
class PlayersFastTransferLearningInferenceParams(PlayersFastTransferLearningParams):
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # 80  # Background + player
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 2
    IMAGES_PER_GPU = 8
    BATCH_SIZE = 8
    DETECTION_MIN_CONFIDENCE = 0.9
