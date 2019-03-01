import ml.mask_rcnn.mrcnn.model as modellib
from ml.mask_rcnn.mrcnn import utils

# Compute VOC-style Average Precision
def compute_batch_ap(image_ids, dataset, model_params, model):
    APs = []
    precs = []
    recs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, model_params,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image]*model_params.BATCH_SIZE, verbose=1)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        precs.append(precisions.tolist())
        recs.append(recalls.tolist())
        APs.append(AP)
    return APs,precs,recs

def avg_lst(precs):
    avg_prec = [float(sum(col)) / len(col) for col in zip(*precs)]
    return avg_prec
