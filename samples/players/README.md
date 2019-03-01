# Players Model Resnet50

## Steps to retrain with transfer learning

1. Extract some Images in a folder
2. Create Contours with [via annotator](http://www.robots.ox.ac.uk/~vgg/software/via/) -> via_region_data.json
3. Merge it with data from your previous DB using [ml.mask_rcnn.occlusions.merge_mrcnn_databases.py](https://github.com/footorama/python-prototypes/ml/mask_rcnn/occlusions/extract_jsons_from_FV3_contours.py)
4. Duplicate or split the data in train and val (validation)
5. Retrain model with transfer:
    - in command prompt:
        ```bash
        activate tensorflow19
        cd D:\Dev\python-prototypes\ml\mask_rcnn\samples\players
        python players.py train --dataset=F:\data\ml\mrcnn\dataset_merged --weights=coco
        ```

6. Copy the last keras saved model in logs to MRCNN_MP_Latest

7. Convert keras model to tensorflow model + image_metas and anchors using: 

    `samples/players/convert_keras_to_tensorflow_model.py`

8. Freeze tensoflow model:
    ```
    python "C:\ProgramData\Anaconda3\Lib\site-packages\tensorflow\python\tools\freeze_graph.py" --input_graph="D:\model\tf_save_player\tf_s.pb" --input_checkpoint="D:\model\tf_save_player\\" --output_node_names="mrcnn_detection/Reshape_1,mrcnn_mask/Reshape_1" --output_graph="D:\model\tf_save_player\OcclusionFrozzen.pb"
    ```

9. Copy OcclusionFrozzen, image_metas_players.csv, anchors_players.csv to the C# solution

10. You can test the Frozen model using:
 `player_predict_engine`


## Predict using a frozen model:

You can use the SingletonFrozenPlayerPredictor class from 
`player_predict_engine.py`.

Main function to call:
 
`predictor.predict_single_image(image)`

Option: if you want to use both the mrcnn config and the frozen model use
 
`load_predict_from_frozen_players.py`

## Test a model and estimate its average precission:

``my_demo_players.py``

Obtains:  mAP @ IoU=50:  0.9921666668057442

## Create a video of the detection:

``video_extractor_occlusions``
 
The script plays a video, lets you click with the mouse to a point on the field, apply player_mrcnn at that point and save the images.


## Architecture details:

The architecture of this model is MRCNN + Resnet50 on 256x256 images and is specified in:
 ``players.py``
