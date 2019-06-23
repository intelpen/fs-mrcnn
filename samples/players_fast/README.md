# Players fast model

This model implements a custom mask-rcnn which runs at 17ms/image

The model is somewhat a hybrid derived from the standard mask-rcnn with the folowing changes:
 
* image size is reduced to 256*256px
* the resnet backbone is reduced from 101 layers to 5 layers
* the fully connected layer of the FPN is reduced from 1024 to 128 nodes
* the number of classes is reduced to 2

# Training

The best way to train this model is described bellow:
* train first on coco to teach the backbone on a big dataset (only for the first train, then just use the same trained model)
* use transfer learninh to specialize the heads of model(last layers) on footbal players with the small anotation database that we have

## Train on Coco    
To be able to train on coco the model is configured with 81 classes. The config of the model trained on coco is found in players_fast_params in class PlayersFastParamsFromCoco().

The saved keras model should be 68MB and should be saved in logs\ 
Use script `players_fast_train_from_coco.py` for this training. The script will automatically downoad coco 2014 dataset if it is not foudn locally.

## (Re)Train by transfer learning on our football players dataset

To improve speed, a new model is done with only 2 classes(background and players).    The config of this fast  trained extend  PlayersFastParamsFromCoco found  players_fast_params.py.

Use script ``player_fast_transfer_learning_players-db.py`` to retrain and give it as an input parameter the filename of the last model saved in the coco training.(usually PlayersFastParamsFromCoco found in logs)
The saved kers model should have 67MB.
The newly generated .h5 model will be exported as explained in the players project  [readme](..\players\README.md) (using players\convert_keras_to_tensoflow_model.py)

Export the resulting 3 files (the tensorflow frozen, the metas and the anchors) to the C# ML Models nuget.
 
#Benchmark
 
To evaluate the model use `bench_player_fast.py`.

This script:
* execute the prediction on whole dataset
* shows an example of one of the obtained segmentation
* prints the average time/image
* outputs an graph of Precission vs Recall and mAp @ IoU= 80 : 89%

![Graph of Precission vs recall](resources\prec_vs_recall.png)
