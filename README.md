# semantic-segmentation

ENS de Lyon Machine Learning project

## To use

`python3 main.py` to run script\
`python3 main.py -h` for help on optional args

/data-sample/train/\*.ply - any point-cloud files in here will be used for training the model.  If you want to train on more point clouds, just add more files into here.
/data-sample/test/\*.ply - likewise, any point-cloud files in here will be used for testing the model.  If you want to test on more point clouds, just add more files into here.

**Important - the file data-sample/y.csv needs to contain the y values for all the points in the training and testing data point clouds.  The full y.csv is 1.5Gb though, so is too big to be in the GitHub repository.  So you will need to maintain this copy yourself.  The file /mapper.csv shows which points correspond to which point cloud.  The simplest solution is just to have all the points in y.csv, but this takes longer to read in.**

## Visualising output in CloudCompare

The truth-value files and prediction files are generated into /visu and can be read into CloudCompare.  The attribute `class` is the inferred type for the point.

### Todo:
- [x] Make a test/train work with simple model
- [ ] Try out another model from the Keras examples [Point cloud segmentation model](https://keras.io/examples/vision/pointnet_segmentation/)
