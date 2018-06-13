#!/usr/bin/env sh

# exe directory
CAFFE_DIR="TODO" # Set the path to the caffe-rc3_fvmtl_ccelc folder
RUNDIR="TODO" # Set the path where the prototxt, the solver file and the current file exists

# fine tuning / continue training from existing model
WEIGHTS="-weights=TODO/ResNet-50-model.caffemodel" # Set the path where the pre-trained ResNet-50 model exists (it can be downloaded by the official website)

GPU="-gpu 0"

#ext FV_MTL with CCE_LC 2048
SOLVER="-solver=$RUNDIR/solver_2048.prototxt"
LOGFILE="log_deploy_2048"

echo " $CAFFE_DIR_FM/build/tools/caffe train  $SOLVER  $RESUME_FILE $WEIGHTS $GPU >> $RUNDIR/$LOGFILE"
$CAFFE_DIR_FM/build/tools/caffe train  $SOLVER  $RESUME_FILE $WEIGHTS $GPU 2>  $RUNDIR/$LOGFILE

