imagePathFile = './exampleVOC07/pathFileTrain.txt';
datasetName = 'VOC07';
chunksz = 32;
numOfConcepts = 20;
labelsetTrainFile =  './exampleVOC07/labelsetTrain.txt';
cor_file = './exampleVOC07/phi_cor_PascalVOC.txt';
stratificationFile = []; % './exampleVOC07/stratificationLabelset'

% create hdf5Data for the labels
hdf5_caffe_multiLabel( imagePathFile, datasetName, chunksz, numOfConcepts, labelsetTrainFile, cor_file, stratificationFile);

% create hdf5Data for the covariance matrix
hdf5_fromCovarianceMatrix(cor_file, datasetName);