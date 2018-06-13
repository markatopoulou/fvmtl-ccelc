# Guidelines

To train a caffe model using the proposed *FV-MTL with CCE-LC* architecture you need to instantiate the *ResNet-50-fvmtl_ccelc_ext1_2048.prototxt* (it can be found in folder *example_prototxt_files*) with some input files. These files can be created using the Matlab scripts in the current folder.

## Sample example on Pascal-VOC 2007 dataset

Here we provide a sample example that uses the Pascal-VOC 2007 dataset. Simply run the `runme.m` script in order to create all the required input files for the Pascal-VOC 2007 dataset (The PASCAL VOC 2007 dataset can be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)).

The script will return the following files that will be used by the *ResNet-50-fvmtl_ccelc_ext1_2048.prototxt* 

- [ ] VOC07_20_bs32.h5shuffle
- [ ] VOC07_covariance.h5
- [ ] labelsetTrain_shuffle.txt
- [ ] semanticDescrTrain_shuffle.txt (not used)
- [ ] VOC07_20_bs32.h5shuffle_keyframes.txt

## How to create files for your own dataset

You can easily change the parameters of the `runme.m` script in order to create the required input files for the dataset you want to use.

So, if you want to create files for your own dataset what you need is to create:

1. A file similar to the exampleVOC07/*pathFileTrain.txt* that consists of two columns, the first column contains the paths to the keyframes/images that will be used for training. The second column contains the number 1.
2. A file similar to the *exampleVOC07/labelsetTrain.txt* that contains the multi-label ground-truth information of the keyframes in the *exampleVOC07/pathFileTrain.txt*
3. A file similar to the *exampleVOC07/phi_cor_PascalVOC.txt* that contains the correlation matrix as described in our paper.
4. Use stratification (optional): Optionally, you can give an additional file as input to the `hdf5_caffe_multiLabel.m` script, i.e., see the parameter *stratificationFile*, if you want to set the way that the keyframes/images will be shuffled. The *stratificationFile* has the same format with the  *pathFileTrain.txt* with one additional column in the end that specifies in which position this keyframe/image will be placed. For example, in the proposed paper we use a stratification technique that tries to return a good distribution of concepts appearing in every batch. If you don't set this parameter then matlab will randomly shuffle the data, which may result to completely missing concept labels for some of the batches.

After creating the above files you can simply modify the *runme.m* script in order to add your own keyframes/images, ground-truth labels, and correlation matrix files.