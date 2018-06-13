This is a modified version of the Caffe deep learning framework (http://caffe.berkeleyvision.org/) that can be used for working with the *FV-MTL with CCE-LC* method presented in our paper: "*Implicit and Explicit Concept Relations in Deep Neural Networks for Multi-Label Video/Image Annotation*".

This modification is based on the official Caffe source code of [caffe_rc3](https://github.com/BVLC/caffe/releases/tag/rc3). It contains one additional layer, the `cost_sigmoid_cross_entropy_loss_layer`, that implements the *CCE-LC* cost function of our paper. The `caffe.proto` file was also modified accordingly.



# Installation

The caffe_rc3 version extended with the `cost_sigmoid_cross_entropy_loss_layer` and the modified `caffe.proto` file can be found in folder *code_caffe-rc3_fvmtl_ccelc*.

See http://caffe.berkeleyvision.org/installation.html for the latest installation instructions of Caffe.
This version was tested with:
Cuda 8.0
cudnn v4.0
python 2.7
ubuntu 14.04

# Details for the cost sigmoid cross entropy loss layer

This is a modification of the Sigmoid Cross-Entropy Loss Layer (http://caffe.berkeleyvision.org/tutorial/layers/sigmoidcrossentropyloss.html)
The cost sigmoid cross entropy loss layer implements the *CCE_LC* method of our paper.

## Parameters

Parameters (`CostSigmoidCrossEntropyLossParameter cost_sigmoid_cross_entropy_loss_param`)
From ./src/caffe/proto/caffe.proto:

`message CostSigmoidCrossEntropyLossParameter {`
	`enum Constraint {`
	 `Label_cor = 1;`
	 `No_const = 2;`
	 `CorMatrix = 3;`
	`}`
	
`	optional Constraint constraint = 1 [default = No_const];`
	`optional float constraint_weight = 2 [default = 1.0];`
	`optional int32 missing_label = 3 [default = -2]; // the number that indicates a missing label in the ground truth`
	`optional bool double_neurons = 4 [default = false];`
	`optional float constraint_cor_threshold = 5 [default = 1.0];`
	`optional bool cost_multiplier = 6 [default = false];`
`}`

# Sample example

A sample example with the way that a caffe model can be trained using the proposed *FV-MTL with CCE-LC* cost function can be found in the *example_prototxt_files* folder.

Also the Matlab scripts that create the required input files for training a caffe model using the *ResNet-50-fvmtl_ccelc_ext1_2048.prototxt* can be found at the *example_input_files* folder.

For more details see the guidelines in each of these two folders (*example_prototxt_files* and *example_input_files*).

# License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE). The BVLC reference models are released for unrestricted use. 

Please, cite our paper if you use this code:

F. Markatopoulou, V. Mezaris, I. Patras, "*Implicit and Explicit Concept Relations in Deep Neural Networks for Multi-Label Video/Image Annotation*", IEEE Transactions on Circuits and Systems for Video Technology, accepted for publication.

# Acknowledgements

This work was supported by the EU's Horizon 2020 research and innovation programme under grant agreement H2020-687786 InVID, and by Nvidia corporation with the donation of a TitanX GPU.