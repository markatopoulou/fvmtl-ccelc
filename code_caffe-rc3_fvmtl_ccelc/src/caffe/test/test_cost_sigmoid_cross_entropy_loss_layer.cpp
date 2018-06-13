#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cost_sigmoid_cross_entropy_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CostSigmoidCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CostSigmoidCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
	    //blob_bottom_correlations_(new Blob<Dtype>(10, 5, 1, 1)),
		// for correlation matrix use this:
		blob_bottom_correlations_(new Blob<Dtype>(10, 25, 1, 1)),
        blob_bottom_targets_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
			// These instantiations are for the backward pass
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Fill the targets vector
	for (int i = 0; i < blob_bottom_targets_->count(); ++i) {
		blob_bottom_targets_->mutable_cpu_data()[i] = caffe_rng_rand() % 2; // to produce 0 or 1 rand() % 2
	}
	blob_bottom_vec_.push_back(blob_bottom_targets_);
		
	// Fill the correlation vector
     FillerParameter correlations_filler_param;
    correlations_filler_param.set_min(-1);
    correlations_filler_param.set_max(1);
    UniformFiller<Dtype> correlations_filler(correlations_filler_param);
    correlations_filler.Fill(blob_bottom_correlations_);
    blob_bottom_vec_.push_back(blob_bottom_correlations_);
	
	blob_top_vec_.push_back(blob_top_loss_);

  }
  virtual ~CostSigmoidCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
	delete blob_bottom_correlations_;
    delete blob_bottom_targets_;
    delete blob_top_loss_;
  }

  Dtype CostSigmoidCrossEntropyLossReference(const int count, const int num, const int numLabels,
                                         const Dtype* input,
                                         const Dtype* target, const Dtype* correlations, bool corVectors, bool costMultiplier) {     
		 
 //cost multiplier
 // Calculate a and b

  float * a = new float[numLabels];
  float * normFactor = new float[numLabels];
  if (costMultiplier){
	  //int numPos = 0;
	  int * numNeg = new int[numLabels];
	  int * numPos = new int[numLabels];

	  // init to zero
	  for (int i = 0; i < numLabels; ++i) {
		  a[i] = 1;
		  numPos[i] = 0;
		  numNeg[i] = 0;
		  normFactor[i] = 1;
	  }

	  int counter = 0;
	  for (int j = 0; j < num; ++j) {
		  for (int i = 0; i < numLabels; ++i) {
			  if (target[counter] == 1) { numPos[i]++; }
			  if (target[counter] == 0) { numNeg[i]++; }
			  counter++;
		  }
	  }
	  for (int i = 0; i < numLabels; ++i) {
		  if (numPos[i] > 0 && numNeg[i] > 0){ a[i] = (float)numNeg[i] / (float)numPos[i]; }
		  if (numPos[i] + numNeg[i] > 0){ normFactor[i] = numPos[i] * a[i] + numNeg[i]; }
	  }
  }
  //end of cost multiplier
		 
	  //calculate the loss
    Dtype totalLoss = 0;
	int counter = 0;
	int beta = 1;
	for (int j = 0; j < num; ++j) {
		int counter_pr_cor = 0; // for every new sample parse the correlation matrix from the start
		for (int i = 0; i < numLabels; ++i) {
		
			const Dtype prediction = 1 / (1 + exp(-input[counter]));
			EXPECT_LE(prediction, 1);
			EXPECT_GE(prediction, 0);
			EXPECT_LE(target[counter], 1);
			EXPECT_GE(target[counter], 0);
			Dtype loss =0;
			loss -= (target[counter] * log(prediction + (target[counter] == Dtype(0))));
			loss -= (1 - target[counter]) * log(1 - prediction + (target[counter] == Dtype(1)));
			//Dtype loss = -(( target[counter] * log(prediction + (target[counter] == Dtype(0))))  -
			//  ((1 - target[counter]) * log(1 - prediction + (target[counter] == Dtype(1)))) ) ; // not sure if this normalization is correct / normFactor[i]
			 
			
			 // correlation constraint
		//if ((~corVectors) || (corVectors && (target[counter] == 1))){
			if ( (corVectors && target[counter] == 1) || corVectors==false){
		  float weight = 0.3;
		  float q_value = input[counter];
		  float cor_sum = 0;
		  int counter_pr = j*numLabels;
		  float cor_threshold = 0.5;
		  int missingLabels = 1;
		  for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
			  float correl_i_t_pr;
			  
			  if (corVectors){
				  correl_i_t_pr = correlations[counter_pr];
				 
			  }else{
				  correl_i_t_pr = correlations[ counter_pr_cor];
			  }
			  if (std::abs(correl_i_t_pr) < cor_threshold){
				  correl_i_t_pr = 0;
				  //missingLabels++;
			  }
			  if (t_pr!=i){
				  if (correl_i_t_pr>=0)
					  cor_sum += std::abs(correl_i_t_pr) * (pow((q_value - input[counter_pr]), 2));
				  else
					  cor_sum += std::abs(correl_i_t_pr) * (pow((beta*q_value + input[counter_pr]), 2));
			  }
			  counter_pr_cor++;
			  counter_pr++;
		  }
		  cor_sum = (weight * cor_sum) / (numLabels - missingLabels);
		 
		  loss += cor_sum;
		  // end of correlation constraint
		}
		// cost multiplier
		if (costMultiplier){
		  float cost_mult = 1;
		  if (target[counter] == 1) { cost_mult = a[i]; }
		  if (target[counter] == 0) { cost_mult = 1; }
		  loss = (cost_mult*loss) / normFactor[i];
		}
		
		totalLoss += loss;
		counter++;
		}
    }
	if (costMultiplier){
		return totalLoss;
	}else{
		return totalLoss/num;
	}


  }

  void TestForward() {
    LayerParameter layer_param;
    const Dtype kLossWeight = 1;
    layer_param.add_loss_weight(kLossWeight);
	//correlation constraint
	const bool costMultiplier = true;
	CostSigmoidCrossEntropyLossParameter* cost_sigmoid_cross_entropy_loss_param = layer_param.mutable_cost_sigmoid_cross_entropy_loss_param();
    cost_sigmoid_cross_entropy_loss_param->set_constraint_weight(0.3);
	cost_sigmoid_cross_entropy_loss_param->set_constraint_cor_threshold(0.5);
    cost_sigmoid_cross_entropy_loss_param->set_constraint(CostSigmoidCrossEntropyLossParameter_Constraint_CorMatrix);
	const bool corVectors = false;
	cost_sigmoid_cross_entropy_loss_param->set_cost_multiplier(costMultiplier);
	
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    
	FillerParameter correlation_filler_param;
    correlation_filler_param.set_min(-1.0);
    correlation_filler_param.set_max(1.0);
    UniformFiller<Dtype> correlations_filler(correlation_filler_param);
	
    Dtype eps = 2e-2;
    for (int i = 0; i < 100; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
	  for (int j = 0; j < this->blob_bottom_targets_->count(); ++j) {
		this->blob_bottom_targets_->mutable_cpu_data()[j] = caffe_rng_rand() % 2; // to produce 0 or 1
	 }
	 // Fill the correlation vector
	 correlations_filler.Fill(this->blob_bottom_correlations_);
	  
      CostSigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      const int count = this->blob_bottom_data_->count();
      const int num = this->blob_bottom_data_->num();
	  const int numLabels = count / num;
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
	  const Dtype* blob_bottom_correlations = this->blob_bottom_correlations_->cpu_data();
      const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data();
	    
      Dtype reference_loss = kLossWeight * CostSigmoidCrossEntropyLossReference(
		  count, num, numLabels, blob_bottom_data, blob_bottom_targets, blob_bottom_correlations, corVectors, costMultiplier);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_correlations_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CostSigmoidCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CostSigmoidCrossEntropyLossLayerTest, TestCostSigmoidCrossEntropyLoss) {
  this->TestForward();
}

TYPED_TEST(CostSigmoidCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 1;
  layer_param.add_loss_weight(kLossWeight);
  //correlation constraint
	const bool costMultiplier = true;
	CostSigmoidCrossEntropyLossParameter* cost_sigmoid_cross_entropy_loss_param = layer_param.mutable_cost_sigmoid_cross_entropy_loss_param();
    cost_sigmoid_cross_entropy_loss_param->set_constraint_weight(0.3);
	cost_sigmoid_cross_entropy_loss_param->set_constraint_cor_threshold(0.5);
    cost_sigmoid_cross_entropy_loss_param->set_constraint(CostSigmoidCrossEntropyLossParameter_Constraint_CorMatrix); //Label_cor, No_const, CorMatrix
	cost_sigmoid_cross_entropy_loss_param->set_cost_multiplier(costMultiplier);
	
  CostSigmoidCrossEntropyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
 //GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);  
  //GradientChecker(const Dtype stepsize, const Dtype threshold,
    //  const unsigned int seed = 1701, const Dtype kink = 0.,
     // const Dtype kink_range = -1)
  
  GradientChecker<Dtype> checker(0.1, 0.3, 1701);  
  
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}


}  // namespace caffe
