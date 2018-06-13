#include <vector>

#include "caffe/layers/cost_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CostSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  cor_threshold = this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint_cor_threshold();
  LOG(INFO) << " cor_threshold " << cor_threshold; //FM: TODO delete
  firstTime = true;
}

template <typename Dtype>
void CostSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().double_neurons()){
	  CHECK_EQ(bottom[0]->count()/2, bottom[1]->count()) <<
		  "COST_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
	  CHECK_EQ(bottom[1]->count(), bottom[2]->count()) <<
		  "COST_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count. Correlations are wrong.";
  }
  else if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint() == 3){ // correlation matrix
	  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
		  "COST_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
	  CHECK_EQ((bottom[0]->count() / bottom[0]->num()) * (bottom[0]->count() / bottom[0]->num()), bottom[2]->count() / bottom[2]->num()) <<
		  "COST_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count. Correlations are wrong. Give the correlation matrix.";
  }
  else if(this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint() == 1){ // correlation vectors per keyframe
	  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
		  "COST_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
	  CHECK_EQ(bottom[0]->count(), bottom[2]->count()) <<
		  "COST_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count. Correlations are wrong. Give the correlation vectors per keyframe.";
  }
  else{ // no correlation constraint
	CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
		"COST_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
	//CHECK_EQ(bottom[0]->count(), bottom[2]->count()) <<
	//	"COST_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count. Correlations are wrong.";
  }

  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void CostSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// The forward pass computes the sigmoid outputs.
	sigmoid_bottom_vec_[0] = bottom[0];
	sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
	// Compute the loss (negative log likelihood)
	const int count = bottom[0]->count();
	const int num = bottom[0]->num();
	int numLabels = count / num;
	// Stable version of loss computation from input data
	const Dtype* input_data = bottom[0]->cpu_data();
	const Dtype* target = bottom[1]->cpu_data();
	const Dtype* correlations2 = bottom[2]->cpu_data();
	if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint() == 3){
		if (firstTime)
		correlations = bottom[2]->cpu_data();
		correlations2 = NULL;
	}

  // double neurons parameter 
  if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().double_neurons()){
	  numLabels = numLabels / 2.0;
	  Dtype* input_data_double = new Dtype[num*numLabels];
	  int data_counter = 0;
	  for (int i = 0; i < count; i+=2) {
		  input_data_double[data_counter] = exp(input_data[i]) / (exp(input_data[i]) + exp(input_data[i + 1]));
		  data_counter++;
	  }
	  input_data = NULL;
	  input_data = input_data_double;
  }

  //cost multiplier// Calculate a and b
  float * a = new float[numLabels];
  float * normFactor = new float[numLabels];
  if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().cost_multiplier()){
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
		  //a[i] = numNeg[i] / (numPos[i] );
		  //normFactor[i] = (numPos[i] * a[i] ) + (numNeg[i]); 
		  if (numPos[i] + numNeg[i] > 0){ normFactor[i] = numPos[i] * a[i] + numNeg[i]; }
		#ifdef _DEBUG
				  cout << " label " << i << ": a= " << a[i] << " norm=" << normFactor[i] << endl; //FM: TODO delete
		#endif
	  }
  }
  //end of cost multiplier
int beta = 1;
  int counter = 0;
  Dtype totalLoss = 0;
  for (int j = 0; j < num; ++j) {
	  int counter_pr_cor = 0; // for every new sample parse the correlation matrix from the start
	  for (int i = 0; i < numLabels; ++i) {
		#ifdef _DEBUG
		  cout << " " << counter << ": x= " << input_data[counter] << " t= " << target[counter] << endl; //FM: TODO delete
		#endif
		  // don't calculate the loss for examples with missing labels
		  if (target[counter] == 0 || target[counter] == 1) {
	
			  Dtype loss = -(input_data[counter] * (target[counter] - (input_data[counter] >= 0)) -
				   log(1 + exp(input_data[counter] - 2 * input_data[counter] * (input_data[counter] >= 0))) ); //not sure if /normFactor[i] is required
		  

			  switch (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint()) {
			  case CostSigmoidCrossEntropyLossParameter_Constraint_No_const:
				  break;
			  case CostSigmoidCrossEntropyLossParameter_Constraint_Label_cor:
				  // add constraints in the loss function only if the target is positive
				  if ( target[counter] == 1) {
					#ifdef _DEBUG
					  double correl_i_t_pr22 = correlations[j*numLabels];
					  LOG(INFO) << "cc1 " << correl_i_t_pr22  << " cc2= " << abs(correl_i_t_pr22) << " cc3= " << std::abs(correl_i_t_pr22); //FM: TODO delete
					  correl_i_t_pr22 = correlations[j*numLabels + 1];
					  LOG(INFO) << " ccplus1 " << correl_i_t_pr22 ; //FM: TODO delete
					  correl_i_t_pr22 = correlations[j*numLabels + (numLabels-1)];
					  LOG(INFO) << " ccplus345 " << correl_i_t_pr22; //FM: TODO delete
					#endif

					  float weight = this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint_weight();
					  float q_value = input_data[counter];
					  float cor_sum = 0;
					  int counter_pr = j*numLabels;
					  int missingLabels = 1;
					  for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
						  float correl_i_t_pr = correlations2[counter_pr];
						  if (std::abs(correl_i_t_pr) < cor_threshold){
							  correl_i_t_pr = 0;
							  //missingLabels++;
						  }
						  if (t_pr != i){
							  if (correl_i_t_pr >= 0)
								  cor_sum += std::abs(correl_i_t_pr) * (pow((q_value - input_data[counter_pr]), 2));
							  else
								  cor_sum += std::abs(correl_i_t_pr) * (pow((beta*q_value + input_data[counter_pr]), 2));
						  }

						  counter_pr++;
					  }
					  cor_sum = (weight * cor_sum) / (numLabels - missingLabels);

					  loss += cor_sum;
				}// constraint was added for this positive label
				  break;
			  case CostSigmoidCrossEntropyLossParameter_Constraint_CorMatrix:
				  firstTime = false;

				  float weight = this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint_weight();
				  float q_value = input_data[counter];
				  float cor_sum = 0;
				  int counter_pr = j*numLabels;
				  int missingLabels = 1;
				  for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
					  float correl_i_t_pr = correlations[counter_pr_cor];
					  if (std::abs(correl_i_t_pr) < cor_threshold){
						  correl_i_t_pr = 0;
						  //missingLabels++;
					  }
					  if (t_pr != i){
						  if (correl_i_t_pr >= 0)
							  cor_sum += std::abs(correl_i_t_pr) * (pow((q_value - input_data[counter_pr]), 2));
						  else
							  cor_sum += std::abs(correl_i_t_pr) * (pow((beta*q_value + input_data[counter_pr]), 2));
					  }
					  counter_pr_cor++;
					  counter_pr++;
				  }
				  cor_sum = (weight * cor_sum) / (numLabels - missingLabels);

				  loss += cor_sum;

				  break;

			  } // end of constraint
			
			  // cost multiplier
			  if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().cost_multiplier()){
				  float cost_mult = 1;
				  if (target[counter] == 1) { cost_mult = a[i]; }
				  if (target[counter] == 0) { cost_mult = 1; }
				  loss = (cost_mult*loss) / normFactor[i];
			  }
		  
		  totalLoss += loss;
		  }
		  counter++;
	  }
	  
  }

  if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().cost_multiplier()){
	  top[0]->mutable_cpu_data()[0] = totalLoss; //edw den xreiazetai num epeidi eidi exoume diairesei me to normFactor
  } else{
	  top[0]->mutable_cpu_data()[0] = totalLoss / num; //edw den eixa num
  }
  
}

template <typename Dtype>
void CostSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
	int numLabels = count / num;
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
	const Dtype* correlations2 = bottom[2]->cpu_data();
	if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint() == 3){
		correlations2 = NULL;
	}
	
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	// double neurons parameter 
	if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().double_neurons()){
		numLabels = numLabels / 2.0;
		Dtype* input_data_double = new Dtype[num*numLabels];
		int data_counter = 0;
		for (int i = 0; i < count; i += 2) {
			input_data_double[data_counter] = exp(sigmoid_output_data[i]) / (exp(sigmoid_output_data[i]) + exp(sigmoid_output_data[i + 1]));
			data_counter++;
		}
		sigmoid_output_data = NULL;
		sigmoid_output_data = input_data_double;
	}

	//cost multiplier
	// Calculate a and b
	float * a = new float[numLabels];
	float * normFactor = new float[numLabels];
	Dtype* a_vector = new Dtype[count];
	if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().cost_multiplier()){
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
			#ifdef _DEBUG
						cout << " label " << i << ": a= " << a[i] << " norm=" << normFactor[i] << endl; //FM: TODO delete
			#endif
		}
	}
	//end of cost multiplier

	int beta = 1;
	Dtype* constraint_vector = new Dtype[count];
	int counter = 0;
	for (int j = 0; j < num; ++j) {
		int counter_pr_cor = 0; // for every new sample parse the correlation matrix from the start
		for (int i = 0; i < numLabels; ++i) {
			// add constraints in the loss function only if the target is not missing
			if (target[counter] == 1 || target[counter] == 0)
			{
				switch (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint()) {
				case CostSigmoidCrossEntropyLossParameter_Constraint_No_const:
					for (int fot = 0; fot < count; ++fot) {
						constraint_vector[fot] = 0;
					}
					break;
				case CostSigmoidCrossEntropyLossParameter_Constraint_Label_cor:
					// Add this constraint only if the target is positive
					if (target[counter] == 1 ){
						float weight = this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint_weight();
						float q_value = sigmoid_output_data[counter];
						float cor_sum = 0;
						int counter_pr = j*numLabels;
						int missingLabels = 1;
						for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
							float correl_i_t_pr = correlations2[counter_pr];
							if (t_pr != i){
								if (std::abs(correl_i_t_pr) < cor_threshold){
									correl_i_t_pr = 0;
									//missingLabels++;
								}
								if (correl_i_t_pr >= 0)
									cor_sum += std::abs(correl_i_t_pr) * ((q_value - pow(q_value, 2))* (q_value - sigmoid_output_data[counter_pr]));
								else
									cor_sum += std::abs(correl_i_t_pr) * ((beta*(q_value - pow(q_value, 2)))* (beta*q_value + sigmoid_output_data[counter_pr]));

							}

							counter_pr++;
						}
						cor_sum = (2 * weight * cor_sum) / (numLabels - missingLabels);

						//LOG(INFO) << " cor_sum " << cor_sum; //FM: TODO delete
						constraint_vector[counter] = cor_sum;
					}
					else{
						constraint_vector[counter] = 0;
					}
					break;
				case CostSigmoidCrossEntropyLossParameter_Constraint_CorMatrix:
					float weight = this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint_weight();
					float q_value = sigmoid_output_data[counter];
					float cor_sum = 0;
					int counter_pr = j*numLabels;
					int missingLabels = 1;
					for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
						float correl_i_t_pr = correlations[counter_pr_cor];
						if (t_pr != i){
							if (std::abs(correl_i_t_pr) < cor_threshold){
								correl_i_t_pr = 0;
								//missingLabels++;
							}
							if (correl_i_t_pr >= 0)
								cor_sum += std::abs(correl_i_t_pr) * ((q_value - pow(q_value, 2))* (q_value - sigmoid_output_data[counter_pr]));
							else
								cor_sum += std::abs(correl_i_t_pr) * ((beta*(q_value - pow(q_value, 2)))* (beta*q_value + sigmoid_output_data[counter_pr]));

						}
						counter_pr_cor++;
						counter_pr++;
					}
					cor_sum = (2 * weight * cor_sum) / (numLabels - missingLabels);

					//LOG(INFO) << " cor_sum " << cor_sum; //FM: TODO delete
					constraint_vector[counter] = cor_sum;
					break;
				}
				// end of constraint vector

				if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().cost_multiplier()){
					float cost_mult = 1;
					if (target[counter] == 1) { cost_mult = a[i]; }
					if (target[counter] == 0) { cost_mult = 1; }
					a_vector[counter] = (float)cost_mult / (float)normFactor[i]; // not sure if normalization here is correct  / normFactor[i]
				}

			}
			else {
				a_vector[counter] = 0;
				constraint_vector[counter] = 0;
			}
			counter++;
		}
	}

	
    const Dtype loss_weight = top[0]->cpu_diff()[0];
	if (this->layer_param_.cost_sigmoid_cross_entropy_loss_param().cost_multiplier()){
		Dtype* bt = new Dtype[count];
		Dtype* btx = new Dtype[count];
		caffe_sub(count, sigmoid_output_data, target, bt);
		caffe_add(count, bt, constraint_vector, btx);
		caffe_mul(count, btx, a_vector, bottom_diff);
		// Scale down gradient
		caffe_scal(count, loss_weight, bottom_diff); // edw den exei num giati hdh ehei diairethei idi me to normFactor
	} else{
		Dtype* bt = new Dtype[count];
		caffe_sub(count, sigmoid_output_data, target, bt);
		caffe_add(count, bt, constraint_vector, bottom_diff);
		// Scale down gradient
		caffe_scal(count, loss_weight / num, bottom_diff); // edw den eixa num
	}
	

	#ifdef _DEBUG
		for (int i = 0; i < 30; i++){
			cout << " " << target[i] << " ";
		}
		cout << endl;
	#endif
  }
}

// you don't need this because you don't have the corresponding cuda file for this class
//#ifdef CPU_ONLY
//STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
//#endif

INSTANTIATE_CLASS(CostSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CostSigmoidCrossEntropyLoss);

}  // namespace caffe
