#include <vector>

#include "caffe/layers/correlation_constraint_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>        // std::abs
namespace caffe {

template <typename Dtype>
void CorrelationConstraintLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  firstTime = true;
  cor_threshold = this->layer_param_.cost_sigmoid_cross_entropy_loss_param().constraint_cor_threshold();
  LOG(INFO) << " cor_threshold " << cor_threshold; //FM: TODO delete

  //pFile = fopen("bilateral.par", "r");
  //CHECK(pFile) << "The file 'bilateral.par' is not found. Please create it with initial bilateral kernel weights.";
  //for (int i = 0; i < channels_; i++) {
//	  fscanf(pFile, "%lf", &this->blobs_[1]->mutable_cpu_data()[i * channels_ + i]);
  //}
  //fclose(pFile);

}

template <typename Dtype>
void CorrelationConstraintLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (this->layer_param_.correlation_constraint_param().double_neurons()){
	  CHECK_EQ(bottom[0]->count()/2, bottom[1]->count()) <<
		  "CORRELATION_CONSTRAINT layer inputs must have the same count.";
  }
  else{
	  const int count = bottom[0]->count();
	  const int num = bottom[0]->num();
	  int numLabels = count / num;
	  CHECK_EQ((numLabels*numLabels)*num, bottom[1]->count()) <<
		  "CORRELATION_CONSTRAINT layer inputs must have the same count.";
  }
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);

}

template <typename Dtype>
void CorrelationConstraintLayer<Dtype>::Forward_cpu(
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
  const Dtype* target = bottom[2]->cpu_data();
  //const Dtype target = 0; //either 0 or 100
  if (firstTime){
	  correlations = bottom[1]->cpu_data(); // GLOBAL VARIABLE
	  firstTime = false;
  }
 // const Dtype* correlations = bottom[1]->cpu_data(); // GLOBAL VARIABLE

  // double neurons parameter 
  if (this->layer_param_.correlation_constraint_param().double_neurons()){
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

  int counter = 0;
  Dtype loss = 0;
  for (int j = 0; j < num; ++j) {
	  int counter_pr_cor = 0; // for every new sample parse the correlation matrix from the start
	  target_loss = 0;
	  for (int i = 0; i < numLabels; ++i) {
		  double q_value = input_data[counter];
		  long double cor_sum = 0;
		  int counter_pr = j*numLabels;

		  #ifdef _DEBUG
		  double correl_i_t_pr22 = correlations[counter_pr_cor];
		  LOG(INFO) << " " << counter << ": x= " << input_data[counter] << " id= " << input_data[counter_pr] << " q= " << q_value << " cc1= " << (pow((q_value - input_data[counter_pr]), 2)) << " cc2= " << correlations[1] << " cc3= " << std::abs(correl_i_t_pr22); //FM: TODO delete
		  #endif 

		// add constraints in the loss function
		switch (this->layer_param_.correlation_constraint_param().constraint()) {
		case CorrelationConstraintParameter_Constraint_Summation: //sum R - Q
			for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
				double correl_i_t_pr = correlations[counter_pr_cor];
				if (t_pr != i){
					if (std::abs(correl_i_t_pr) < cor_threshold){
						correl_i_t_pr = 0;
					}
					if (correl_i_t_pr >= 0)
						cor_sum += target[counter_pr] * (std::abs(correl_i_t_pr) - (pow((q_value - input_data[counter_pr]), 2)));
					else
						cor_sum += target[counter_pr] * (std::abs(correl_i_t_pr) - (pow((q_value + input_data[counter_pr]), 2)));
					target_loss += std::abs(correl_i_t_pr);
				}
				counter_pr_cor++;
				counter_pr++;
			}
			break;
		case CorrelationConstraintParameter_Constraint_Multiplication:  //use multiplication R*Q
			
			for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
				double correl_i_t_pr = correlations[counter_pr_cor];
				if (t_pr!=i){
					if (correl_i_t_pr>=0)
						cor_sum += target[counter_pr] * (std::abs(correl_i_t_pr) * (pow((q_value - input_data[counter_pr]), 2)));
					else
						cor_sum += target[counter_pr] * (std::abs(correl_i_t_pr) * (pow((q_value + input_data[counter_pr]), 2)));
					target_loss += std::abs(correl_i_t_pr);
				}
				//LOG(INFO) << " " << counter << " corAct= " << abs(correl_i_t_pr); //FM: TODO delete
				counter_pr_cor++;
				counter_pr++;
			}
			
			break;
		}
		//LOG(INFO) << " " << counter <<  " cor= " << cor_sum; //FM: TODO delete
		cor_sum = cor_sum / (numLabels - 1);
		loss += cor_sum;
		counter++;
	  }
	  
  }
  //top[0]->mutable_cpu_data()[0] = loss / normFactor;

  // minimize or maximize
  //target = 1810.7;
  target_loss = 0;
  loss = loss / (num * Dtype(2)); //normalize across batch
 // loss = pow((target_loss - loss), 2);
  top[0]->mutable_cpu_data()[0] = loss;
  LOG(INFO) << "Loss correlation label: " << loss; //FM: TODO delete
}

template <typename Dtype>
void CorrelationConstraintLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
	 // LOG(INFO) << "Target correlation label: " << target; //FM: TODO delete
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
	int numLabels = count / num;
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
	//const Dtype* correlations = bottom[1]->cpu_data(); //GLOBAL VARIABLE
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* target = bottom[2]->cpu_data();

	// double neurons parameter 
	if (this->layer_param_.correlation_constraint_param().double_neurons()){
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

	
	Dtype* constraint_vector = new Dtype[count];
	int counter = 0;
	for (int j = 0; j < num; ++j) {
		int counter_pr_cor = 0; // for every new sample parse the correlation matrix from the start
		for (int i = 0; i < numLabels; ++i) {
			double q_value = sigmoid_output_data[counter];
			long double cor_sum = 0;
			int counter_pr = j*numLabels;
			// add constraints in the loss function
			switch (this->layer_param_.correlation_constraint_param().constraint()) {
			case CorrelationConstraintParameter_Constraint_Summation:
				for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
					double correl_i_t_pr = correlations[counter_pr_cor];
					if (t_pr != i){
						if (std::abs(correl_i_t_pr) < cor_threshold){
							correl_i_t_pr = 0;
						}
						if (correl_i_t_pr >= 0)
							cor_sum += target[counter_pr]*(2 * std::abs(correl_i_t_pr) - (q_value - sigmoid_output_data[counter_pr]));
						else
							cor_sum += target[counter_pr] * (2 * std::abs(correl_i_t_pr) - (sigmoid_output_data[counter_pr] - q_value));
					}
					counter_pr_cor++;
					counter_pr++;
				}
				break;
			case CorrelationConstraintParameter_Constraint_Multiplication:
				
				for (int t_pr = 0; t_pr < numLabels; ++t_pr) {
					double correl_i_t_pr = correlations[counter_pr_cor];
					if (t_pr != i){
						if (correl_i_t_pr >= 0)
							cor_sum += target[counter_pr] * (2 * std::abs(correl_i_t_pr) * (q_value - sigmoid_output_data[counter_pr]));
						else
							cor_sum += target[counter_pr] * (2 * std::abs(correl_i_t_pr) * (sigmoid_output_data[counter_pr] - q_value));
					}
					counter_pr_cor++;
					counter_pr++;
				}
				
				break;
			}
			cor_sum = cor_sum / (numLabels - 1);
			constraint_vector[counter] = cor_sum;
			counter++;
		}
	}


	// minimize or maximize
	Dtype* target_vector = new Dtype[count];
	for (int j = 0; j < count; ++j) {
		target_vector[j] = target_loss;
	}
	caffe_sub(count, constraint_vector, target_vector, bottom_diff);
	// Scale down gradient
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	//caffe_scal(count, loss_weight , bottom_diff);
	caffe_scal(count, loss_weight / num, bottom_diff); // normalize across batch
	LOG(INFO) << "Loss correlation label backward: " << bottom_diff[0];

  }
}

// you don't need this because you don't have the corresponding cuda file for this class
//#ifdef CPU_ONLY
//STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
//#endif

INSTANTIATE_CLASS(CorrelationConstraintLayer);
REGISTER_LAYER_CLASS(CorrelationConstraint);

}  // namespace caffe
