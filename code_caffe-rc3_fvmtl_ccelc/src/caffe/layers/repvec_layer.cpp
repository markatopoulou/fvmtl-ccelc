#include <cfloat>
#include <vector>

#include "caffe/layers/repvec_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RepvecLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RepvecLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	int height_ = bottom[0]->height();
	int neurons_b0 = bottom[0]->channels();
	int neurons_b1 = bottom[1]->channels();
	int width_ = bottom[0]->width();

  for (int i = 1; i < bottom.size(); ++i) {
    //CHECK(bottom[i]->shape() % bottom[0]->shape() == 0);
	  CHECK(neurons_b0 %  bottom[i]->channels() == 0);
  }
  
  int new_neurons = neurons_b0 / neurons_b1;
  top[0]->Reshape(bottom[0]->num(), new_neurons, height_,
	  width_);

}

template <typename Dtype>
void RepvecLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const int top_count = top[0]->count();
  const int top_neurons = top[0]->channels();
  const int top_width = top[0]->width();
 // const int top_channels = top[0]->channels();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_num = top[0]->num();

  const Dtype* bottom_data_b0 = bottom[0]->cpu_data();
  const Dtype* bottom_data_b1 = bottom[1]->cpu_data();
  const int bottom_neurons_b1 = bottom[1]->channels();

  for (int i = 0; i < top_count; ++i) {
	  top_data[i] = 0;
  }
  int b0DataCounter = 0;
  int b1DataCounter = 0;
  int topCounter = 0;
  // The main loop
  for (int tn = 0; tn < top_num; ++tn) { // loop for each training example in the batch
	  float currentSum = 0;
	  for (int th = 0; th < top_neurons; ++th) { //loop for each target label
		  for (int b1h = 0; b1h < bottom_neurons_b1; ++b1h){ // loop for each part of the bottom 2 layer
			  currentSum += bottom_data_b0[b0DataCounter] * bottom_data_b1[b1DataCounter];
			  b0DataCounter++;
			  b1DataCounter++;
		  }
		  top_data[topCounter] = currentSum;
		  topCounter++;
		  b1DataCounter = bottom_neurons_b1 * tn + 1;
	  }
	 
  }

}

template <typename Dtype>
void RepvecLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  int top_num = top[0]->num(); //batch size
  int top_num_labels = count / top_num;
  const int bottom_count_b0 = bottom[0]->count();
  
  const Dtype* top_data = top[0]->cpu_data(); //LS
  const Dtype* top_diff = top[0]->cpu_diff(); //(LS)'



  // gradient method A
  const Dtype* bottom_data_b0 = bottom[0]->cpu_data();
  const Dtype* bottom_data_b1 = bottom[1]->cpu_data();
  const int bottom_neurons_b1 = bottom[1]->channels();
  Dtype* bottom_data2 = new Dtype[count];
  Dtype* bottom_diff = top[0]->mutable_cpu_diff(); // This will be calculated and will be backpropagated

  for (int i = 0; i < count; ++i) {
	  bottom_data2[i] = 0;
  }
  int b0DataCounter = 0;
  int b1DataCounter = 0;
  int topCounter = 0;
  // The main loop
  for (int tn = 0; tn < top_num; ++tn) { // loop for each training example in the batch
	  float currentSum = 0;
	  for (int th = 0; th < top_num_labels; ++th) { //loop for each target label
		  for (int b1h = 0; b1h < bottom_neurons_b1; ++b1h){ // loop for each part of the bottom 2 layer
			  currentSum += bottom_data_b0[b0DataCounter] * bottom_data_b1[b1DataCounter];
			  b0DataCounter++;
			  b1DataCounter++;
		  }
		  bottom_data2[topCounter] = currentSum;
		  topCounter++;
		  b1DataCounter = bottom_neurons_b1 * tn + 1;
	  }

  }
  caffe_mul(count, bottom_data2, top_diff, bottom_diff);

#ifdef _DEBUG
  for (int i = 0; i < 30; i++){
	  cout << " " << top_diff[i] << " ";
  }
  cout << endl;

  for (int i = 0; i < 30; i++){
	  cout << " " << bottom_diff[i] << " ";
  }
  cout << endl;
  for (int i = 0; i < 30; i++){
	  cout << " " << bottom_data2[i] << " ";
  }
  cout << endl;
#endif

  // gradient method b
  //for (int i = 0; i < bottom.size(); ++i) {
  //  if (propagate_down[i]) {
		//const Dtype* bottom_data = bottom[i]->cpu_data(); // first iteration S second iteration L
		//Dtype* bottom_diff = bottom[i]->mutable_cpu_diff(); // This will be calculated and will be backpropagated
		//
		//Dtype* bottom_data2 = new Dtype[count];
		//int bottom_count = bottom[i]->count(); // total size of the vector i.e., bs*number of neurons
		//int bottom_num = bottom[i]->num(); //batch size
		//int num_neurons = bottom_count / bottom_num;
		//int counter = 0;
		//int counter_b = 0;
		//if (bottom[i]->count() < bottom_count_b0){ // L layer
		//	for (int b = 0; b < bottom_num; b++){
		//		float sumWeights = 0;
		//		for (int n = 0; n < num_neurons; n++){
		//			sumWeights += bottom_data[counter_b];
		//			counter_b++;
		//		}
		//		bottom_data2[counter] = sumWeights;
		//		counter++;
		//	}
		//}
		//else{ // S layer
		//	int k_value = num_neurons / top_num_labels;
		//	for (int b = 0; b < bottom_num; b++){
		//		
		//		for (int n = 0; n < top_num_labels; n++){
		//			float sumWeights = 0;
		//			for (int k = 0; k < k_value; k++){
		//				sumWeights += bottom_data[counter_b];
		//				counter_b++;
		//			}
		//			bottom_data2[counter] = sumWeights;
		//			counter++;
		//		}
		//		
		//	}

		//}
		//bottom_data = NULL;
		//bottom_data = bottom_data2;

  //      caffe_div(count, top_data, bottom_data, bottom_diff);
  //      //caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  //  }
  //}
}

// you don't need this because you don't have the corresponding cuda file for this class
//#ifdef CPU_ONLY
//STUB_GPU(EltwiseLayer);
//#endif

INSTANTIATE_CLASS(RepvecLayer);
REGISTER_LAYER_CLASS(Repvec);

}  // namespace caffe
