#include <cmath>
#include <vector>

#include "caffe/layers/sigmoidNeg_layer.hpp"

namespace caffe {

template <typename Dtype>

inline Dtype sigmoidNeg(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void SigmoidNegLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype* target = bottom[1]->cpu_data();
  for (int i = 0; i < count; ++i) {
	  if (target[i] >= 0){
		  top_data[i] = sigmoidNeg(bottom_data[i]);
	  }
	  else{
		#ifdef _DEBUG
		  cout << " s" << i << ": " << bottom_data[i] << " " << target[i]; //FM: TODO delete
		  cout << " " << i << ": " << (-1 * bottom_data[i]) << " " << (-1 * target[i]); //FM: TODO delete
		  Dtype lossPrev = sigmoidNeg(bottom_data[i]);
		#endif
		  top_data[i] = sigmoidNeg((-1*bottom_data[i]));
	  }
   
  }
}

template <typename Dtype>
void SigmoidNegLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidNegLayer);
#endif

INSTANTIATE_CLASS(SigmoidNegLayer);


}  // namespace caffe
