#include <vector>

#include "caffe/layers/dot_product_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

	template <typename Dtype>
	void DotProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
		CHECK_EQ(bottom[0]->height(), bottom[1]->height());
		CHECK_EQ(bottom[0]->width(), bottom[1]->width());
		top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
	}

	template <typename Dtype>
	void DotProductLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data_a = bottom[0]->cpu_data();
		const Dtype* bottom_data_b = bottom[1]->cpu_data();
		const int num = bottom[0]->num();
		const int dim = bottom[0]->count() / num;
		Dtype* top_data = top[0]->mutable_cpu_data();

		for (int i = 0; i < num; ++i) {
			top_data[i] = caffe_cpu_dot(
				dim,
				bottom_data_a + i*dim,
				bottom_data_b + i*dim);
		}
	}

	template <typename Dtype>
	void DotProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->cpu_diff();

		const Dtype* bottom_data_a = bottom[0]->cpu_data();
		const Dtype* bottom_data_b = bottom[1]->cpu_data();
		const int num = bottom[0]->num();
		const int dim = bottom[0]->count() / num;

		if (propagate_down[0]) {
			Dtype* bottom_diff_a = bottom[0]->mutable_cpu_diff();
			for (int i = 0; i < num; ++i) {
				caffe_cpu_scale(
					dim,
					top_diff[i],
					bottom_data_b + i*dim,
					bottom_diff_a + i*dim);
			}
		}

		if (propagate_down[1]) {
			Dtype* bottom_diff_b = bottom[1]->mutable_cpu_diff();
			for (int i = 0; i < num; ++i) {
				caffe_cpu_scale(
					dim,
					top_diff[i],
					bottom_data_a + i*dim,
					bottom_diff_b + i*dim);
			}
		}
	}

//#ifdef CPU_ONLY
//	STUB_GPU(DotProductLayer);
//#endif

	INSTANTIATE_CLASS(DotProductLayer);
	REGISTER_LAYER_CLASS(DotProduct);
}  // namespace caffe