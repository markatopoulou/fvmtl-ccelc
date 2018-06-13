#include <vector>

#include "caffe/layers/dot_product_matrix_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

	template <typename Dtype>
	void DotProductMatrixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		CHECK_EQ(bottom[0]->channels(), 1);
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
		CHECK_EQ(bottom[0]->width(), bottom[1]->height());
		//CHECK_EQ(bottom[0]->width(), bottom[1]->width());
		top[0]->Reshape(bottom[0]->num(), 1, 1, bottom[1]->width());
		
		// Figure out the dimensions
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		// Dimensions starting from "axis" are "flattened" into a single
		// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
		// and axis == 1, N inner products with dimension CHW are performed.
		K_ = bottom[0]->count(axis);
		// The first "axis" dimensions are independent inner products; the total
		// number of these is M_, the product over these dimensions.
		M_ = bottom[0]->count(0, axis);
		N_ = bottom[1]->width();
		transpose_ = false;
	}

	template <typename Dtype>
	void DotProductMatrixLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data_a = bottom[0]->cpu_data(); //vector
		const Dtype* bottom_data_b = bottom[1]->cpu_data(); // matrix	
		Dtype* top_data = top[0]->mutable_cpu_data();

		for (int i = 0; i < M_; ++i) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
				1, N_, K_, (Dtype)1.,
				bottom_data_a + (i*K_), bottom_data_b + (i*K_*N_), (Dtype)0., top_data+(i*N_));
		}

		#ifdef _DEBUG
		const int www = top[0]->width();
		const int hhh = top[0]->height();
		for (int i = 0; i < 30; i++){
			cout << " " << top_data[i] << " ";
		}
		cout << endl;
		
		#endif
	}

	template <typename Dtype>
	void DotProductMatrixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
		const Dtype* top_diff = top[0]->cpu_diff();

		const Dtype* bottom_data_a = bottom[0]->cpu_data(); //vector
		const Dtype* bottom_data_b = bottom[1]->cpu_data(); // matrix

		#ifdef _DEBUG
		const int www = top[0]->width();
		const int hhh = top[0]->height();
		for (int i = 0; i < 30; i++){
			cout << " " << top_diff[i] << " ";
		}
		cout << endl;

		#endif

		if (propagate_down[0]) {
			Dtype* bottom_diff_a = bottom[0]->mutable_cpu_diff();
			// with respect to bottom 0 (vector)
			for (int i = 0; i < M_; ++i) {
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					1, K_, N_,
					(Dtype)1., top_diff + (i*N_), bottom_data_b + (i*K_*N_),
					(Dtype)1., bottom_diff_a + (i*K_));
			}
		}


		

		if (propagate_down[1]) {
			Dtype* bottom_diff_b = bottom[1]->mutable_cpu_diff();
			// with respect to bottom 1 (matrix)
			for (int i = 0; i < M_; ++i) {
				caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
					K_, N_, 1,
					(Dtype)1., bottom_data_a + (i*K_), top_diff + (i*N_), 
					(Dtype)1., bottom_diff_b + (i*K_*N_));
			}
		}
	}

	//#ifdef CPU_ONLY
	//	STUB_GPU(DotProductLayer);
	//#endif

	INSTANTIATE_CLASS(DotProductMatrixLayer);
	REGISTER_LAYER_CLASS(DotProductMatrix);
}  // namespace caffe