#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_mtl_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/calculate_SVD.hpp"
//#include <Eigen/Eigen>
//using namespace Eigen;

namespace caffe {

template <typename Dtype>

void InnerProductMtlLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_mtl_param().num_output();
  bias_term_ = this->layer_param_.inner_product_mtl_param().bias_term();
  transpose_ = this->layer_param_.inner_product_mtl_param().transpose();
  N_ = num_output;
  k_value_ = this->layer_param_.inner_product_mtl_param().k_value();
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_mtl_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_mtl_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_mtl_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductMtlLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_mtl_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductMtlLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductMtlLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	if (this->param_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		// Gradient with respect to weight
		#ifdef _DEBUG
			const Dtype* weight = this->blobs_[0]->cpu_data();
			const Dtype* weight_diff_mut = this->blobs_[0]->mutable_cpu_diff();
			const Dtype* data_diff = bottom[0]->cpu_diff();
			const Dtype* weight_diff = this->blobs_[0]->cpu_diff();
			for (int i = 0; i < 10; i++){
				cout << " " << weight[i] << " ";
			}
			cout << endl;
			cout << endl;

			for (int i = 0; i < 10; i++){
				cout << " " << weight_diff_mut[i] << " ";
			}
			cout << endl;
			cout << endl;

			for (int i = 0; i < 10; i++){
				cout << " " << weight_diff[i] << " ";
			}
			cout << endl;
			cout << endl;

			for (int i = 0; i < 10; i++){
				cout << " " << data_diff[i] << " ";
			}
			cout << endl;
			cout << endl;

		#endif

		if (transpose_) {
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				K_, N_, M_,
				(Dtype)1., bottom_data, top_diff,
				(Dtype)1., this->blobs_[0]->mutable_cpu_diff());
		}
		else {
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				N_, K_, M_,
				(Dtype)1., top_diff, bottom_data,
				(Dtype)1., this->blobs_[0]->mutable_cpu_diff());
		}

	}
	if (bias_term_ && this->param_propagate_down_[1]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		// Gradient with respect to bias
		caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
			bias_multiplier_.cpu_data(), (Dtype)1.,
			this->blobs_[1]->mutable_cpu_diff());
	}

	//FM: SVD
	const Dtype* weight = this->blobs_[0]->cpu_data(); // nomizw pos to weight mporei na figei teleiws kai na douleuw mono me to weight_mut
	Dtype* weight_mut = this->blobs_[0]->mutable_cpu_data();

	int num_neurons = this->blobs_[0]->num(); // T tasks
	int dim_features = this->blobs_[0]->count() / num_neurons; // feature dimension
	CalculateSVD(weight, weight_mut, num_neurons, dim_features, k_value_);
	// FM: End of SVD


	if (propagate_down[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		// Gradient with respect to bottom data
		if (transpose_) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				M_, K_, N_,
				(Dtype)1., top_diff, weight,
				(Dtype)0., bottom[0]->mutable_cpu_diff());
		}
		else {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
				M_, K_, N_,
				(Dtype)1., top_diff, weight,
				(Dtype)0., bottom[0]->mutable_cpu_diff());
		}
	}

	#ifdef _DEBUG
		const Dtype* weight_diff_mut = this->blobs_[0]->mutable_cpu_diff();
		const Dtype* data_diff = bottom[0]->cpu_diff();
		const Dtype* weight_diff = this->blobs_[0]->cpu_diff();
		for (int i = 0; i < 10; i++){
			cout << " " << weight[i] << " ";
		}
		cout << endl;
		cout << endl;

		for (int i = 0; i < 10; i++){
			cout << " " << weight_diff_mut[i] << " ";
		}
		cout << endl;
		cout << endl;

		for (int i = 0; i < 10; i++){
			cout << " " << weight_diff[i] << " ";
		}
		cout << endl;
		cout << endl;

		for (int i = 0; i < 10; i++){
			cout << " " << data_diff[i] << " ";
		}
		cout << endl;
		cout << endl;

	#endif

}

//template <typename Dtype>
//void InnerProductMtlLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down,
//    const vector<Blob<Dtype>*>& bottom) {
//  if (this->param_propagate_down_[0]) {
//    const Dtype* top_diff = top[0]->cpu_diff();
//    const Dtype* bottom_data = bottom[0]->cpu_data();
//    // Gradient with respect to weight
//	#ifdef _DEBUG
//		const Dtype* weight = this->blobs_[0]->cpu_data();
//		const Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
//		for (int i = 0; i < 10; i++){
//			cout << " " << weight[i] << " ";
//		}
//		cout << endl;
//		cout << endl;
//
//		for (int i = 0; i < 10; i++){
//			cout << " " << weight_diff[i] << " ";
//		}
//		cout << endl;
//		cout << endl;
//
//	#endif
//
//    if (transpose_) {
//      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//          K_, N_, M_,
//          (Dtype)1., bottom_data, top_diff,
//          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
//    } else {
//      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//          N_, K_, M_,
//          (Dtype)1., top_diff, bottom_data,
//          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
//    }
//
//  }
//  if (bias_term_ && this->param_propagate_down_[1]) {
//    const Dtype* top_diff = top[0]->cpu_diff();
//    // Gradient with respect to bias
//    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
//        bias_multiplier_.cpu_data(), (Dtype)1.,
//        this->blobs_[1]->mutable_cpu_diff());
//  }
//
//  //FM: SVD
//  const Dtype* weight = this->blobs_[0]->cpu_data();
//  Dtype* weight_mut = this->blobs_[0]->mutable_cpu_data();
//
//#ifdef _DEBUG
//  //initial weights
//  for (int i = 0; i < 10; i++){
//	  cout << " " << weight[i] << " ";
//  }
//  cout << endl;
//  cout << endl;
//#endif
//
//  int num_neurons = this->blobs_[0]->num(); // T tasks
//  int dim_features = this->blobs_[0]->count() / num_neurons; // feature dimension
//  //Dtype* weight2 = new Dtype[num_neurons*dim_features];
//
//  //Instantiate the weight matrix
//  MatrixXf m = MatrixXf::Random(num_neurons, dim_features);
//  int counter = 0;
//  for (int i = 0; i < num_neurons; ++i) {
//	  for (int j = 0; j < dim_features; ++j) {
//		  m(i, j) = weight[counter];
//		  counter++;
//	  }
//
//  }
//
//
//  //Perform SVD
//  // cout << "Here is the matrix m:" << endl << m << endl;
//  JacobiSVD<MatrixXf> svd(m, ComputeThinU | ComputeThinV);
//
//  // Set with the new values
//  counter = 0;
//  MatrixXf m_inner = svd.matrixU() * (svd.singularValues().asDiagonal() * svd.matrixV().transpose());
//
//  for (int i = 0; i < num_neurons; ++i) {
//	  for (int j = 0; j < dim_features; ++j) {
//		  weight_mut[counter] = m_inner(i, j); // j,i because the matrix now has features in the columns (each column corresponds to one neuron)
//		  counter++;
//	  }
//
//  }
//
//#ifdef _DEBUG
//  MatrixXf diff = m_inner - m;
//  cout << "diff:\n" << diff.array().abs().sum() << "\n";
//  //cout << "Its singular values are:" << endl << svd.singularValues() << endl;
//  //cout << "Its left singular vectors are the columns of the thin L-U matrix:" << endl << svd.matrixU() << endl;
//  //cout << "Its right singular vectors are the columns of the thin S-V matrix:" << endl << svd.matrixV() << endl;
//
//  for (int i = 0; i < 10; i++){
//	  cout << " " << weight[i] << " ";
//  }
//  cout << endl;
//  cout << endl;
//
//  for (int i = 0; i < 10; i++){
//	  cout << " " << weight_mut[i] << " ";
//  }
//  cout << endl;
//  cout << endl;
//
//#endif
//
//  // FM: End of SVD
//
//  if (propagate_down[0]) {
//    const Dtype* top_diff = top[0]->cpu_diff();
//    // Gradient with respect to bottom data
//    if (transpose_) {
//      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
//          M_, K_, N_,
//		  (Dtype)1., top_diff, weight,
//          (Dtype)0., bottom[0]->mutable_cpu_diff());
//    } else {
//      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
//          M_, K_, N_,
//		  (Dtype)1., top_diff, weight,
//          (Dtype)0., bottom[0]->mutable_cpu_diff());
//    }
//  }
//}

#ifdef CPU_ONLY
STUB_GPU(InnerProductMtlLayer);
#endif

INSTANTIATE_CLASS(InnerProductMtlLayer);
REGISTER_LAYER_CLASS(InnerProductMtl);

}  // namespace caffe
