#ifndef CPU_ONLY
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_mtl_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/calculate_SVD.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductMtlLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductMtlLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
   //FM: SVD
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_mut = this->blobs_[0]->mutable_cpu_data();

  int num_neurons = this->blobs_[0]->num(); // T tasks
  int dim_features = this->blobs_[0]->count() / num_neurons; // feature dimension
  CalculateSVD(weight, weight_mut, num_neurons, dim_features, k_value_);
  // FM: End of SVD
  
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
		  (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
		  (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}
//
//template <typename Dtype>
//void InnerProductMtlLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down,
//    const vector<Blob<Dtype>*>& bottom) {
//  if (this->param_propagate_down_[0]) {
//    const Dtype* top_diff = top[0]->gpu_diff();
//    const Dtype* bottom_data = bottom[0]->gpu_data();
//    // Gradient with respect to weight
//    if (transpose_) {
//      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//          K_, N_, M_,
//          (Dtype)1., bottom_data, top_diff,
//          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
//    } else {
//      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//          N_, K_, M_,
//          (Dtype)1., top_diff, bottom_data,
//          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
//    }
//  }
//  if (bias_term_ && this->param_propagate_down_[1]) {
//    const Dtype* top_diff = top[0]->gpu_diff();
//    // Gradient with respect to bias
//    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
//        bias_multiplier_.gpu_data(), (Dtype)1.,
//        this->blobs_[1]->mutable_gpu_diff());
//  }
//   //FM: SVD
//  const Dtype* weight = this->blobs_[0]->gpu_data();
//  Dtype* weight_mut = this->blobs_[0]->mutable_gpu_data();
//
//  int num_neurons = this->blobs_[0]->num(); // T tasks
//  int dim_features = this->blobs_[0]->count() / num_neurons; // feature dimension
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
//  //Perform SVD
//  JacobiSVD<MatrixXf> svd(m, ComputeThinU | ComputeThinV);
//
//  // Set with the new values
//  counter = 0;
//  MatrixXf m_inner = svd.matrixU() * (svd.singularValues().asDiagonal() * svd.matrixV().transpose());
//  for (int i = 0; i < num_neurons; ++i) {
//	  for (int j = 0; j < dim_features; ++j) {
//		  weight_mut[counter] = m_inner(i, j); // j,i because the matrix now has features in the columns (each column corresponds to one neuron)
//		  counter++;
//	  }
//
//  }
//  // FM: End of SVD
//  
//  if (propagate_down[0]) {
//    const Dtype* top_diff = top[0]->gpu_diff();
//    // Gradient with respect to bottom data
//    if (transpose_) {
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
//          M_, K_, N_,
//          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
//          (Dtype)0., bottom[0]->mutable_gpu_diff());
//    } else {
//      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
//          M_, K_, N_,
//         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
//         (Dtype)0., bottom[0]->mutable_gpu_diff());
//    }
//  }
//}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductMtlLayer);

}  // namespace caffe
#endif