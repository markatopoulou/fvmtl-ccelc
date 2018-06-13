#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/util/calculate_SVD.hpp"
#include <Eigen/SVD>
using namespace Eigen;

namespace caffe {

template <typename Dtype>
void CalculateSVD(const Dtype* weight,
    Dtype* weight_mut, int num_neurons, int dim_features, int k_value) {
  

  //FM: SVD
  //const Dtype* weight = this->blobs_[0]->cpu_data();
  //Dtype* weight_mut = this->blobs_[0]->mutable_cpu_data();

  //Instantiate the weight matrix
  MatrixXf m = MatrixXf::Random(num_neurons, dim_features);
  int counter = 0;
  for (int i = 0; i < num_neurons; ++i) {
	  for (int j = 0; j < dim_features; ++j) {
		  m(i, j) = weight[counter];
		  counter++;
	  }

  }

  //Perform SVD
  // cout << "Here is the matrix m:" << endl << m << endl;
  JacobiSVD<MatrixXf> svd(m, ComputeThinU | ComputeThinV);

  // Set with the new values
  counter = 0;
  //MatrixXf m_inner = svd.matrixU() * (svd.singularValues().asDiagonal() * svd.matrixV().transpose()); //original
  int u_rows = svd.matrixU().rows();
  int v_rows = svd.matrixV().rows();
  int s_cols = svd.singularValues().cols();
  //MatrixXf m_inner = svd.matrixU().block<u_rows, k>(0, 0) * (svd.singularValues().asDiagonal() * svd.matrixV().transpose()); //low-rank decomposition
  MatrixXf m_inner = svd.matrixU().block(0, 0, u_rows, k_value) * (svd.singularValues().block(0, 0, k_value, s_cols).asDiagonal() * svd.matrixV().block(0, 0, v_rows, k_value).transpose()); //low-rank decomposition
  
  #ifdef _DEBUG
	  MatrixXf diff = m_inner - m;
	  cout << "diff:\n" << diff.array().abs().sum() << "\n";
  #endif

  for (int i = 0; i < num_neurons; ++i) {
	  for (int j = 0; j < dim_features; ++j) {
		  weight_mut[counter] = m_inner(i, j);
		  counter++;
	  }

  }
  // FM: End of SVD
}

template void CalculateSVD<float>(const float* weight, float* weight_mut, int num_neurons, int dim_features, int k_value);
template void CalculateSVD<double>(const double* weight, double* weight_mut, int num_neurons, int dim_features, int k_value);

}  // namespace caffe
