#include <vector>

#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void CalculateSVD(const Dtype* weight,
    Dtype* weight_mut, int num_neurons, int dim_features, int k_value);


}  // namespace caffe


