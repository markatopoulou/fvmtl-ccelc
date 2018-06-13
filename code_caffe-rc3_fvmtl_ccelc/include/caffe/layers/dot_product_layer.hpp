#ifndef CAFFE_DOT_PRODUCT_LAYER_HPP_ //FM: I added this definition
#define CAFFE_DOT_PRODUCT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Computes @f$ y = x_1 \cdot x_2 @f$
	*/
	template <typename Dtype>
	class DotProductLayer : public Layer < Dtype > {
	public:
		explicit DotProductLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		//virtual inline LayerParameter_LayerType type() const { //FM: I commented this three lines
		//	return LayerParameter_LayerType_DOT_PRODUCT;
		//}
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "DotProduct"; } //FM: I added this line

	protected:
		/**
		* @param bottom input Blob vector (length 2)
		*   -# @f$ (N \times C \times H \times W) @f$
		*      the inputs @f$ x_1 @f$
		*   -# @f$ (N \times C \times H \times W) @f$
		*      the inputs @f$ x_2 @f$
		* @param top output Blob vector (length 1)
		*   -# @f$ (N \times 1 \times 1 \times 1) @f$
		*      the computed output @f$
		*        y = x_1 \cdot x_2
		*      @f$
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			//const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);//FM: I added this line
		//virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			//const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom); //FM: I added this line
	};
}  // namespace caffe

#endif  // CAFFE_DOT_PRODUCT_LAYER_HPP_