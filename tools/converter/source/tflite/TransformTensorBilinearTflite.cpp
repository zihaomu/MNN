//
//  CustomTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "liteOpConverter.hpp"
#include "TfliteUtils.hpp"
#include "flatbuffers/flexbuffers.h"

DECLARE_OP_COVERTER(TransformTensorBilinearTflite);

MNN::OpType TransformTensorBilinearTflite::opType(int quantizedModel) {
//    DCHECK(!quantizedModel) << "Not support quantized model";
    return MNN::OpType_TransformTensorBilinear;
}

MNN::OpParameter TransformTensorBilinearTflite::type(int quantizedModel) {
    return MNN::OpParameter_TransformTensorBilinearParam;
}

void TransformTensorBilinearTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel) {
    DCHECK(tfliteOp->inputs.size() == 2) << "TransformTensorBilinear should have 2 inputs!";
    DCHECK(tfliteOp->outputs.size() == 1) << "TransformTensorBilinear should have 1 outputs!";

    auto paramSave = new MNN::TransformTensorBilinearParamT;
    dstOp->main.value = paramSave;

    // set input output index
    dstOp->inputIndexes.resize(2);
    dstOp->outputIndexes.resize(1);

    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->inputIndexes[1]  = tfliteOp->inputs[1];

    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(TransformTensorBilinearTflite, BuiltinOperator_TransformTensorBilinear);
