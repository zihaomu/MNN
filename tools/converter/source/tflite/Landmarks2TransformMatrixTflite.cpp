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

DECLARE_OP_COVERTER(Landmarks2TransformMatrixTflite);

MNN::OpType Landmarks2TransformMatrixTflite::opType(int quantizedModel) {
//    DCHECK(!quantizedModel) << "Not support quantized model";
    return MNN::OpType_Landmarks2TransformMatrix;
}

MNN::OpParameter Landmarks2TransformMatrixTflite::type(int quantizedModel) {
    return MNN::OpParameter_Landmarks2TransformMatrixParam;
}

void Landmarks2TransformMatrixTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel) {
    auto paramSave = new MNN::Landmarks2TransformMatrixParamT;
    const uint8_t *customOptionBufferDataPtr = tfliteOp->custom_options.data();
    const auto size                          = tfliteOp->custom_options.size();
    const flexbuffers::Map &m                = flexbuffers::GetRoot(customOptionBufferDataPtr, size).AsMap();

    paramSave->left_rotation_idx  = m["left_rotation_idx"].AsInt32();
    paramSave->right_rotation_idx = m["right_rotation_idx"].AsInt32();
    paramSave->output_height = m["output_height"].AsInt32();
    paramSave->output_width = m["output_width"].AsInt32();
    paramSave->scale_x = m["scale_x"].AsFloat();
    paramSave->scale_y = m["scale_y"].AsFloat();

    if (m["target_rotation_radians"].IsNull())
        paramSave->target_rotation_radians = 0;

    std::vector<int> subset_vec;

    auto fbData = m["subset_idxs"].AsTypedVector();
    int dataSize = fbData.size();

    for (int i = 0; i < dataSize; i++)
    {
        subset_vec.emplace_back(fbData[i].AsInt32());
    }

    paramSave->subset_idxs = std::vector<int>(subset_vec.size());
    ::memcpy(paramSave->subset_idxs.data(), subset_vec.data(), sizeof(int) * subset_vec.size());

    DCHECK(tfliteOp->inputs.size() == 1) << "Landmarks2TransformMatrix should have 1 inputs!";
    DCHECK(tfliteOp->outputs.size() == 1) << "Landmarks2TransformMatrix should have 1 outputs!";

    dstOp->main.value = paramSave;

    // set input output index
    dstOp->inputIndexes.resize(1);
    dstOp->outputIndexes.resize(1);

    dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
    dstOp->outputIndexes[0] = tfliteOp->outputs[0];
}

using namespace tflite;
REGISTER_CONVERTER(Landmarks2TransformMatrixTflite, BuiltinOperator_Landmarks2TransformMatrix);
