//
//  CustomTflite.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfliteUtils.hpp"
#include "flatbuffers/flexbuffers.h"
#include "liteOpConverter.hpp"

DECLARE_OP_COVERTER(CustomTflite);

MNN::OpType CustomTflite::opType(int quantizedModel) {

    if (quantizedModel == -1)
        return MNN::OpType_Landmarks2TransformMatrix;
    else if (quantizedModel == -2)
        return MNN::OpType_TransformTensorBilinear;
    else if (quantizedModel == -3)
        return MNN::OpType_TransformLandmarks;

    DCHECK(!quantizedModel) << "Not support quantized model";
    return MNN::OpType_DetectionPostProcess;
}

MNN::OpParameter CustomTflite::type(int quantizedModel) {
    if (quantizedModel == -1)
        return MNN::OpParameter_Landmarks2TransformMatrixParam;
    else if (quantizedModel == -2)
        return MNN::OpParameter_TransformTensorBilinearParam;
    else if (quantizedModel == -3 )
        return MNN::OpParameter_TransformLandmarksParam;
    return MNN::OpParameter_DetectionPostProcessParam;
}

void CustomTflite::run(MNN::OpT *dstOp, const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                       const std::vector<std::unique_ptr<tflite::TensorT> > &tfliteTensors,
                       const std::vector<std::unique_ptr<tflite::BufferT> > &tfliteModelBuffer,
                       const std::vector<std::unique_ptr<tflite::OperatorCodeT> > &tfliteOpSet, int quantizedModel) {
    auto &customOPCode = tfliteOpSet[tfliteOp->opcode_index]->custom_code;

    if (customOPCode == "Landmarks2TransformMatrix")
    {
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
        return;
    }

    if (customOPCode == "TransformTensorBilinear")
    {
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
        return;
    }

    if (customOPCode == "TransformLandmarks")
    {
        DCHECK(tfliteOp->inputs.size() == 2) << "TransformLandmarks should have 2 inputs!";
        DCHECK(tfliteOp->outputs.size() == 1) << "TransformLandmarks should have 1 outputs!";


        auto paramSave = new MNN::TransformLandmarksParamT;
        dstOp->main.value = paramSave;

        // set input output index
        dstOp->inputIndexes.resize(2);
        dstOp->outputIndexes.resize(1);

        dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
        dstOp->inputIndexes[1]  = tfliteOp->inputs[1];

        dstOp->outputIndexes[0] = tfliteOp->outputs[0];
        return;
    }

    DCHECK(customOPCode == "TFLite_Detection_PostProcess")
        << "Now Only support Custom op of 'TFLite_Detection_PostProcess'";

    auto postProcessParam    = new MNN::DetectionPostProcessParamT;
    auto customOptionsFormat = tfliteOp->custom_options_format;
    DCHECK(customOptionsFormat == tflite::CustomOptionsFormat_FLEXBUFFERS) << "custom options format ERROR!";
    const uint8_t *customOptionBufferDataPtr = tfliteOp->custom_options.data();
    const auto size                          = tfliteOp->custom_options.size();
    const flexbuffers::Map &m                = flexbuffers::GetRoot(customOptionBufferDataPtr, size).AsMap();

    postProcessParam->maxDetections          = m["max_detections"].AsInt32();
    postProcessParam->maxClassesPerDetection = m["max_classes_per_detection"].AsInt32();
    if (m["detections_per_class"].IsNull()) {
        postProcessParam->detectionsPerClass = 100;
    } else {
        postProcessParam->detectionsPerClass = m["detections_per_class"].AsInt32();
    }
    if (m["use_regular_nms"].IsNull()) {
        postProcessParam->useRegularNMS = false;
    } else {
        postProcessParam->useRegularNMS = m["use_regular_nms"].AsBool();
    }
    postProcessParam->nmsScoreThreshold = m["nms_score_threshold"].AsFloat();
    postProcessParam->iouThreshold      = m["nms_iou_threshold"].AsFloat();
    postProcessParam->numClasses        = m["num_classes"].AsInt32();
    postProcessParam->centerSizeEncoding.push_back(m["y_scale"].AsFloat());
    postProcessParam->centerSizeEncoding.push_back(m["x_scale"].AsFloat());
    postProcessParam->centerSizeEncoding.push_back(m["h_scale"].AsFloat());
    postProcessParam->centerSizeEncoding.push_back(m["w_scale"].AsFloat());

    dstOp->main.value = postProcessParam;

    DCHECK(tfliteOp->inputs.size() == 3) << "TFLite_Detection_PostProcess should have 3 inputs!";
    DCHECK(tfliteOp->outputs.size() == 4) << "TFLite_Detection_PostProcess should have 4 outputs!";
}

using namespace tflite;
REGISTER_CONVERTER(CustomTflite, BuiltinOperator_CUSTOM);
