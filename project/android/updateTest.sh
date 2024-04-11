#!/bin/bash
DIR=MNN

make -j16
adb push ./libMNN.so /data/local/tmp/$DIR/libMNN.so
adb push ./libMNN_CL.so /data/local/tmp/$DIR/libMNN_CL.so
adb push ./libMNN_Vulkan.so /data/local/tmp/$DIR/libMNN_Vulkan.so
adb push ./libMNN_GL.so /data/local/tmp/$DIR/libMNN_GL.so
adb push ./libMNN_Express.so /data/local/tmp/$DIR/libMNN_Express.so
adb push ./MNNV2Basic.out /data/local/tmp/$DIR/MNNV2Basic.out
adb shell "cd /data/local/tmp/$DIR && rm -r output"
adb shell "cd /data/local/tmp/$DIR && mkdir output"
adb push ./unitTest.out /data/local/tmp/$DIR/unitTest.out
adb push ./testModel.out /data/local/tmp/$DIR/testModel.out
adb push ./testModelWithDescribe.out /data/local/tmp/$DIR/testModelWithDescribe.out
adb push ./backendTest.out /data/local/tmp/$DIR/backendTest.out
adb push ./timeProfile.out /data/local/tmp/$DIR/timeProfile.out

adb push ./train.out /data/local/tmp/$DIR/train.out
adb push ./benchmark.out /data/local/tmp/$DIR/benchmark.out
adb push ./benchmarkExprModels.out /data/local/tmp/$DIR/benchmarkExprModels.out
adb push ./run_test.out /data/local/tmp/$DIR/run_test.out

adb push /Users/mzh/work/models/models_migu/self_seg.onnx.mnn /data/local/tmp/$DIR/self_seg.onnx.mnn
# adb push /Users/mzh/work/models/models_migu/hand_driver/hand3d_20231130_head.onnx.mnn /data/local/tmp/$DIR/hand3d_20231130_head.onnx.mnn
# adb push /Users/mzh/work/models/models_migu/hand_driver/hand3d_20231130_head_v1.onnx.mnn /data/local/tmp/$DIR/hand3d_20231130_head_v1.onnx.mnn
# adb push /Users/mzh/work/models/models_migu/hand_driver/hand3d_20231130_head_v2.onnx.mnn /data/local/tmp/$DIR/hand3d_20231130_head_v2.onnx.mnn
# adb push /Users/mzh/work/models/models_migu/hand_driver/hand3d_20231130_head_v3.onnx.mnn /data/local/tmp/$DIR/hand3d_20231130_head_v3.onnx.mnn
# adb push /Users/mzh/work/models/tinghua/mobilenetv3_large_100.mnn /data/local/tmp/$DIR/mobilenetv3_large_100.mnn
# adb push /Users/mzh/work/models/body_driver/body3d_model_cspnext_sim_v3.mnn /data/local/tmp/$DIR/body3d_model_cspnext_sim_v3.mnn
# adb push /Users/mzh/work/models/body_driver/body3d_model_cspnext_sim_v2.mnn /data/local/tmp/$DIR/body3d_model_cspnext_sim_v2.mnn
# adb push /Users/mzh/work/models/body_driver/body3d_model_cspnext_sim_v4.mnn /data/local/tmp/$DIR/body3d_model_cspnext_sim_v4.mnn
# adb push /Users/mzh/work/models/body_driver/body3d_model_cspnext_sim_v5.mnn /data/local/tmp/$DIR/body3d_model_cspnext_sim_v5.mnn
# adb push /Users/mzh/work/models/body_driver/body3d_model_cspnext_no_sim.mnn /data/local/tmp/$DIR/body3d_model_cspnext_no_sim.mnn
# adb push /Users/mzh/work/models/body_driver/unet_13_256.mnn /data/local/tmp/$DIR/unet_13_256.mnn

# adb push /Users/mzh/work/github/onnx-modifier/modified_onnx/body3d_model_cspnext_v1.mnn /data/local/tmp/$DIR/body3d_model_cspnext_v1.mnn
# adb push /Users/mzh/work/github/onnx-modifier/modified_onnx/body3d_model_cspnext_v2.mnn /data/local/tmp/$DIR/body3d_model_cspnext_v2.mnn
# adb push /Users/mzh/work/github/onnx-modifier/modified_onnx/body3d_model_cspnext_v3.mnn /data/local/tmp/$DIR/body3d_model_cspnext_v3.mnn
# adb push /Users/mzh/work/github/onnx-modifier/modified_onnx/body3d_model_cspnext_v4.mnn /data/local/tmp/$DIR/body3d_model_cspnext_v4.mnn
# adb push /Users/mzh/work/github/onnx-modifier/modified_onnx/body3d_model_cspnext_v5.mnn /data/local/tmp/$DIR/body3d_model_cspnext_v5.mnn
# adb push /Users/mzh/work/github/onnx-modifier/modified_onnx/body3d_model_cspnext_v6.mnn /data/local/tmp/$DIR/body3d_model_cspnext_v6.mnn
# adb push /Users/mzh/work/github/onnx-modifier/modified_onnx/modified_body3d_np_post.onnx.mnn /data/local/tmp/$DIR/modified_body3d_np_post.onnx.mnn
# adb push /Users/mzh/work/my_project/mediapipe_cmake/mpp_vision/interactive_segmenter/models/magic_touch.mnn /data/local/tmp/$DIR/magic_touch.mnn

# adb push /Users/mzh/work/my_project/flutter_ml/flutter_pose_landmark_project/assets/model/pose_landmark_full_sim.mnn /data/local/tmp/$DIR/pose_landmark_full_sim.mnn
# adb push /Users/mzh/work/github/MNN_Debug/zihaomu_MNN/benchmark/models/mobilenetV3.mnn /data/local/tmp/$DIR/mobilenetV3.mnn
# adb push /Users/mzh/work/models/tflite/resnet50.tflite.mnn /data/local/tmp/$DIR/resnet50.tflite.mnn
# adb push /Users/mzh/work/models/tflite/mobilenetv1.tflite.mnn /data/local/tmp/$DIR/mobilenetv1.tflite.mnn
# adb push /Users/mzh/work/models/tflite/mobilenetv2.tflite.mnn /data/local/tmp/$DIR/mobilenetv2.tflite.mnn
# adb push /Users/mzh/work/models/tflite/mobilenetv3_small.tflite.mnn /data/local/tmp/$DIR/mobilenetv3_small.tflite.mnn
# adb push /Users/mzh/work/models/tflite/mobilenetv3_large.tflite.mnn /data/local/tmp/$DIR/mobilenetv3_large.tflite.mnn
