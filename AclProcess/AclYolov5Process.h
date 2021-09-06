#pragma once

#include "iostream"
#include "acl/acl.h"
#include "ModelProcess.h"
#include "opencv2/opencv.hpp"
#include "sys/time.h"
#include "Yolov5PostProcess.h"
#include "dataType.h"
using namespace std;
using namespace cv;

class AclYolov5Process{
public:
    AclYolov5Process();
    ~AclYolov5Process();
    int Init(aclrtContext& context, string modelPath, int _inputShape, int _classNum, float _thresObj, float _thresNms);
    void Process(Mat& img, DETECTIONS& detections);
private:
    aclError PostProcess(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes, DETECTIONS& detections);
    Yolov5PostProcess yolov5PostProcess;
    std::vector<void *> inputBuffers;
    std::vector<size_t> inputSizes;
    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    aclrtContext context_;
    aclrtStream stream_;
    std::shared_ptr<ModelProcess> m_modelProcess;
    aclmdlDesc *m_modelDesc;
    void* hostPtr[3];
    int origin_height, origin_width;
    int inputShape;
    int classNum;
    float thresObj;
    float thresNms;
};
