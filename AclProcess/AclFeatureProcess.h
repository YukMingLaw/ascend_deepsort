#pragma once

#include "iostream"
#include "acl/acl.h"
#include "ModelProcess.h"
#include "opencv2/opencv.hpp"
#include "sys/time.h"
#include "dataType.h"
using namespace std;
using namespace cv;

class AclFeatureProcess{
public:
    AclFeatureProcess();
    ~AclFeatureProcess();
    int Init(aclrtContext& context, string modelPath);
    void Process(Mat& img, DETECTIONS& detections);
private:
    std::vector<void *> inputBuffers;
    std::vector<size_t> inputSizes;
    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    aclrtContext context_;
    aclrtStream stream_;
    std::shared_ptr<ModelProcess> m_modelProcess;
    aclmdlDesc *m_modelDesc;
};
