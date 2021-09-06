#include "AclFeatureProcess.h"
#include <sys/time.h>
AclFeatureProcess::AclFeatureProcess()
{
}
AclFeatureProcess::~AclFeatureProcess()
{
    m_modelProcess = nullptr;
    aclError ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_ERROR_NONE)
    {
        cout << "some tasks in stream not done, ret = " << ret << endl;
    }
    cout << "all tasks in stream done" << endl;
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_ERROR_NONE)
    {
        cout << "Destroy Stream faild, ret = " << ret << endl;
    }
    cout << "Destroy Stream successfully" << endl;
}

int AclFeatureProcess::Init(aclrtContext& context, string modelPath)
{
    context_ = context;
    aclError ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE)
    {
        cout << "Failed to set current context, ret = " << ret << endl;
        return ret;
    }
    cout << "set context successfully" << endl;
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE)
    {
        cout << "Failed to create stream, ret = " << ret << endl;
        return ret;
    }
    cout << "Create stream successfully" << endl;
    //Load model
    if (m_modelProcess == nullptr)
    {
        m_modelProcess = std::make_shared<ModelProcess>("");
    }
    ret = m_modelProcess->Init(modelPath);

    if (ret != ACL_ERROR_NONE)
    {
        cout << "Failed to initialize m_modelProcess, ret = " << ret << endl;
        return ret;
    }
    m_modelDesc = m_modelProcess->GetModelDesc();
    //get model input description and malloc them
    size_t inputSize = aclmdlGetNumInputs(m_modelDesc);
    for (size_t i = 0; i < inputSize; i++)
    {
        size_t bufferSize = aclmdlGetInputSizeByIndex(m_modelDesc, i);
        void *inputBuffer = nullptr;
        aclError ret = aclrtMalloc(&inputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE)
        {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
            return ret;
        }
        inputBuffers.push_back(inputBuffer);
        inputSizes.push_back(bufferSize);
    }
    //get model output description and malloc them
    size_t outputSize = aclmdlGetNumOutputs(m_modelDesc);
    for (size_t i = 0; i < outputSize; i++)
    {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(m_modelDesc, i);
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE)
        {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
            return ret;
        }
        outputBuffers.push_back(outputBuffer);
        outputSizes.push_back(bufferSize);
    }
    vector<int> modelInputShape(4);
    aclmdlIODims dims;
    aclmdlGetInputDims(m_modelDesc, 0, &dims);
    if (dims.dimCount == 4)
    {
        modelInputShape[0] = dims.dims[0];
        modelInputShape[2] = dims.dims[1];
        modelInputShape[3] = dims.dims[2];
        modelInputShape[1] = dims.dims[3];
    }
    cout << "finish init AclProcess" << endl;
    return ACL_ERROR_NONE;
}

void AclFeatureProcess::Process(Mat &img, DETECTIONS& detections)
{
    aclError ret = ACL_ERROR_NONE;
    Mat imgResize;
    //get input dims
    int origin_height, origin_width;
    origin_height = img.rows;
    origin_width = img.cols;
    int batch, channels, height, width;
    aclmdlIODims dims;
    aclmdlGetInputDims(m_modelDesc, 0, &dims);
    if (dims.dimCount == 4)
    {
        batch = dims.dims[0];
        height = dims.dims[1];
        width = dims.dims[2];
        channels = dims.dims[3];
    }

    for (DETECTION_ROW &dbox : detections)
    {
        cv::Rect roi = cv::Rect(int(dbox.tlwh(0)),
                                int(dbox.tlwh(1)),
                                int(dbox.tlwh(2)),
                                int(dbox.tlwh(3)));

        roi.x -= (roi.height * 0.5 - roi.width) * 0.5;
        roi.width = roi.height * 0.5;
        roi.x = (roi.x >= 0 ? roi.x : 0);
        roi.y = (roi.y >= 0 ? roi.y : 0);
        roi.width = (roi.x + roi.width <= img.cols ? roi.width : (img.cols - roi.x));
        roi.height = (roi.y + roi.height <= img.rows ? roi.height : (img.rows - roi.y));

        resize(img(roi), imgResize, Size(width, height));
        aclrtMemcpy(inputBuffers[0], inputSizes[0], imgResize.data, imgResize.cols * imgResize.rows * imgResize.channels(), ACL_MEMCPY_HOST_TO_DEVICE);
        //forward
        ret = m_modelProcess->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes);
        if (ret != ACL_ERROR_NONE)
        {
            cout << "model run faild.ret = " << ret << endl;
        }
        aclrtMemcpy(dbox.feature.data(), outputSizes[0], outputBuffers[0], outputSizes[0], ACL_MEMCPY_DEVICE_TO_HOST);
    }
}
