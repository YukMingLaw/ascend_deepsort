#include "AclYolov5Process.h"
#include <sys/time.h>
AclYolov5Process::AclYolov5Process()
{
    
}
AclYolov5Process::~AclYolov5Process()
{
    m_modelProcess = nullptr;
    aclError ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "some tasks in stream not done, ret = " << ret <<endl;
    }
    cout << "all tasks in stream done" << endl;
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Destroy Stream faild, ret = " << ret <<endl;
    }
    cout << "Destroy Stream successfully" << endl;
}

int AclYolov5Process::Init(aclrtContext& context, string modelPath, int _inputShape, int _classNum, float _thresObj, float _thresNms)
{
    inputShape = _inputShape;
    classNum = _classNum;
    thresObj = _thresObj;
    thresNms = _thresNms;
    context_ = context;
    aclError ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to set current context, ret = " << ret << endl;
        return ret;
    }
    cout << "set context successfully" << endl;
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to create stream, ret = " << ret << endl;
        return ret;
    }
    cout << "Create stream successfully" << endl;
    //Load model
    if (m_modelProcess == nullptr) {
        m_modelProcess = std::make_shared<ModelProcess>("");
    }
    ret = m_modelProcess->Init(modelPath);

    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to initialize m_modelProcess, ret = " << ret << endl;
        return ret;
    }
    m_modelDesc = m_modelProcess->GetModelDesc();
    //get model input description and malloc them
    size_t inputSize = aclmdlGetNumInputs(m_modelDesc);
    for (size_t i = 0; i < inputSize; i++) {
        size_t bufferSize = aclmdlGetInputSizeByIndex(m_modelDesc, i);
        void *inputBuffer = nullptr;
        aclError ret = aclrtMalloc(&inputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
            return ret;
        }
        inputBuffers.push_back(inputBuffer);
        inputSizes.push_back(bufferSize);
    }
    //get model output description and malloc them
    size_t outputSize = aclmdlGetNumOutputs(m_modelDesc);
    for (size_t i = 0; i < outputSize; i++) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(m_modelDesc, i);
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
            return ret;
        }
        outputBuffers.push_back(outputBuffer);
        outputSizes.push_back(bufferSize);
    }
    
    for (size_t j = 0; j < outputSizes.size(); j++) {
        aclError ret = (aclError)aclrtMallocHost(&hostPtr[j], outputSizes[j]);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc output buffer of model on host, ret = " << ret << endl;
            return ret;
        }
    }
    
    vector<int> modelInputShape(4);
    aclmdlIODims dims;
    aclmdlGetInputDims(m_modelDesc,0,&dims);
    if(dims.dimCount == 4){
        modelInputShape[0] = dims.dims[0];
        modelInputShape[2] = dims.dims[1];
        modelInputShape[3] = dims.dims[2];
        modelInputShape[1] = dims.dims[3];
    }
    cout << "finish init AclProcess" << endl;
    return ACL_ERROR_NONE;
}

void AclYolov5Process::Process(Mat& img, DETECTIONS& detections)
{
    aclError ret = ACL_ERROR_NONE;
    Mat imgResize;
    //get input dims
    origin_height = img.rows;
    origin_width = img.cols;
    int batch,channels,height,width;
    aclmdlIODims dims;
    aclmdlGetInputDims(m_modelDesc,0,&dims);
    if(dims.dimCount == 4){
        batch = dims.dims[0];
        height = dims.dims[1];
        width = dims.dims[2];
        channels = dims.dims[3];
    }
    /*warp affine image
    **what does the code do:               416
    **           1920                |-------------|
    **       _____________           |_____________|
    ** 1080 |  /______\   |  =====>  |  /______\   |
    **      |  |# n # |   |      416 |  |# n # |   |
    **      |-------------|          |-------------|
    **                               |_____________|
    **if your input image is with a static size
    **you can accelerate this process with aipp->padding
    **more detail:https://support.huaweicloud.com/ti-atc-A300_3000_3010/altasatc_16_009.html*/
    Mat M;
    float r = origin_height * 1.0 / origin_width;
    if(r > 1)
    {
        float offset = (width - origin_width * (height * 1.0 / origin_height)) / 2.0;
        M = getAffineTransform(
                vector<Point2f>({Point2f(0, 0),Point2f(0, origin_height),Point2f(origin_width, 0)}),
                vector<Point2f>({Point2f(offset, 0),Point2f(offset, height),Point2f((width - offset), 0)})
        );
    }
    else
    {
        float offset = (height - origin_height * (width * 1.0 / origin_width)) / 2.0;
        M = getAffineTransform(
                vector<Point2f>({Point2f(0, 0),Point2f(origin_width, 0),Point2f(0, origin_height)}),
                vector<Point2f>({Point2f(0, offset),Point2f(width, offset),Point2f(0, (height-offset))})
        );
    }
    warpAffine(img, imgResize, M, Size(height,width),INTER_NEAREST);
    //warp image infomation data
    aclrtMemcpy(inputBuffers[0], inputSizes[0], imgResize.data, imgResize.cols * imgResize.rows * imgResize.channels(), ACL_MEMCPY_HOST_TO_DEVICE);

    //forward
    ret = m_modelProcess->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes);
    if (ret != ACL_ERROR_NONE) {
        cout<<"model run faild.ret = "<< ret <<endl;
    }
    //postprocess
    PostProcess(outputBuffers, outputSizes, detections);
}

aclError AclYolov5Process::PostProcess(std::vector<void *> outputBuffers, std::vector<size_t> outputSizes, DETECTIONS& detections)
{
    if (outputSizes.size() <= 0) {
        cout << "Failed to get model output data" << endl;;
        return -4;
    }
    aclError ret;
    for (size_t j = 0; j < outputSizes.size(); j++) {
        ret = (aclError)aclrtMemcpy(hostPtr[j], outputSizes[j], outputBuffers[j],
                                    outputSizes[j], ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to copy output buffer of model from device to host, ret = " << ret << endl;
            return ret;
        }
    }
    
    vector<vector<float>> results;
    yolov5PostProcess.run(inputShape, classNum, thresObj, thresNms, hostPtr[2], hostPtr[1], hostPtr[0], results);
    for (auto result : results)
    {
        DETECTION_ROW tmpRow;
        float r = origin_width * 1.f / inputShape;
        int offset = (inputShape - origin_height / r) / 2;
        tmpRow.tlwh = DETECTBOX((result[0] - result[2] / 2) * r, //x
                                (result[1] - result[3] / 2 - offset) * r, //y
                                result[2] * r, //w
                                result[3]* r); //h
        tmpRow.confidence = result[4] * result[6];
        detections.push_back(tmpRow);
    }
    return ACL_ERROR_NONE;
}
