/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ModelProcess.h"

ModelProcess::ModelProcess(const std::string& modelName)
{
    modelName_ = modelName;
}

ModelProcess::ModelProcess()
{

}

ModelProcess::~ModelProcess()
{
    if (!isDeInit_) {
        DeInit();
    }
}

void ModelProcess::DestroyDataset(aclmdlDataset *dataset)
{
    // Just release the DataBuffer object and DataSet object, remain the buffer, because it is managerd by user
    if (dataset != nullptr) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); i++) {
            aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
            if (dataBuffer != nullptr) {
                aclDestroyDataBuffer(dataBuffer);
                dataBuffer = nullptr;
            }
        }
        aclmdlDestroyDataset(dataset);
        dataset = nullptr;
    }
}

aclmdlDesc *ModelProcess::GetModelDesc()
{
    return modelDesc_.get();
}

int ModelProcess::ModelInference(std::vector<void *> &inputBufs, std::vector<size_t> &inputSizes,
    std::vector<void *> &ouputBufs, std::vector<size_t> &outputSizes, size_t dynamicBatchSize)
{
    aclmdlDataset *input = nullptr;
    input = CreateAndFillDataset(inputBufs, inputSizes);
    if (input == nullptr) {
        cout << "CreateAndFillDataset Faild." << endl;
        return -1;
    }
    aclError ret = 0;
    if (dynamicBatchSize != 0) {
        size_t index;
        ret = aclmdlGetInputIndexByName(modelDesc_.get(), ACL_DYNAMIC_TENSOR_NAME, &index);
        if (ret != ACL_ERROR_NONE) {
            cout << "aclmdlGetInputIndexByName failed, maybe static model" << endl;;
            return -2;
        }
        ret = aclmdlSetDynamicBatchSize(modelId_, input, index, dynamicBatchSize);
        if (ret != ACL_ERROR_NONE) {
            cout << "dynamic batch set failed, modelId_=" << modelId_ << ", input=" << input << ", index=" << index
                     << ", dynamicBatchSize=" << dynamicBatchSize <<endl;;
            return -2;
        }
        cout << "set dynamicBatchSize successfully, dynamicBatchSize=" << dynamicBatchSize;
    }
    aclmdlDataset *output = nullptr;
    output = CreateAndFillDataset(ouputBufs, outputSizes);
    if (output == nullptr) {
        DestroyDataset(input);
        input = nullptr;
        cout << "CreateAndFillDataset Faild." << endl;
        return -1;
    }

    ret = aclmdlExecute(modelId_, input, output);
    if (ret != ACL_ERROR_NONE) {
        cout << "aclmdlExecute failed, ret[" << ret << "]." << endl;
        return ret;
    }

    DestroyDataset(input);
    DestroyDataset(output);
    return ACL_ERROR_NONE;
}

int ModelProcess::DeInit()
{
    cout << "ModelProcess:Begin to deinit instance." << endl;
    isDeInit_ = true;
    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_ERROR_NONE) {
        cout << "aclmdlUnload  failed, ret["<< ret << "]." << endl;
        return ret;
    }

    for (size_t i = 0; i < inputBuffers_.size(); i++) {
        if (inputBuffers_[i] != nullptr) {
            aclrtFree(inputBuffers_[i]);
            inputBuffers_[i] = nullptr;
        }
    }

    for (size_t i = 0; i < outputBuffers_.size(); i++) {
        if (outputBuffers_[i] != nullptr) {
            aclrtFree(outputBuffers_[i]);
            outputBuffers_[i] = nullptr;
        }
    }

    inputSizes_.clear();
    outputSizes_.clear();
    cout << "ModelProcess:Finished deinit instance." << endl;
    return ACL_ERROR_NONE;
}

aclError ModelProcess::Init(std::string modelPath)
{

    cout << "ModelProcess:Begin to init instance." << endl;
    //Load model from file
    aclError ret = aclmdlLoadFromFile(modelPath.c_str(),&modelId_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Load model from file failed, ret[" << ret << "]." << endl;
        return ret;
    }
    //get current context
    ret = aclrtGetCurrentContext(&contextModel_);
    if (ret != ACL_ERROR_NONE) {
        cout << "aclrtMalloc weight_ptr failed, ret[" << ret << "]." << endl;
        return ret;
    }
    // get input and output size
    aclmdlDesc *modelDesc = aclmdlCreateDesc();
    if (modelDesc == nullptr) {
        cout << "aclmdlCreateDesc failed." << endl;
        return ret;
    }
    //get model description
    ret = aclmdlGetDesc(modelDesc, modelId_);
    if (ret != ACL_ERROR_NONE) {
        cout << "aclmdlGetDesc ret fail, ret:" << ret << "." << endl;
        return ret;
    }
    modelDesc_.reset(modelDesc, aclmdlDestroyDesc);
    return ACL_ERROR_NONE;
}

aclmdlDataset *ModelProcess::CreateAndFillDataset(std::vector<void *> &bufs, std::vector<size_t> &sizes)
{
    aclError ret = ACL_ERROR_NONE;
    aclmdlDataset *dataset = aclmdlCreateDataset();
    if (dataset == nullptr) {
        cout << "ACL_ModelInputCreate failed." << endl;
        return nullptr;
    }

    for (size_t i = 0; i < bufs.size(); ++i) {
        aclDataBuffer *data = aclCreateDataBuffer(bufs[i], sizes[i]);
        if (data == nullptr) {
            DestroyDataset(dataset);
            cout << "aclCreateDataBuffer failed." << endl;
            return nullptr;
        }

        ret = aclmdlAddDatasetBuffer(dataset, data);
        if (ret != ACL_ERROR_NONE) {
            DestroyDataset(dataset);
            cout << "ACL_ModelInputDataAdd failed, ret[" << ret << "]." << endl;
            return nullptr;
        }
    }
    return dataset;
}
