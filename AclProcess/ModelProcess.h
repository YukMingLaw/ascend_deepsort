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
#pragma once

#include <cstdio>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <iostream>
#include <fstream>
#include "acl/acl.h"

using namespace std;

// Class of model inference
class ModelProcess {
public:

    // Construct a new Model Process object for model in the device
    ModelProcess(const std::string& modelName);
    ModelProcess();
    ~ModelProcess();

    int Init(std::string modelPath);
    int DeInit();

    int ModelInference(std::vector<void *> &inputBufs, std::vector<size_t> &inputSizes, std::vector<void *> &ouputBufs,
        std::vector<size_t> &outputSizes, size_t dynamicBatchSize = 0);
    aclmdlDesc *GetModelDesc();

    std::vector<void *> inputBuffers_ = {};
    std::vector<size_t> inputSizes_ = {};
    std::vector<void *> outputBuffers_ = {};
    std::vector<size_t> outputSizes_ = {};

private:
    aclmdlDataset *CreateAndFillDataset(std::vector<void *> &bufs, std::vector<size_t> &sizes);
    void DestroyDataset(aclmdlDataset *dataset);

    std::string modelName_ = "";
    uint32_t modelId_ = 0; // Id of import model
    aclrtContext contextModel_ = nullptr;
    shared_ptr<aclmdlDesc> modelDesc_ = nullptr;
    bool isDeInit_ = false;
};
