//
// Created by yo on 21-6-5.
//

#include "Yolov5PostProcess.h"
#include <sys/time.h>
//#define PRINT_TIME
Yolov5PostProcess::Yolov5PostProcess() {

}

void Yolov5PostProcess::run(int _img_size, int _classes, float _obj_conf, float _nms_threshold, void *stride_32_data, void *stride_16_data, void *stride_8_data, vector<vector<float>>& results) {
    obj_conf = _obj_conf;
    nms_threshold = _nms_threshold;
    img_size = _img_size;
    classes = _classes;

    float *data[3];
    data[0] = (float*)stride_8_data;
    data[1] = (float*)stride_16_data;
    data[2] = (float*)stride_32_data;
#ifdef PRINT_TIME
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
#endif
    vector<vector<float>> threshold_results = ThresholdFliter(data, obj_conf);
    if(threshold_results.size() < 1){
        return;
    }
#ifdef PRINT_TIME
    gettimeofday(&end,NULL);
    cout<<"ThresholdFliter time:"<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec) / 1000.0 <<"ms"<<endl;

    gettimeofday(&start,NULL);
#endif
    std::sort(threshold_results.begin(), threshold_results.end(), BoxSortDecendScore);
#ifdef PRINT_TIME
    gettimeofday(&end,NULL);
    cout<<"sort time:"<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec) / 1000.0 <<"ms"<<endl;
#endif
    vector<int> idxes;
#ifdef PRINT_TIME
    gettimeofday(&start,NULL);
#endif
    nms(threshold_results, idxes, nms_threshold);
#ifdef PRINT_TIME
    gettimeofday(&end,NULL);
    cout<<"nms time:"<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec) / 1000.0 <<"ms"<<endl;
#endif   
    int num = idxes.size();
    
    for(int i = 0; i < num; i++){
        results.push_back(threshold_results[idxes[i]]);
    }
}

float Yolov5PostProcess::fastSigmoid(float x)
{
    //return (x / (1.f + abs(x))) * 0.5f + 0.5f;
    return (float)(1.f / (1.f + exp(-1.f * x)));
}

vector<vector<float>> Yolov5PostProcess::ThresholdFliter(float **data, float threahold)
{
    vector<vector<float>> results;
    vector<vector<float>> temp[3];
    int stride[] = {8, 16, 32};
#pragma omp parallel for
    for(int i = 0; i < 3; i++)
    {
        float *stride_data = data[i];
        int stride_len = (img_size / stride[i]) * (img_size / stride[i]);
        int stride_w = (img_size / stride[i]);
        int stride_h = (img_size / stride[i]);
        int offset = 5 + classes;
        for(int c = 0; c < 3; c++){
            float* x1_data = stride_data + stride_len * (c * offset + 0);
            float* y1_data = stride_data + stride_len * (c * offset + 1);
            float* x2_data = stride_data + stride_len * (c * offset + 2);
            float* y2_data = stride_data + stride_len * (c * offset + 3);
            float* conf_data = stride_data + stride_len * (c * offset + 4);
            float* classes_data = stride_data + stride_len * (c * offset + 5);
            for(int h = 0; h < stride_h; h++){
                for(int w = 0; w < stride_w; w++){
                    int idx = w + h * stride_w;
                    if(fastSigmoid(conf_data[idx]) > threahold){
                        vector<float> result(7);

                        result[0] = (fastSigmoid(x1_data[idx]) * 2.f - 0.5f + w) * stride[i];
                        result[1] = (fastSigmoid(y1_data[idx]) * 2.f - 0.5f + h) * stride[i];
                        result[2] = (float)pow(fastSigmoid(x2_data[idx]) * 2.f, 2) * anchors[i][c * 2 + 0];
                        result[3] = (float)pow(fastSigmoid(y2_data[idx]) * 2.f, 2) * anchors[i][c * 2 + 1];
                        result[4] = fastSigmoid(conf_data[idx]);

                        float max_idx = 0;
                        float max_score = -1;
                        #pragma omp parallel for
                        for(int cls = 0; cls < classes; cls++){
                            float score = fastSigmoid(classes_data[idx + cls * stride_len]);
                            if(score > max_score){
                                max_score = score;
                                max_idx = cls;
                            }
                        }
                        result[5] = max_idx;
                        result[6] = max_score;
                        
                        temp[i].push_back(result);
                        
                    }
                }
            }
        }
    }
    for(int i = 0; i < 3; i++){
        for(int k = 0; k < temp[i].size(); k++){
            results.push_back(temp[i][k]);
        }
    }
    return results;
}

bool BoxSortDecendScore(vector<float>& box1, vector<float>& box2) {
    return box1[4] > box2[4];
}

float Yolov5PostProcess::overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float Yolov5PostProcess::box_intersection(BOX& a, BOX& b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float Yolov5PostProcess::box_iou(BOX& a, BOX& b)
{
    float inter = box_intersection(a, b);
    float b1_area = ((a.x + a.w / 2) - (a.x - a.w / 2) + 1) * ((a.y + a.h / 2) - (a.y - a.h / 2) + 1 );
    float b2_area = ((b.x + b.w / 2) - (b.x - b.w / 2) + 1) * ((b.y + b.h / 2) - (b.y - b.h / 2) + 1 );
    float u = inter / (b1_area + b2_area - inter);
    return u;
}

void Yolov5PostProcess::nms(vector<vector<float>>& boxes, vector<int>& idxes, float threshold) {
    map<int, int> idx_map;
    for (int i = 0; i < boxes.size() - 1; ++i) {
        if (idx_map.find(i) != idx_map.end()) {
            continue;
        }
        for (int j = i + 1; j < boxes.size(); ++j) {
            if (idx_map.find(j) != idx_map.end()) {
                continue;
            }
            BOX Bbox1, Bbox2;
            Bbox1.x = boxes[i][0];
            Bbox1.y = boxes[i][1];
            Bbox1.w = boxes[i][2];
            Bbox1.h = boxes[i][3];

            Bbox2.x = boxes[j][0];
            Bbox2.y = boxes[j][1];
            Bbox2.w = boxes[j][2];
            Bbox2.h = boxes[j][3];

            float iou = box_iou(Bbox1, Bbox2);
            if (iou >= threshold) {
                idx_map[j] = 1;
            }
        }
    }
    for (int i = 0; i < boxes.size(); ++i) {
        if (idx_map.find(i) == idx_map.end()) {
            idxes.push_back(i);
        }
    }
}
