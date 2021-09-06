#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <map>
#include <algorithm>

struct BOX{
  float x;
  float y;
  float w;
  float h;
};

using namespace std;
bool BoxSortDecendScore(vector<float>& box1, vector<float>& box2);
class Yolov5PostProcess {
public:
    Yolov5PostProcess();
    void run(int _img_size, int _classes, float _obj_conf, float _nms_threshold, void *stride_32_data, void *stride_16_data, void *stride_8_data, vector<vector<float>>& results);
private:
    vector<vector<float>> ThresholdFliter(float **data, float threahold);
    void nms(vector<vector<float>>& result, vector<int>& idxes,float threshold);
    float fastSigmoid(float x);

    float overlap(float x1, float w1, float x2, float w2);
    float box_intersection(BOX& a, BOX& b);
    float box_iou(BOX& a, BOX& b);
    float obj_conf;
    float nms_threshold;
    int img_size;
    int classes;
    float anchors[3][6] = {{10.f, 13.f, 16.f, 30.f, 33.f, 23.f} ,{30.f, 61.f, 62.f, 45.f, 59.f, 119.f} ,{116.f, 90.f, 156.f, 198.f, 373.f, 326.f}}; //8, 16, 32
};