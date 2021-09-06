#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/types_c.h>

#include "KalmanFilter/tracker.h"
#include "AclProcess/AclYolov5Process.h"
#include "AclProcess/AclFeatureProcess.h"

//Ascend parameter
const int device_id = 0;
const int inputShape = 640;
const int classNum = 80;
const float obj_threshold = 0.4;
const float nms_threshold = 0.45;

//Deep SORT parameter
const int nn_budget = 50;
const float max_cosine_distance = 0.2;

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame);

int main(int argc, char **argv)
{
    struct timeval start;
    struct timeval end;
    struct timeval g_start;
    struct timeval g_end;
    
    aclError ret = aclInit(nullptr); // Initialize ACL
    if (ret != ACL_ERROR_NONE)
    {
        cout << "Failed to init acl, ret = " << ret << endl;
        return ret;
    }
    aclrtContext context;
    ret = aclrtCreateContext(&context, device_id);
    if (ret != ACL_ERROR_NONE)
    {
        cout << "Failed to set current context, ret = " << ret << endl;
        return ret;
    }
    //deep SORT
    tracker mytracker(max_cosine_distance, nn_budget);

    //yolo
    AclYolov5Process aclYolov5Process;
    aclYolov5Process.Init(context, argv[1], inputShape, classNum, obj_threshold, nms_threshold);

    //feature extraction
    AclFeatureProcess aclFeatureProcess;
    aclFeatureProcess.Init(context, argv[2]);

    // Open a video file or an image file or a camera stream.
    std::string outputFile = "deepsort_result.avi";
    cv::VideoCapture cap;
    cv::VideoWriter video;
    cv::Mat frame;

    try
    {
        cap.open(argv[3]);
    }
    catch (...)
    {
        std::cout << "Could not open the input image/video stream" << std::endl;
        return 0;
    }

    // Get the video writer initialized to save the output video

    video.open( outputFile, 
                cv::VideoWriter::fourcc('H', '2', '6', '4'), 
                25.0,
                cv::Size(
                    static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), 
                    static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                    )
                );
    

    // Create a window
    //static const std::string kWinName = "Multiple Object Tracking";
    //namedWindow(kWinName, cv::WINDOW_NORMAL);

    // Process frames.
    //while (cv::waitKey(1) < 0)
    while (true)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty())
        {
            std::cout << "Done processing !!!" << std::endl;
            std::cout << "Output file is stored as " << outputFile << std::endl;
            //cv::waitKey(3000);
            break;
        }
        //yolo infer
        DETECTIONS detections;
        gettimeofday(&g_start,NULL);
        gettimeofday(&start,NULL);
        aclYolov5Process.Process(frame, detections);
        gettimeofday(&end,NULL);
        cout<<"yolov5 infer time:"<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec) / 1000.0 <<"ms"<<endl;
        std::cout << "Detections size:" << detections.size() << std::endl;

        gettimeofday(&start,NULL);
        aclFeatureProcess.Process(frame, detections);
        gettimeofday(&end,NULL);
        cout<<"feature infer time:"<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec) / 1000.0 <<"ms"<<endl;

        gettimeofday(&start,NULL);
        mytracker.predict();
        gettimeofday(&end,NULL);
        cout<<"predict time:"<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec) / 1000.0 <<"ms"<<endl;
        
        gettimeofday(&start,NULL);
        mytracker.update(detections);
        gettimeofday(&end,NULL);
        cout<<"update time:"<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec) / 1000.0 <<"ms"<<endl;
        gettimeofday(&g_end,NULL);
        cout <<"fps:" << 1000.0 / ((g_end.tv_sec-g_start.tv_sec)*1000+(g_end.tv_usec-g_start.tv_usec) / 1000.0) << endl;
        cout << "===============================" << endl;
        system( "clear" );
        
        std::vector<RESULT_DATA> result;
        for (Track &track : mytracker.tracks)
        {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
        }

        for (unsigned int k = 0; k < detections.size(); k++)
        {
            DETECTBOX tmpbox = detections[k].tlwh;
            cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 4);

            for (unsigned int k = 0; k < result.size(); k++)
            {
                DETECTBOX tmp = result[k].second;
                cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);

                std::string label = cv::format("%d", result[k].first);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
            }
        }

        // Write the frame with the detection boxes
        cv::Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);

        //imshow(kWinName, frame);
    }
    ret = aclrtDestroyContext(context);
    if (ret != ACL_ERROR_NONE) {
        cout << "Destroy Context faild, ret = " << ret <<endl;
    }
    cout << "Destroy Context successfully" << endl;
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to deinit acl, ret = " << ret <<endl;
    }
    cout << "acl deinit successfully" << endl;
    cap.release();
    video.release();
    return 0;
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}