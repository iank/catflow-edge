#ifndef YOLO_H
#define YOLO_H

#include <opencv2/opencv.hpp>

std::vector<std::string> load_class_list(std::string filename);
void load_net(cv::dnn::Net &net, std::string weights);

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output,
            const unsigned int n_classes, float confidence_threshold);

#endif
