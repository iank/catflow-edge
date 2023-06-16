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
            const unsigned int n_classes);

const std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 0)};

#endif
