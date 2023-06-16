#include "yolo.h"
#include <iostream>
#include <string>

#define TEST(condition, message)                                               \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            std::cerr << "Test " << message << " failed: `" #condition "`"     \
                      << std::endl;                                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (false)

int main(int argc, char **argv)
{
    constexpr unsigned int YOLOV5M_N_CLASSES = 80;
    TEST(argc == 3, "catflow-edge-test image.png model.onnx");

    // Load model
    cv::dnn::Net net;
    load_net(net, argv[2]);

    // Load test image
    cv::Mat frame = cv::imread(argv[1], cv::IMREAD_COLOR);
    TEST(!frame.empty(), "Image is not empty");

    // Detect
    std::vector<Detection> output;
    detect(frame, net, output, YOLOV5M_N_CLASSES);

    TEST(output.size() == 1, "Detect one object");
    TEST(output[0].class_id == 2, "Detected a car");

    std::exit(EXIT_SUCCESS);
}
