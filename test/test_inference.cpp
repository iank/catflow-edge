#include <iostream>
#include <string>

#include "CLI11.hpp"
#include "yolo.h"

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
    CLI::App app{"YOLOv5 inference test"};
    std::string modelweights_filename = "";
    std::string image_filename = "";

    app.add_option("model", modelweights_filename, "File containing ONNX model")
        ->required();
    app.add_option("image", image_filename, "Image to test")->required();

    CLI11_PARSE(app, argc, argv);

    constexpr unsigned int YOLOV5M_N_CLASSES = 80;
    constexpr unsigned int YOLOV5M_CONFIDENCE = 0.5;

    // Load model
    cv::dnn::Net net;
    load_net(net, modelweights_filename);

    // Load test image
    cv::Mat frame = cv::imread(image_filename, cv::IMREAD_COLOR);
    TEST(!frame.empty(), "Image is not empty");

    // Detect
    std::vector<Detection> output;
    detect(frame, net, output, YOLOV5M_N_CLASSES, YOLOV5M_CONFIDENCE);

    TEST(output.size() == 1, "Detect one object");
    TEST(output[0].class_id == 2, "Detected a car");

    std::exit(EXIT_SUCCESS);
}
