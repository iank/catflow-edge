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
    bool show_image = false;

    app.add_option("model", modelweights_filename, "File containing ONNX model")
        ->required();
    app.add_option("image", image_filename, "Image to test")->required();
    app.add_flag("-s,--show-image", show_image);

    CLI11_PARSE(app, argc, argv);

    constexpr unsigned int yolov5m_n_classes = 80;
    constexpr unsigned int yolov5m_confidence = 0.5;

    // Load model
    cv::dnn::Net net;
    load_net(net, modelweights_filename);

    // Load test image
    cv::Mat frame = cv::imread(image_filename, cv::IMREAD_COLOR);
    TEST(!frame.empty(), "Image is not empty");

    // Detect
    std::vector<Detection> output;
    detect(frame, net, output, yolov5m_n_classes, yolov5m_confidence);

    if (show_image)
    {
        // Draw boxes on frame
        for (int i = 0; i < output.size(); i++)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto class_id = detection.class_id;
            auto confidence = detection.confidence;
            const auto color = cv::Scalar(255, 255, 0);
            cv::rectangle(frame, box, color, 3);

            cv::rectangle(frame, cv::Point(box.x, box.y - 20),
                          cv::Point(box.x + box.width, box.y), color,
                          cv::FILLED);

            auto label =
                std::to_string(class_id) + ": " + std::to_string(confidence);

            std::cout << class_id << ": " << confidence << std::endl;
            cv::putText(frame, label.c_str(), cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        cv::imshow("output", frame);
        cv::waitKey(0);
    }

    TEST(output.size() == 1, "Detect one object");
    TEST(output[0].class_id == 2, "Detected a car");

    std::exit(EXIT_SUCCESS);
}
