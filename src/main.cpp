#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "CLI11.hpp"
#include "yolo.h"

const std::vector<cv::Scalar> CLASS_COLORS = {
    cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 0)};

int main(int argc, char **argv)
{
    CLI::App app{"Run YOLOv5 inference on a video"};
    std::string classlist_filename = "";
    std::string modelweights_filename = "";
    std::string video_filename = "";
    float confidence_threshold = 0.5;
    bool headless = false;
    bool camera = false;

    app.add_option("model", modelweights_filename, "File containing ONNX model")
        ->required();
    app.add_option("classes", classlist_filename,
                   "A .txt file containing one class name per line")
        ->required();
    app.add_option("video", video_filename,
                   "Video filename or camera index (-W)")
        ->required();

    app.add_flag("-n,--no-gui", headless, "Don't open a GUI window");
    app.add_flag("-w,--camera", camera, "Video argument is a camera index");
    app.add_option("-c,--confidence", confidence_threshold,
                   "Confidence threshold");

    CLI11_PARSE(app, argc, argv);

    std::vector<std::string> class_list = load_class_list(classlist_filename);

    // Open video file or camera
    cv::VideoCapture capture;
    try
    {
        if (camera)
        {
            int camera_index = std::stoi(video_filename);
            capture.open(camera_index);
            capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        }
        else
        {
            capture.open(video_filename);
        }
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Invalid camera index: " << video_filename
                  << ". The camera index must be an integer." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch (const std::out_of_range &e)
    {
        std::cerr << "Camera index out of range: " << video_filename
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (!capture.isOpened())
    {
        std::cerr << "Error opening camera or video file" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cv::dnn::Net net;
    load_net(net, modelweights_filename);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;

    cv::Mat frame;
    while (capture.read(frame))
    {
        // Pass frame through model
        std::vector<Detection> output;
        detect(frame, net, output, class_list.size(), confidence_threshold);

        int detections = output.size();

        // FPS timer
        frame_count++;
        if (frame_count >= 5)
        {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 /
                  std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                        start)
                      .count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            std::cout << fps_label_str << std::endl;
        }

        // Skip GUI code below if headless flag given
        if (headless)
            continue;

        // Draw boxes on frame
        for (int i = 0; i < detections; i++)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto class_id = detection.class_id;
            const auto color = CLASS_COLORS[class_id % CLASS_COLORS.size()];
            cv::rectangle(frame, box, color, 3);

            cv::rectangle(frame, cv::Point(box.x, box.y - 20),
                          cv::Point(box.x + box.width, box.y), color,
                          cv::FILLED);
            cv::putText(frame, class_list[class_id].c_str(),
                        cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 0));
        }

        // Display frame
        cv::imshow("output", frame);
        if (cv::waitKey(1) != -1)
        {
            capture.release();
            break;
        }
    }

    std::exit(EXIT_SUCCESS);
}
