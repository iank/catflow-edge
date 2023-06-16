#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "yolo.h"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: catflow-edge model.onnx classes.txt video.mp4"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::string> class_list = load_class_list(argv[2]);

    cv::Mat frame;
    cv::VideoCapture capture(argv[3]);
    if (!capture.isOpened())
    {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    cv::dnn::Net net;
    load_net(net, argv[1]);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;

    while (capture.read(frame))
    {
        // Pass frame through model
        std::vector<Detection> output;
        detect(frame, net, output, class_list.size());

        int detections = output.size();

        // Draw boxes on frame
        for (int i = 0; i < detections; i++)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(frame, box, color, 3);

            cv::rectangle(frame, cv::Point(box.x, box.y - 20),
                          cv::Point(box.x + box.width, box.y), color,
                          cv::FILLED);
            cv::putText(frame, class_list[classId].c_str(),
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
    }

    std::exit(EXIT_SUCCESS);
}
