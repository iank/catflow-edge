# catflow-edge

YOLOv5 inference in C++ using ONNX

## Set-up

This repository uses [pre-commit](https://pre-commit.com/#install), install it and run `pre-commit install` after cloning.

## Prerequisites

```
apt install cmake g++ libopencv-dev make clang-tidy clang-format
```

clang-tidy and clang-format should be relatively recent. I use `-17` from [apt.llvm.org](https://apt.llvm.org/).

I used OpenCV 4.7.0. The OpenCV 4.2.0 package distributed with Ubuntu 20.04 could not load my model. The OpenCV 4.5.4 package distributed with Ubuntu 22.04 can load the model if it is [exported with `--opset 12`](https://github.com/ultralytics/yolov5/issues/10665).

## Building

```
mkdir build && cd build
cmake ..
make
ARGS=--output-on-failure make test
```

## Attribution

Derived from https://github.com/doleron/yolov5-opencv-cpp-python/
