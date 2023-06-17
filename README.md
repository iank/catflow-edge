# catflow-edge

YOLOv5 inference in C++ using ONNX.

I created this to test performance of CPU inference on a potential target platform and am setting it down for now.

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

## Cross-compilation

### OpenCV

[OpenCV instructions](https://docs.opencv.org/4.x/d0/d76/tutorial_arm_crosscompile_with_cmake.html).

For example:

```
cmake -DCMAKE_TOOLCHAIN_FILE=platforms/linux/arm-gnueabi.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/../../prefix/usr/local \
    -DINSTALL_C_EXAMPLES=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_calib3d=OFF \
    -DCMAKE_SHARED_LINKER_FLAGS='-latomic'
    ..
```

### catflow-edge

```cmake -DOPENCV_DIR=../prefix/ -DCMAKE_TOOLCHAIN_FILE=cmake/arm-toolchain.cmake ..`

## Attribution

This is a cleaned up and packaged version of code derived from https://github.com/doleron/yolov5-opencv-cpp-python/
