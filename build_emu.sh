#!/bin/bash

# Compilacion para x86_64 (Emulador)
g++ main.cpp -o pipeonnx_emu -Wl,--no-as-needed \
    -std=c++17 -O3 -DNDEBUG -Wall -Wno-unknown-pragmas \
    -I$INTELFPGAOCLSDKROOT/host/include \
    -L$INTELFPGAOCLSDKROOT/host/linux64/lib \
    -lOpenCL -lalteracl -lacl_emulator_kernel_rt \
    -lelf -lstdc++ \
    -I/usr/include \
    -I/usr/include/opencv4 \
    -I/usr/local/include/onnxruntime \
    -I/usr/local/include/onnxruntime/core/session \
    -L/usr/local/lib \
    -lonnxruntime \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_dnn \
    -lpthread