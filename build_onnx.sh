export AOCL_LIB=/root/aocl-rte-18.1.0-625.arm32/host/arm32/lib
export AOCL_INC=/root/aocl-rte-18.1.0-625.arm32/host/include
export AOCL_BOARD_LIB=/root/aocl-rte-18.1.0-625.arm32/board/c5soc/arm32/lib

g++  main.cpp -o pipeonnx -Wl,--no-as-needed \
    -std=c++17 -O3 -DNDEBUG -DUSE_ACL -Wall -Wno-unknown-pragmas \
	-I$AOCL_INC \
	-L$AOCL_LIB \
	-lOpenCL \
	-L/root/aocl-rte-18.1.0-625.arm32/host/arm32/lib/ -lalteracl -lacl_emulator_kernel_rt\
    -L/root/aocl-rte-18.1.0-625.arm32/board/c5soc/arm32/lib/ -lintel_soc32_mmd \
    -lelf -lstdc++ \
    -I/usr/include \
    -I/usr/include/opencv4 \
    -I/usr/local/include/onnxruntime \
    -I/usr/local/include/onnxruntime/core/session \
    -I/usr/local/include/ArmComputeLibrary/include \
    -I/usr/local/include/ArmComputeLibrary/support \
    -I/usr/local/include/half \
    -L/usr/lib/arm-linux-gnueabihf \
    -L/usr/local/lib \
    -lonnxruntime \
    -lonnxruntime_providers_shared \
    -larm_compute -larm_compute_core -larm_compute_graph \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
	-lpthread
