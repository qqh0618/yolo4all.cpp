# CMakeList.txt: YOLO4ALL 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project ("YOLOv8-CPP-Inference")
set(OpenCV_DIR "E:/software/opencv4.9/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

# 将源代码添加到此项目的可执行文件。
add_executable (YOLO4ALL "YOLO4ALL.cpp" "YOLO4ALL.h")

# TODO: 如有需要，请添加测试并安装目标。

target_link_libraries(YOLO4ALL ${OpenCV_LIBS})