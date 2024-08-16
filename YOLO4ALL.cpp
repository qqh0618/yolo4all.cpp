// YOLO4ALL.cpp: 定义应用程序的入口点。
//

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

#include "YOLO4ALL.h"

using namespace std;
using namespace cv;

int main()
{
    string onnxmodel="E:/code/YOLOv8-CPP-Inference/yolov8n.onnx";
    string input_img = "E:/code/YOLOv8-CPP-Inference/0004.jpg";
    cout<<"---------"<<endl;
    dnn::Net model = dnn::readNetFromONNX(onnxmodel);
    cout<<"---------"<<endl;
    Mat original_image = cv::imread(input_img);

    int height = original_image.size().height;
    int width = original_image.size().width;
    int length = std::max(height, width);

    cv::Mat image = cv::Mat::zeros(cv::Size(length, length), CV_8UC3);
    original_image.copyTo(image(cv::Rect(0, 0, width, height)));

    int scale = length/640;

    cout<<"---------"<<endl;
    Mat blob = dnn::blobFromImage(image, 1.0/255, Size (640,640), true, false);
    model.setInput(blob);
    cout<<"----"<<endl;
    vector<Mat> outputs;
    model.forward(outputs, model.getUnconnectedOutLayersNames());
    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

}
