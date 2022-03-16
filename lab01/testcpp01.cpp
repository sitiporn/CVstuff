#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
    cout << "Hello World!" << endl;
    Mat image = Mat::zeros(300, 600, CV_8UC3);
    circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
    circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
    imshow("Display Window", image);
    waitKey(0);
    return 0;
}
