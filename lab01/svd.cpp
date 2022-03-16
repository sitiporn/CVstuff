
// C++


#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    double adData[] = { 3, 2, 4, 8, 4, 2, 1, 3, 2 };
    cv::Mat matA( 3, 3, CV_64F, adData );
    cout << "A:" << endl << matA << endl;
    cv::SVD svdA( matA, SVD::FULL_UV );
    cout << "U:" << endl << svdA.u << endl;
    cout << "W:" << endl << svdA.w << endl;
    cout << "Vt:" << endl << svdA.vt << endl;
    return 0;
}