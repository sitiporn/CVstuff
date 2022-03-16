
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <stdlib.h>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

// point
double adDataP1[] = { 2, 4, 2 };
double adDataP2[] = { 6, 3, 3 };
double adDataP3[] = { 1, 2, 0.5 };
double adDataP4[] = { 16, 8, 4};
double adDataL[] = { 8, -4, 0};

// The four points and line as OopenCV matrices
Mat matP1(3, 1, CV_64F, adDataP1);
Mat matP2(3, 1, CV_64F, adDataP2);
Mat matP3(3, 1, CV_64F, adDataP3);
Mat matP4(3, 1, CV_64F, adDataP4);
Mat matL(3, 1, CV_64F, adDataL);

// Test whether the points are on the line or not
Mat matResult1 = matP1.t() * matL;
Mat matResult2 = matP2.t() * matL;
Mat matResult3 = matP3.t() * matL;
Mat matResult4 = matP4.t() * matL;

int main( int argc, char** argv )
{  

    if(fabs(matResult1.at<double>(0,0))< 1e-6)
    {
        cout << "Point " <<matP1.t() << "is on line " << matL.t()
            << " (P'*L = " << matResult1.at<double>(0,0) << ")" << endl;
    }
    if (fabs(matResult2.at<double>(0,0)) < 1e-6)        
    {
        cout << "Point " <<matP2.t() << "is on line " << matL.t()
            << " (P'*L = " << matResult2.at<double>(0,0) << ")" << endl;
    }
    if (fabs(matResult3.at<double>(0,0)) < 1e-6)        
    {
        cout << "Point " <<matP3.t() << "is on line " << matL.t()
            << " (P'*L = " << matResult3.at<double>(0,0) << ")" << endl;
    }
    if (fabs(matResult4.at<double>(0,0) < 1e-6))        
    {
        cout << "Point " <<matP4.t() << "is on line " << matL.t()
            << " (P'*L = " << matResult4.at<double>(0,0) << ")" << endl;
    }

}
