
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <stdlib.h>
#include <vector>

using namespace cv;
using namespace std;

bool is_on_the_line(Mat arg_line,Mat& arg_point)
{
    bool is_line = false;
    //arg_line = arg_line.t();
    cout<<"##################"<<endl;

    // cout<<arg_line<<endl;
    // cout<<arg_point<<endl;

    if (abs(arg_line.dot(arg_point))<= 1e-6)
    {
        cout <<arg_point<<endl<<"is On the Line"<<endl;
        is_line = true;
        
    } 
    else
        cout <<arg_point<<endl <<"is Not on the Line"<<endl;
      
    return is_line;
}

int main( int argc, char** argv )
{  
   // point
   double p[][3] = {{2, 4, 2},{6, 3, 3},{1, 2, 0.5},{16, 8, 4}};
   // line
   double line[] = {8, -4, 0};
   // convert arry to matrix
   cv::Mat *vpoint = new Mat[4];
   vector<int> onLine;

   cv::Mat matline( 3, 1, CV_64F, line);
   for(int i =0;i<4;i++)
   {
       vpoint[i] = Mat( 3, 1, CV_64F, p[i]);
    //    cout <<vpoint[i]<<endl;

       bool on_the_line = is_on_the_line(matline,vpoint[i]);

               
   }

  
delete [] vpoint;
vpoint = NULL; 
  
//    cv::Mat vect_line( 3, 1, CV_64F, line);
//    cv::Mat *points[4] = {vect_p1,vect_p2,vect_p3,vect_p4}



}