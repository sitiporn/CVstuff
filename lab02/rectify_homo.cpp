// An OpenCV mouse handler function has 5 parameters

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// In C++, you can define constants variable using #define
#define VIDEO_FILE "Lab01-robot-video.mp4"
#define ROTATE false
Mat matPauseScreen, matResult, matFinal;
Point point;
vector<Point> pts;
int var = 0;
int drag = 0;

// Create mouse handler function
void mouseHandler(int, int, int, int, void*);


int main(int argc, char** argv)
{
    Mat matFrameCapture;
    Mat matFrameDisplay;
    int key = -1;

    // --------------------- [STEP 1: Make video capture from file] ---------------------
    // Open input video file
    VideoCapture videoCapture(VIDEO_FILE);
    if (!videoCapture.isOpened()) {
        cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return -1;
    }

    // Capture loop
    while (key < 0)        // play video until press any key
    {
        // Get the next frame
        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty()) {   // no more frame capture from the video
            // End of video file
            break;
        }
        cvtColor(matFrameCapture, matFrameCapture, COLOR_BGR2BGRA);

        // Rotate if needed, some video has output like top go down, so we need to rotate it
#if ROTATE
        rotate(matFrameCapture, matFrameCapture, RotateFlags::ROTATE_180);   //rotate 180 degree and put the image to matFrameDisplay
#endif

        float ratio = 640.0 / matFrameCapture.cols;
        resize(matFrameCapture, matFrameDisplay, cv::Size(), ratio, ratio, INTER_LINEAR);

        // Display
        imshow(VIDEO_FILE, matFrameDisplay); // Show the image in window named "robot.mp4"
        key = waitKey(30);

        // --------------------- Good :)[STEP 2: pause the screen and show an image] ---------------------
        if (key >= 0)
        {
            matPauseScreen = matFrameCapture;  // transfer the current image to process
            matFinal = matPauseScreen.clone(); // clone image to final image
        }
    }

    // --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
    if (!matFrameCapture.empty())
    {
        var = 0;   // reset number of saving points
        pts.clear(); // reset all points
        namedWindow("Source", WINDOW_AUTOSIZE);  // create a windown named source
        setMouseCallback("Source", mouseHandler, NULL); // set mouse event handler "mouseHandler" at Window "Source"
        cout<<"Done click"<<endl;
        imshow("Source", matPauseScreen); // Show the image
        waitKey(0); // wait until press anykey
        destroyWindow("Source"); // destroy the window
    }
    else
    {
        cout << "You did not pause the screen before the video finish, the program will stop" << endl;
        return 0;
    }
    
      
    
    if (pts.size() == 4)
    {    
        cout<<"======Show pts======= :"<<endl;
    
        for (vector<Point>::const_iterator i = pts.begin(); i != pts.end(); ++i)
        {
            //cout << *i << ' ';
          //  cout << "XX"<<endl;

        }
        cout<<"====== === === ======== :"<<endl;

        Point2f src[4];
        
        for (int i = 0; i < 4; i++)
        {
            src[i].x = pts[i].x * 1.0;
            src[i].y = pts[i].y * 1.0;
        }



        Point2f reals[4];
        reals[0] = Point2f(800.0, 800.0);
        reals[1] = Point2f(1000.0, 800.0); 
        reals[2] = Point2f(1000.0, 1000.0);
        reals[3] = Point2f(800.0, 1000.0);
        // src matrix from 4 points form source images
        // corresponding 4 points in destination imgs
        Mat mat;
        for(int i=0; i < 4;i++)
        {  Mat row;
           // aggregrate before 
           double p1 =  -1.0  * pts[i].x;
           double p2 =  -1.0  * pts[i].y;
           double p3 =   pts[i].x * reals[i].x;
           double p4 =   pts[i].y * reals[i].x;

           double row_1[9] = {p1, p2, -1, 0, 0, 0, p3,  p4, reals[i].x};   

           p3 =  pts[i].x*  reals[i].y;
           p4 =  pts[i].y* reals[i].y;
           
           double row_2[9] = { 0, 0, 0, p1, p2, -1,p3, p4, reals[i].y};    

           Mat matA( 1, 9, CV_64F, row_1);
           Mat matB( 1, 9, CV_64F, row_2);
           vconcat(matA, matB, row);
           cout<<"Row matrix :"<<row<<endl;
           mat.push_back(row);
          // vconcat(mat,row, mat);
        }

        cout<<" Mat :  "<<mat<<endl;
        // To do before Doing SVD  
        //  A'@ A => A
        //  svd (A)  -> select 
        //  
        Mat mat_T = mat.clone().t();
        cout<<"Mat transpose "<<mat_T<<endl;

        cout<<"Original Mat "<<mat<<endl;

        Mat A = mat_T * mat;
        cout<<"After Dot product :"<<A<<endl;
        cv::SVD svdA(A, cv::SVD::FULL_UV );
        /*cout << "U:" << endl << svdA.u << endl;
        cout << "W:" << endl << svdA.w << endl;
        */
       
        

        cout << "Vt:" << endl << svdA.vt << endl;
        cv::Size s = mat.size();
        int rows = svdA.vt.rows;
        int cols = svdA.vt.cols;
        
        cout<<"rows :"<<rows<<endl;
        cout<<"Column :"<<cols<<endl;
        // Mat homo_row[9] = svdA.vt.row(8);
        Mat h(1,9, CV_64F); 
        h = svdA.vt.row(8);
        
        cout<<"h:"<<h<<endl;
        
        cout<<"Point p1' :"<<reals[0]<<endl;
       
        double p0_x = 1.0 * pts[0].x;
        double p0_y = 1.0 * pts[0].y; 
        double p_click[3] = {p0_x,p0_y,1.0};

        
        Mat mat_click(3, 1,CV_64F,p_click);
        Mat homo = h.reshape (3,3);
        cout<<"h"<<endl;
        cout<<"rows:"<<homo.rows<<endl;
        cout<<"cols:"<<homo.cols<<endl;
        cout<<"1st P click:"<<mat_click<<endl;
        cout<<"Homo graphy:"<<homo*mat_click<<endl;
        cout<<"Real Point :"<<reals[0]<<endl;

        //cout<<"row ; H*x :"<<homo_row.dot(mat_click)<<endl;
            
        //Mat homography_matrix = getPerspectiveTransform(src, reals);
        /*std::cout << "Estimated Homography Matrix is:" << std::endl;
        std::cout << homography_matrix << std::endl;

        / perspective transform operation using transform matrix
        
        cv::warpPerspective(matPauseScreen, matResult, homography_matrix, matPauseScreen.size(), cv::INTER_LINEAR);
        imshow("Source", matPauseScreen);
        imshow("Result", matResult);
        
        waitKey(0); */
    }

    return 0;
}

// An OpenCV mouse handler function has 5 parameters

void mouseHandler(int event, int x, int y, int, void*)
{
    if (var >= 4) // If we already have 4 points, do nothing
        return;
    if (event == EVENT_LBUTTONDOWN) // Left button down
    {
        drag = 1; // Set it that the mouse is in pressing down mode
        matResult = matFinal.clone(); // copy final image to draw image
        point = Point(x, y); // memorize current mouse position to point var
        if (var >= 1) // if the point has been added more than 1 points, draw a line
        {
            line(matResult, pts[var - 1], point, Scalar(0, 255, 0, 255), 2); // draw a green line with thickness 2
        }
        circle(matResult, point, 2, Scalar(0, 255, 0), -1, 8, 0); // draw a current green point
        imshow("Source", matResult); // show the current drawing
    }
    if (event == EVENT_LBUTTONUP && drag) // When Press mouse left up
    {
        drag = 0; // no more mouse drag
        pts.push_back(point);  // add the current point to pts
        var++; // increase point number
        matFinal = matResult.clone(); // copy the current drawing image to final image
        if (var >= 4) // if the homograpy points are done
        {
            line(matFinal, pts[0], pts[3], Scalar(0, 255, 0, 255), 2); // draw the last line
            fillPoly(matFinal, pts, Scalar(0, 120, 0, 20), 8, 0); // draw polygon from points

            setMouseCallback("Source", NULL, NULL); // remove mouse event handler
        }
        imshow("Source", matFinal);
    }
    if (drag) // if the mouse is dragging
    {
        matResult = matFinal.clone(); // copy final images to draw image
        point = Point(x, y); // memorize current mouse position to point var
        if (var >= 1) // if the point has been added more than 1 points, draw a line
        {
            line(matResult, pts[var - 1], point, Scalar(0, 255, 0, 255), 2); // draw a green line with thickness 2
        }
        circle(matResult, point, 2, Scalar(0, 255, 0), -1, 8, 0); // draw a current green point
        imshow("Source", matResult); // show the current drawing
    }
}
