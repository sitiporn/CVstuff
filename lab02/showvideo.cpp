#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// In C++, you can define constants variable using #define
#define VIDEO_FILE "Lab01-robot-video.mp4"
#define ROTATE false

int main(int argc, char** argv)
{   
    int terminate_flag = 0;
    Mat matFrameCapture;
    Mat matFrameDisplay;
    int iKey = -1;
    int count_frame = 0;

    //Open input video file
    VideoCapture videoCapture(VIDEO_FILE);

    if (!videoCapture.isOpened()) {
        cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return -1;
    }
    
    while(true)
    {   
        count_frame++;

        videoCapture.read(matFrameCapture);

        if(matFrameCapture.empty())
        {
            break;
        }

        // we can rotate image
#if ROTATE
    rotate(matFrameCapture, matFrameDisplay, RotateFlags:: ROTATE_180); //rotate 180 degree
#else
    matFrameDisplay = matFrameCapture;
#endif
        float ratio = 480.0 / matFrameDisplay.rows;
        resize(matFrameDisplay, matFrameDisplay, cv::Size(), ratio, ratio, INTER_LINEAR);  //resize image to 480p for showing

        //Display
        // frame number, total frames, and user control actions.
        string text = "Press spacebar to next frame or q to exit :"s + "Current frames :"s + to_string(count_frame);
        namedWindow("R obot",WINDOW_GUI_EXPANDED);
        displayOverlay("Robot",text,10000); 
        // which flags to set up resizable but keep aspect ratio display 
        // and expanded GUI
      
        
        imshow("Robot", matFrameDisplay);
        //iKey = waitKey(30);
        // wait to press space to skip next to advance frame
        // press q to terminate
        while(true)
        {   
          
            iKey = waitKey(30);
            
            if(iKey == int(' '))
            {
                break;
            }
            
            if(iKey == int('q'))
            {
               return 0;
            }
            
        }
       
        

    }
    return 0;
}

//     // Capture loop
//     while (iKey != int(' '))        // play video until user presses <space>
//     {
//         // Get the next frame
//         videoCapture.read(matFrameCapture);
//         if (matFrameCapture.empty())
//         {
//             // End of video file
//             break;
//         }

//         // We can rotate the image easily if needed.
// #if ROTATE iKey = waitKey(30);
//         matFrameDisplay = matFrameCapture;
// #endif

//         float ratio = 480.0 / matFrameDisplay.rows;
//         resize(matFrameDisplay, matFrameDisplay, cv::Size(), ratio, ratio, INTER_LINEAR); // resize image to 480p for showing

//         // Display
//         imshow(VIDEO_FILE, matFrameDisplay); // Show the image in window named "robot.mp4"
//         iKey = waitKey(30); // Wait 30 ms to give a realistic playback speed

