
#include <opencv2/opencv.hpp>
#include <iostream>
#include "HomographyData.h"

using namespace cv;
using namespace std;

#define VIDEO_FILE "robot.mp4"
#define HOMOGRAPHY_FILE "robot-homography.yml"

void displayFrame(cv::Mat& matFrameDisplay, int iFrame, int cFrames, HomographyData* pHomographyData) {
    for (int i = 0; i < pHomographyData->cPoints; i++) {
        cv::circle(matFrameDisplay, pHomographyData->aPoints[i], 10, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
    }
    imshow(VIDEO_FILE, matFrameDisplay);
    stringstream ss;
    ss << "Frame " << iFrame << "/" << cFrames;
    ss << ": hit <space> for next frame or 'q' to quit";
    //cv::displayOverlay(VIDEO_FILE, ss.str(), 0);  // for linux + qt
    putText(matFrameDisplay, ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 3);
}

int main()
{
    cv::Mat matFrameCapture;
    cv::Mat matFrameDisplay;
    int cFrames;
    HomographyData homographyData;

    cv::VideoCapture videoCapture(VIDEO_FILE);
    if (!videoCapture.isOpened()) {
        cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return -1;
    }
    cFrames = (int)videoCapture.get(cv::CAP_PROP_FRAME_COUNT);

    // Create a named window that will be used later to display each frame
    cv::namedWindow(VIDEO_FILE, (unsigned int)cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);

    // Read homography from file
    if (!readHomography(HOMOGRAPHY_FILE, &homographyData)) {
        cerr << "ERROR! Unable to read homography data file " << HOMOGRAPHY_FILE << endl;
        return -1;
    }

    int iFrame = 0;
    while (true) {

        // Block for next frame

        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty()) {
            // End of video file
            break;
        }

        displayFrame(matFrameCapture, iFrame, cFrames, &homographyData);

        int iKey;
        do {
            iKey = cv::waitKey(10);
            if (getWindowProperty(VIDEO_FILE, cv::WND_PROP_VISIBLE) == 0) {
                return 0;
            }
            if (iKey == (int)'q' || iKey == (int)'Q') {
                return 0;
            }
        } while (iKey != (int)' ');
        iFrame++;
    }

    return 0;
}

