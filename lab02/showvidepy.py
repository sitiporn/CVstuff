import cv2
import numpy as np
import sys

VIDEO_FILE = 'robot.mp4'
ROTATE = False

if __name__ == '__main__':

    key = -1;

    # Open input video file
    videoCapture = cv2.VideoCapture(VIDEO_FILE);
    if not videoCapture.isOpened():
        print('Error: Unable to open input video file', VIDEO_FILE)
        sys.exit('Unable to open input video file')

    width  = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # Capture loop 
    while (key != ord(' ')):        # play video until user presses <space>
        # Get the next frame
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:
            # End of video
            break

        # Rotate if needed
        if ROTATE:
            _, matFrameDisplay = cv2.rotate(matFrameCapture, cv2.ROTATE_180)
        else:
            matFrameDisplay = matFrameCapture;

        ratio = 480.0 / height
        dim = (int(width * ratio), int(height * ratio))
        # resize image to 480p for display
        matFrameDisplay = cv2.resize(matFrameDisplay, dim)

        # Show the image in window named "robot.mp4"
        cv2.imshow(VIDEO_FILE, matFrameDisplay)
        key = cv2.waitKey(30)
