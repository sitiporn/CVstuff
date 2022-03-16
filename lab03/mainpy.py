import cv2
import numpy as np

VIDEO_FILE = "robot.mp4"
HOMOGRAPHY_FILE = "robot-homography.yml"

matResult = None
matFinal = None
matPauseScreen = None

point = (-1, -1)
pts = []
var = 0 
drag = 0

# Mouse handler function has 5 parameters input (no matter what)
def mouseHandler(event, x, y, flags, param):
    global point, pts, var, drag, matFinal, matResult

    if (var >= 4):                           # if homography points are more than 4 points, do nothing
        return
    if (event == cv2.EVENT_LBUTTONDOWN):     # When Press mouse left down
        drag = 1                             # Set it that the mouse is in pressing down mode
        matResult = matFinal.copy()          # copy final image to draw image
        point = (x, y)                       # memorize current mouse position to point var
        if (var >= 1):                       # if the point has been added more than 1 points, draw a line
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8, 0)             # draw a current green point
        cv2.imshow("Source", matResult)      # show the current drawing
    if (event == cv2.EVENT_LBUTTONUP and drag):  # When Press mouse left up
        drag = 0                             # no more mouse drag
        pts.append(point)                    # add the current point to pts
        var += 1                             # increase point number
        matFinal = matResult.copy()          # copy the current drawing image to final image
        if (var >= 4):                                                      # if the homograpy points are done
            cv2.line(matFinal, pts[0], pts[3], (0, 255, 0, 255), 2)   # draw the last line
            cv2.fillConvexPoly(matFinal, np.array(pts, 'int32'), (0, 120, 0, 20))        # draw polygon from points
        cv2.imshow("Source", matFinal);
    if (drag):                                    # if the mouse is dragging
        matResult = matFinal.copy()               # copy final images to draw image
        point = (x, y)                            # memorize current mouse position to point var
        if (var >= 1):                            # if the point has been added more than 1 points, draw a line
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8, 0)         # draw a current green point
        cv2.imshow("Source", matResult)           # show the current drawing

def main():
    global matFinal, matResult, matPauseScreen
    key = -1;

    videoCapture = cv2.VideoCapture(VIDEO_FILE)
    if not videoCapture.isOpened():
        print("ERROR! Unable to open input video file ", VIDEO_FILE)
        return -1

    width  = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # Capture loop 
    while (key < 0):
        # Get the next frame
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:
            # End of video file
            break

        ratio = 640.0 / width
        dim = (int(width * ratio), int(height * ratio))
        matFrameDisplay = cv2.resize(matFrameDisplay, dim)

        cv2.imshow(VIDEO_FILE, matFrameDisplay)
        key = cv2.waitKey(30)

        if (key >= 0):
            cv2.destroyWindow(VIDEO_FILE)
            matPauseScreen = matFrameCapture
            matFinal = matPauseScreen.copy()
            cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("Source", mouseHandler)
            cv2.imshow("Source", matPauseScreen)
            cv2.waitKey(0)
            cv2.destroyWindow("Source")

            if (len(pts) < 4):
                return

            src = np.array(pts).astype(np.float32)
            reals = np.array([(800, 800),
                                (1000, 800),
                                (1000, 1000),
                                (800, 1000)], np.float32)

            homography_matrix = cv2.getPerspectiveTransform(src, reals);
            print("Estimated Homography Matrix is:")
            print(homography_matrix)

            h, w, ch = matPauseScreen.shape
            matResult = cv2.warpPerspective(matPauseScreen, homography_matrix, (w, h), cv2.INTER_LINEAR)
            matPauseScreen = cv2.resize(matPauseScreen, dim)
            cv2.imshow("Source", matPauseScreen)
            matResult = cv2.resize(matResult, dim)
            cv2.imshow("Result", matResult)

            cv2.waitKey(0)

main()
