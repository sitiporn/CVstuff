import  numpy as np
import cv2
import numpy as np
import sys

VIDEO_FILE = 'Lab01-robot-video.mp4'
ROTATE = False

def mouseHandler(event, x, y, flags, param):

    global point, pts, var, drag, matFinal, matResult   # call global variable to use in this function

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
        point = (x, y)                   # memorize current mouse position to point var
        if (var >= 1):                            # if the point has been added more than 1 points, draw a line
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8, 0)         # draw a current green point
        cv2.imshow("Source", matResult)           # show the current drawing

if __name__ == '__main__':
    

    global matFinal, matResult, matPauseScreen 
    point = (-1, -1)
    pts = []
    var = 0 
    drag = 0

    key = -1;

    # --------------------- [STEP 1: Make video capture from file] ---------------------
    # Open input video file
    videoCapture = cv2.VideoCapture(VIDEO_FILE);
   
    if not videoCapture.isOpened():
        print("ERROR! Unable to open input video file ", VIDEO_FILE)
        sys.exit('Unable to open input video file')

    width  = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # Capture loop 
    while (key < 0):        # play video until press any key
        # Get the next frame
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:   # no more frame capture from the video
            # End of video file
            break

        # Rotate if needed, some video has output like top go down, so we need to rotate it
        if ROTATE:
            _, matFrameDisplay = cv2.rotate(matFrameCapture, cv2.ROTATE_180)   #rotate 180 degree and put the image to matFrameDisplay
        else:
            matFrameDisplay = matFrameCapture;

        ratio = 640.0 / width
        dim = (int(width * ratio), int(height * ratio))
        # resize image to 480 * 640 for showing
        matFrameDisplay = cv2.resize(matFrameDisplay, dim)

        # Show the image in window named "robot.mp4"
        cv2.imshow(VIDEO_FILE, matFrameDisplay)
        key = cv2.waitKey(30)

        # --------------------- [STEP 2: pause the screen and show an image] ---------------------
        if (key >= 0):
            matPauseScreen = matFrameCapture     # transfer the current image to process
            matFinal = matPauseScreen.copy()     # copy image to final image

    # --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
    if (matFrameCapture is not None):
        var = 0                                             # reset number of saving points
        pts.clear()                                         # reset all points
        cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)      # create a windown named source
        cv2.setMouseCallback("Source", mouseHandler)        # set mouse event handler "mouseHandler" at Window "Source"
        cv2.imshow("Source", matPauseScreen)                # Show the image
        cv2.waitKey(0)                                      # wait until press anykey
        cv2.destroyWindow("Source")                         # destroy the window
    else:
        print("No pause before end of video finish. Exiting.")

    if (len(pts) == 4):
        src = np.array(pts).astype(np.float32)

        reals = np.array([(800, 800),
                          (1000, 800),
                          (1000, 1000),
                          (800, 1000)], np.float32)
        row_coef = []


        for i in range(4):
       
           p1 =  -1.0  * pts[i][0]
           p2 =  -1.0  * pts[i][1]
           p3 =   pts[i][0] * reals[i][0]
           p4 =   pts[i][1] * reals[i][0]


           row_coef.append([p1, p2, -1, 0, 0, 0, p3,  p4, reals[i][0]])
            
            
           p3 =  pts[i][0]*  reals[i][1]
           p4 =  pts[i][1]* reals[i][1]
           
           row_coef.append([0, 0, 0, p1, p2, -1,p3, p4, reals[i][1]])    
           


        A  = np.asarray(row_coef)
        
        print("====Design Matrix====")
        print(A)

        w, u, vt = cv2.SVDecomp(A) 

        ## slice the last column of v
        ## so select the row of vt
        
        homography_matrix = vt[-1,:]

        homography_matrix = homography_matrix.reshape((3,3))

        homography_matrix = homography_matrix / homography_matrix[-1,-1]

        

        #homography_matrix = cv2.getPerspectiveTransform(src, reals);
        print("Estimated Homography Matrix is:")
        print(homography_matrix)

        # perspective transform operation using transform matrixi

        p0_x = 1.0 * pts[0][0];
        p0_y = 1.0 * pts[0][1]; 
        p0 = np.array([p0_x,p0_y,1.0]).reshape(3,1)
         
        print("P click :",p0)
        res = homography_matrix @ p0
        res = res/res[-1]
        print("Prime hat:",res)
        print("P_prime:",reals[0])
        h, w, ch = matPauseScreen.shape
        
        # Doing bilnear interpolation
        # 1) loop through to every position of pause image same as click image
        # 2) phat_pos <= H_inv @ x
        
        print("====Wrap image=====")

        print(matPauseScreen.shape)
        #y ~ heigh , x ~ width
        #(1080, 1920, 3)
       
        y_size = h
        x_size = w
        h_inv =  np.linalg.pinv(homography_matrix)  
         
        target_img = np.copy(matPauseScreen)
        print("shape:",x_size)
        for y_idx in range(y_size):
            for x_idx in range(x_size):
            
               p_prime = [x_idx, y_idx,1]

               p = h_inv @ p_prime
               # interpolate 
               p = p / p[-1]
               # let's say (2.7,3.5) 
               # x 2 <---> 3
               # y 3 <---> 4
               if p[0] < 0 :
                  p[0] = -1 * p[0]
               
               if p[1] < 0:
                  p[1] = -1 * p[1]


               if p[1] >= h:
                   p[1] = h-1
               if p[0] >= w:
                   p[0] = w -1
        

               up_bound =  int(p[1]) + 1
               low_bound = int(p[1]) 

               left_bound = int(p[0]) 
               right_bound = int(p[0]) + 1
               
             #  x = p[0]
             #  y=  p[1]
               
               # f(x,y) ~    f(0,0)(1-x)(1-y) + f(1,0)x(1-y)  + f(0,1)(1-x)y  +  f(1,1)xy
               # (y,x,channel)

               if right_bound >= w and left_bound>= h:
                   

                      blue_prime = 0 
                                   
                      green_prime = 0
                                   
                      red_prime =  0

               if p[0] < 0 or p[1] < 1:
                      
                      blue_prime = 0 
                                   
                      green_prime = 0
                                   
                      red_prime =  0






               elif right_bound >= w:
                   print(y_idx,x_idx)
                   print(int(p[1]),int(p[2]))
                   print(p)
                   print(up_bound,low_bound,left_bound,right_bound)
                 
                   blue_prime = (matPauseScreen[low_bound,left_bound,0] * (1-p[0]) * (1-p[1])) + (matPauseScreen[up_bound,left_bound,0] * (1-p[0]) * p[1]) 

                   green_prime = (matPauseScreen[low_bound,left_bound,1] * (1-p[0]) * (1-p[1])) +  (matPauseScreen[up_bound,left_bound,1] * (1-p[0]) * p[1])

                   red_prime =  (matPauseScreen[low_bound,left_bound,2] * (1-p[0]) * (1-p[1])) + (matPauseScreen[up_bound,left_bound,2] * (1-p[0]) * p[1]) 

               elif up_bound >= h:    
                   blue_prime = (matPauseScreen[low_bound,left_bound,0] * (1-p[0]) * (1-p[1])) + (matPauseScreen[low_bound,right_bound,0]*p[0]* (1-p[1])) 
               
                   green_prime = (matPauseScreen[low_bound,left_bound,1] * (1-p[0]) * (1-p[1])) + (matPauseScreen[low_bound,right_bound,1]*p[0]* (1-p[1])) 
 
                   red_prime =  (matPauseScreen[low_bound,left_bound,2] * (1-p[0]) * (1-p[1])) + (matPauseScreen[low_bound,right_bound,2]*p[0]* (1-p[1]))  
               else:
           
                   '''
                   [1.91900000e+03 2.36779466e+02 1.00000000e+00]
                   237 236 1919 1920
                   
                   '''

                   '''

349 348 1919 1920
522 0
1079 1
[1.91900000e+03 1.07939467e+03 1.00000000e+00]
1080 1079 1919 1920
Traceback (most recent call last):
  File "rectify_python.py", line 218, in <module>
    blue_prime = (matPauseScreen[low_bound,left_bound,0] * (1-p[0]) * (1-p[1])) + (matPauseScreen[up_bound,left_bound,0] * (1-p[0]) * p[1]) 
IndexError: index 1080 is out of bounds for axis 0 with size 1080
                   '''

                   print(y_idx,x_idx)
                   print(int(p[1]),int(p[2]))
                   print(p)
                   print(up_bound,low_bound,left_bound,right_bound)
                 



                   blue_prime = (matPauseScreen[low_bound,left_bound,0] * (1-p[0]) * (1-p[1])) + (matPauseScreen[low_bound,right_bound,0]*p[0]* (1-p[1])) + (matPauseScreen[up_bound,left_bound,0] * (1-p[0]) * p[1]) + (matPauseScreen[up_bound,right_bound,0] * p[0] * p[1])
                   
                   green_prime = (matPauseScreen[low_bound,left_bound,1] * (1-p[0]) * (1-p[1])) + (matPauseScreen[low_bound,right_bound,1]*p[0]* (1-p[1])) + (matPauseScreen[up_bound,left_bound,1] * (1-p[0]) * p[1]) + (matPauseScreen[up_bound,right_bound,1] * p[0] * p[1])
                   
 
                   red_prime =  (matPauseScreen[low_bound,left_bound,2] * (1-p[0]) * (1-p[1])) + (matPauseScreen[low_bound,right_bound,2]*p[0]* (1-p[1])) + (matPauseScreen[up_bound,left_bound,2] * (1-p[0]) * p[1]) + (matPauseScreen[up_bound,right_bound,2] * p[0] * p[1])
                   
               target_img[y_idx,x_idx,0] = blue_prime
               target_img[y_idx,x_idx,1] = green_prime
               target_img[y_idx,x_idx,2] = red_prime
     

       


       



                

           


       # for px in range(h):
       #     for py in range(w):
       #          
       #         px = px + 1
       #         py = py + 1
       #         # first we have to get 
       #         pos_x = 1.0 * px

       #         pos_y = 1.0 * py
       #        
       #         print("Pos_x:",pos_x)
       #         print("Pos_y:",pos_y)
       #         break
       #     break


               # p_src = np.array([pos_x,pos_y,1.0]).reshape(3,1)
               #
               # p_prime = homography_matrix @ p_src
               # ## Convert into homogeous form
               # p_prime = p_prime/p_prime[-1]

               #  
               # h_inv = np.linalg.pinv(homography_matrix)
               # p_src_fake = h_inv @ p_prime
               # 
               # print(p_src_fake)
                
                
               # transform 
                # Bilinear interpolation 
                
              #  for idx_ch in range(ch):
              #      matPauseScreen[i,j,] 
       # matResult = cv2.warpPerspective(matPauseScreen, homography_matrix, (w, h), cv2.INTER_LINEAR)
        matPauseScreen = cv2.resize(matPauseScreen, dim)
        cv2.imshow("Source", matPauseScreen)
        matResult = cv2.resize(target_img, dim)
        cv2.imshow("Result", target_img)
         
        cv2.waitKey(0)








'''

H = np.array([[-0.00045632663, -0.00080100836, 0.70602891755],
              [-0.00015413196, -0.00110850468, 0.70818139183],
              [-0.00000022383, -0.00000062992, 0.0004259910224381828]])

p1 = np.array([[638],[634],[1]])

res = H @ p1


res/res[-1]

InHomo genous form
[[800.011487  ]
 [800.01142796]
 [  1.        ]]
 
'''
