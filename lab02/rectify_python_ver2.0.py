import  numpy as np
import sys
#import ruamel.yaml
from pathlib import Path
import cv2
import numpy as np
import sys
from ruamel import yaml
from readYaml import ReadYaml 
VIDEO_FILE = 'Lab01-robot-video.mp4'
ROTATE = False

path ='config.yaml'

#config_file = Path('config.yaml')
#yaml = ruamel.yaml.YAML(typ='safe')



###
#@yaml.register_class
##

def from_yaml(node):
        
    array= []

    for x in node:
        if x =='--':
            sub_array = []
            array.append(sub_array)
            continue
        if x == '-':
            continue
        sub_array.append(int(x))

        return numpy.array(array)

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
    
    
   # data = yaml.load(config_file)
   # print(type(data['score']))
   # print(data)
   # homo_str = np.array(data['score'])
   # print("Homo :",homo_str)
   # print(type(homo_str))
   # homo_arr = from_yaml(homo_str)
   # print(homo_arr)

    
    
    
    homo  = ReadYaml('config.yaml')
    homo_mat = homo.readTonumpyArray()

    print(homo_mat)
    print(type(homo_mat))
    
    
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
        A = A.T @ A 

        print("====Design Matrix====")
        print(A)

        w, u, vt = cv2.SVDecomp(A) 

        ## slice the last column of v
        ## so select the row of vt
        
        homography_matrix = vt[-1,:]
      
        homography_matrix = homography_matrix.reshape((3,3))
        homography_matrix_dummy = homography_matrix / homography_matrix[-1,-1]
        #homography_matrix = cv2.getPerspectiveTransform(src, reals);
        print("Estimated Homography Matrix is:")
        print(homography_matrix_dummy)

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
        
        invH = np.linalg.pinv(homography_matrix)
        p_dum = np.array([reals[0][0],reals[0][1], 1]).reshape((3,1))
        res2 =  invH @ p_dum
         
        print("res2:",res2/res2[-1])
       # print("show image channels:")
       # cv2.imshow("show image channels 0:",matPauseScreen[:,:,0])
       # cv2.imshow("show image channels 1:",matPauseScreen[:,:,1])
       # cv2.imshow("show image channels 2:",matPauseScreen[:,:,2])
        cv2.waitKey(0)
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
        
        target_img = np.zeros_like(matPauseScreen)

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
               
               x_1 = int(p[0]) 
               x_2 = int(p[0]) + 1
               y_1 = int(p[1]) 
               y_2 = int(p[1]) + 1
              
               x = p[0]
               y = p[1]

             #  x = p[0]
             #  y=  p[1]
               
               # f(x,y) ~    f(0,0)(1-x)(1-y) + f(1,0)x(1-y)  + f(0,1)(1-x)y  +  f(1,1)xy
               # (y,x,channel)
                
               if int(p[0]) >= h-1 or int(p[1]) >= w-1 or int(p[0]) <=0 or int(p[1])<=0: 
                   
                   p_b = 0
                   p_g = 0
                   p_r = 0
               else:
                                    
                  x_neg = -1 *x
                  x_1_neg = -1 * x_1

                  vect_x = np.array([x_2,x_neg,x,x_1_neg]).reshape((1,4))

                    
                  #y2 and p[1] : 1080 1079
                  
                  if y_2 >= 1080:
                      print("y2 and p[1] :",y_2,int(p[1]))
                  
                  color_b = np.array([matPauseScreen[y_1,x_1,0],matPauseScreen[y_2,x_1,0],matPauseScreen[y_1,x_2,0],matPauseScreen[y_2,x_2,0]]).reshape((2,2))
                  
                  color_g = np.array([matPauseScreen[y_1,x_1,1],matPauseScreen[y_2,x_1,1],matPauseScreen[y_1,x_2,1],matPauseScreen[y_2,x_2,1]]).reshape((2,2))
                  
                  color_r = np.array([matPauseScreen[y_1,x_1,2],matPauseScreen[y_2,x_1,2],matPauseScreen[y_1,x_2,2],matPauseScreen[y_2,x_2,2]]).reshape((2,2))
                  

                  diff_1 = x_2 - x
                  diff_2 = x - x_1
                  diff_3 = y_2 - y
                  diff_4 = y - y_1
                 
                  vect_x = np.array([diff_1,diff_2]).reshape((1,2))
                  vect_y = np.array([diff_3,diff_4]).reshape((2,1))


                  coef = 1/((x_2-x_1)*(y_2-y_1))
                   
                  p_b = coef * vect_x @ color_b @ vect_y   
                  p_g = coef * vect_x @ color_g @ vect_y   
                  p_r = coef * vect_x @ color_r @ vect_y   
                    

                  # r1_b = (x_2 - x)/(x_2 - x_1) * matPauseScreen[y_1,x_1,0] + (x-x_1)/(x_2 - x_1) * matPauseScreen[y_1,x_2,0]
                  # r1_g = (x_2 - x)/(x_2 - x_1) * matPauseScreen[y_1,x_1,1] + (x-x_1)/(x_2 - x_1) * matPauseScreen[y_1,x_2,1]
                  # r1_r = (x_2 - x)/(x_2 - x_1) * matPauseScreen[y_1,x_1,2] + (x-x_1)/(x_2 - x_1) * matPauseScreen[y_1,x_2,2]

                  # r2_b = (x_2 - x)/(x_2 -x_1) * matPauseScreen[y_2,x_1,0] + (x-x_1)/(x_2 - x_1) * matPauseScreen[y_2,x_2,0]
                  # r2_g = (x_2 - x)/(x_2 -x_1) * matPauseScreen[y_2,x_1,1] + (x-x_1)/(x_2 - x_1) * matPauseScreen[y_2,x_2,1]
                  # r2_r = (x_2 - x)/(x_2 -x_1) * matPauseScreen[y_2,x_1,2] + (x-x_1)/(x_2 - x_1) * matPauseScreen[y_2,x_2,2]
                  # 
                  # p_b = (y_2 - y)/(y_2 - y_1) * r1_b + (y-y_1)/(y_2 - y_1) * r2_b
                  # p_g = (y_2 - y)/(y_2 - y_1) * r1_g + (y-y_1)/(y_2 - y_1) * r2_g
                  # p_r = (y_2 - y)/(y_2 - y_1) * r1_r + (y-y_1)/(y_2 - y_1) * r2_r

                   # print(y_idx,x_idx)
                   # print(int(p[1]),int(p[2]))
                   # print(p)利用者:Einstee/共1次内挿
                   # print(up_bound,low_bound,left_bound,right_bound)
                                    

                   
               


               if p_b >255 or p_b < 0 or p_g > 255 or p_g < 0 or p_r > 255 or p_r < 0:
                  
                   print("Problem occurs (B,G,R) :",p_b,p_g,p_r)
               
               #print(blue_prime, green_prime, red_prime)
               target_img[y_idx,x_idx,0] = p_b 
               target_img[y_idx,x_idx,1] = p_g 
               target_img[y_idx,x_idx,2] = p_r 
      

       


       



                

           


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
        cv2.imshow("Result", matResult)
         
        cv2.waitKey(0)

