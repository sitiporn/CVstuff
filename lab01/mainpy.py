import numpy as np
import cv2


def isOntheLine(line,point):

    
    print("##################")
    if abs(np.dot(line,point)) <= 1e-6:
        print(str(point)+"  is On the Line")
        ontheLine = True
    else:
        print(str(point)+ " is Not on the line")



p1 = np.array([2, 4, 2])
p2 = np.array([6, 3, 3])
p3 = np.array([1, 2, 0.5])
p4 = np.array([16, 8, 4])

pt = np.array([p1,p2,p3,p4])
line = np.array([8, -4, 0])

for i in range(4):
    isOntheLine(line,pt[i])