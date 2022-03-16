import cv2
import numpy as np
import torch
IMAGE_FILE = "lab4-segmentation-training/FloorData/Images/frame-018.png"
ONNX_NETWORK_DEFINITION = "fcn_resnet18.onnx" 

aStringClasses = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

aColorClasses = [
        ( 0, 0, 0 ), ( 255, 0, 0 ), ( 0, 255, 0 ), ( 0, 255, 120 ), ( 0, 0, 255 ), ( 255, 0, 255 ), ( 70, 70, 70 ),
        ( 102, 102, 156 ), ( 190, 153, 153 ), ( 180, 165, 180 ), ( 150, 100, 100 ), ( 153, 153, 153 ),
        ( 250, 170, 30 ), ( 220, 220, 0 ), ( 107, 142, 35 ), ( 192, 128, 128 ), ( 70, 130, 180 ), ( 220, 20, 60 ),
        ( 0, 0, 142 ), ( 0, 0, 70 ), ( 119, 11, 32 )
]

nClasses = len(aColorClasses)
print("nClasses: ",nClasses)
# Read CNN definition
net = cv2.dnn.readNetFromONNX(ONNX_NETWORK_DEFINITION)

# Read input image
matFrame = cv2.imread(IMAGE_FILE);

#print(net.eval())

if (matFrame is None):

    print("Cannot open image file ", IMAGE_FILE)
    

print("matFrame.shape :",matFrame.shape)

matInputTensor = torch.from_numpy(matFrame)
print(type(matInputTensor))
net.setInput(matInputTensor)

matScore = net.forward()


# Colorize the image and display

matColored = cv2.colorizeSegmentation(matFrame, matScore, aColorClasses, aStringClasses, nClasses)

# Add timing information
layersTimes = 0
freq = cv2.getTickFrequency() / 1000;
t = net.getPerfProfile(layersTimes) / freq;
label = "Inference time: " + str(t) + " ms"
cv2.putText(matColored, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

# Display
cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
cv2.imshow(WINDOW_NAME, matColored)
cv2.waitKey(0)
