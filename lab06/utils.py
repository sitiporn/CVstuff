from stats import Stats
import cv2
from typing import List #use it for :List[...]

def drawBoundingBox(image, bb):
    """
    Draw the bounding box from the points set

    Parameters
    ----------
        image (array):
            image which you want to draw
        bb (List):
            points array set
    """
    color = (0, 0, 255)
    for i in range(len(bb) - 1):
        b1 = (int(bb[i][0]), int(bb[i][1]))
        b2 = (int(bb[i + 1][0]), int(bb[i + 1][1]))
        cv2.line(image, b1, b2, color, 2)
    b1 = (int(bb[len(bb) - 1][0]), int(bb[len(bb) - 1][1]))
    b2 = (int(bb[0][0]), int(bb[0][1]))
    cv2.line(image, b1, b2, color, 2)

def drawStatistics(image, stat: Stats):
    """
    Draw the statistic to images

    Parameters
    ----------
        image (array):
            image which you want to draw
        stat (Stats):
            statistic values
    """
    font = cv2.FONT_HERSHEY_PLAIN

    str1, str2, str3, str4, str5 = stat.to_strings()

    shape = image.shape

    cv2.putText(image, str1, (0, shape[0] - 120), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str2, (0, shape[0] - 90), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str3, (0, shape[0] - 60), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str5, (0, shape[0] - 30), font, 2, (0, 0, 255), 3)

def printStatistics(name: str, stat: Stats):
    """
    Print the statistic

    Parameters
    ----------
        name (str):
            image which you want to draw
        stat (Stats):
            statistic values
    """
    print(name)
    print("----------")
    str1, str2, str3, str4, str5 = stat.to_strings()
    print(str1)
    print(str2)
    print(str3)
    print(str4)
    print(str5)
    print()

def Points(keypoints):
    res = []
    for i in keypoints:
        res.append(i)
    return res
