import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'/Volumes/Local/HUST/ImageProcessing/Project/Project1Set2/object2.jpg'
img = cv2.imread(path)

if img is None:
    print("check path")
    raise

try:
    cv2.imshow('Origin', img)

    # Lay anh da muc xam
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', grayImage)

    # Loc nhieu
    # equalizeHistImg = cv2.equalizeHist(grayImage)
    # cv2.imshow(" equalizeHistImg", equalizeHistImg)

    threshImage = cv2.threshold(grayImage, 115, 255, cv2.THRESH_BINARY)[1]
    kernal = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(threshImage, kernal, iterations=2)
    cv2.imshow('ThreshImage', dilation)
    # Phat hien bien (Loc canny)

    # cannyImage = cv2.Canny(equalizeHistImg,50,250)
    # cv2.imshow("CannyImage",cannyImage)

    # (contours, hierarchy) = cv2.findContours(cannyImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   
    # sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    # print("Number of object", len(contours))

    # for (i,c) in enumerate(sorted_contours):
    #     M= cv2.moments(c)
        # print(M)
        # cx= int(M['m10']/M['m00'])
        # cy= int(M['m01']/M['m00'])
        # cv2.putText(img, text= str(i+1), org=(cx,cy),
        #     fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
        #     thickness=2, lineType=cv2.LINE_AA)
 

    # cv2.imshow("Contour",img)

    # print("Number of object", len(contours))

except Exception as e:
    print(e)
cv2.waitKey()
