import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'ObjectCounting/Project1Set2/object1.jpg'
img = cv2.imread(path)

if img is None:
    print("check path")
    raise

try:
    cv2.imshow('Origin', img)

    # Lay anh da muc xam
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', grayImage)

    ret2 = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,-2)
    cv2.imshow("Adaptive Threshold 2", ret2)

    blur = cv2.GaussianBlur(ret2, (5, 5), 0)
    cv2.imshow("Gauss", blur)

    cannyImage = cv2.Canny(blur,10,250)
    cv2.imshow("CannyImage",cannyImage)

    dilatedKernel = np.ones((3, 3), np.uint8)
    dilatedImg = cv2.dilate(cannyImage, dilatedKernel, iterations=3)
    cv2.imshow("Dilated Img", dilatedImg)

    erodedKernel = np.ones((3, 3), np.uint8)
    erodedImg = cv2.erode(dilatedImg, erodedKernel, iterations=1)
    cv2.imshow("Eroded Img", erodedImg)


    contours, hierarchy = cv2.findContours(erodedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = img.copy()
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Contour", rgb)
    print("Number of object", len(contours))

except Exception as e:
    print(e)
cv2.waitKey()
