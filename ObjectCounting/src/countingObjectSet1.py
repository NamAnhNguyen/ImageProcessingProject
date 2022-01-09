import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'ObjectCounting\Project1Set1\fade.png'
img = cv2.imread(path)

if img is None:
    print("check path")
    raise
try:
    cv2.imshow('Origin', img)

    # Lay anh da muc xam
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', grayImage)

    medianImage = cv2.medianBlur(grayImage, 5)
    cv2.imshow('Median Image', medianImage)

    ret2 = cv2.adaptiveThreshold(medianImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,127,-1)
    cv2.imshow("Adaptive Threshold 2", ret2)

    medianImage2 = cv2.medianBlur(ret2, 7)
    cv2.imshow('Median Image2', medianImage2)

    erodedKernel = np.ones((5, 5), np.uint8)
    erodedImg = cv2.erode(medianImage2, erodedKernel, iterations=5)
    cv2.imshow("Eroded Img", erodedImg)

    # dilatedKernel = np.ones((5, 5), np.uint8)
    # dilatedImg = cv2.dilate(erodedImg, dilatedKernel, iterations=5)
    # cv2.imshow("Dilated Img", dilatedImg)

    rice = medianImage2 - erodedImg
    cv2.imshow("Rice Img", rice)

    th1, ret1 = cv2.threshold(rice, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("Adaptive Threshold", ret1)

    contours, hierarchy = cv2.findContours(ret1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rgb = img.copy()
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)

    cv2.imshow("Contour", rgb)

    print("Number of object", len(contours))

except Exception as e:
    print(e)
cv2.waitKey()