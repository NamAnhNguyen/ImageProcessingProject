import cv2
import numpy as np
import matplotlib.pyplot as plt

path = r'/Volumes/Local/HUST/ImageProcessing/Project/Project1Set2/object1.jpg'
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
    guassianBlur = cv2.GaussianBlur(grayImage, (9, 9), 0)
    cv2.imshow("GuassianBlur",guassianBlur)

    # Phat hien bien (Loc canny)

    cannyImage = cv2.Canny(guassianBlur,30,125)
    cv2.imshow("CannyImage",cannyImage)

    (contours, hierarchy) = cv2.findContours(cannyImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = img.copy()
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)
 
    cv2.imshow("Contour",rgb)

    print("Number of object", len(contours))

except Exception as e:
    print(e)
cv2.waitKey()
