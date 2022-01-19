import cv2
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
figR = 2
figC = 3

path = r'ObjectCounting/Project1Set2/object2.jpg'
img = cv2.imread(path)

if img is None:
    print("check path")
    raise

try:
    #=============== PRE-PROCESSING =================
    fig.add_subplot(figR, figC, 1), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    # Lay anh da muc xam
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig.add_subplot(figR, figC, 2), plt.imshow(grayImage, cmap='gray')
    plt.title('Grey Scale Image'), plt.xticks([]), plt.yticks([])

    # Tang do tuong phan
    gammaCorrect = np.array(255 * (grayImage / 255) ** 1.2 , dtype='uint8')
    fig.add_subplot(figR, figC, 3), plt.imshow(gammaCorrect, cmap='gray')
    plt.title('Gamma Correction Image'), plt.xticks([]), plt.yticks([])

    #=============== PROCESSING ================
    # Phan vung su dung nguong thich nghi
    thresh = cv2.adaptiveThreshold(gammaCorrect, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
    thresh = cv2.bitwise_not(thresh)
    fig.add_subplot(figR, figC, 4), plt.imshow(thresh, cmap='gray')
    plt.title('Adaptive Threshold Image'), plt.xticks([]), plt.yticks([])

    # Dilatation + erosion
    kernel = np.ones((16,16), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation,kernel, iterations=1)

    # Clean noise by median
    img_erode = cv2.medianBlur(img_erode, 7)
    fig.add_subplot(figR, figC, 5), plt.imshow(img_erode, cmap='gray')
    plt.title('Closing Image'), plt.xticks([]), plt.yticks([])

    #=============== COUNTING ================
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rgb = img.copy()
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)
    fig.add_subplot(figR, figC, 6), plt.imshow(rgb)
    plt.title('Contour Image'), plt.xticks([]), plt.yticks([])

    print("Number of object", len(contours))
    plt.show()

except Exception as e:
    print(e)
cv2.waitKey()
