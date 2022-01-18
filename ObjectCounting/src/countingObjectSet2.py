import cv2
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
figR = 2
figC = 3

path = r'ObjectCounting/Project1Set2/object4.jpg'
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

    gammaCorrect = np.array(255 * (grayImage / 255) ** 1.2 , dtype='uint8')
    fig.add_subplot(figR, figC, 3), plt.imshow(gammaCorrect, cmap='gray')
    plt.title('Gamma Correction Image'), plt.xticks([]), plt.yticks([])

    #=============== PROCESSING ================
    thresh = cv2.adaptiveThreshold(gammaCorrect, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
    thresh = cv2.bitwise_not(thresh)
    fig.add_subplot(figR, figC, 4), plt.imshow(thresh, cmap='gray')
    plt.title('Adaptive Threshold Image'), plt.xticks([]), plt.yticks([])

    # Dilatation + erosion
    kernel = np.ones((16,16), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation,kernel, iterations=1)

    # clean all noise after dilatation and erosion
    img_erode = cv2.medianBlur(img_erode, 7)
    fig.add_subplot(figR, figC, 5), plt.imshow(img_erode, cmap='gray')
    plt.title('Closing Image'), plt.xticks([]), plt.yticks([])

# Labeling
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rgb = img.copy()
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)

    fig.add_subplot(figR, figC, 6), plt.imshow(rgb)
    plt.title('Contour Image'), plt.xticks([]), plt.yticks([])

    print("Number of object", len(contours))

    # ret, labels = cv2.connectedComponents(img_erode)
    # label_hue = np.uint8(179 * labels / np.max(labels))
    # blank_ch = 255 * np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # labeled_img[label_hue == 0] = 0

    # fig.add_subplot(figR, figC, 6), plt.imshow(labeled_img, cmap='gray')
    # plt.title('Labeled Image'), plt.xticks([]), plt.yticks([])

    # print('Number of objects is:', ret-1)
    plt.show()

except Exception as e:
    print(e)
cv2.waitKey()
