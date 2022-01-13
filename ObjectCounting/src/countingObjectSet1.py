import cv2
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
figR = 3
figC = 4

def sinusNoise(img):
    try:
        cv2.imshow('image', img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum_pre = 20 * np.log(np.abs(dft_shift))

        rows, cols = img.shape
        crow, ccol = int(rows/2), int(cols/2)

        mask = np.zeros((rows, cols), np.uint8)
        mask[0:rows, 0:cols] = 1

        # mask[crow-30:crow+30,ccol-30:ccol+30]=0
        mask[crow, ccol-10:ccol-5] = 0
        mask[crow, ccol+5:ccol+10] = 0

        # mask = np.ndarray((rows,cols),)

        fshift = dft_shift*mask

        magnitude_spectrum_after = magnitude_spectrum_pre * mask

        f_ishift = np.fft.ifftshift(fshift)

        img_back = np.abs(np.fft.ifft2(f_ishift))
        img_back = img_back.astype(np.uint8)

        fig.add_subplot(figR, figC, 1), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        fig.add_subplot(figR, figC, 2), plt.imshow(
            magnitude_spectrum_pre, cmap='gray')
        plt.title('Magnitude Spectrum Pre'), plt.xticks([]), plt.yticks([])
        fig.add_subplot(figR, figC, 3), plt.imshow(
            magnitude_spectrum_after, cmap='gray')
        plt.title('Magnitude Spectrum After'), plt.xticks([]), plt.yticks([])
        fig.add_subplot(figR, figC, 4), plt.imshow(img_back, cmap='gray')
        plt.title('Inverse'), plt.xticks([]), plt.yticks([])
        img_back = img_back[0:460, 0:460]
        return img_back

    except Exception as e:
        print("eeeeee", e)

path = r'ObjectCounting/Project1Set1/fade.png'
img = cv2.imread(path)

if img is None:
    print("check path")
    raise
try:
    cv2.imshow('Origin', img)

    # Lay anh da muc xam
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', grayImage)
    print(grayImage.shape)

    filterSinus = sinusNoise(grayImage)

    medianImage = cv2.medianBlur(filterSinus, 5)
    fig.add_subplot(figR, figC, 5), plt.imshow(medianImage, cmap='gray')
    plt.title('Median'), plt.xticks([]), plt.yticks([])
    ret2 = cv2.adaptiveThreshold(
        medianImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, -1)
    # cv2.imshow("Adaptive Threshold 2", ret2)
    fig.add_subplot(figR, figC, 6), plt.imshow(ret2, cmap='gray')
    plt.title('Adaptive Threshold 2'), plt.xticks([]), plt.yticks([])

    medianImage2 = cv2.medianBlur(ret2, 7)
    # cv2.imshow('Median Image2', medianImage2)
    fig.add_subplot(figR, figC, 7), plt.imshow(medianImage2, cmap='gray')
    plt.title('Median Image 2'), plt.xticks([]), plt.yticks([])

    erodedKernel = np.ones((5, 5), np.uint8)
    erodedImg = cv2.erode(medianImage2, erodedKernel, iterations=5)
    # cv2.imshow("Eroded Img", erodedImg)
    fig.add_subplot(figR, figC, 8), plt.imshow(erodedImg, cmap='gray')
    plt.title('Erode Image'), plt.xticks([]), plt.yticks([])

    # dilatedKernel = np.ones((5, 5), np.uint8)
    # dilatedImg = cv2.dilate(erodedImg, dilatedKernel, iterations=5)
    # cv2.imshow("Dilated Img", dilatedImg)

    rice = medianImage2 - erodedImg
    # cv2.imshow("Rice Img", rice)
    fig.add_subplot(figR, figC, 9), plt.imshow(rice, cmap='gray')
    plt.title('Rice Image'), plt.xticks([]), plt.yticks([])

    th1, ret1 = cv2.threshold(
        rice, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("Adaptive Threshold", ret1)
    fig.add_subplot(figR, figC, 10), plt.imshow(ret1, cmap='gray')
    plt.title('Threshhold Image'), plt.xticks([]), plt.yticks([])

    contours, hierarchy = cv2.findContours(
        ret1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rgb = img.copy()
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)

    # cv2.imshow("Contour", rgb)
    fig.add_subplot(figR, figC, 11), plt.imshow(rgb)
    plt.title('Contour Image'), plt.xticks([]), plt.yticks([])

    print("Number of object", len(contours))
    plt.show()

except Exception as e:
    print(e)
cv2.waitKey()
