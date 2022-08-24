import cv2
import numpy as np
import pytesseract

myconfig = r"--psm 6 --oem 3"


def imageBinarization(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 3)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def imageNoiseRemoval(image):
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    return image


def imageThinner(image):
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=2)
    return erosion


def main():
    img = cv2.imread("ExtractedImages/Part#.jpg")
    cv2.imshow("BEFORE", img)
    img = imageBinarization(img)
    img = imageThinner(img)
    text = pytesseract.image_to_string(img, config=myconfig)
    print(text)
    cv2.imshow("AFTER", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
