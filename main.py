import pandas as pd
import pytesseract
import cv2
import numpy as np
import os
from Scripts import imagepreprocessing as ip

myconfig = r"--psm 6 --oem 3"
MAX_FEATURE = 1006
KEEP_PERCENT = 0.24
ROI = [[(472, 52), (1112, 200), 'Part#'],
       [(246, 718), (302, 795), 'Mold'],
       [(374, 426), (842, 495), 'Lot#'],
       [(394, 624), (742, 697), 'Batch#'],
       [(208, 530), (346, 593), 'Qty']]


def align_image(src_image, des_image, max_features, keep_percent):
    """
    Used for image alignment
    :param src_image:
    :param des_image:
    :param max_features:
    :param keep_percent:
    :return: aligned image
    """

    h, w, c = src_image.shape
    # Convert images to grayscale
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    des_image = cv2.cvtColor(des_image, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector
    orb = cv2.ORB_create(max_features)

    # Detect keypoints and compute the descriptors
    kp1, des1 = orb.detectAndCompute(src_image, None)
    kp2, des2 = orb.detectAndCompute(des_image, None)

    # Initialize the Matcher for matching keypoints and match the keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)

    # Sort the matched keypoints according to the distance and keep given percent keypoints
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * keep_percent)]

    src_points = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    img_scan = cv2.warpPerspective(des_image, M, (w, h))
    return img_scan


def data_extractor(image):
    """
    Extract data from given image
    :param image:
    :return: data array
    """
    name = "image01.jpg"
    cv2.imwrite(name, image)
    image_test = cv2.imread(name)
    os.remove(name)

    img_show = image_test.copy()
    img_mask = np.zeros_like(img_show)

    data = []
    for x, r in enumerate(ROI):
        cv2.rectangle(img_mask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
        img_show = cv2.addWeighted(img_show, 0.99, img_mask, 0.1, 0)
        img_crop = image[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        name = "temp/" + r[2] + ".jpg"
        cv2.imwrite(name, img_crop)
        img0 = cv2.imread(name)
        os.remove(name)
        img_crop_ = ip.imageBinarization(img0)
        img_crop_ = ip.imageThinner(img_crop_)
        text = pytesseract.image_to_string(img_crop_, config=myconfig)
        data.append(text.strip())
    return data


# Read images
img_template = cv2.imread('image0.jpg')
path = 'TestImages'
image_list = os.listdir(path)
# data_list = []
# titles = []
# for title in ROI:
#     titles.append(title[2])
# data_list.append(titles)
# for i in image_list:
#     img = cv2.imread(path + "/" + i)
#     aligned_image = align_image(img_template, img, MAX_FEATURE, KEEP_PERCENT)
#     cv2.imshow("IMG", aligned_image)
#     cv2.waitKey(0)
    # extracted_text = data_extractor(aligned_image)
    # data_list.append(extracted_text)
# print(data_list)
# df = pd.DataFrame(data_list[1:], columns=data_list[:1])
# df.to_csv("Data/data.csv")
img = cv2.imread(path + "/image8.jpg")
aligned_image0 = align_image(img_template, img, max_features=1000, keep_percent=0.25)
cv2.imshow("Before", aligned_image0)
# cv2.imwrite("testImage.jpg", aligned_image0)
# aligned_image1 = cv2.imread("testImage.jpg")
# aligned_image = align_image(img_template, aligned_image1, max_features=1000, keep_percent=0.25)
# text = data_extractor(aligned_image)
# print(text)
# cv2.imshow("IMG", aligned_image)
cv2.waitKey(0)
#extracted_text = data_extractor(aligned_image)
#6a75dcbc0976d995ebb55ede81bd7ebab4fc97bac3259f21d7314744a5ff794d