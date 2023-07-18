from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_contours_on_canvas(img, contours, color=(255, 255, 255), thickness=2):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    big_canvas = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype=np.uint8)
    canvas = np.zeros(img.shape, dtype=np.uint8)
    canvas = cv2.drawContours(canvas, contours, -1, color, thickness)
    big_canvas[: img.shape[0], : img.shape[1]] = img
    big_canvas[: img.shape[0], img.shape[1] :] = canvas
    return big_canvas


def plot_thresh(thresh: List):
    for i in range(len(thresh)):
        plt.scatter(i, thresh[i], color="black")
    plt.show()


def plot_duplicates(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))
    big_canvas = np.zeros((img1.shape[0], img1.shape[1] * 2, 3), dtype=np.uint8)
    big_canvas[: img1.shape[0], : img1.shape[1]] = img1
    big_canvas[: img1.shape[0], img1.shape[1] :] = img2
    return big_canvas
