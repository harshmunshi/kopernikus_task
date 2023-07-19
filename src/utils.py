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
    p = range(len(thresh))
    plt.scatter(p, thresh, color="blue")
    plt.xlabel("Frame Number")
    plt.ylabel("All Contours")
    plt.grid(True)
    # plt.show()
    plt.savefig("areaclustering.png")


def plot_duplicates(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))
    big_canvas = np.zeros((img1.shape[0], img1.shape[1] * 2, 3), dtype=np.uint8)
    big_canvas[: img1.shape[0], : img1.shape[1]] = img1
    big_canvas[: img1.shape[0], img1.shape[1] :] = img2
    return big_canvas


def plot_curr_prev_thresh(img1, img2, contours1, contours2):
    canvas = np.zeros((img1.shape[0] * 2, img1.shape[1] * 2, 3), dtype=np.uint8)
    canvas[: img1.shape[0], : canvas.shape[1]] = draw_contours_on_canvas(
        img1, contours1, color=(255, 255, 255)
    )
    canvas[img1.shape[0] :, : canvas.shape[1]] = draw_contours_on_canvas(
        img2, contours2, color=(255, 255, 255)
    )
    return canvas
