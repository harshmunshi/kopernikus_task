import math
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_contours_on_canvas(
    img, contours=None, color=(255, 255, 255), thickness=2, thresh=None
):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    big_canvas = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype=np.uint8)
    canvas = np.zeros(img.shape, dtype=np.uint8)
    if thresh is None:
        canvas = cv2.drawContours(canvas, contours, -1, color, thickness)
    else:
        canvas = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
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


def plot_curr_prev_thresh(img1, img2, contours1, thresh):
    canvas = np.zeros((img1.shape[0] * 2, img1.shape[1] * 2, 3), dtype=np.uint8)
    canvas[: img1.shape[0], : canvas.shape[1]] = draw_contours_on_canvas(
        img1, contours1, color=(255, 255, 255)
    )
    canvas[img1.shape[0] :, : canvas.shape[1]] = draw_contours_on_canvas(
        img2, color=(255, 255, 255), thresh=thresh
    )
    return canvas


def viz_frames_in_window(img_list: List):
    img_f = img_list[0]
    canvas = np.zeros((img_f.shape[0] * 2, img_f.shape[1], 3))
    for i in len(img_list):
        img = img_list[0]
        img_size = int(img.shape[0] / 5), int(img.shape[1])
        img = cv2.resize(img, (img_size[0], img_size[1]))
        canvas[
            i * img_size[0] : i * img_size[0] + img_size[0],
            i * img_size[1] : i * img_size[1] + img_size[1],
        ] = img
    return canvas


def log(a):
    return math.log(a)


def plot_complexity():
    f1 = lambda a: a * log(a) + a**2
    f5 = lambda a: a * log(a) + a * 10
    f2 = lambda a: a * log(a) + a
    f3 = lambda a: a * log(a) + 2 * a
    f4 = lambda a: a * log(a) + 3 * a

    pts = range(1, 1000)
    f1_pts = list(map(f1, pts))
    f2_pts = list(map(f2, pts))
    f3_pts = list(map(f3, pts))
    f4_pts = list(map(f4, pts))
    f5_pts = list(map(f5, pts))
    plt.grid(True)
    # plt.plot(pts, f1_pts, c="blue", label="O(nlogn) + O(n2)")
    plt.plot(pts, f2_pts, c="green", label="O(nlogn) + O(n)")
    plt.plot(pts, f3_pts, c="black", label="O(nlogn) + 2O(n)")
    plt.plot(pts, f4_pts, c="red", label="O(nlogn) + 3O(n)")
    plt.plot(pts, f5_pts, c="yellow", label="O(nlogn) + O(kn)")
    plt.legend()
    plt.savefig("complexity_sim.png")


plot_complexity()
