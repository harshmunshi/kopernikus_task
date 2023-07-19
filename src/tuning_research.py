from typing import List, Tuple, Union

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from src.core import load_data, preprocess_image, sort_images
from src.imaging_interview import preprocess_image_change_detection
from src.utils import draw_contours_on_canvas, plot_thresh


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        print(cv2.contourArea(c))
        # if cv2.contourArea(c) < min_contour_area:
        #     continue

        res_cnts.append(cv2.contourArea(c))
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


def tune_parameters(src: str, resize_w: int = 640, resize_h: int = 480) -> List:
    image_list = load_data(src)
    image_list = sort_images(image_list)

    # take the first image as the reference
    # and compare all the reamining images for frame difference
    prev_frame = None
    scores = []
    countours = []
    window = 5

    # Step 1: Process the delta between the the first frame and the subsequent frames
    for i, img in enumerate(image_list):
        if img.endswith(".jpg") or img.endswith(".png"):
            if i == 0:
                im = cv2.imread(img)
                if type(im) == type(None):
                    continue
                prev_frame = preprocess_image(im, resize_w, resize_h)
                continue

            search_range = min(i + window, len(image_list))
            for j in range(i, search_range):
                curr_frame = cv2.imread(img)
                if type(curr_frame) == type(None):
                    continue
                curr_frame = preprocess_image(curr_frame, resize_w, resize_h)
                score, res_cnts, thresh = compare_frames_change_detection(
                    prev_frame, curr_frame, 500
                )
                scores.append(score)
                countours.extend(res_cnts)
        else:
            continue

    # plot the score
    plot_thresh(countours)
    kmeans_scores = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(
        np.array(scores).reshape(-1, 1)
    )
    kmeans_carea = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(
        np.array(countours).reshape(-1, 1)
    )
    max_threshold = min(kmeans_scores.cluster_centers_)
    min_threshold = min(kmeans_carea.cluster_centers_)

    return min_threshold, max_threshold * 0.05


tune_parameters("/home/harsh/Downloads/dataset-2")
