import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.imaging_interview import (compare_frames_change_detection,
                                   preprocess_image_change_detection)
from src.tuning_research import tune_parameters
from utils import (draw_contours_on_canvas, plot_curr_prev_thresh,
                   plot_duplicates, plot_thresh)


def load_data(src: str) -> List:
    """A helper function to load the image path(s) as a list

    Args:
        src (str): folder path source

    Returns:
        List: list of image paths under the given folder
    """
    image_list = glob(src + "/*")
    return image_list


def sort_images(image_list: List[str]) -> List[str]:
    """A helper function to sort the image paths in ascending order based on the datetime

    Args:
        image_list (List[str]): list of image paths

    Returns:
        List[str]: sorted list of image paths
    """

    def extract_datetime(entry: str):
        entry = entry.split("/")[-1].split(".")[0]  # Get the filename
        entry = entry[4:]  # Remove the first three characters
        try:
            # Check if the entry has the format YYYY_MM_DD__HH_MM_SS
            dt = datetime.strptime(entry, "%Y_%m_%d__%H_%M_%S")
            print(dt)
        except ValueError:
            try:
                # Check if the entry has the format -UNIX_TIMESTAMP
                dt = datetime.fromtimestamp(int(entry) / 1000)
            except (IndexError, ValueError):
                # If the format doesn't match either of the expected formats, set a default datetime value
                dt = datetime.min
        return dt

    sorted_image_list = sorted(image_list, key=extract_datetime)
    return sorted_image_list


def compute_delta(
    prev_frame: NDArray, next_frame: NDArray, area_th: float = 500
) -> NDArray:
    """A helper function to compute the delta between two frames

    Args:
        prev_frame (np.ndarray): previous frame
        next_frame (np.ndarray): next frame

    Returns:
        np.ndarray: delta between the two frames
    """
    score, res_cnts, thresh = compare_frames_change_detection(
        prev_frame, next_frame, area_th
    )
    return score, res_cnts, thresh


def preprocess_image(img: NDArray, resize_w: int = 640, resize_h: int = 480) -> NDArray:
    """A helper function to preprocess the image"""
    img = cv2.resize(img, (resize_w, resize_h))
    img = preprocess_image_change_detection(img)
    return img


def prune_images_generic(
    src: str,
    resize_w: int = 640,
    resize_h: int = 480,
    debug: bool = False,
    score_threshold: int = 3000,
    image_mean_threshold: int = 2,
) -> None:
    image_list = load_data(src)
    image_list = sort_images(image_list)

    # take the first image as the reference
    # and compare all the reamining images for frame difference
    prev_frame = None
    del_idx = []
    window = 5

    # Step 0: compute the threshold for the first frame
    area_th, filter_th = tune_parameters(src, resize_w, resize_h)

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
                score, res_cnts, thresh = compute_delta(prev_frame, curr_frame, area_th)
                print(score)
                if score == 0 and len(res_cnts) == 0:
                    del_idx.append(img)
                    prev_frame = curr_frame

                if score < filter_th:
                    del_idx.append(img)
                    prev_frame = curr_frame

                else:
                    i += j
                    prev_frame = curr_frame
                    break

        else:
            continue
    for im_del in del_idx:
        os.remove(im_del)
        print(f"removed {im_del}")


def prune_images_On(
    src: str,
    resize_w: int = 640,
    resize_h: int = 480,
    debug: bool = False,
    score_threshold: int = 3000,
    image_mean_threshold: int = 2,
) -> None:
    """Function to check for duplicates and prune them

    Args:
        src (str): folder path source

    Returns:
        None
    """

    image_list = load_data(src)
    image_list = sort_images(image_list)

    # take the first image as the reference
    # and compare all the reamining images for frame difference
    prev_frame = None
    scores = []
    contours = []
    thresh_all = []

    # Step 1: Process the delta between the the first frame and the subsequent frames
    for i, img in enumerate(image_list):
        if img.endswith(".jpg") or img.endswith(".png"):
            print(img)
            if i == 0:
                im = cv2.imread(img)
                if type(im) == type(None):
                    continue
                prev_frame = preprocess_image(im, resize_w, resize_h)
                continue
            im = cv2.imread(img)
            if type(im) == type(None):
                continue
            next_frame = preprocess_image(im, resize_w, resize_h)
            score, res_cnts, thresh = compute_delta(prev_frame, next_frame)
            contours.append(res_cnts)
            thresh_all.append(thresh)
            scores.append(score)
        else:
            continue

    # Step 2: Run a secondary loop to check the duplicates based on the delta and remove the duplicates
    for i in range(len(scores)):
        if i == 0:
            continue
        if scores[i] == scores[i - 1]:
            duplicates = True
            os.remove(image_list[i + 1])
            print(f"removed {image_list[i+1]}")

        elif abs(scores[i] - scores[i - 1]) < score_threshold:
            # check if the contours are the same
            if len(contours[i]) == len(contours[i - 1]):
                # check if the images are the same
                try:
                    if np.allclose(
                        np.mean(cv2.imread(image_list[i + 1])),
                        np.mean(cv2.imread(image_list[i])),
                        atol=image_mean_threshold,
                    ):
                        os.remove(image_list[i + 1])
                        print(f"removed {image_list[i+1]}")
                except:
                    continue


if __name__ == "__main__":
    src = "/home/harsh/Downloads/dataset-2"
    pruned_images = prune_images(src)
