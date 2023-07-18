from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List
from numpy.typing import ArrayLike, NDArray

import cv2
from utils import draw_contours_on_canvas, plot_thresh, plot_duplicates
import numpy as np

from src.imaging_interview import (compare_frames_change_detection,
                                   preprocess_image_change_detection)


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

def compute_delta(prev_frame: NDArray, next_frame: NDArray) -> NDArray:
    """A helper function to compute the delta between two frames

    Args:
        prev_frame (np.ndarray): previous frame
        next_frame (np.ndarray): next frame

    Returns:
        np.ndarray: delta between the two frames
    """
    score, res_cnts, thresh = compare_frames_change_detection(prev_frame, next_frame, 500)
    return score, res_cnts, thresh

def preprocess_image(img: NDArray, resize_w: int = 640, resize_h: int = 480) -> NDArray:
    """A helper function to preprocess the image
    """
    img = cv2.resize(img, (resize_w, resize_h))
    img = preprocess_image_change_detection(img)
    return img



def prune_images(src: str, resize_w: int = 640, resize_h: int = 480, debug: bool = True) -> None:
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
    duplicates = 0
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
    
    #plot_thresh(scores)

    # run a secondary for loop to check for duplicates
    for i in range(len(scores)):
        if i == 0:
            continue
        if scores[i] == scores[i-1]:
            duplicates += 1
            print("Duplicate found")
            d = plot_duplicates(image_list[i], image_list[i-1])
            cv2.imshow("frame", d)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        elif abs(scores[i] - scores[i-1]) < 3000:
            # check if the contours are the same
            print(len(contours[i]), len(contours[i-1]))
            if len(contours[i]) == len(contours[i-1]):
                duplicates += 1
                print("Duplicate found")
                d = plot_duplicates(image_list[i], image_list[i-1])
                cv2.imshow("frame", d)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        

        # if debug:
        #     cv2.imshow("frame", frame)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
                #prev_frame = curr_frame



if __name__ == "__main__":
    src = "/home/harsh/Downloads/dataset-3"
    pruned_images = prune_images(src)
