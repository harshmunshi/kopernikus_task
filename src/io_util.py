from datetime import datetime
from glob import glob
from typing import List, Tuple, Union

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


def preprocess_image(img: NDArray, resize_w: int = 640, resize_h: int = 480) -> NDArray:
    """A helper function to preprocess the image"""
    img = cv2.resize(img, (resize_w, resize_h))
    img = preprocess_image_change_detection(img)
    return img


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
