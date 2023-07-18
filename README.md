# Dataset Analysis

* The given dataset consists of multiple cameras (3 different IDs) which captured frames at different timestamps.
* The images follow two distinct timestamps: python datetime and unix.
* Illumination conditions vary based on time of the day even though there is no object variance.
* There is a radial distortion in the images (beacuse of angle and the lens).
* In some cases there is heavy occlusion of objects.
* Some images are noise / nonetype.

# Algorithm
![Alt Text](./data/algo_overview.png)

The high-level overview of the algorithm is give in the figure above. It consists of the following steps:
* Read the file paths of all the images in a list.
* Sort the list based on the timestamp. This is necessary since images around the same timestamp have higher likelihood to be similar.
* Run a for loop to preprocess the images and compute the delta (using `compare_frames_change_detection`) and append the scores, contours and thresh images as separate lists.
* Run a secondary for loop to compute the difference in the scores as a first pass. If the images are exactly similar, this will hit. Else, check for an acceptable range of tolerance. Also check of the mean of the images are similar, to make sure they are the same images.

