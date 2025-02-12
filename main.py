import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from utils.split_and_merge import Region, split, merge, split_and_merge

from utils.lib import save_image

images = [
    "images/landscape1.jpeg", 
    "images/peperoni.png"
]

split_thresholds = [
    250,
]

split_depths = [
    6,
    7
]

merge_thresholds = [
    1000000,
    800000,
    600000,
    400000
]

for image_path in images:
    for split_depth in split_depths:
        for split_threshold in split_thresholds: 
            for merge_threshold in merge_thresholds:
                print(f"Processing {image_path} with split_depth={split_depth}, split_threshold={split_threshold}, merge_threshold={merge_threshold}")

                input_image = cv2.imread(image_path)

                start_time = time.time()

                result_image = split_and_merge(input_image, split_depth, split_threshold, merge_threshold)

                end_time = time.time()
                execution_time = end_time - start_time

                save_image(result_image, image_path, split_depth, split_threshold, merge_threshold, execution_time)