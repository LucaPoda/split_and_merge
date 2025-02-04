class Region:
    def __init__(self, x, y, width, height):
        """
        Initialize a Region instance.

        :param x: X-coordinate of the top-left corner of the region.
        :param y: Y-coordinate of the top-left corner of the region.
        :param width: Width of the region.
        :param height: Height of the region.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.subregions = []  # Initially empty

    def split_into_subregions(self):
        """
        Split the current region into subregions.

        - If the region is large enough, it is split into 4 equal subregions.
        - If the region is too small to be split into 4, it will be split into 2 subregions
        (either horizontally or vertically, based on dimensions).
        - If the region is too small to be split at all, an exception is raised.
        """
        if self.width < 2 and self.height < 2:
            raise ValueError("Region is too small to split into any subregions.")

        if self.width >= 2 and self.height >= 2:
            # Split into 4 subregions
            half_width = self.width // 2
            half_height = self.height // 2

            self.subregions = [
                Region(self.x, self.y, half_width, half_height),  # Top-left
                Region(self.x + half_width, self.y, self.width - half_width, half_height),  # Top-right
                Region(self.x, self.y + half_height, half_width, self.height - half_height),  # Bottom-left
                Region(self.x + half_width, self.y + half_height, self.width - half_width, self.height - half_height),  # Bottom-right
            ]
        elif self.width >= 2:
            # Split into 2 vertical subregions
            half_width = self.width // 2

            self.subregions = [
                Region(self.x, self.y, half_width, self.height),  # Left
                Region(self.x + half_width, self.y, self.width - half_width, self.height),  # Right
            ]
        elif self.height >= 2:
            # Split into 2 horizontal subregions
            half_height = self.height // 2

            self.subregions = [
                Region(self.x, self.y, self.width, half_height),  # Top
                Region(self.x, self.y + half_height, self.width, self.height - half_height),  # Bottom
            ]


    def __repr__(self):
        """
        String representation of the Region.
        """
        return f"Region(x={self.x}, y={self.y}, width={self.width}, height={self.height}, subregions={len(self.subregions)})"
    
import numpy as np
from skimage.color import rgb2gray
from skimage.draw import rectangle
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

def calculate_homogeneity(image, region):
    """
    Calculate the homogeneity of a region in an image based on pixel variance.

    :param image: The input image as a NumPy array.
    :param region: A Region object representing the region of interest.
    :return: Homogeneity score (lower is more homogeneous).
    """

    # Extract the region of interest
    x, y, width, height = region.x, region.y, region.width, region.height
    roi = image[y:y + height, x:x + width]

    # Calculate variance of the region
    variance = np.var(roi)

    return variance

def split_image_until_homogeneous(image, region, threshold, max_depth=10, depth=0):
    """
    Recursively split an image region until it reaches a specified homogeneity threshold.

    :param image: The input image as a NumPy array.
    :param region: A Region object representing the region of interest.
    :param threshold: Homogeneity threshold (lower values mean stricter requirements).
    :param max_depth: Maximum recursion depth to avoid infinite splitting.
    :param depth: Current depth of recursion (used internally).
    :return: A list of homogeneous regions.
    """

    if len(image.shape) == 3:  # If the image has multiple channels (e.g., RGB)
        gray_image = rgb2gray(image)
        gray_image = img_as_ubyte(gray_image)
    else:
        gray_image = image

    # Calculate the homogeneity of the current region
    homogeneity = calculate_homogeneity(gray_image, region)

    # Stop splitting if homogeneity is below threshold or max depth is reached
    if homogeneity <= threshold or depth >= max_depth:
        return [region]
    
    # Split the region into 4 subregions
    region.split_into_subregions()

    # Recursively process each subregion
    homogeneous_regions = []
    for subregion in region.subregions:
        homogeneous_regions.extend(
            split_image_until_homogeneous(image, subregion, threshold, max_depth, depth + 1)
        ) 

    return homogeneous_regions

import cv2
import numpy as np

# Assume the Region class and calculate_homogeneity function are already defined
# Include the split_image_until_homogeneous function as well

def visualize_homogeneous_regions(image, homogeneous_regions):
    """
    Visualize homogeneous regions on an image by drawing rectangles.

    :param image: The input image as a NumPy array.
    :param homogeneous_regions: List of Region objects representing homogeneous regions.
    :return: Image with rectangles drawn around homogeneous regions.
    """
    output_image = image.copy()
    for region in homogeneous_regions:
        cv2.rectangle(
            output_image,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            (0, 255, 0),  # Green color
            2  # Line thickness
        )
    return output_image

def visualize_homogeneous_regions_with_avg_color(image, homogeneous_regions):
    """
    Visualize homogeneous regions on an image by filling each region
    with its average color.

    :param image: The input image as a NumPy array.
    :param homogeneous_regions: List of Region objects representing homogeneous regions.
    :return: Image with regions filled with their average color.
    """
    # Create a blank canvas with the same dimensions as the original image
    output_image = image.copy()

    for region in homogeneous_regions:
        # Extract the region of interest
        x, y, width, height = region.x, region.y, region.width, region.height
        roi = image[y:y + height, x:x + width]

        # Calculate the average color of the region
        avg_color = cv2.mean(roi)[:3]  # Ignore the alpha channel if present

        # Fill the region with the average color
        output_image[y:y + height, x:x + width] = np.uint8(avg_color)

    return output_image
