import cv2
import numpy as np

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
        self.mean = None
        self.variance = None
        self.value = None
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

    def calculate_mean_variance(self, image):
        """
        Calculate the homogeneity of a region in an image based on pixel variance.

        :param image: The input image as a NumPy array.
        :param region: A Region object representing the region of interest.
        :return: Homogeneity score (lower is more homogeneous).
        """
        # Ensure the image is grayscale
        if len(image.shape) == 3:  # If the image has multiple channels (e.g., RGB)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Extract the region of interest
        x, y, width, height = self.x, self.y, self.width, self.height
        roi = gray_image[y:y + height, x:x + width]

        # Calculate variance of the region
        self.variance = np.var(roi, ddof=1)

        # Calculate mean of the image
        self.mean = np.mean(roi)

        # Calculate sum of all points
        self.value = np.sum(roi)

    def is_adjacent_to(self, other: "Region"):
        """
        Determine if two regions are adjacent.

        :param region1: A Region object.
        :param region2: A Region object.
        :return: True if the regions are adjacent, False otherwise.
        """
        # Horizontal adjacency (region1 above or below region2)
        horizontally_adjacent = (
            self.y + self.height >= other.y and  # region1's bottom edge touches or overlaps region2's top edge
            other.y + other.height >= self.y     # region2's bottom edge touches or overlaps region1's top edge
        )
        horizontal_overlap = (
            self.x <= other.x + other.width and  # region1's left edge touches or overlaps region2's right edge
            self.x + self.width >= other.x     # region1's right edge touches or overlaps region2's left edge
        )

        # Vertical adjacency (region1 left or right of region2)
        vertically_adjacent = (
            self.x + self.width >= other.x and  # region1's right edge touches or overlaps region2's left edge
            other.x + other.width >= self.x     # region2's right edge touches or overlaps region1's left edge
        )
        vertical_overlap = (
            self.y <= other.y + other.height and  # region1's top edge touches or overlaps region2's bottom edge
            self.y + self.height >= other.y     # region1's bottom edge touches or overlaps region2's top edge
        )

        return (horizontally_adjacent and horizontal_overlap) or (vertically_adjacent and vertical_overlap)

    def get_size(self):
        return self.width * self.height

    def __repr__(self):
        """
        String representation of the Region.
        """
        return f"Region(x={self.x}, y={self.y}, width={self.width}, height={self.height}, subregions={len(self.subregions)})"
    
def split(image, region: Region, threshold: int, max_depth=10, depth=0):
    """
    Recursively split an image region until it reaches a specified homogeneity threshold.

    :param image: The input image as a NumPy array.
    :param region: A Region object representing the region of interest.
    :param threshold: Homogeneity threshold (lower values mean stricter requirements).
    :param max_depth: Maximum recursion depth to avoid infinite splitting.
    :param depth: Current depth of recursion (used internally).
    :return: A list of homogeneous regions.
    """
    # Calculate the homogeneity of the current region
    region.calculate_mean_variance(image)

    # Stop splitting if homogeneity is below threshold or max depth is reached
    if region.variance <= threshold or depth >= max_depth:
        return [region]
    
    # Split the region into 4 subregions
    region.split_into_subregions()

    # Recursively process each subregion
    homogeneous_regions = []
    for subregion in region.subregions:
        homogeneous_regions.extend(
            split(image, subregion, threshold, max_depth, depth + 1)
        ) 

    return homogeneous_regions
