import cv2
import numpy as np
import pathlib

import matplotlib.pyplot as plt

# Assume the Region class and calculate_homogeneity function are already defined
# Include the split_image_until_homogeneous function as well

def visualize_homogeneous_regions(image, homogeneous_regions):
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

from utils.split_and_merge import Node, Leaf, ParentNode

import cv2
import numpy as np


def visualize_regions_with_avg_color(input_image, regions: list[Node]):
    # Create a blank canvas with the same dimensions as the original image
    image_with_mean_color = np.zeros(input_image.shape, dtype=np.uint8)

    for region in regions:
        mask = region.get_mask(input_image.shape)

        # Ensure the mask is single-channel (8-bit grayscale)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] 

        # Compute the mean color values inside the masked region
        mean_value = cv2.mean(input_image, mask=mask)

        mean_bgr = mean_value[:3]  # Get the BGR values (ignore alpha)

        # Apply the mean color to the region defined by the mask
        image_with_mean_color[mask == 255] = mean_bgr

    return image_with_mean_color

def visualize_regions_with_borders(input_image, regions: list[Node]):
    # Create a blank canvas with the same dimensions as the original image
    image_with_mean_color = np.zeros(input_image.shape, dtype=np.uint8)
    for region in regions:
        mask = region.get_mask(input_image.shape)

        # Ensure the mask is single-channel (8-bit grayscale)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] 

        # Compute the mean color values inside the masked region
        mean_value = cv2.mean(input_image, mask=mask)

        mean_bgr = mean_value[:3]  # Get the BGR values (ignore alpha)

        # Apply the mean color to the region defined by the mask
        image_with_mean_color[mask == 255] = mean_bgr

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours
        image_with_borders = cv2.drawContours(image_with_mean_color, contours, -1, (0, 255, 0), 2)

    return image_with_borders

import pathlib
import cv2
import os
import re

def save_image(image, image_path, split_depth, split_threshold, merge_threshold, time):
    path = pathlib.Path(image_path)
    image_name = path.stem
    image_ext = path.suffix

    time = int(time * 1000)  # Convert time to milliseconds
    output_path = f"previous_runs/{image_name}_{split_depth}_{split_threshold}_{merge_threshold}_{time}{image_ext}"

    # Regex pattern to match existing runs with the same parameters
    pattern = re.compile(rf"{image_name}_{split_depth}_{split_threshold}_{merge_threshold}_(\d+){image_ext}")

    existing_files = []
    
    # Check existing files in the folder
    for file in os.listdir("previous_runs"):
        match = pattern.match(file)
        if match:
            existing_files.append((file, int(match.group(1))))  # Store filename and execution time

    # Delete all slower runs (keep only the fastest)
    for file, exec_time in existing_files:
        if exec_time > time:
            os.remove(os.path.join("previous_runs", file))
            print(f"Removed slower run: {file}")

    # Save the new image
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")



def plot_previous_runs():
    import os
    import re
    import matplotlib.pyplot as plt
    from PIL import Image

    # Folder containing processed images
    folder_path = "previous_runs"
    # Folder containing original images
    original_folder = "images"

    # Regex pattern to extract parameters from filenames
    pattern = re.compile(r"(\w+)_(\d+)_(\d+)_(\d+)_(\d+)\.(jpeg|png)")

    # Dictionary to organize images
    image_data = {}

    # Read filenames and organize data
    for filename in sorted(os.listdir(folder_path)):
        match = pattern.match(filename)
        if match:
            base_name, param1, param2, param3, exec_time, ext = match.groups()
            key = (param1, param2, param3)  # Columns based on these params
            if base_name not in image_data:
                image_data[base_name] = {}  # Rows based on base_name
            image_data[base_name][key] = (os.path.join(folder_path, filename), exec_time)

    # Get unique column keys in sorted order
    column_keys = sorted(set(k for d in image_data.values() for k in d))

    # Add an extra column for the original image
    num_cols = len(column_keys) + 1
    fig, axes = plt.subplots(len(image_data), num_cols, figsize=(5 * num_cols, 5 * len(image_data)))

    # Ensure axes is always a 2D array
    if len(image_data) == 1:
        axes = [axes]

    # Plot images
    for row_idx, (base_name, col_data) in enumerate(image_data.items()):
        # Load and plot the original image
        original_path = os.path.join(original_folder, f"{base_name}.png")  # Assuming original images are PNG
        if not os.path.exists(original_path):
            original_path = os.path.join(original_folder, f"{base_name}.jpeg")  # Check for JPEG

        ax = axes[row_idx][0]
        if os.path.exists(original_path):
            original_img = Image.open(original_path)
            ax.imshow(original_img)
        ax.set_title(base_name, fontsize=14, fontweight="bold")
        ax.axis("off")

        # Plot processed images
        for col_idx, col_key in enumerate(column_keys):
            ax = axes[row_idx][col_idx + 1]
            if col_key in col_data:
                img_path, exec_time = col_data[col_key]
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f"Time: {exec_time} ms", fontsize=10)
            ax.axis("off")

    # Set column titles for parameters (properly centered)
    for col_idx, col_key in enumerate(column_keys):
        param_label = f"({col_key[0]}, {col_key[1]}, {col_key[2]})"
        col_x_pos = (col_idx + 1) / num_cols  # Adjusted to fit exactly above each column
        fig.text(0.1 + col_x_pos, 0.98, param_label, ha="center", fontsize=12, fontweight="bold")

    # General layout adjustments
    plt.tight_layout()
    plt.show()
