# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.split import Region
from utils.split import split
from utils.lib import visualize_homogeneous_regions
from utils.lib import visualize_homogeneous_regions_with_avg_color

# %%
image_path = "images/image1.jpeg"  # Replace with the path to your image
# image_path = "images/tetris.png"  # Replace with the path to your image

# Load an example image
input_image = cv2.imread(image_path) 

# %%
# Define the initial region
initial_region = Region(0, 0, input_image.shape[1], input_image.shape[0])
# Split the image into homogeneous regions
homogeneous_regions = split(input_image, initial_region, threshold=250, max_depth=6)
# %%
output_with_rectangles = visualize_homogeneous_regions(input_image, homogeneous_regions)
plt.imshow(cv2.cvtColor(output_with_rectangles, cv2.COLOR_BGR2RGB))
plt.show()

# %%

output_with_avg_color = visualize_homogeneous_regions_with_avg_color(input_image, homogeneous_regions)
plt.imshow(cv2.cvtColor(output_with_avg_color, cv2.COLOR_BGR2RGB))
plt.show()

# %%
from utils.merge import merge

# %%
regions = merge(homogeneous_regions, input_image, 20)

# %%
print(regions)

# %%

from utils.lib import visualize_regions_with_avg_color

image_with_borders = visualize_regions_with_avg_color(input_image, regions)
plt.imshow(cv2.cvtColor(image_with_borders, cv2.COLOR_BGR2RGB))
plt.show()


