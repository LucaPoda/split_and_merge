import cv2
import numpy as np
from sortedcontainers import SortedDict
from itertools import chain

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


class SortedListWithCosts:
    def __init__(self):
        self.sorted_dict = SortedDict()

    def is_empty(self):
        return len(self.sorted_dict) == 0

    def add(self, link: "Link"):
        # Use (cost, id) as the key
        self.sorted_dict[(link.get_cost(), link.id)] = link

    def remove(self, link: "Link"):
        # Remove using (cost, id) as the key
        key = (link.get_cost(), link.id)
        if key in self.sorted_dict:
            del self.sorted_dict[key]
            return True
        else:
            return False

    def update(self, link: "Link"):
        if self.remove(link):
            # Update cost by removing and re-adding with new cost
            link.reset_cost()
            self.add(link)
            return True
        else: 
            return False
        
    def refresh(self, link: "Link"):
      
        done = set()
        for l in chain(link.region_a.get_root().links, link.region_b.get_root().links):
            if l.id in done:
                continue
            if l.region_a.get_root() == l.region_b.get_root():
                self.remove(l)
                continue
            
            self.update(l)

            done.add(l.id)

    def get_lowest_cost_link(self):
        _ , lowest_node = self.sorted_dict.popitem(0)
        return lowest_node
    def __repr__(self):
        return str([f"(Cost: {key[0]}, ID: {key[1]})" for key in self.sorted_dict.keys()])


class Node: 
    class TreeIterator:
        def __init__(self, root: "ParentNode"):
            self.stack = [root]

        def __iter__(self):
            return self

        def __next__(self) -> "Leaf":
            while self.stack:
                node = self.stack.pop()
                if isinstance(node, Leaf):
                    return node
                elif isinstance(node, ParentNode):
                    # Add children to the stack (right child first so left is processed first)
                    self.stack.append(node.child_b)
                    self.stack.append(node.child_a)
            raise StopIteration

    def __init__(self):
        self.parent : "ParentNode" = None
        self.size : 0

        self.n = None
        self.mean = None
        self.variance = None

    def set_parent(self, parent: "ParentNode"):
        self.parent = parent

    def get_parent(self) -> "ParentNode":
        return self.parent
    
    def get_root(self) -> "ParentNode":
        if self.get_parent() == None:
            return self
        
        return self.get_parent().get_root() 
    
    def get_stats(self):
        return

    def get_mask(self, image_shape) -> cv2.typing.MatLike:
        return

    def __iter__(self):
        return Node.TreeIterator(self)
    
    def __repr__(self):
        return str.join("-", [str(r.id) for r in self])


class ParentNode(Node): 
    def __init__(self, child_a: "Node", child_b: "Node"):
        super().__init__()
        self.child_a: Node = child_a
        self.child_b: Node = child_b
        self.child_a.set_parent(self)
        self.child_b.set_parent(self)

        self.links = chain(child_a.links, child_b.links)
        self.size = self.child_a.size + self.child_b.size

        self.n = self.child_a.n + self.child_b.n
        self.value = child_a.value + child_b.value
        self.mean = self.value / self.n


    def get_stats(self) -> float:
        if self.n != None and self.mean != None and self.variance != None:
            return self.variance

        # Compute variances
        N1, mu1, var1 = self.child_a.n, self.child_a.mean, self.child_a.get_stats()
        N2, mu2, var2 = self.child_b.n, self.child_b.mean, self.child_b.get_stats()

        # Calcolo della media combinata
        self.mean = (N1 * mu1 + N2 * mu2) / (self.n)
        
        # Calcolo della varianza combinata
        self.variance = (
            (N1 * (var1 + (mu1 - self.mean)**2) + N2 * (var2 + (mu2 - self.mean)**2))
            / (N1 + N2)
        )
        
        return self.variance
    
    def calculate_cost_old(self) -> float:
        for i, region in enumerate(self):
            if i == 0:
                cumulative_n = region.region.get_size()
                cumulative_mean = region.region.mean
                cumulative_variance = region.region.variance
            else:
                n_i = region.region.get_size()
                mean_i = region.region.mean
                variance_i = region.region.variance
                
                # Total number of elements after merging
                new_n = cumulative_n + n_i

                # Update cumulative mean
                new_mean = (cumulative_n * cumulative_mean + n_i * mean_i) / new_n

                # Update cumulative variance
                new_variance = (
                    (cumulative_n * cumulative_variance + n_i * variance_i) / new_n + (cumulative_n * n_i * (cumulative_mean - mean_i) ** 2) / (new_n ** 2)
                )

                # Update values for the next iteration
                cumulative_n, cumulative_mean, cumulative_variance = new_n, new_mean, new_variance

        return cumulative_variance

    def calculate_cost(self) -> float:
        
        mean = self.mean
        tot = 0
        for i, region in enumerate(self):
            tot += region.n*((region.mean-mean)**2)

        return tot/self.n

    def get_cost(self) -> float: 
        if self.cost != None:
            return self.cost

        n, _, variance = self.get_stats()        

        N1, _, var1 = self.child_a.get_stats()
        N2, _, var2 = self.child_b.get_stats()

         # Calcolo della varianza media delle regioni originali
        weighted_var = (N1 * var1 + N2 * var2) / (n)
        
        # Calcolo della perdita di omogeneità
        self.cost = variance - weighted_var

    def get_mask(self, image_shape) -> cv2.typing.MatLike:
        """
        Create a mask for a polygon made by adjacent rectangles.

        :param rectangles: List of rectangles, where each rectangle is a tuple (x, y, width, height).
        :param image_shape: Shape of the image (height, width) to define the mask size.
        :return: A binary mask (0 for background, 255 for the region covered by the rectangles).
        """
        # Create an empty black mask (same size as the image)
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Draw each rectangle on the mask
        for rect in self:
            x, y, w, h = rect.region.x, rect.region.y, rect.region.width, rect.region.height
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # -1 to fill the rectangle

        return mask


class Leaf(Node):
    def __init__(self, region: Region, i):
        super().__init__()
        self.region : Region = region
        self.links = []

        self.value = region.value
        self.n = self.region.get_size()
        self.mean = self.value / self.n
        self.size = 1
        self.id = i

    def get_stats(self):
        return self.region.variance
    
    def calculate_cost(self): 
        return self.get_stats()
    
    def calculate_cost_old(self):
        return self.get_stats()
    
    def get_mask(self, image_shape) -> cv2.typing.MatLike:
        """
        Create a mask for a polygon made by adjacent rectangles.

        :param rectangles: List of rectangles, where each rectangle is a tuple (x, y, width, height).
        :param image_shape: Shape of the image (height, width) to define the mask size.
        :return: A binary mask (0 for background, 255 for the region covered by the rectangles).
        """
        # Create an empty black mask (same size as the image)
        mask = np.zeros(image_shape, dtype=np.uint8)

        x, y, w, h = self.region.x, self.region.y, self.region.width, self.region.height
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # -1 to fill the rectangle

        return mask

    def add_link(self, link: "Link"):
        self.links.append(link)


class Link:
    n_links = 0
    def __init__(self, region_a: Leaf, region_b: Leaf):
        self.id = Link.n_links
        Link.n_links += 1

        self.region_a : Leaf = region_a
        self.region_b : Leaf = region_b
        region_a.add_link(self)
        region_b.add_link(self)

        self.size = region_a.size + region_b.size
        self.n = self.region_a.get_root().n + self.region_b.get_root().n

        self.cost = None
        self.mean = None
        self.variance = None

    def merge(self) -> ParentNode:
        region_a = self.region_a.get_root()
        region_b = self.region_b.get_root()

        if region_a == region_b:
            return
        
        return ParentNode(region_a, region_b)

    def get_cost(self) -> float: 
        if self.cost != None:
            return self.cost

        region_a = self.region_a.get_root()
        region_b = self.region_b.get_root()

        N1, mu1 = region_a.n, region_a.mean
        N2, mu2 = region_b.n, region_b.mean

        # Compute variances
        var1 = region_a.calculate_cost()
        var2 = region_b.calculate_cost()

        # Calcolo della media combinata
        self.mean = (N1 * mu1 + N2 * mu2) / (N1 + N2)
        
        # Calcolo della varianza combinata
        self.variance = (
            (N1 * (var1 + (mu1 - self.mean)**2) + N2 * (var2 + (mu2 - self.mean)**2)) / (N1 + N2)
        )
        
        # Calcolo della varianza media delle regioni originali
        weighted_var = (N1 * var1 + N2 * var2) / (N1 + N2)
        
        # Calcolo della perdita di omogeneità
        self.cost = self.variance - weighted_var

        return self.cost
    
    def reset_cost(self):
        self.cost = None
        return self.get_cost()

    def get_size(self):
        return self.region_a.get_root().n + self.region_b.get_root().n

    def __lt__(self, other):
        # Compare the cost property of the instances
        if isinstance(other, Link):
            return self.get_cost() < other.get_cost()
        return NotImplemented

    def __gt__(self, other):
        # Compare the cost property of the instances
        if isinstance(other, Link):
            return self.get_cost() > other.get_cost()
        return NotImplemented


class RAG: 
    def __init__(self, regions: list[Region]):
        self.links : SortedListWithCosts = SortedListWithCosts()
        # self.links : LinkHeap = LinkHeap()
        leaves = [Leaf(r, i) for i, r in enumerate(regions)]

        # adding links
        for i in range(0, len(regions)):
            for j in range(0, i):
                if regions[i].is_adjacent_to(regions[j]):
                    new_link = Link(leaves[i], leaves[j])
                    self.links.add(new_link)
    
    def get_all_regions(links: list[Link]) -> list[Node]:
        """Return all unique regions involved in the links."""
        unique_regions = set()
        
        # Iterate over each link and add the regions to the set
        for link in links:
            unique_regions.add(link.region_a.get_root())
            unique_regions.add(link.region_b.get_root())
        
        # Convert the set to a list and return it
        return list(unique_regions)

    def merge(self, alpha) -> list[Node]:

        link : Link = self.links.get_lowest_cost_link()
        result = []

        while not self.links.is_empty():
            merge_threshold = alpha / (link.get_size())
            
            # print(f"{link.variance/merge_threshold*100} - {link.variance} < {merge_threshold} - {link.cost} - {link.size}")

            if link.variance > merge_threshold:
                result.append(link)
            else:
                _ = link.merge()
                self.links.refresh(link)

            if not self.links.is_empty():
                link = self.links.get_lowest_cost_link()

        return RAG.get_all_regions(result)


def split(image: cv2.typing.MatLike, region: Region, threshold: int, max_depth=10, depth=0):
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


def merge(regions: list[Region], threshold):
    rag = RAG(regions)
    return rag.merge(threshold)


def split_and_merge(input_image: cv2.typing.MatLike, split_depth: int = 6, split_threshold: float = 250, merge_threshold: float = 6000000):
    initial_region = Region(0, 0, input_image.shape[1], input_image.shape[0])

    homogeneous_regions = split(input_image, initial_region, threshold=split_threshold, max_depth=split_depth)

    regions = merge(homogeneous_regions, threshold=merge_threshold)


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

