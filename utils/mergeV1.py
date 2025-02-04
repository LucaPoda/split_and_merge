import heapq

from utils.split import Region

import numpy as np

class Link:
    def __init__(self, region_a: "RegionNode", region_b: "RegionNode"):
        self.region_a: RegionNode = region_a
        self.region_b: RegionNode = region_b
        
        # Add this link to each region's set of links
        region_a.links.add(self)
        region_b.links.add(self)

        self.mean: float = 0
        self.variance: float = 0
        self.cost: float = 0
        self.mean, self.variance, self.cost = self.estimate_cost()

    def compute_cost(self, image) -> float:
        """
        Compute the precise cost (variance) of merging all regions in this RegionNode.
        
        :param image: Input image as a NumPy array (grayscale).
        :return: Precise cost (variance) of merging the regions.
        """
        # Combine all regions into a single list of pixel values
        pixel_a = []
        for region in self.region_a.regions:
            roi = image[region.y:region.y + region.height, region.x:region.x + region.width].flatten()
            pixel_a.extend(roi)

        pixel_b = []
        for region in self.region_b.regions:    
            roi = image[region.y:region.y + region.height, region.x:region.x + region.width].flatten()
            pixel_b.extend(roi)
        
        merged_pixels = pixel_a + pixel_b
        # Convert to a NumPy array and compute variance
        pixel_a = np.array(pixel_a)
        pixel_b = np.array(pixel_b)
        merged_pixels = np.array(merged_pixels)

        var_a = np.var(pixel_a)
        var_b = np.var(pixel_b)
        merged_variance = np.var(merged_pixels)

        loss = merged_variance - (var_a*len(pixel_a) + var_b*len(pixel_b)) / (len(pixel_a)+len(pixel_b))

        return loss

    def combined_variance(self):
        # Compute variances
        N1, mu1, var1 = self.region_a.estimate_cost()  # ddof=1 for sample variance
        N2, mu2, var2 = self.region_b.estimate_cost()

        # Calcolo della media combinata
        mu_c = (N1 * mu1 + N2 * mu2) / (N1 + N2)
        
        # Calcolo della varianza combinata
        sigma_c2 = (
            (N1 * (var1 + (mu1 - mu_c)**2) + N2 * (var2 + (mu2 - mu_c)**2))
            / (N1 + N2)
        )
        
        # Calcolo della varianza media delle regioni originali
        weighted_var = (N1 * var1 + N2 * var2) / (N1 + N2)
        
        # Calcolo della perdita di omogeneità
        delta_var = sigma_c2 - weighted_var

        # print(f"var1: {var1}, var2: {var2}, sigma_c: {sigma_c2}, weighted_var: {weighted_var}, delta: {delta_var}")
        
        return mu_c, sigma_c2, delta_var

    def estimate_cost(self):      # Compute number of pixels
        

        # Compute pooled variance
        # pooled_variance = ((pixel_a - 1) * var_a + (pixel_b - 1) * var_b) / (pixel_a + pixel_b - 2)

        # loss = pooled_variance - (var_a*pixel_a + var_b*pixel_b) / (pixel_a+pixel_b)

        # print(f"{loss} = {pooled_variance} - ({var_a} * {pixel_a} + {var_b} * {pixel_b}) / {pixel_a+pixel_b}")

        mu_c, variance, loss = self.combined_variance()

        return mu_c, variance, loss

    def __lt__(self, other):
        # Compare the cost property of the instances
        if isinstance(other, Link):
            return self.cost < other.cost
        return NotImplemented

    def __gt__(self, other):
        # Compare the cost property of the instances
        if isinstance(other, Link):
            return self.cost > other.cost
        return NotImplemented

    def __repr__(self):
        return f"{self.region_a.id} - {self.region_b.id}"

class LinkHeap:
    def __init__(self):
        """Initialize the priority queue (min-heap) for links."""
        self.heap = []  # The heap containing links as (cost, Link)

    def add_link(self, link: Link):
        """
        Add a link to the heap.

        :param link: The Link object to add.
        """
        heapq.heappush(self.heap, link)

    def get_lowest_cost_link(self):
        """
        Retrieve the link with the lowest cost, ensuring it is valid.

        :return: The Link object with the lowest cost, or None if no valid links exist.
        """
        link = heapq.heappop(self.heap)

        return link

    def remove_links_involving(self, region: "RegionNode"):
        """
        Remove all links involving the specified region.
        
        :param region: The region to remove links for.
        """
        self.heap = [link for link in self.heap if link.region_a != region and link.region_b != region]

        heapq.heapify(self.heap)
            

    def is_empty(self):
        return len(self.heap) == 0

    def __repr__(self):
        return f"{self.heap}"

class RegionNode:
    def __init__(self, id: str, regions: list[Region]):
        """
        Initialize a RegionAdjacencyGraphNode instance.

        :param region: A Region object representing the initial region.
        :param adjacent_nodes: A list of initial adjacent nodes (default is None).
        """

        self.id : str = id
        self.regions : list[Region] = regions  # Vector of regions (initially contains only the region)
        self.links : set[Link] = set()  # Set of links connected to this node

        self.mean = np.mean([r.mean for r in self.regions ])

    def compute_cost(self, image):
        """
        Compute the precise cost (variance) of merging all regions in this RegionNode.
        
        :param image: Input image as a NumPy array (grayscale).
        :return: Precise cost (variance) of merging the regions.
        """
        # Combine all regions into a single list of pixel values
        merged_pixels = []
        for region in self.regions:
            roi = image[region.y:region.y + region.height, region.x:region.x + region.width].flatten()
            merged_pixels.extend(roi)
        
        # Convert to a NumPy array and compute variance
        merged_pixels = np.array(merged_pixels)
        precise_variance = np.var(merged_pixels)
        return precise_variance

    def estimate_cost(self):
        # Initialize with the first region
        cumulative_n = self.regions[0].height * self.regions[0].height
        cumulative_mean = self.regions[0].mean
        cumulative_variance = self.regions[0].variance

        for i in range(1, len(self.regions)):
            n_i = self.regions[i].height * self.regions[i].width 
            mean_i = self.regions[i].mean
            variance_i = self.regions[i].variance
            
            # Total number of elements after merging
            new_n = cumulative_n + n_i

            # Update cumulative mean
            new_mean = (cumulative_n * cumulative_mean + n_i * mean_i) / new_n

            # Update cumulative variance
            new_variance = (
                (cumulative_n * cumulative_variance + n_i * variance_i) / new_n +
                (cumulative_n * n_i * (cumulative_mean - mean_i) ** 2) / (new_n ** 2)
            )

            # Update values for the next iteration
            cumulative_n, cumulative_mean, cumulative_variance = new_n, new_mean, new_variance

        return cumulative_n, cumulative_mean, cumulative_variance

    def merge(self, other_node: "RegionNode"):
        """
        Merge another node into this node by combining its regions and adjacent nodes.

        :param other_node: A RegionAdjacencyGraphNode to merge with the current node.
        """

        # Merge regions
        self.regions.extend(other_node.regions)

        # Merge adjacent nodes, avoiding duplicates and excluding the current node
        for node in other_node.links:
            if node is not self and node not in self.links:
                self.links.append(node)

        # Remove references to the merged node from its former adjacent nodes
        for node in other_node.links:
            if other_node in node.adjacent_nodes:
                node.adjacent_nodes.remove(other_node)

    def get_size(self):
        size = 0
        for region in self.regions:
            size += region.height * region.width

        return size

    def __repr__(self):
        return (
            f"{self.id}"
        )
    
def are_regions_adjacent(region_a: Region, region_b: Region):
    """
    Determine if two regions are adjacent.

    :param region1: A Region object.
    :param region2: A Region object.
    :return: True if the regions are adjacent, False otherwise.
    """
    # Horizontal adjacency (region1 above or below region2)
    horizontally_adjacent = (
        region_a.y + region_a.height >= region_b.y and  # region1's bottom edge touches or overlaps region2's top edge
        region_b.y + region_b.height >= region_a.y     # region2's bottom edge touches or overlaps region1's top edge
    )
    horizontal_overlap = (
        region_a.x <= region_b.x + region_b.width and  # region1's left edge touches or overlaps region2's right edge
        region_a.x + region_a.width >= region_b.x     # region1's right edge touches or overlaps region2's left edge
    )

    # Vertical adjacency (region1 left or right of region2)
    vertically_adjacent = (
        region_a.x + region_a.width >= region_b.x and  # region1's right edge touches or overlaps region2's left edge
        region_b.x + region_b.width >= region_a.x     # region2's right edge touches or overlaps region1's left edge
    )
    vertical_overlap = (
        region_a.y <= region_b.y + region_b.height and  # region1's top edge touches or overlaps region2's bottom edge
        region_a.y + region_a.height >= region_b.y     # region1's bottom edge touches or overlaps region2's top edge
    )

    return (horizontally_adjacent and horizontal_overlap) or \
           (vertically_adjacent and vertical_overlap)

def merge(regions: list[Region], image, merge_threshold):
    # RAG init
    nodes : list[RegionNode] = [RegionNode(f"{i}", [r]) for i, r in enumerate(regions)] 
    links : LinkHeap = LinkHeap()

    # adding links
    for i in range(0, len(regions)):
        for j in range(0, i):
            if are_regions_adjacent(regions[i], regions[j]):
                new_link = Link(nodes[i], nodes[j])

                links.add_link(new_link)

    n = 0
    outlier_threshold = 80
    link = links.get_lowest_cost_link()
    result = []
    while not links.is_empty():
        n = sum([r.height * r.width for r in (link.region_a.regions + link.region_b.regions)])
        print(f"{link.variance} < {merge_threshold}")

        if link.variance > merge_threshold:
            result.append(link)
            print("ignore")
        else:
            merge_regions(link.region_a, link.region_b, links, image)
            print("merge")

        if links.heap.__len__() > 0:
            link = links.get_lowest_cost_link()

    print(f"{link.cost} - {link.mean} - {link.variance} < {merge_threshold}")

    return get_all_regions(result)
    
def get_all_regions(links):
    """Return all unique regions involved in the links."""
    unique_regions = set()
    
    # Iterate over each link and add the regions to the set
    for link in links:
        unique_regions.add(link.region_a)
        unique_regions.add(link.region_b)
    
    # Convert the set to a list and return it
    return list(unique_regions)

import time

def merge_regions(region_a: RegionNode, region_b: RegionNode, link_heap: LinkHeap, image) -> RegionNode:
    """
    Merge two regions into a new region and update the link heap.

    :param region_a: The first region to merge.
    :param region_b: The second region to merge.
    :param link_heap: The LinkHeap instance managing all links.
    :return: The new merged region.
    """
    start_time = time.perf_counter()  # Record the start time

    # Create a new region with a combined ID and regions
    new_region_id = f"{region_a.id}_{region_b.id}"
    new_region = RegionNode(new_region_id, list(set(region_a.regions + region_b.regions)))

    current_time = time.perf_counter()
    print(f"Created new region (Elapsed: {current_time - start_time:.6f} seconds)")

    # Update all links associated with the merged regions
    updated_links = set()

    for link in list(region_a.links | region_b.links):
        # Skip the link connecting region_a and region_b (self-link after merging)
        if (link.region_a == region_a and link.region_b == region_b) or \
           (link.region_a == region_b and link.region_b == region_a):
            continue

        # Update the link to point to the new region
        if link.region_a in {region_a, region_b}:
            link.region_a = new_region
        if link.region_b in {region_a, region_b}:
            link.region_b = new_region

        # Add the updated link to the new region's links
        updated_links.add(link)

    current_time = time.perf_counter()
    print(f"Link updated (Elapsed: {current_time - start_time:.6f} seconds)")

    # Assign updated links to the new region
    new_region.links = updated_links

    # Remove the old regions from the heap and adjacent links
    link_heap.remove_links_involving(region_a)
    link_heap.remove_links_involving(region_b)

    current_time = time.perf_counter()
    print(f"Links removed (Elapsed: {current_time - start_time:.6f} seconds)")

    # Add new or updated links to the heap
    for link in updated_links:
        link.mean, link.variance, link.cost = link.estimate_cost()
        link_heap.add_link(link)

    current_time = time.perf_counter()
    print(f"Added new links (Elapsed: {current_time - start_time:.6f} seconds)")

    return new_region
