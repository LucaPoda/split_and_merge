from abc import abstractmethod

import heapq

from utils.split import Region

import time
import numpy as np

global_total_time = 0  # Accesso alla variabile globale

from sortedcontainers import SortedDict

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
        global global_total_time

        start_time = time.time()
        
        done = set()
        for l in link.region_a.get_root().links + link.region_b.get_root().links:
            if l.id in done:
                continue
            if l.region_a.get_root() == l.region_b.get_root():
                self.remove(l)
                continue
            
            self.update(l)

            done.add(l.id)


        elapsed_time = time.time() - start_time
        global_total_time += elapsed_time  # Accumula il tempo nella variabile globale
        print(f"time: {elapsed_time:.6f} s")                

    def get_lowest_cost_link(self):
        _ , lowest_node = self.sorted_dict.popitem(0)
        return lowest_node
    def __repr__(self):
        return str([f"(Cost: {key[0]}, ID: {key[1]})" for key in self.sorted_dict.keys()])

class LinkHeap:
    def __init__(self):
        """Initialize the priority queue (min-heap) for links."""
        self.heap = []  # The heap containing links as (cost, Link)

    def add(self, link: "Link"):
        """
        Add a link to the heap.

        :param link: The Link object to add.
        """
        heapq.heappush(self.heap, link)

    def get_lowest_cost_link(self) -> "Link":
        """
        Retrieve the link with the lowest cost, ensuring it is valid.

        :return: The Link object with the lowest cost, or None if no valid links exist.
        """
        link = heapq.heappop(self.heap)

        return link

    def refresh(self, link: "Link"):
        """
        Remove all links involving the specified region.
        
        :param region: The region to remove links for.
        """
        global global_total_time

        for l in self.heap:
            if (
                l.region_a == link.region_a or
                l.region_a == link.region_b or
                l.region_b == link.region_a or
                l.region_b == link.region_b
            ):
                
                start_time = time.time()
                
                l.reset_cost()
                
                elapsed_time = time.time() - start_time
                
                global_total_time += elapsed_time  # Accumula il tempo nella variabile globale

                print(f"time: {elapsed_time:.6f} s")



        heapq.heapify(self.heap)
        
        start_time = time.time()
        elapsed_time = time.time() - start_time
        
        global_total_time += elapsed_time  # Accumula il tempo nella variabile globale

        print(f"Heap size: {len(self.heap)}\t - time: {elapsed_time:.6f} s")
            

    def is_empty(self):
        return len(self.heap) == 0

    def __repr__(self):
        return f"{self.heap}"

class TreeIterator:
    def __init__(self, root: "ParentNode"):
        self.stack = [root]

    def __iter__(self):
        return self

    def __next__(self):
        while self.stack:
            node = self.stack.pop()
            if isinstance(node, Leaf):
                return node
            elif isinstance(node, ParentNode):
                # Add children to the stack (right child first so left is processed first)
                self.stack.append(node.child_b)
                self.stack.append(node.child_a)
        raise StopIteration

class Node: 
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
    
    @abstractmethod
    def get_stats(self):
        return
    
    def __iter__(self):
        return TreeIterator(self)
    
    def __repr__(self):
        return str.join("-", [str(r.id) for r in self])
        
class ParentNode(Node): 
    def __init__(self, child_a: "Node", child_b: "Node"):
        super().__init__()
        self.child_a: Node = child_a
        self.child_b: Node = child_b
        self.links = child_a.links
        self.links.extend(child_b.links)

        self.size = self.child_a.size + self.child_b.size

        self.n = self.child_a.n + self.child_b.n
        self.value = child_a.value + child_b.value
        self.mean = self.value / self.n

        self.child_a.set_parent(self)
        self.child_b.set_parent(self)

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
        
        # print(f"var1: {var1}, var2: {var2}, sigma_c: {sigma_c2}, weighted_var: {weighted_var}, delta: {delta_var}")
        
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

        # print(f"n1: {N1}, mu1: {mu1}, var1: {var1}, n2: {N2}, mu2: {mu2}, var2: {var2}, mean: {self.mean}, variance: {self.variance}, weighted_var: {weighted_var}, delta: {self.cost}")
        
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
    
    def merge(self, alpha):
        global global_total_time

        link : Link = self.links.get_lowest_cost_link()
        result = []

        print(link.n)
        while not self.links.is_empty():
            merge_threshold = alpha / link.get_size()
            print(f"{link.variance/merge_threshold*100} - {link.variance} < {merge_threshold} - {link.cost} - {link.size}")

            if link.variance > merge_threshold:
                result.append(link)
            else:
                _ = link.merge()
                self.links.refresh(link)

            if not self.links.is_empty():
                link = self.links.get_lowest_cost_link()

        return get_all_regions(result)
    
def get_all_regions(links: list[Link]):
    """Return all unique regions involved in the links."""
    unique_regions = set()
    
    # Iterate over each link and add the regions to the set
    for link in links:
        unique_regions.add(link.region_a.get_root())
        unique_regions.add(link.region_b.get_root())
    
    # Convert the set to a list and return it
    return list(unique_regions)

def merge(regions: list[Region], alpha):
    rag = RAG(regions)
    return rag.merge(alpha)