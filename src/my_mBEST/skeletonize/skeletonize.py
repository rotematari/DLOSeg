import numpy as np
import cv2

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from my_mBEST.utils.utils import *

from itertools import combinations
import numpy as np
from skimage.draw import line  # Assumed to be used for drawing connecting lines
import time
from scipy.spatial import distance_matrix
class Dlo_skeletonize:
    def __init__(self):
        self.mask = None
        self.skeleton = None
        self.distance = None
        
        self.end_point_kernel = np.array(([1, 1, 1], [1, 10, 1], [1, 1, 1]), dtype=np.uint16)
        self.adjacent_pixel_clusterer = DBSCAN(eps=2, min_samples=1)
        
        self.colors =[[0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [0, 255, 255], [255, 255, 0], [255, 0, 255],
                        [0, 127, 0], [0, 0, 127], [127, 0, 0],
                        [0, 127, 127], [127, 127, 0], [127, 0, 127],
                        [0, 64, 0], [0, 0, 64], [64, 0, 0],
                        [0, 64, 64], [64, 64, 0], [64, 0, 64]]
        self.skeleton = None
        self.intersections = None
        self.ends = None
        
        
    def set_image(self, mask,blur_size=5):
        
        self.image = mask # [H, W]
        self.blurred_image = cv2.blur(mask, (blur_size, blur_size))
        self.skeleton, self.distance = get_skeleton_from_mask(mask,method='medial_axis')
        
    def _set_params(self):
        self.delta = self.distance.max() * 2
        self.delta_i = int(round(self.distance.max() * 1.5))
        self.epsilon = self.distance.max() * 10

    def _detect_keypoints(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Detects keypoints in a skeletonized image by applying a convolution
        with a kernel to identify endpoints and intersections.
        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays of endpoints and intersections.
        """
        # Use cv2.copyMakeBorder to pad the image with zeros
        padded_img = cv2.copyMakeBorder(self.skeleton, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
        
        # Apply the convolution filter with explicit border handling to keep consistency
        res = cv2.filter2D(src=padded_img, ddepth=-1, kernel=self.end_point_kernel, borderType=cv2.BORDER_CONSTANT)
        
        # Endpoints: combine conditions where the result equals 10 or 11
        self.ends = np.argwhere((res == 10) | (res == 11)) - 1
        # Intersections: pixels with a result greater than 12
        self.intersections = np.argwhere(res > 12) - 1


    def _prune_split_ends_from_inerections(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Prune noisy branch segments from a skeletonized image and update the endpoints and intersections.
        
        The function starts at each intersection and follows each branch until either:
        - It reaches an endpoint (no more connected neighbors), or
        - It encounters another intersection.
        
        If the branch length is below a given threshold, the branch is pruned (removed from the skeleton).
        If all branches emerging from an intersection are pruned, then the intersection itself is pruned.
        
        Parameters:
            skeleton (np.ndarray): Binary image representing the skeleton (nonzero pixels are part of the skeleton).
            ends (np.ndarray): Array of endpoint coordinates (e.g. shape (N, 2)).
            intersections (np.ndarray): Array of intersection coordinates (e.g. shape (M, 2)).
        
        Returns:
            tuple[np.ndarray, np.ndarray]: Updated endpoints and intersections arrays after pruning.
        """
        
        # Set branch length threshold: use self.branch_threshold if defined, else default to 10.
        branch_threshold = getattr(self, 'delta', 10)
        # Convert intersections and endpoints arrays into sets of tuples for fast membership testing.
        intersections_set = {tuple(coord) for coord in self.intersections}
        ends_set = {tuple(coord) for coord in self.ends}
        
        interactions_pruned = []
        ends_pruned = []
        # For each intersection pixel, explore each branch.
        for inter in intersections_set:
            # Get the neighboring pixels that are part of the skeleton.
            neighbors = get_neighbors_global(img=self.skeleton,pixel_coord=inter) 
            branchs_to_prune = np.ones((len(neighbors),), dtype=bool)
            branchs = []
            # Process each branch emerging from the intersection.
            for i,neighbor in enumerate(neighbors):
                branch_pixels = [tuple(neighbor)]
                prev = inter
                current = neighbor
                # Follow the branch until an endpoint or another intersection is reached.
                while True:
                    nb = get_neighbors_global( self.skeleton , current)
                    # Exclude the pixel we just came from to avoid backtracking.
                    nb = [tuple(p) for p in nb if tuple(p) != tuple(prev)]
                    # If there are no additional neighbors, we are at the end of the branch.
                    if not nb:
                        break
                    # If one of the neighbors is an intersection, end the branch.
                    if any(p in intersections_set for p in nb):
                        branch_pixels.append(nb[0])
                        branchs_to_prune[i] = False
                        break
                    # Otherwise, continue following the branch.
                    next_pixel = nb[0]
                    branch_pixels.append(next_pixel)
                    prev, current = current, next_pixel
                    if len(branch_pixels) > branch_threshold:
                        branchs_to_prune[i] = False
                        break
                branchs.append(branch_pixels)
            
            if not any(branchs_to_prune) :                    
                nb = [tuple(p) for p in nb if tuple(p) != prev]
                interactions_pruned.append(tuple(list(inter)))
            
            # Prune the branch if its length is below the threshold.
            for i,branch in enumerate(branchs):
                if branchs_to_prune[i]:
                    for x, y in branch:
                        self.skeleton[x, y] = 0

                # if all the branches are pruned, remove the intersection itself.
                if all(branchs_to_prune):
                    self.skeleton[inter[0], inter[1]] = 0

        for end in ends_set:
            branch_pixels = [end]
            current = end
            prev = end
            prune = True
            while True:
                nb = get_neighbors_global( self.skeleton,current)
                neighbors = [tuple(p) for p in nb if tuple(p) != prev]
                
                if len(neighbors) == 1:
                    if not any(p in intersections_set for p in neighbors):
                        branch_pixels.append(neighbors[0])
                        prev = current
                        current = neighbors[0]
                    else:
                        prune = False
                        break
                else:
                    if len(branch_pixels) > branch_threshold:
                        prune = False
                    break
                if len(branch_pixels) > branch_threshold:
                    prune = False
                    break
            if prune:
                for x, y in branch_pixels:
                    self.skeleton[x, y] = 0
            else:
                ends_pruned.append(tuple(end))
                
        self.ends = ends_pruned
        self.intersections = interactions_pruned

    def _cluster_and_match_intersections(self) -> tuple[list, dict]:
        """
        Cluster adjacent intersection pixels using DBSCAN, and then match and merge Y-branch intersections.
        
        This function performs a two-phase process:
        1. Cluster adjacent intersection pixels into a set of averaged points.
        2. Examine each averaged intersection points local window in the skeleton (sk) to extract branch endpoints.
            - If the window yields 3 segments, classify as a Y-intersection.
            - If 4 segments are found, consider it an already correct (X) intersection.
            Then, pair Y-intersections based on the Euclidean distance between them.
            Closest pairs (within a threshold self.epsilon) are matched, a new intersection is computed,
            and associated branch endpoints are recorded.
        
        Parameters:
            intersections (np.ndarray): Array of intersection pixel coordinates.
            sk (np.ndarray): The skeleton image used for extracting local boundary information.
        
        Returns:
            correct_intersections (list): List of intersection coordinates that are either X-intersections or unmatched Y-intersections.
            new_intersections (dict): Dictionary with keys:
                "inter": List of new intersection coordinates computed from matched Y-branches.
                "ends": List of paired endpoints for the matched intersections. 4 ends for each intersection.
                "matches": List of matched Y-branch pairs (coordinates).
        """
        # Initialize temporary lists and dictionaries to store results.
        temp_intersections = []  # Averaged intersection points from each DBSCAN cluster.
        correct_intersections = []  # Already correct intersections (X-shaped).
        Y_intersections = {"inter": [], "ends": []}  # Y-intersections needing matching.
        new_intersections = {"inter": [], "ends": [],"matches": []}  # New intersections from matching Y-branches.
        np_intersections = np.asarray(self.intersections)
        # --- Phase 1: Clustering via DBSCAN ---
        # Cluster adjacent intersection pixels.
        self.adjacent_pixel_clusterer.fit(np_intersections)
        
        for label in np.unique(self.adjacent_pixel_clusterer.labels_):
            # Average the pixels in each cluster and round to uint16.
            cluster_mean = np.round(np.mean(np_intersections[self.adjacent_pixel_clusterer.labels_ == label], axis=0))
            temp_intersections.append(cluster_mean.astype(np.uint16))
        temp_intersections = np.asarray(temp_intersections)


        # --- Phase 2: Matching Y-intersections via Pairwise Distances ---
        for x, y in temp_intersections:
            
            # figure out if the intersection is Y or X
            nb = get_neighbors_global(img=self.skeleton, pixel_coord=(x,y))
            
            if len(nb[:,0]) == 4:
                # 4 neighbors: likely an X-intersection.
                correct_intersections.append((x, y))
                continue
            elif len(nb[:,0]) == 3:
                # 3 neighbors: likely a Y-intersection.
                Y_intersections["inter"].append((x, y))
                continue
            else:
                # Not a valid intersection, skip it.
                continue    
        # Convert the list of Y-intersection coordinates to an array for easier manipulation.
        Y_intersections_np = np.asarray(Y_intersections["inter"])
        num_Y_intersections = Y_intersections_np.shape[0]

        matching_arr = []  # Will hold all possible pairs and their distances.
        # Create all possible pairs of Y-intersections and compute the Euclidean distance between them.
        for i in range(num_Y_intersections):
            p1 = Y_intersections_np[i].astype(float)
            for j in range(i + 1, num_Y_intersections):
                p2 = Y_intersections_np[j].astype(float)
                distance = np.linalg.norm(p1 - p2)
                matching_arr.append([i, j, distance]) # [Y_intersection index 1, Y_intersection index 2, distance]

        # Sort pairs by distance in descending order (so we can pop the smallest distance pair).
        matching_arr.sort(key=lambda pair: pair[2], reverse=True)

        # Match the closest Y-intersections that have not been matched yet.
        available_nodes = set(range(num_Y_intersections))
        while matching_arr:
            i, j, distance = matching_arr.pop()
            # Skip if either intersection is already matched.
            if i not in available_nodes or j not in available_nodes:
                continue
            # If the distance exceeds a threshold, stop matching.
            if distance > self.epsilon:
                break
            # Compute the new intersection as the average of the two matched intersections.
            new_inter = np.round(np.mean(Y_intersections_np[[i, j]], axis=0)).astype(np.uint16)
            new_intersections["inter"].append(tuple(new_inter))
            
        
            # Compute the new endpoints as the average of the two matched endpoints.
            # Extract a local window shpae n=window_size*2+1 (n,n) around the averaged intersection.
            window_size = 5
            # Get boundary pixels in the window and reshape them into coordinate pairs.
            boundary_global = get_boundary_pixels_in_global_frame(skeleton=self.skeleton,window_size=window_size,inter=new_inter)
            # add the new endpoints to ends 
            self.ends.extend(boundary_global)
            
            # Adjust local endpoint coordinates to match the full image space.
            new_intersections["ends"].append(boundary_global)
            # Record the matched Y-intersection indices.
            new_intersections["matches"].append(Y_intersections_np[[i, j]].tolist())
            # Remove the matched indices from the available set.
            available_nodes.discard(i)
            available_nodes.discard(j)

            # Delete the matched Y-intersections from the skeleton.
            self.skeleton[new_inter[0] - window_size + 1 : new_inter[0] + window_size , new_inter[1] - window_size + 1 : new_inter[1] + window_size] = 0
            
        # Any remaining unmatched Y-intersections are added to the correct intersections list.
        for i in available_nodes:
            correct_intersections.append(Y_intersections["inter"][i])
        self.intersections = correct_intersections
        self.intersections.extend(new_intersections["inter"])
        
        return correct_intersections, new_intersections
    
    
    def _generate_paths(self, intersection_paths: dict) -> tuple[list, dict]:
        """
        Generate continuous paths by traversing the skeleton starting from each endpoint.
        When a precomputed intersection path is encountered (its key is in intersection_paths),
        the function appends that path and updates the current endpoint accordingly.
        
        Parameters:
            skeleton (np.ndarray): A binary (or otherwise labeled) image representing the skeleton.
            ends (np.ndarray): An array of endpoints (coordinates) from which to start path tracing.
            intersection_paths (dict): A dictionary where keys are intersection identifiers (formatted
                as "x,y") and values are precomputed paths (iterables of pixel coordinates) to follow.
        
        Returns:
            paths (list): A list of NumPy arrays, each representing a traced path (with an offset of -1 applied).
            intersection_path_id (dict): A mapping from intersection key (str "x,y") to a unique path ID.
        """
        # Convert the endpoints array into a list of int16 coordinates so we can modify it (e.g., pop endpoints)
        endpoints = self.ends 
        paths = []  # List to store the complete paths
        path_id = 0  # Counter to assign a unique ID to each generated path
        intersection_path_id = {}  # Map intersection key (as a string) to its corresponding path ID

        # 
        # Process endpoints until none remain
        while endpoints:
            # Start a new path from an endpoint (pop from the list)
            curr_pixel = endpoints.pop()
            path = []  # Initialize the current path as an empty list
            visited_intersections = set()  # Keep track of intersections already visited to prevent cycles

            # Continue traversing until the path is completed
            while True:
                # Extend the current path by traversing from the current pixel.
                # traverse_skeleton() returns an iterable of pixels along the branch.
                path.extend(list(traverse_skeleton(self.skeleton, curr_pixel)))
                
                # Get the last pixel in the current path and create an identifier string (e.g., "3,5")
                p_x, p_y = path[-1]
                pixel_id = "{},{}".format(p_x, p_y)
                
                # If a precomputed intersection path is available at this pixel...
                if pixel_id in intersection_paths:
                    # If this intersection was already visited, a cycle exists; finalize the path.
                    if pixel_id in visited_intersections:
                        paths.append(np.asarray(path) - 1)  # subtract 1 for offset correction
                        break
                    # Mark this intersection as visited.
                    visited_intersections.add(pixel_id)
                    # Append the precomputed intersection path to the current path.
                    path.extend(list(intersection_paths[pixel_id]))
                    # Update the current pixel to the last point of the appended intersection path.
                    curr_pixel = np.array([path[-1][0], path[-1][1]], dtype=np.int16)
                    # Record the current path ID for this intersection.
                    intersection_path_id[pixel_id] = path_id
                    # Continue traversing from the updated current pixel.
                    continue
                else:
                    # No intersection found – the current branch has ended.
                    paths.append(np.asarray(path) - 1)  # Finalize this path with an offset of -1.
                    # Remove the final endpoint from the list to avoid re-traversal in the opposite direction.
                    remove_from_array(endpoints, path[-1])
                    break  # Exit the inner while loop for this path

            path_id += 1  # Increment the path ID counter for the next path

        return paths, intersection_path_id


# ---------------------------------------------------------------------
    @staticmethod
    def _compute_minimal_bending_energy_paths(ends: np.ndarray, inter: np.ndarray) -> dict:
        """
        Compute the pairing of endpoints that minimizes the total bending energy of the intersection.
        
        The function handles two cases:
        - When there are 4 endpoints: It considers pairing endpoints in two pairs,
            and computes the cumulative curvature (via ut.compute_cumulative_curvature) for each possible pair grouping.
            The grouping that yields the minimum total curvature is chosen, and best_paths is updated accordingly.
        - When there are 3 endpoints: It computes pairwise curvature (via ut.compute_curvature) for all pairs,
            and pairs the two endpoints with minimal curvature while designating the third endpoint (computed as 
            total_sum - (v1 + v2)) to terminate the path.
        
        Parameters:
            ends (np.ndarray): Array of endpoint coordinates.
            inter (np.ndarray): The intersection coordinate.
            
        Returns:
            best_paths (dict): A dictionary mapping each endpoint index to its paired endpoint index.
                            For a 3-endpoint case, the unpaired endpoint is assigned a value of None.
                            Returns False if endpoints are not 3 or 4.
        """
        indices = list(range(len(ends)))
        # Generate all pairs of endpoint indices
        all_pairs = list(combinations(indices, 2))
        best_paths = {}

        if len(ends) == 4:
            possible_path_pairs = set()
            already_added = set()
            # For 4 endpoints, the sum of indices is fixed
            total_index_sum = sum(indices)  # Should be 0+1+2+3 = 6

            # Identify pairs of pairs that together cover all indices.
            for pair1 in all_pairs:
                if pair1 in already_added:
                    continue
                for pair2 in all_pairs:
                    # Skip same pair and ensure the two pairs cover all endpoints.
                    if pair1 == pair2 or (sum(pair1) + sum(pair2) != total_index_sum):
                        continue
                    already_added.add(pair1)
                    already_added.add(pair2)
                    possible_path_pairs.add((pair1, pair2))

            minimum_total_curvature = np.inf
            # Evaluate each possible pairing combination.
            for (a1, a2), (b1, b2) in possible_path_pairs:
                try:
                    total_curvature = compute_cumulative_curvature(ends[a1], ends[a2],
                                                                    ends[b1], ends[b2], inter)
                except ZeroDivisionError:
                    return False

                if total_curvature < minimum_total_curvature:
                    minimum_total_curvature = total_curvature
                    # Record the best pairing: endpoints a1<->a2 and b1<->b2.
                    best_paths[a1] = a2
                    best_paths[a2] = a1
                    best_paths[b1] = b2
                    best_paths[b2] = b1

        elif len(ends) == 3:
            minimum_curvature = np.inf
            total = sum(indices)  # For 3 endpoints: 0+1+2 = 3
            for v1, v2 in all_pairs:
                curvature = compute_curvature(ends[v1], ends[v2], inter)
                if curvature < minimum_curvature:
                    minimum_curvature = curvature
                    best_paths[v1] = v2
                    best_paths[v2] = v1
                    # The remaining index is computed by subtracting the sum of the paired indices.
                    third_index = total - (v1 + v2)
                    best_paths[third_index] = None

        else:
            # Not enough endpoints to compute a path.
            return False

        return best_paths

    # ---------------------------------------------------------------------
    @staticmethod
    def _construct_path(best_paths: dict, ends: np.ndarray, inter: np.ndarray, paths_to_ends: dict) -> bool:
        """
        Construct a continuous path for the intersection based on the best_paths pairing.
        
        For each endpoint:
        - If its pairing is not defined (i.e. best_paths[index] is None), mark the intersection as three-way.
        - Otherwise, if the current index is less than its paired index,
            the path is constructed by concatenating:
                generated_path from the current endpoint,
                the intersection point,
                and the reversed generated_path from the paired endpoint with the endpoint added.
        - If the reverse path is already constructed (current index > paired index), reuse the reversed path.
        
        The generated path for each endpoint is stored in the paths_to_ends dictionary using a key formatted as "x,y".
        
        Parameters:
            best_paths (dict): Mapping of endpoint indices to their pair indices (or None).
            ends (np.ndarray): Array of endpoint coordinates.
            inter (np.ndarray): The intersection coordinate.
            paths_to_ends (dict): Dictionary to store the constructed paths keyed by endpoint coordinates.
            
        Returns:
            three_way (bool): True if a three-way intersection was detected; otherwise False.
        """
        # Precompute generated paths from each endpoint to the intersection.
        # 'line' is used to generate a straight line between two points.
        generated_paths = [list(np.asarray(line(e[0], e[1], inter[0], inter[1])).T[:-1]) for e in ends]
        three_way = False

        for i, (x1, y1) in enumerate(ends):
            if best_paths.get(i) is None:
                # A None pairing indicates a three-way intersection.
                three_way = True
                continue

            x2, y2 = ends[best_paths[i]]
            if i < best_paths[i]:
                # Construct path: current generated path + [inter] + reversed generated path from paired endpoint + endpoint coordinate.
                constructed_path = generated_paths[i] + [inter] + generated_paths[best_paths[i]][::-1] + [[x2, y2]]
                constructed_path = np.asarray(constructed_path, dtype=np.int16)
            else:
                # If the reverse path is already constructed, flip and reuse it.
                key = "{},{}".format(x2, y2)
                constructed_path = np.flip(paths_to_ends[key], axis=0)
                # Remove first element and append the endpoint.
                constructed_path[:-1] = constructed_path[1:]
                constructed_path[-1] = [x2, y2]

            # Store the constructed path keyed by the current endpoint coordinate.
            paths_to_ends["{},{}".format(x1, y1)] = constructed_path

        return three_way

    # ---------------------------------------------------------------------
    def _generate_intersection_paths(self, correct_intersections, new_intersections) -> tuple[dict, dict]:
        """
        Generate intersection paths based on the clustered intersections.
        
        The intersections tuple is unpacked into:
        - correct_intersections: intersections already topologically correct.
        - new_intersections: dictionary with keys "inter", "ends", "matches" for Y-branch intersections.
        
        First, for matched Y-branches, the function determines which branch endpoints should be connected,
        prunes the branches from the skeleton, traverses the remaining endpoints, computes minimal bending energy paths,
        constructs continuous paths, and determines crossing order.
        
        Then, for correct intersections (including standalone Y-branches and X-branches), it extracts a local window
        of the skeleton, gets boundary pixels, computes paths, and sets crossing order.
        
        Parameters:
            skeleton (np.ndarray): The skeletonized binary image.
            intersections (tuple): Tuple (correct_intersections, new_intersections) where:
                correct_intersections: list of intersection coordinates.
                new_intersections: dict with keys "inter", "ends", "matches" for Y-branch intersections.
                
        Returns:
            paths_to_ends (dict): Mapping from endpoint identifiers to constructed paths.
            crossing_orders (dict): Mapping from endpoint identifiers to crossing order (0 or 1).
        """
        paths_to_ends = {}
        crossing_orders = {}

        # Process matched Y-branches first.
        for inter, ends, matches in zip(new_intersections["inter"],
                                        new_intersections["ends"],
                                        new_intersections["matches"]):
            ends = np.asarray(ends)

            best_paths = self._compute_minimal_bending_energy_paths(ends, inter)
            if not best_paths:
                continue

            three_way = self._construct_path(best_paths, ends, inter, paths_to_ends)
            if three_way:
                continue

            self._determine_crossing_order(best_paths, ends, paths_to_ends, crossing_orders)

        # Now process correct intersections (X-branches and unmatched Y-branches).
        handled_areas = np.zeros_like(self.skeleton)
        for inter in correct_intersections:
            x, y = inter
            k_size = 5
            if handled_areas[x, y]:
                continue

            boundary_global = get_boundary_pixels_in_global_frame(skeleton=self.skeleton,window_size=k_size,inter=inter)
            best_paths = self._compute_minimal_bending_energy_paths(boundary_global, np.asarray(inter))
            
            if not best_paths:
                continue

            three_way = self._construct_path(best_paths, boundary_global, inter, paths_to_ends)
            if three_way:
                continue

            self._determine_crossing_order(best_paths, boundary_global, paths_to_ends, crossing_orders)

        return paths_to_ends, crossing_orders

# ---------------------------------------------------------------------
    def _determine_crossing_order(self, best_paths: dict, ends: np.ndarray, 
                                paths_to_ends: dict, crossing_orders: dict) -> None:
        """
        Determine the crossing order for an intersection based on the standard deviation along two candidate paths.
        
        This function uses the precomputed paths (stored in paths_to_ends) and extracts two candidate paths
        based on the best_paths pairing. It computes the sum of standard deviations (from self.blurred_image)
        along these paths to decide which branch should have which crossing order.
        
        Parameters:
            best_paths (dict): Mapping of endpoint indices to their paired index.
            ends (np.ndarray): Array of endpoint coordinates.
            paths_to_ends (dict): Dictionary containing precomputed paths, keyed by "x,y" of the endpoint.
            crossing_orders (dict): Dictionary to be updated with the crossing order for each endpoint (keys "x,y").
        
        Returns:
            None. The function updates crossing_orders in-place.
        """
        # Define possible path labels (e.g., 1, 2, 3) and remove the one corresponding to best_paths[0]
        possible_paths = [1, 2, 3]
        # best_paths[0] should be defined; remove its value from possible_paths.
        possible_paths.remove(best_paths[0])
        
        # Unpack endpoints: first pair and the remaining two.
        x11, y11 = ends[0]
        x12, y12 = ends[best_paths[0]]
        x21, y21 = ends[possible_paths[0]]
        x22, y22 = ends[possible_paths[1]]
        
        # Create keys for accessing paths from paths_to_ends.
        key11 = "{},{}".format(x11, y11)
        key12 = "{},{}".format(x12, y12)
        key21 = "{},{}".format(x21, y21)
        key22 = "{},{}".format(x22, y22)
        
        # Retrieve the corresponding paths.
        path1 = paths_to_ends[key11]
        path2 = paths_to_ends[key21]
        
        # Compute the sum of standard deviations along each path from the blurred image.
        std1 = self.blurred_image[path1[:, 0], path1[:, 1]].std(axis=0).sum()
        std2 = self.blurred_image[path2[:, 0], path2[:, 1]].std(axis=0).sum()
        
        # Assign crossing orders based on which path has a larger standard deviation.
        if std1 > std2:
            crossing_orders[key11] = 0
            crossing_orders[key12] = 0
            crossing_orders[key21] = 1
            crossing_orders[key22] = 1
        else:
            crossing_orders[key11] = 1
            crossing_orders[key12] = 1
            crossing_orders[key21] = 0
            crossing_orders[key22] = 0

    def _compute_radii(self, paths: list) -> tuple[np.ndarray, list]:
        """
        Compute radii information for a set of skeleton paths using the distance image.

        This function performs two tasks:
        1. Rounds the entire distance image (self.distance) to produce an integer-valued array (path_radii).
        2. For each path in paths, it calculates the average distance value along that path,
            rounds it, and converts it to an integer to yield the average radius for that path.
        
        Parameters:
            paths (list): A list of NumPy arrays. Each array contains pixel coordinates (rows and columns)
                        representing a skeleton path.
        
        Returns:
            path_radii (np.ndarray): The rounded version of self.distance as an int32 array.
            path_radii_avgs (list): A list of integers where each entry is the average radius computed along a path.
        """
        # Round the distance image and convert to int32.
        path_radii = np.round(self.distance/2).astype(np.int32)
        
        # For each path, index the distance image using the pixel coordinates,
        # compute the mean distance along the path, round it, and convert to an integer.
        path_radii_avgs = [
            int(np.round(self.distance[path[:, 0], path[:, 1]].mean()))
            for path in paths
        ]
        # path_radii_avgs = self.distance
        return path_radii, path_radii_avgs


    def _plot_paths(self, paths: list, intersection_paths: dict, 
                    intersection_path_id: dict, crossing_orders: dict, 
                    path_radii_data: tuple, intersection_color: list = None) -> np.ndarray:
        """
        Plot the DLO paths and intersections onto a blank image using the provided radii and colors.
        
        The function creates an image (of the same size as self.image) and draws circles along each
        path. The circle radii are taken from two sources:
        - 'path_radii': the pixel-wise rounded distance image.
        - 'path_radii_avgs': an average radius computed for each path.
        
        Depending on the position along the path (e.g. at the ends or in the middle) and its location relative
        to image boundaries, the function uses either the pixel-wise radius or the average radius.
        
        After drawing the paths, the function then processes intersections:
        - For each intersection (keyed by "x,y" string), if its crossing order is 0, it draws circles with the 
            corresponding color from self.colors.
        - For crossing order 1, it uses either self.colors or the provided intersection_color.
        
        Parameters:
            paths (list): List of NumPy arrays; each array is a sequence of (x, y) pixel coordinates representing a path.
            intersection_paths (dict): Dictionary mapping intersection IDs ("x,y") to precomputed paths (list of (x,y) coordinates).
            intersection_path_id (dict): Mapping from intersection ID to the corresponding path ID.
            crossing_orders (dict): Dictionary mapping intersection ID ("x,y") to a crossing order (0 or 1).
            path_radii_data (tuple): Tuple containing:
                - path_radii (np.ndarray): Rounded distance image (same shape as self.distance) as int32.
                - path_radii_avgs (list): List of average radii (integers) for each path.
            intersection_color (list, optional): If provided, this color (BGR or RGB) is used for intersections with crossing order 1.
            
        Returns:
            np.ndarray: An image (same size as self.image) with drawn paths and intersections.
        """
        # Create a blank image (same size as self.image) to draw the paths.
        path_img = np.zeros_like(self.image)

        # Unpack distance information.
        path_radii, path_radii_avgs = path_radii_data
        
        # Define parameters for drawing endpoints.
        end_lengths = int(round(self.epsilon))
        end_buffer = 10 if end_lengths > 10 else int(end_lengths * 0.5)
        
        # Note: self.image.shape is assumed to be (height, width, channels); however,
        # here img_height and img_width are swapped to match coordinate conventions.
        img_height, img_width = self.image.shape[1], self.image.shape[0]
        left_limit   = end_lengths
        right_limit  = img_width - int(end_lengths * 0.5)
        bottom_limit = end_lengths
        top_limit    = img_height - int(end_lengths * 0.5)

        # Draw the segmentation along the DLO paths.
        for i, path in enumerate(paths):
            # Draw the initial segment of the path (first end_buffer points).
            for x, y in path[:end_buffer]:
                cv2.circle(path_img, (y, x), path_radii[x, y], self.colors[i], -1)
            # Draw the segment from end_buffer to end_lengths.
            for x, y in path[end_buffer:end_lengths]:
                # Use pixel-wise radius if near image boundary; otherwise, use the average radius.
                if x < left_limit or x > right_limit or y < bottom_limit or y > top_limit:
                    cv2.circle(path_img, (y, x), path_radii[x, y], self.colors[i], -1)
                else:
                    cv2.circle(path_img, (y, x), path_radii_avgs[i], self.colors[i], -1)
            # Draw the middle segment of the path.
            for x, y in path[end_lengths:-end_lengths]:
                cv2.circle(path_img, (y, x), path_radii_avgs[i], self.colors[i], -1)
            # Draw the segment from -end_lengths to -end_buffer.
            for x, y in path[-end_lengths:-end_buffer]:
                if x < left_limit or x > right_limit or y < bottom_limit or y > top_limit:
                    cv2.circle(path_img, (y, x), path_radii[x, y], self.colors[i], -1)
                else:
                    cv2.circle(path_img, (y, x), path_radii_avgs[i], self.colors[i], -1)
            # Draw the last segment of the path.
            for x, y in path[-end_buffer:]:
                cv2.circle(path_img, (y, x), path_radii[x, y], self.colors[i], -1)

        # Process intersections based on crossing orders.
        # For intersections with crossing order 0:
        for id_key, p_id in intersection_path_id.items():
            if id_key not in crossing_orders or crossing_orders[id_key] == 1:
                continue
            color = self.colors[p_id]
            for x, y in intersection_paths[id_key]:
                # Adjust coordinates by subtracting 1 to account for any offset.
                cv2.circle(path_img, (y - 1, x - 1), path_radii_avgs[p_id], color, -1)
        # For intersections with crossing order 1:
        for id_key, p_id in intersection_path_id.items():
            if id_key not in crossing_orders or crossing_orders[id_key] == 0:
                continue
            color = self.colors[p_id] if intersection_color is None else intersection_color
            for x, y in intersection_paths[id_key]:
                cv2.circle(path_img, (y - 1, x - 1), path_radii_avgs[p_id], color, -1)

        return path_img
    def visualize_keypoints(self):
        """
        Visualize the skeleton with colored keypoints for easier analysis.
        
        Parameters
        ----------
        skeleton : numpy.ndarray
            Binary image of the skeleton (2D array with values 0 and 1)
        endpoints : numpy.ndarray
            Array of endpoint coordinates, shape (n, 2) with each row being [y, x]
        intersections : numpy.ndarray
            Array of intersection coordinates, shape (m, 2) with each row being [y, x]
        
        Returns
        -------
        numpy.ndarray
            Colored image of the skeleton with highlighted keypoints
        """
        # Convert grayscale skeleton to color
        colored_skeleton = cv2.cvtColor(self.skeleton.astype(np.uint16) * 255, cv2.COLOR_GRAY2BGR)
        
        # Set skeleton pixels to white
        colored_skeleton[self.skeleton > 0] = [255, 255, 255]

        # Color endpoints in green
        for end in self.ends:
            x, y = end
            colored_skeleton[x, y] = [0, 255, 0]  # Green in BGR

        # Color intersections in blue
        for inter in self.intersections:
            x, y = inter
            colored_skeleton[x, y] = [0, 0, 255]  # Blue in BGR

        # Display the colored skeleton
        plt.figure()
        plt.imshow(cv2.cvtColor(colored_skeleton, cv2.COLOR_BGR2RGB))
        plt.title('Skeleton with Keypoints (Green: Endpoints, Blue: Intersections)')
        # plt.show()
        plt.pause(0.001)    
        return colored_skeleton

if __name__ == '__main__':
    
    mask = cv2.imread('/home/admina/segmetation/DLOSeg/src/my_mBEST/skeletonize/mask.png', cv2.IMREAD_GRAYSCALE)
    mask = mask/255
    mask = mask.astype(np.uint16)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    start_time = time.time()
    dlo_skel = Dlo_skeletonize()
    dlo_skel.set_image(mask)

    dlo_skel._set_params()
    
    dlo_skel._detect_keypoints()
    # print(ends)
    # print(interactions)
    
    # dlo_skel.visualize_keypoints()

    # ends_pruned, interactions_pruned,skeleton_pruned =dlo_skel._prune_split_ends(skeleton, ends, interactions)
    dlo_skel._prune_split_ends_from_inerections()
    # dlo_skel.visualize_keypoints()

    if len(dlo_skel.intersections) > 0:
        
        correct_intersections, new_intersections =dlo_skel._cluster_and_match_intersections()
        
        
        dlo_skel.visualize_keypoints()
        intersection_paths, crossing_orders = dlo_skel._generate_intersection_paths(correct_intersections, new_intersections)
        
        
        paths, intersection_path_id = dlo_skel._generate_paths(intersection_paths)
        
        
        path_radii = dlo_skel._compute_radii(paths)
        
        path_img = dlo_skel._plot_paths(paths, intersection_paths, intersection_path_id,
                            crossing_orders, path_radii, intersection_color=[255, 0, 0])
        end_time = time.time()
        print("Time taken to process: ", end_time - start_time, " seconds")
        plt.imshow(path_img)
        plt.show()
    # cv2.imwrite('skeleton.png', skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows