import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis , thin
import matplotlib.pyplot as plt
from skimage.draw import line

class Dlo_skeletonize:
    def __init__(self):
        self.mask = None
        self.skeleton = None
        self.distance = None
        
        self.end_point_kernel = np.array(([1, 1, 1], [1, 10, 1], [1, 1, 1]), dtype=np.uint8)
    def set_image(self, mask):
        self.mask = mask
    def _set_params(self, max_distance):

        self.delta = max_distance * 2
        self.delta_i = int(round(max_distance * 1.5))
        self.epsilon = max_distance * 10
        
        
    def _detect_keypoints(self, skeleton):
        padded_img = np.zeros((skeleton.shape[0]+2, skeleton.shape[1]+2), dtype=np.uint8)
        padded_img[1:-1, 1:-1] = skeleton
        res = cv2.filter2D(src=padded_img, ddepth=-1, kernel=self.end_point_kernel)
        ends = np.argwhere(res == 11) - 1
        intersections = np.argwhere(res > 12) - 1
        return ends, intersections
    
    def get_boundary_pixels(self,mask: np.ndarray) -> np.ndarray:
        """
        Extracts the boundary pixels from an inner region of the given binary mask.
        
        # Apply the convolution filter with explicit border handling to keep consistency
        res = cv2.filter2D(src=padded_img, ddepth=-1, kernel=self.end_point_kernel, borderType=cv2.BORDER_CONSTANT)
        
        # Endpoints: combine conditions where the result equals 10 or 11
        self.ends = np.argwhere((res == 10) | (res == 11)) - 1
        # Intersections: pixels with a result greater than 12
        self.intersections = np.argwhere(res > 12) - 1
        # Convert to uint16 for consistency
        self.ends = self.ends.astype(np.uint8)
        self.intersections = self.intersections.astype(np.uint8)

    def _prune_split_ends_from_inerections(self, skeleton: np.ndarray, ends: np.ndarray, intersections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prune noisy branch segments from a skeletonized image and update the endpoints and intersections.
        
        The function starts at each intersection and follows each branch until either:
        - It reaches an endpoint (no more connected neighbors), or
        - It encounters another intersection.
        
        If the branch length is below a given threshold, the branch is pruned (removed from the skeleton).
        If all branches emerging from an intersection are pruned, then the intersection itself is pruned.
        
        Parameters:
            mask (np.ndarray): A 2D binary array representing the mask.
            
        Returns:
            np.ndarray: An array of shape (n, 2) with the coordinates of the boundary pixels.
        """
        # Get the inner region of the mask (exclude outer border)
        inner_mask = mask[1:-1, 1:-1].copy()
        # Zero out the interior of inner_mask so that only its border remains
        inner_mask[1:-1, 1:-1] = 0

        # Find coordinates (row, col) where inner_mask is nonzero.
        # np.where returns arrays for each dimension; we combine them into a list of [x, y] pairs.
        vecs = list(np.column_stack(np.where(inner_mask)))
        # Alternatively, you could use: vecs = list(np.asarray(np.where(inner_mask)).T)
        
        # Iterate through the list, removing pixels that satisfy a certain condition.
        # Since we may remove elements, we use a while loop.
        curr_ele = 0
        while curr_ele < len(vecs):
            x, y = vecs[curr_ele]
            # Ensure indices for the 3x3 window do not go out of bounds
            x_i = max(x - 1, 0)
            y_i = max(y - 1, 0)
            
            # Check two conditions:
            # 1. The sum of the 3x3 neighborhood in the inner_mask is > 1.
            # 2. The sum of the corresponding 3x3 neighborhood in the original mask equals 3.
            # If both are true, we remove the current pixel from vecs.
            if np.sum(inner_mask[x_i:x_i+3, y_i:y_i+3]) > 1 and np.sum(mask[x:x+3, y:y+3]) == 3:
                vecs.pop(curr_ele)
            else:
                curr_ele += 1

        # Convert the final list of coordinates into a NumPy array of type int16
        return np.asarray(vecs, dtype=np.int16)
    
    
    def _prune_split_ends(self, skeleton: np.ndarray, ends: np.ndarray, intersections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prunes noisy branch segments from a skeletonized image and updates the endpoints and intersection points.

        Parameters:
            skeleton (np.ndarray): Binary image (2D array) representing the skeleton.
            ends (np.ndarray): Array of endpoint coordinates (e.g., shape (n, 2)).
            intersections (np.ndarray): Array of intersection coordinates (e.g., shape (m, 2)).

        Returns:
            pruned_ends (np.ndarray): Updated endpoints after pruning.
            updated_intersections (np.ndarray): Updated intersections after removing those 
                                                that were encountered during branch tracing.
        """
        # Make a working copy of the skeleton for branch tracing
        processed_skel = skeleton.copy()
        
        # Initialize a boolean list marking each intersection as valid (True initially)
        intersection_valid = [True] * len(intersections)
        # Create a dictionary that maps each intersection coordinate (as a tuple) to its index
        intersection_index = {(x, y): i for i, (x, y) in enumerate(intersections)}
        
        # Convert endpoints to a list to allow appending new endpoints (if needed)
        endpoints = list(ends)
        valid_end_indices = []  # This will record the indices of endpoints that are not pruned

        # Loop over all endpoints with their index in the endpoints list
        for idx, endpoint in enumerate(endpoints):
            current_pixel = endpoint  # Current pixel coordinates (endpoint of a branch)
            branch_path = [current_pixel]  # List to store the pixels in the current branch
            prune_branch = False  # Flag to indicate if the branch should be pruned
            branch_terminated = False  # Flag if the branch ends without intersection
            
            # Follow the branch until termination or an intersection is found
            while True:
                x, y = current_pixel
                # Remove the current pixel from the processed skeleton to avoid revisiting it
                processed_skel[x, y] = 0
                
                # Find neighboring skeleton pixels in a 3x3 window centered at (x, y)
                neigh_x, neigh_y = np.where(processed_skel[x-1:x+2, y-1:y+2])
                # Adjust indices to global coordinates
                neigh_x += x - 1
                neigh_y += y - 1
                num_neighbors = len(neigh_x)
                
                if num_neighbors == 1:
                    # One neighbor found: branch continues
                    current_pixel = [neigh_x[0], neigh_y[0]]
                    branch_path.append(current_pixel)
                elif num_neighbors == 0:
                    # No neighbor found: reached an endpoint
                    branch_terminated = True
                    break
                else:
                    # More than one neighbor: an intersection is encountered
                    # Mark the intersection at the current pixel as invalid
                    key = (x, y)
                    if key in intersection_index:
                        intersection_valid[intersection_index[key]] = False

                    # Also mark nearby intersection pixels (in a 5x5 window) as invalid
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            neighbor_key = (x + dx, y + dy)
                            if neighbor_key in intersection_index:
                                intersection_valid[intersection_index[neighbor_key]] = False

                    prune_branch = True
                    break

                # If the branch path length exceeds a threshold, we assume it's a valid segment
                if len(branch_path) > self.delta:
                    break

            # If the branch terminated without encountering an intersection, skip further processing
            if branch_terminated:
                continue

            # Convert the branch path to a NumPy array for easier indexing
            branch_path_arr = np.asarray(branch_path)

            if prune_branch:
                # Remove the branch pixels from the original skeleton image
                skeleton[branch_path_arr[:, 0], branch_path_arr[:, 1]] = 0
                # Get the last pixel in the branch, which is near an intersection
                x_end, y_end = branch_path_arr[-1]
                # Get boundary pixels in a 9x9 window around the endpoint
                boundary_pixels = self.get_boundary_pixels(skeleton[x_end-4:x_end+5, y_end-4:y_end+5])

                # If exactly two boundary pixels are found, reconnect them by drawing a line
                if len(boundary_pixels) == 2:
                    # Adjust boundary pixel coordinates to match the full image coordinates
                    boundary_pixels[:, 0] += x_end - 3
                    boundary_pixels[:, 1] += y_end - 3
                    # Draw a line between the two points (using skimage.draw.line)
                    reconnected_line = line(boundary_pixels[0][0], boundary_pixels[0][1],
                                            boundary_pixels[1][0], boundary_pixels[1][1])
                    skeleton[reconnected_line[0], reconnected_line[1]] = 1

                # If only one boundary pixel is found, treat it as a new endpoint and add it
                elif len(boundary_pixels) == 1:
                    new_end = boundary_pixels.squeeze()
                    new_end[0] += x_end - 3
                    new_end[1] += y_end - 3
                    endpoints.append(new_end)

                # Update the processed skeleton in the local region to reflect the changes
                processed_skel[x_end-2:x_end+3, y_end-2:y_end+3] = skeleton[x_end-2:x_end+3, y_end-2:y_end+3]
            else:
                # If no pruning occurred, mark this endpoint as valid
                valid_end_indices.append(idx)

        # Convert endpoints back to a NumPy array and select only those marked as valid
        endpoints = np.asarray(endpoints)
        pruned_endpoints = endpoints[valid_end_indices]
        # Update intersections by filtering out those marked as invalid
        updated_intersections = intersections[np.array(intersection_valid)]

        return pruned_endpoints, updated_intersections
    
    # def _prune_split_ends(self, skeleton, ends, intersections):
    #     my_skeleton = skeleton.copy()
    #     inter_indices = [True for _ in intersections]
    #     inter_index_dict = {"{},{}".format(x, y): i for i, (x, y) in enumerate(intersections)}
    #     valid_ends = []
    #     ends = list(ends)  # so that we can append new ends

    #     for i, e in enumerate(ends):
    #         curr_pixel = e
    #         path = [curr_pixel]
    #         prune = False
    #         found_nothing = False

    #         while True:
    #             x, y = curr_pixel
    #             my_skeleton[curr_pixel[0], curr_pixel[1]] = 0
    #             c_x, c_y = np.where(my_skeleton[x-1:x+2, y-1:y+2])
    #             c_x += x-1
    #             c_y += y-1

    #             num_neighbors = len(c_x)
    #             # Keep following segment
    #             if num_neighbors == 1:
    #                 curr_pixel = [c_x[0], c_y[0]]
    #                 path.append(curr_pixel)
    #             # We've reached an end
    #             elif num_neighbors == 0:
    #                 found_nothing = True
    #                 break
    #             # Found an intersection
    #             else:
    #                 # Remove intersection pixels from list
    #                 ind = "{},{}".format(curr_pixel[0], curr_pixel[1])
    #                 inter_indices[inter_index_dict[ind]] = False

    #                 for j in range(-2, 3):
    #                     for k in range(-2, 3):
    #                         ind = "{},{}".format(j+curr_pixel[0], k+curr_pixel[1])
    #                         if ind in inter_index_dict:
    #                             inter_indices[inter_index_dict[ind]] = False

    #                 prune = True
    #                 break

    #             # This is most likely a valid segment
    #             if len(path) > self.delta:
    #                 break

    #         if found_nothing: continue
    #         path = np.asarray(path)

    #         # Prune noisy segment from skeleton.
    #         if prune:
    #             skeleton[path[:, 0], path[:, 1]] = 0
    #             x, y = path[-1]
    #             vecs = ut.get_boundary_pixels(skeleton[x-4:x+5, y-4:y+5])

    #             # Reconnect the segments together after pruning a branch
    #             if len(vecs) == 2:
    #                 vecs[:, 0] += x - 3
    #                 vecs[:, 1] += y - 3
    #                 reconnected_line = line(vecs[0][0], vecs[0][1], vecs[1][0], vecs[1][1])
    #                 skeleton[reconnected_line[0], reconnected_line[1]] = 1

    #             # We created a new end so add it.
    #             elif len(vecs) == 1:
    #                 vecs = vecs.squeeze()
    #                 vecs[0] += x - 3
    #                 vecs[1] += y - 3
    #                 ends.append(vecs)

    #             # Reflect the changes on our skeleton copy.
    #             my_skeleton[x-2:x+3, y-2:y+3] = skeleton[x-2:x+3, y-2:y+3]

    #         else:
    #             valid_ends.append(i)

    #     ends = np.asarray(ends)
    #     return ends[valid_ends], intersections[inter_indices]
    
    def get_skeleton_from_mask(self, mask, method='skimage'):
        """
        Extract skeleton from a binary mask.
        
        Parameters:
        -----------
        mask : ndarray
            Binary mask from which to extract the skeleton.
            Should be a 2D array with values 0 and 1 (or boolean).
        method : str, optional
            The skeletonization method to use. Can be 'skimage' (default),
            'medial_axis', or 'opencv'.
            
        Returns:
        --------
        ndarray
            A binary image containing the skeleton.
        """
        # Ensure mask is binary
        if not np.all(np.isin(mask, [0, 1])):
            mask = (mask > 0).astype(np.uint8)
        
        if method == 'skimage':
            # Zhang-Suen thinning algorithm
            return skeletonize(mask,method='zhang').astype(np.uint8)
        elif method == 'medial_axis':
            # Medial axis skeletonization
            skel, distance = medial_axis(mask, return_distance=True)
            return skel.astype(np.uint8), distance
        elif method == 'thin':
            return thin(mask).astype(np.uint8)

    def _generate_intersection_paths(self, skeleton: np.ndarray, intersections: tuple) -> (dict, dict):
        """
        Generate paths for intersections by replacing matched Y-branches with X-branches 
        and handling standalone X-branches or unmatched Y-branches.
        
        Parameters:
            skeleton (np.ndarray): 2D binary image representing the skeleton.
            intersections (tuple): A tuple containing:
                - correct_intersections: list of intersection coordinates that are topologically correct.
                - new_intersections: dictionary with keys:
                    "inter": list of intersection coordinates computed from matched Y-branches,
                    "ends": list of endpoints for each matched pair,
                    "matches": list of matched branch pairs (coordinates) for each intersection.
        
        Returns:
            paths_to_ends (dict): A dictionary mapping endpoint identifiers to precomputed paths.
            crossing_orders (dict): A dictionary defining the crossing order at each intersection.
        """
        # Unpack the intersections tuple into correct intersections and newly computed intersections
        correct_intersections, new_intersections = intersections

        paths_to_ends = {}       # Will store the computed path for each intersection-end pair
        crossing_orders = {}     # Will record the crossing order for each intersection

        # --- Process matched Y-branches first ---
        # For each newly computed intersection (from matched Y-branches), along with its associated endpoints and matches:
        for inter, ends, matches in zip(new_intersections["inter"],
                                        new_intersections["ends"],
                                        new_intersections["matches"]):
            # Split endpoints for the two branches into separate lists.
            ends1 = ends[0]
            ends2 = ends[1]

            # Compute pairwise distances between every endpoint in ends1 and every endpoint in ends2.
            pair_combinations = {}
            for i in range(len(ends1)):
                e1 = ends1[i].astype(float)
                for j in range(len(ends2)):
                    e2 = ends2[j].astype(float)
                    # Use a key of the form "i_j" to store the Euclidean distance.
                    pair_combinations[f"{i}_{j}"] = np.linalg.norm(e1 - e2)

            # If no valid pairs exist, skip to the next intersection.
            if not pair_combinations:
                continue

            # Sort the pairs by distance (smallest distance last since we will pop from the end)
            sorted_pairs = sorted(pair_combinations.items(), key=lambda x: x[1], reverse=True)

            # Check for ambiguous cases: if the two smallest distances are equal, leave the intersection unmodified.
            if len(sorted_pairs) > 1 and sorted_pairs[-1][1] == sorted_pairs[-2][1]:
                correct_intersections.append(inter)
                continue

            # Get the indices of the pair with the smallest distance (the two ends that connect)
            connected_indices = sorted_pairs[-1][0].split("_")
            skip_e1 = int(connected_indices[0])
            skip_e2 = int(connected_indices[1])

            # The remaining endpoints (not used for connection) form the "outer" endpoints.
            outer_ends = []
            for i, e in enumerate(ends1):
                if i != skip_e1:
                    outer_ends.append(e)
            for i, e in enumerate(ends2):
                if i != skip_e2:
                    outer_ends.append(e)
            outer_ends = np.asarray(outer_ends)

            # Zero out a small window around the matched endpoints in the skeleton
            x1, y1 = matches[0]
            x2, y2 = matches[1]
            skeleton[x1-1:x1+2, y1-1:y1+2] = 0
            skeleton[x2-1:x2+2, y2-1:y2+2] = 0

            # For each outer endpoint, traverse a fixed number of pixels along the skeleton
            for idx in range(len(outer_ends)):
                outer_ends[idx] = ut.traverse_skeleton_n_pixels(skeleton, outer_ends[idx], self.delta_i)

            # Compute the best (minimal bending energy) path for these endpoints
            best_paths = self._compute_minimal_bending_energy_paths(outer_ends, inter)
            if not best_paths:
                continue

            # Construct a path from the best paths and update the paths_to_ends dictionary
            three_way = self._construct_path(best_paths, outer_ends, inter, paths_to_ends)
            if three_way:
                continue

            # Determine the crossing order (i.e., which branch crosses over which)
            self._determine_crossing_order(best_paths, outer_ends, paths_to_ends, crossing_orders)

        # --- Process X-branches and standalone Y-branches ---
        handled_areas = np.zeros_like(skeleton)
        for inter in correct_intersections:
            x, y = inter
            k_size = 5  # Defines the size of the local window

            # Skip if this area has already been handled
            if handled_areas[x, y]:
                continue

            # Extract a window around the intersection and get boundary pixels
            window = skeleton[x - k_size - 2: x + k_size + 3, y - k_size - 2: y + k_size + 3]
            ends_local = ut.get_boundary_pixels(window)
            handled_areas[x - k_size - 1: x + k_size + 2, y - k_size - 1: y + k_size + 2] = 1

            # Reshape the endpoints and adjust coordinates to the full image space
            ends_local = ends_local.reshape((-1, 2))
            ends_local[:, 0] += x - k_size - 1
            ends_local[:, 1] += y - k_size - 1

            # Compute best paths for the local endpoints with respect to the intersection
            best_paths = self._compute_minimal_bending_energy_paths(ends_local, np.asarray(inter))
            if not best_paths:
                continue

            three_way = self._construct_path(best_paths, ends_local, inter, paths_to_ends)
            if three_way:
                continue

            self._determine_crossing_order(best_paths, ends_local, paths_to_ends, crossing_orders)

        return paths_to_ends, crossing_orders

    def _cluster_and_match_intersections(self, intersections: np.ndarray, sk: np.ndarray) -> (list, dict):
        """
        Cluster adjacent intersection pixels using DBSCAN, and then match and merge Y-branch intersections.
        
        This function performs a two-phase process:
        1. Cluster adjacent intersection pixels into a set of averaged points.
        2. Examine each averaged intersection point’s local window in the skeleton (sk) to extract branch endpoints.
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
                "ends": List of paired endpoints for the matched intersections.
                "matches": List of matched Y-branch pairs (coordinates).
        """
        # Initialize temporary lists and dictionaries to store results.
        temp_intersections = []  # Averaged intersection points from each DBSCAN cluster.
        correct_intersections = []  # Already correct intersections (X-shaped).
        Y_intersections = {"inter": [], "ends": []}  # Y-intersections needing matching.
        new_intersections = {"inter": [], "ends": [], "matches": []}  # New intersections from matching Y-branches.

        # --- Phase 1: Clustering via DBSCAN ---
        # Cluster adjacent intersection pixels.
        self.adjacent_pixel_clusterer.fit(intersections)
        for label in np.unique(self.adjacent_pixel_clusterer.labels_):
            # Average the pixels in each cluster and round to uint16.
            cluster_mean = np.round(np.mean(intersections[self.adjacent_pixel_clusterer.labels_ == label], axis=0))
            temp_intersections.append(cluster_mean.astype(np.uint16))
        temp_intersections = np.asarray(temp_intersections)

        # --- Phase 2: Matching via Pairwise Distances ---
        for x, y in temp_intersections:
            # Extract a local window (7x7) around the averaged intersection.
            window = sk[x - 3:x + 4, y - 3:y + 4].copy()
            # Get boundary pixels in the window and reshape them into coordinate pairs.
            ends_local = ut.get_boundary_pixels(window).reshape((-1, 2))
            num_segments = ends_local.shape[0]

            # Adjust local endpoint coordinates to match the full image space.
            ends_local[:, 0] += x - 2
            ends_local[:, 1] += y - 2

            # Classify the intersection based on the number of segments found.
            if num_segments == 3:
                # Likely a Y-intersection: store it for later matching.
                Y_intersections["inter"].append([x, y])
                Y_intersections["ends"].append(ends_local)
            else:  # Assume num_segments == 4 (or other value) indicates an X-intersection.
                correct_intersections.append([x, y])

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
                matching_arr.append([i, j, distance])

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
            new_intersections["inter"].append(new_inter)
            # Record the endpoints from both intersections for potential branch replacement.
            new_intersections["ends"].append([Y_intersections["ends"][i], Y_intersections["ends"][j]])
            # Record the matched Y-intersection indices.
            new_intersections["matches"].append(Y_intersections_np[[i, j]])
            # Remove the matched indices from the available set.
            available_nodes.discard(i)
            available_nodes.discard(j)

        # Any remaining unmatched Y-intersections are added to the correct intersections list.
        for i in available_nodes:
            correct_intersections.append(Y_intersections["inter"][i])

        return correct_intersections, new_intersections

if __name__ == '__main__':
    
    mask = cv2.imread('/home/admina/segmetation/DLOSeg/src/my_mBEST/skeletonize/mask.png', cv2.IMREAD_GRAYSCALE)
    mask = mask/255
    mask = mask.astype(np.uint8)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    dlo_skel = Dlo_skeletonize()
    skeleton , distance = dlo_skel.get_skeleton_from_mask(mask, method='medial_axis')
    plt.imshow(skeleton, cmap='gray')
    plt.show()
    dlo_skel._set_params(distance.max())
    
    ends, interactions = dlo_skel._detect_keypoints(skeleton)
    print(ends)
    print(interactions)
    
    
    colored_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    # Set skeleton pixels to white
    colored_skeleton[skeleton > 0] = [255, 255, 255]

    # Color end points in green
    for end in ends:
        y, x = end
        colored_skeleton[y, x] = [0, 255, 0]  # Green in RGB

    # Color intersection points in blue
    for inter in interactions:
        y, x = inter
        colored_skeleton[y, x] = [0, 0, 255]  # Blue in RGB

    # Display the colored skeleton
    plt.figure()
    plt.imshow(colored_skeleton)
    plt.title('Skeleton with keypoints (green: endpoints, blue: intersections)')
    plt.show()
    
    ends_pruned, interactions_pruned =dlo_skel._prune_split_ends(skeleton, ends, interactions)
    
    # cv2.imwrite('skeleton.png', skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows