import numpy as np
from skimage.morphology import skeletonize, medial_axis , thin
import matplotlib.pyplot as plt
import cv2
import time
# TODO: transform coordinates to global coordinates
def get_boundary_edges_local(graph):
    graph[1 : - 1, 1 : - 1] = 0

    edges = np.where(graph == 1)
    new_edges = []
    for edge in zip(*edges):
        new_edges.append(edge)
    # print(new_edges)
    new_edges = np.asarray(new_edges)

    best_edges = []
    # top row
    best_indx = np.where(new_edges[:,0] == 0)[0]
    best_edges.extend(new_edges[[best_indx[0], best_indx[-1]]])
    # bottom row
    best_indx = np.where(new_edges[:,0] == 14)[0]
    best_edges.extend(new_edges[[best_indx[0], best_indx[-1]]])
    # left column
    best_indx = np.where(new_edges[:,1] == 0)[0]
    if len(best_indx) > 1:
        best_edges.extend(new_edges[[best_indx[0], best_indx[-1]]])

    # # right column
    best_indx = np.where(new_edges[:,1] == 14)[0]
    if len(best_indx) > 1:
        best_edges.extend(new_edges[[best_indx[0], best_indx[-1]]])

    best_edges = [tuple(x) for x in best_edges ]
    best_edges = list(set(best_edges))
    
    return best_edges[:4]

def get_boundary_edges_global(skeleton: np.ndarray, window_size: int, inter: tuple[int,int]):
    
    
    boundary_global = []
    # make sure the window size is odd
    assert window_size % 2 == 1, "window size should be odd"
    # Get the bounding box of the mask
    window = skeleton[inter[0]-window_size:inter[0] + window_size + 1, inter[1] - window_size:inter[1] + window_size + 1].copy()
    
    boundary_local = get_boundary_edges_local(window)
    
    # convert the local coordinates to global coordinates  
    for pixel in boundary_local:
        global_pixel = (pixel[0] + inter[0] - window_size, pixel[1] + inter[1] - window_size)
        boundary_global.append(global_pixel)
    return boundary_global
def get_boundary_pixels_in_global_frame(skeleton: np.ndarray, window_size: int, inter: tuple[int,int]) -> np.ndarray:
    """
    Get boundary pixels specifically set to 1 in the mask.

    Parameters:
        mask (np.ndarray): Binary mask; nonzero values indicate object.

    Returns:
        np.ndarray: Array of coordinates (y, x) of boundary pixels that are 1 in the original mask.
    """
    boundary_local = []
    boundary_global = []
    # make sure the window size is odd
    assert window_size % 2 == 1, "window size should be odd"
    # Get the bounding box of the mask
    window = skeleton[inter[0]-window_size:inter[0] + window_size + 1, inter[1] - window_size:inter[1] + window_size + 1].copy()
    
    # Get neighbors of the center pixel
    neighbors = get_neighbors_global(window,(window_size,window_size))
    # traverse each neighbor brach
    center_pixel = (window_size, window_size)
    for nb in neighbors:
        edge = traverse_to_edges(window,nb,center_pixel)
        if edge[1]:
            # if the endpoint is reached, add the pixel to the boundary
            boundary_local.append(edge[0])
            continue
        else:
            inter_nb = get_neighbors_global(window,edge[0])
            between_inter_nb = edge[2]
            inter_nb =[tuple(x) for x in inter_nb if tuple(x) != tuple(between_inter_nb)]
            previous_pixel = edge[0]
            
            for i in inter_nb:
                # traverse on each branch
                edge = traverse_to_edges(window,i,previous_pixel)
                if edge[1]:
                    # if the endpoint is reached, add the pixel to the boundary
                    boundary_local.append(edge[0])
                    continue
                else:
                    print("intersection found")

    # convert the local coordinates to global coordinates  
    for pixel in boundary_local:
        global_pixel = (pixel[0] + inter[0] - window_size, pixel[1] + inter[1] - window_size)
        boundary_global.append(global_pixel)

    return boundary_global
def traverse_to_edges(skeleton: np.ndarray,start_coord: tuple[int,int],previous_pixel: tuple[int,int]) -> np.ndarray:
    """
    Traverses along a path until an endpoint or intersection is reached.
    
    Returns:
    tuple[int,int]
    coordinates of the endpoint or intersection.
    
    bool 
    is_end
    True if the endpoint is reached, False if an intersection is found.
    
    """
    current_pixel = start_coord
    
    while True:
        # travers the branch
        next_nb = get_neighbors_global(skeleton,current_pixel)
        # remove the previous pixel from the neighbors
        next_nb = [tuple(x) for x in next_nb if tuple(x) != tuple(previous_pixel)]
        # no neighbors found edge of the branch
        if len(next_nb) == 0:
            # end reached return the current pixel
            is_end = True
            return current_pixel , is_end
            
        # one neighbor found keep traversing
        elif len(next_nb) == 1:
            previous_pixel = current_pixel
            current_pixel = next_nb[0]
        else:
            # more than one neighbor found, intersection reached
            is_end = False
            break
    return current_pixel , is_end, previous_pixel

def get_skeleton_from_mask(mask: np.ndarray , method='skimage') -> np.ndarray:
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
    ndarray 
        If method is 'medial_axis', also returns the distance transform.
    """
    # Ensure mask is binary
    if not np.all(np.isin(mask, [0, 1])):
        mask = (mask > 0).astype(np.uint8)
    
    if method == 'skimage':
        # Zhang-Suen thinning algorithm
        skeleton = skeletonize(mask, method='zhang').astype(np.uint8)
    elif method == 'medial_axis':
        # Medial axis skeletonization
        skeleton, distance = medial_axis(mask, return_distance=True)
    elif method == 'thin':
        skeleton = thin(mask)
    # compute the distance transform
    elif method == 'opencv':
        # OpenCV skeletonization
        skeleton = cv2.ximgproc.thinning(mask)
    # Compute the distance transform
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return skeleton.astype(np.uint8) , distance


def visualize_keypoints( skeleton, endpoints, intersections):
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
    colored_skeleton = cv2.cvtColor(skeleton.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    
    # Set skeleton pixels to white
    colored_skeleton[skeleton > 0] = [255, 255, 255]

    # Color endpoints in green
    for end in endpoints:
        x, y = end
        colored_skeleton[x, y] = [0, 255, 0]  # Green in BGR

    # Color intersections in blue
    for inter in intersections:
        x, y = inter
        colored_skeleton[x, y] = [0, 0, 255]  # Blue in BGR

    # Display the colored skeleton
    plt.figure()
    plt.imshow(cv2.cvtColor(colored_skeleton, cv2.COLOR_BGR2RGB))
    plt.title('Skeleton with Keypoints (Green: Endpoints, Blue: Intersections)')
    # plt.show()
    plt.pause(0.001)    
    return colored_skeleton


def traverse_skeleton_n_pixels(sk: np.ndarray, curr_pixel: np.ndarray, n: int) -> np.ndarray:
    """
    Traverse the skeleton image `sk` from the starting pixel `curr_pixel` for `n` steps.
    
    At each step:
      - The current pixel is marked as visited (set to 0 in `sk`).
      - The 3x3 neighborhood (centered on the current pixel) is examined.
      - The next pixel is chosen from the neighborhood (excluding the center)
        in a fixed order:
            top-left, top-center, top-right,
            left,         right,
            bottom-left, bottom-center, bottom-right.
      - The relative position from the 3x3 view is converted to global coordinates.
    
    If no neighboring pixel is found, the traversal stops and the current pixel is returned.
    
    Parameters:
        sk (np.ndarray): 2D array representing the skeleton image.
        curr_pixel (np.ndarray): Starting pixel coordinate as [x, y].
        n (int): Number of pixels (steps) to traverse.
    
    Returns:
        np.ndarray: The final pixel coordinate reached after n steps or when no neighbor is found.
    """
    # Define neighbor offsets (relative to the top-left corner of the 3x3 window)
    # Order: top-left, top-center, top-right, left, right, bottom-left, bottom-center, bottom-right.
    neighbor_offsets = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2)
    ]
    
    for _ in range(n):
        x, y = curr_pixel
        # Mark the current pixel as visited by setting it to 0.
        sk[x, y] = 0
        
        # Extract a 3x3 neighborhood centered at (x, y)
        view = sk[x-1:x+2, y-1:y+2]
        
        # Look for a non-zero (i.e. skeleton) neighbor in the defined order.
        found_next = False
        for r, c in neighbor_offsets:
            if view[r, c]:
                # Found a neighbor: set the new relative coordinate from the view.
                new_relative = np.array([r, c], dtype=np.int16)
                found_next = True
                break
        
        # If no neighbor was found, return the current pixel immediately.
        if not found_next:
            return curr_pixel
        
        # Convert the relative coordinate to the global coordinate.
        # The view covers rows x-1 to x+1, so add (x-1, y-1) to the relative offset.
        curr_pixel = new_relative + np.array([x - 1, y - 1], dtype=np.int16)
    
    return curr_pixel


def get_unit_vector(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute the unit vector pointing from vec1 to vec2.
    
    Parameters:
        vec1 (np.ndarray): The starting point (coordinate).
        vec2 (np.ndarray): The ending point (coordinate).
    
    Returns:
        np.ndarray: The unit vector (as float32) from vec1 to vec2.
    """
    # Compute the difference vector and cast to float32
    u_vec = (vec2 - vec1).astype(np.float32)
    norm = np.linalg.norm(u_vec)
    if norm != 0:
        u_vec /= norm
    else:
        # Avoid division by zero by returning a zero vector
        u_vec = np.zeros_like(u_vec, dtype=np.float32)
    return u_vec

def compute_curvature(v1: np.ndarray, v2: np.ndarray, inter: np.ndarray) -> float:
    """
    Compute the curvature of a branch defined by two segments:
    from v1 to inter and from inter to v2.
    
    The curvature is defined as:
        kappa = 2 * np.cross(unit_vector1, unit_vector2) / (1 + dot(unit_vector1, unit_vector2))
    and the function returns the absolute value of kappa.
    
    Parameters:
        v1 (np.ndarray): The first endpoint of the branch.
        v2 (np.ndarray): The second endpoint of the branch.
        inter (np.ndarray): The intersection point connecting the two segments.
        
    Returns:
        float: The absolute curvature value.
    """
    # Get unit vectors for each segment
    vec1 = get_unit_vector(v1, inter)
    vec2 = get_unit_vector(inter, v2)
    
    # Compute curvature using the provided formula
    # TODO: implement amore efficent method to compute cross product and dot product
    kappa = 2 * np.cross(vec1, vec2) / (1 + np.dot(vec1, vec2))
    curvature = np.abs(kappa)
    return curvature

def compute_cumulative_curvature(a1: np.ndarray, a2: np.ndarray, 
                                 b1: np.ndarray, b2: np.ndarray, 
                                 inter: np.ndarray) -> float:
    """
    Compute the cumulative curvature for two branches meeting at an intersection.
    
    The function computes curvature for the first branch (from a1 to inter to a2)
    and for the second branch (from b1 to inter to b2) using:
        kappa = 2 * np.cross(unit_vector_start, unit_vector_end) / (1 + dot(unit_vector_start, unit_vector_end))
    and returns the sum of their absolute curvature values.
    
    Parameters:
        a1 (np.ndarray): Start point of the first branch.
        a2 (np.ndarray): End point of the first branch.
        b1 (np.ndarray): Start point of the second branch.
        b2 (np.ndarray): End point of the second branch.
        inter (np.ndarray): The intersection point common to both branches.
        
    Returns:
        float: The sum of the absolute curvature values of the two branches.
    """
    # Compute unit vectors for each segment of both branches.
    vec_a1 = get_unit_vector(a1, inter)
    vec_a2 = get_unit_vector(inter, a2)
    vec_b1 = get_unit_vector(b1, inter)
    vec_b2 = get_unit_vector(inter, b2)
    
    # Calculate curvature for each branch using the formula.
    kappa1 = 2 * np.cross(vec_a1, vec_a2) / (1 + np.dot(vec_a1, vec_a2))
    kappa2 = 2 * np.cross(vec_b1, vec_b2) / (1 + np.dot(vec_b1, vec_b2))
    
    # Total curvature is the sum of absolute curvatures.
    total_curvature = np.abs(kappa1) + np.abs(kappa2)
    return total_curvature



def remove_from_array(base_array: list, array_to_remove: np.ndarray) -> None:
    """
    Remove the first element from base_array that is equal to array_to_remove.

    The equality is tested using np.array_equal.
    
    Parameters:
        base_array (list): A list of numpy arrays.
        array_to_remove (np.ndarray): The numpy array to be removed from base_array.
    
    Returns:
        None. The function modifies base_array in-place.
    """
    for index, element in enumerate(base_array):
        if np.array_equal(element, array_to_remove):
            base_array.pop(index)
            break


def get_neighbors_local(img:np.ndarray) -> list[tuple[int, int]]:
    """Return the list of 8-connected neighbor coordinates for a given pixel (coord).
    
    parameters:
        img (np.ndarray): 2D binary image. shpape (3, 3)
    
    returns:
        list[tuple[int, int]]: List of neighbor coordinates (x, y).
    """
    assert img.shape == (3, 3), "Image must be of shape (3, 3) "
        
    temp = img.copy()
    temp[1, 1] = 0
    neighbors = np.argwhere(temp > 0)
    return neighbors


def get_neighbors_global(img:np.ndarray,pixel_coord:tuple[int, int]) -> list[tuple[int, int]]:
    """Return the list of 8-connected neighbor coordinates for a given pixel (coord).
    
    parameters:
        img (np.ndarray): 2D binary image. 
        pixel_coord (tuple[int, int]): Coordinates of the pixel in the image.
        
    returns:
        list[tuple[int, int]]: List of neighbor coordinates (x, y).
    """
    # Extract the 3x3 neighborhood around the pixel_coord
    croped_img = img[pixel_coord[0] - 1:pixel_coord[0] + 2, pixel_coord[1] - 1: pixel_coord[1] + 2]
    if croped_img.shape != (3, 3):
        return []
    neighbors_local = get_neighbors_local(croped_img)
    
    # Convert local coordinates to global coordinates
    neighbors_global = neighbors_local + pixel_coord - (1,1)
    
    return neighbors_global

def traverse_skeleton(skeleton: np.ndarray, current_pixel: tuple[int,int]) -> list[tuple[int,int]]:
    """
    Traverse a skeleton image starting at curr_pixel until no neighbor is found.

    At each step:
      - Mark the current pixel as visited by setting sk[x, y] to 0.
      - Extract the 3x3 neighborhood (view) around the current pixel.
      - Check neighbors in a fixed order:
            (0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)
      - If a neighbor is found (nonzero in the view), compute its global coordinates.
      - Append the new pixel to the path.
    The traversal stops when no neighbor is found (indicated by setting the new pixel to [-1, -1]).

    Parameters:
        sk (np.ndarray): 2D skeleton image.
        curr_pixel (np.ndarray): Starting pixel coordinates as an array [x, y] (dtype=int16).

    Returns:
        np.ndarray: A 2D array (shape (num_steps, 2)) containing the global coordinates of the traversed path.
    """
    path = [current_pixel]
    while True:
         # travers the branch
        next_nb = get_neighbors_global(skeleton,current_pixel)
        # Check if any neighbor is found
        if len(next_nb[0]) == 0:
            break
                # one neighbor found keep traversing
        elif len(next_nb[0]) == 1:
            path.append(next_nb[0])
            current_pixel = next_nb[0]

    # Convert the path list to a NumPy array and return it.
    return path

















