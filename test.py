import numpy as np

import numpy as np

import numpy as np

def get_extreme_points_hv(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Finds the coordinates of extreme '1' pixels based on row and column extents.

    Specifically, it finds:
    - Leftmost and rightmost '1's in the overall topmost row with '1's.
    - Leftmost and rightmost '1's in the overall bottommost row with '1's.
    - Topmost and bottommost '1's in the overall leftmost column with '1's.
    - Topmost and bottommost '1's in the overall rightmost column with '1's.

    Parameters:
        mask (np.ndarray): A 2D binary array where nonzero (usually 1)
                           values indicate the object/shape.

    Returns:
        list[tuple[int, int]]: A list of unique (row, col) tuples for the
                               identified extreme points. Returns an empty list
                               if no '1's are found. The order is not guaranteed.
    """
    # Find coordinates (row, col) of all pixels marked as 1
    coords = np.argwhere(mask == 1)

    if coords.size == 0:
        # If there are no '1' pixels, return an empty list
        return []

    # Determine the overall min/max row and column indices containing '1's
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)

    # Use a set to store the resulting points, ensuring uniqueness
    extreme_points = set()

    # --- 1. Process Topmost Row ---
    top_row_coords = coords[coords[:, 0] == min_row]
    if top_row_coords.size > 0:
        extreme_points.add(tuple(top_row_coords[top_row_coords[:, 1].argmin()])) # Leftmost
        extreme_points.add(tuple(top_row_coords[top_row_coords[:, 1].argmax()])) # Rightmost

    # --- 2. Process Bottommost Row ---
    bottom_row_coords = coords[coords[:, 0] == max_row]
    if bottom_row_coords.size > 0:
        extreme_points.add(tuple(bottom_row_coords[bottom_row_coords[:, 1].argmin()])) # Leftmost
        extreme_points.add(tuple(bottom_row_coords[bottom_row_coords[:, 1].argmax()])) # Rightmost

    # --- 3. Process Leftmost Column ---
    left_col_coords = coords[coords[:, 1] == min_col]
    if left_col_coords.size > 0:
        extreme_points.add(tuple(left_col_coords[left_col_coords[:, 0].argmin()])) # Topmost
        extreme_points.add(tuple(left_col_coords[left_col_coords[:, 0].argmax()])) # Bottommost

    # --- 4. Process Rightmost Column ---
    right_col_coords = coords[coords[:, 1] == max_col]
    if right_col_coords.size > 0:
        extreme_points.add(tuple(right_col_coords[right_col_coords[:, 0].argmin()])) # Topmost
        extreme_points.add(tuple(right_col_coords[right_col_coords[:, 0].argmax()])) # Bottommost

    # Convert the set of unique tuples back to a list
    return list(extreme_points)

# --- Example Usage with your specific mask ---
mask_example_2 = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],  # Row 0 -> min_row. Left=4, Right=9
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Col 4 -> min_col. Top=0, Bottom=8
    [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # Col 9 -> max_col. Top=0, Bottom=0
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]   # Row 10 -> max_row. Left=0, Right=6. Col 0 -> min_col. Top=10, Bottom=10
])                                      # Col 6 -> max_col (NO, max_col is 9). Col 6: Top=2, Bot=10

# Calculation Trace:
# min_row=0, max_row=10, min_col=0, max_col=9
# Top Row (0): Left=(0,4), Right=(0,9) -> {(0,4), (0,9)}
# Bottom Row (10): Left=(10,0), Right=(10,6) -> {(0,4), (0,9), (10,0), (10,6)}
# Left Col (0): Top=(10,0), Bottom=(10,0) -> {(0,4), (0,9), (10,0), (10,6)} (no change)
# Right Col (9): Top=(0,9), Bottom=(0,9) -> {(0,4), (0,9), (10,0), (10,6)} (no change)

extreme_pixels = get_extreme_points_hv(mask_example_2)
print(f"Calculated: {sorted(extreme_pixels)}") # Sorting for consistent comparison
print(f"Expected:   {sorted([(10, 0), (10, 6), (0, 9), (0, 4)])}")

# Output should match:
# Calculated: [(0, 4), (0, 9), (10, 0), (10, 6)]
# Expected:   [(0, 4), (0, 9), (10, 0), (10, 6)]

# --- Test previous example too ---
mask_example_1 = np.array([
    [0, 0, 0, 0, 1, 1, 0], # row 0. min_r=0. L=4, R=5
    [0, 0, 0, 1, 0, 0, 0], # col 0. min_c=0. T=5, B=6
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0], # col 5. max_c=5. T=0, B=6
    [1, 0, 0, 0, 0, 1, 0]  # row 6. max_r=6. L=0, R=5
])
# min_r=0, max_r=6, min_c=0, max_c=5
# Top R(0): L=(0,4), R=(0,5) -> {(0,4), (0,5)}
# Bot R(6): L=(6,0), R=(6,5) -> {(0,4), (0,5), (6,0), (6,5)}
# Left C(0): T=(5,0), B=(6,0) -> {(0,4), (0,5), (6,0), (6,5), (5,0)}
# Right C(5): T=(0,5), B=(6,5) -> {(0,4), (0,5), (6,0), (6,5), (5,0)}

extreme_pixels_1 = get_extreme_points_hv(mask_example_1)
print(f"\nExample 1 Calculated: {sorted(extreme_pixels_1)}")
# Expected: [(0, 4), (0, 5), (5, 0), (6, 0), (6, 5)]