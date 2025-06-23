import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_bspline_function(points_2d):
    """
    Fits a B-spline to a set of 2D points and returns a function
    that can be evaluated at any parameter 'u'.
    
    Args:
        points_2d (np.ndarray): Array of shape (N, 2) representing the points.

    Returns:
        callable: A function that takes a parameter u (0 to 1) and returns the (x, y) coordinates.
        tck: The raw spline representation from scipy (knots, coefficients, degree).
    """
    # Transpose points for splprep which expects coordinates in columns
    points_2d_t = points_2d.T
    
    # Fit the B-spline. s=0 forces interpolation through all points.
    # k=3 for a cubic spline, which is common.
    # You might need to adjust 's' if your points are noisy.
    tck, u = splprep([points_2d_t[0], points_2d_t[1]], s=0, k=3)
    
    def spline_eval(u_interp):
        return np.array(splev(u_interp, tck)).T
        
    return spline_eval, tck

def find_correspondences(spline_func_left, spline_func_right, tck_right, num_samples=100):
    """
    Finds corresponding points between two B-splines using the epipolar constraint
    (y_left = y_right for rectified images).

    Args:
        spline_func_left (callable): Function for the left B-spline.
        spline_func_right (callable): Function for the right B-spline.
        tck_right: The tck tuple for the right spline, used for root finding.
        num_samples (int): Number of points to sample along the left spline.

    Returns:
        tuple: (matched_points_left, matched_points_right), each as an (M, 2) np.ndarray.
    """
    matched_points_left = []
    matched_points_right = []
    
    # Generate samples along the left spline
    u_samples = np.linspace(0, 1, num_samples)
    left_points = spline_func_left(u_samples)
    
    for p_left in left_points:
        y_target = p_left[1]
        
        # Define a function whose root is the 'u' parameter on the right spline
        # where the y-coordinate matches the target y.
        def y_error_func(u_right):
            x_r, y_r = splev(u_right, tck_right)
            return y_r - y_target
            
        # Use fsolve to find the 'u' parameter for the right spline.
        # We need a good initial guess. We'll scan for a close 'u'.
        # This is a simplification; for complex curves, multiple solutions may exist.
        u_guess = np.linspace(0, 1, 100)
        y_right_samples = splev(u_guess, tck_right)[1]
        initial_guess_idx = np.argmin(np.abs(y_right_samples - y_target))
        
        u_solution, = fsolve(y_error_func, u_guess[initial_guess_idx])
        
        # Check if the solution is within the valid parameter range [0, 1]
        if 0 <= u_solution <= 1:
            p_right = np.array(splev(u_solution, tck_right))
            
            matched_points_left.append(p_left)
            matched_points_right.append(p_right)

    if not matched_points_left:
        raise ValueError("Could not find any corresponding points. Check spline overlap and y-coordinates.")
        
    return np.array(matched_points_left), np.array(matched_points_right)

def triangulate_and_reconstruct(calib_data, bspline_left_pts, bspline_right_pts):
    """
    Main pipeline function to perform 3D reconstruction of a B-spline.

    Args:
        calib_data (dict): Dictionary with ZED camera calibration data.
        bspline_left_pts (np.ndarray): (N, 2) array of points for the left B-spline.
        bspline_right_pts (np.ndarray): (M, 2) array of points for the right B-spline.

    Returns:
        tuple: (points_3d, spline_3d_func)
               points_3d: The raw triangulated 3D points (K, 3).
               spline_3d_func: A callable function for the final smooth 3D B-spline.
    """
    # --- Step 1: Create B-spline functions from input points ---
    print("Step 1: Fitting 2D B-splines...")
    spline_func_left, tck_left = get_bspline_function(bspline_left_pts)
    spline_func_right, tck_right = get_bspline_function(bspline_right_pts)

    # --- Step 2: Find corresponding points ---
    print("Step 2: Finding correspondences using epipolar constraint...")
    points_l, points_r = find_correspondences(spline_func_left, spline_func_right, tck_right)

    # --- Step 3: Triangulate points ---
    print("Step 3: Triangulating 2D point pairs...")
    P1 = calib_data['P1']
    P2 = calib_data['P2']
    
    # The points need to be in shape (2, N) and type float32 for cv2.triangulatePoints
    points_l_t = points_l.T.astype(np.float32)
    points_r_t = points_r.T.astype(np.float32)

    # Triangulate
    points_4d_hom = cv2.triangulatePoints(P1, P2, points_l_t, points_r_t)

    # Convert from homogeneous to cartesian coordinates
    points_3d = (points_4d_hom[:3, :] / points_4d_hom[3, :]).T
    
    # --- Step 4: Fit a 3D B-spline to the reconstructed points ---
    print("Step 4: Fitting a smooth 3D B-spline...")
    if len(points_3d) < 4:
         print("Warning: Not enough points for a cubic 3D spline. Returning raw points.")
         return points_3d, None

    points_3d_t = points_3d.T
    tck_3d, u_3d = splprep([points_3d_t[0], points_3d_t[1], points_3d_t[2]], s=0.1, k=3)
    
    def spline_3d_func(u_interp):
        return np.array(splev(u_interp, tck_3d)).T
        
    print("Reconstruction complete.")
    return points_3d, spline_3d_func

def visualize_reconstruction(bspline_left_pts, bspline_right_pts, points_3d, spline_3d_func):
    """Visualizes the entire process."""
    fig = plt.figure(figsize=(18, 6))
    
    # 2D Splines
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(bspline_left_pts[:, 0], bspline_left_pts[:, 1], 'bo-', label='Left B-spline Pts')
    ax1.plot(bspline_right_pts[:, 0], bspline_right_pts[:, 1], 'ro-', label='Right B-spline Pts')
    ax1.set_title('Input 2D B-Splines (Rectified)')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.legend()
    ax1.invert_yaxis() # Image coordinates
    ax1.set_aspect('equal', adjustable='box')

    # 3D Point Cloud
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='g', marker='o', label='Triangulated Pts')
    ax2.set_title('Reconstructed 3D Point Cloud')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    
    # Final 3D B-Spline
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='g', marker='o', label='Triangulated Pts', alpha=0.3)
    if spline_3d_func:
        u_fine = np.linspace(0, 1, 200)
        smooth_3d_points = spline_3d_func(u_fine)
        ax3.plot(smooth_3d_points[:, 0], smooth_3d_points[:, 1], smooth_3d_points[:, 2], 'r-', label='Fitted 3D B-spline')
    ax3.set_title('Final 3D B-Spline')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.legend()

    # Set same limits for comparison
    max_range = np.array([points_3d[:, 0].max()-points_3d[:, 0].min(), 
                          points_3d[:, 1].max()-points_3d[:, 1].min(), 
                          points_3d[:, 2].max()-points_3d[:, 2].min()]).max() / 2.0
    mid_x = (points_3d[:, 0].max()+points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max()+points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max()+points_3d[:, 2].min()) * 0.5
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # =========================================================================
    # 1. DEFINE YOUR INPUTS HERE
    # =========================================================================
    
    # --- Mock Calibration Data (REPLACE WITH YOUR ACTUAL ZED DATA) ---
    # This data is for a hypothetical ZED camera setup (e.g., 1080p)
    # P1 and P2 are the most important for triangulation.
    # Dimensions are in pixels for K and P, meters for T.
    fx = 1050.0
    fy = 1050.0
    cx = 960.0
    cy = 540.0
    baseline = 0.12  # 120mm baseline in meters

    K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    P1 = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])
    P2 = np.array([[fx, 0, cx, -fx * baseline], [0, fy, cy, 0], [0, 0, 1, 0]])

    calib_data = {
        'K1': K1, 'D1': np.zeros(5), # Assuming rectified, so distortion is zero
        'K2': K1, 'D2': np.zeros(5),
        'R': np.identity(3),
        'T': np.array([[-baseline], [0], [0]]), # Translation vector
        'P1': P1,
        'P2': P2,
        'baseline': baseline,
        'resolution': '1920x1080'
    }

    # --- Mock B-Spline 2D Points (REPLACE WITH YOUR ACTUAL SPLINE POINTS) ---
    # These points simulate a cable curving away from the camera.
    # The right spline has a horizontal shift (disparity) due to the baseline.
    y_coords = np.linspace(200, 800, 10)
    x_coords_l = 800 + 100 * np.sin(y_coords / 200)
    
    # Disparity is larger for closer points (smaller y)
    disparity = 60 - (y_coords / 800) * 40
    x_coords_r = x_coords_l - disparity

    bspline_left_pts = np.vstack([x_coords_l, y_coords]).T
    bspline_right_pts = np.vstack([x_coords_r, y_coords]).T

    # =========================================================================
    # 2. RUN THE RECONSTRUCTION
    # =========================================================================
    try:
        points_3d, spline_3d_func = triangulate_and_reconstruct(
            calib_data, 
            bspline_left_pts, 
            bspline_right_pts
        )

        # =========================================================================
        # 3. VISUALIZE AND USE THE RESULTS
        # =========================================================================
        visualize_reconstruction(
            bspline_left_pts, 
            bspline_right_pts, 
            points_3d, 
            spline_3d_func
        )
        
        # Example of how to use the final 3D spline function
        if spline_3d_func:
            # Get 10 points along the smooth 3D spline
            u_values = np.linspace(0, 1, 10)
            ten_points_on_3d_spline = spline_3d_func(u_values)
            print("\n10 sampled points on the final 3D spline:\n", ten_points_on_3d_spline)
            
    except (ValueError, RuntimeError) as e:
        print(f"\nAn error occurred during reconstruction: {e}")