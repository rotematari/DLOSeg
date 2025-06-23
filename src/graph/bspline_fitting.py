import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from scipy.interpolate import UnivariateSpline
import time
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
def smooth_2d_branch_savgol(coords: np.ndarray,
                            window_length: int = 11,
                            polyorder: int = 2) -> np.ndarray:
    """
    Smooth a 2D branch using a Savitzky-Golay filter.

    This is a very fast method for smoothing noisy data.
    """
    # Ensure window_length is odd and less than the number of points
    window_length = min(window_length, len(coords) - 1)
    if window_length % 2 == 0:
        window_length += 1

    # Apply the filter to each dimension separately
    x_smooth = savgol_filter(coords[:, 0], window_length, polyorder)
    y_smooth = savgol_filter(coords[:, 1], window_length, polyorder)

    return np.vstack((x_smooth, y_smooth)).T
def smooth_2d_branch_splprep(coords: np.ndarray,
                             s: float = 0,
                             num_samples: int = 10) -> np.ndarray:
    """
    Smooth/interpolate a 2D branch using a parametric B-spline.

    This is often faster than fitting two separate UnivariateSplines.
    """
    # 1) Unpack coordinates
    x, y = coords.T

    # 2) Fit a parametric spline
    # splprep returns the B-spline representation (tck) and the parameterization (u)
    tck, u = splprep([x, y], s=s)

    # 3) Evaluate the spline at new parameter values
    M = num_samples or len(coords)
    u_new = np.linspace(u.min(), u.max(), M)
    x_new, y_new = splev(u_new, tck)

    return np.vstack((x_new, y_new)).T

def smooth_2d_branch(coords: np.ndarray,
                     s: float = 0,
                     num_samples: int = None,
                     jump: int = 10) -> np.ndarray:
    """
    Smooth/interpolate a 2D branch via two UnivariateSplines.

    Parameters
    ----------
    coords : (N,2) array of (x,y) points
    s      : smoothing factor (s=0 → interpolate exactly; increase for more smoothing)
    num_samples : number of output points (default = N)

    Returns
    -------
    out : (M,2) array of smoothed (x,y), where M = num_samples or N
    """
    # 1) parameterize by cumulative arc‐length t in [0,1]
    step1_time = time.time()
    diffs = np.diff(coords, axis=0)
    dists = np.hypot(diffs[:,0], diffs[:,1])
    t = np.concatenate(([0], np.cumsum(dists)))
    t /= t[-1]

    M = num_samples or len(coords)
    t_new = np.linspace(0, 1, M)
    out = np.zeros((M, 2))

    for i, dim in enumerate((0, 1)):
        # Determine spline degree: at most 3, but less if not enough points
        k = min(3, len(coords) - 1)
        if k < 1:
            # Not enough points to even fit a linear spline
            out[:, dim] = np.repeat(coords[0, dim], M)
            continue
        spl = UnivariateSpline(t, coords[:, dim], k=k, s=s)
        out[:, dim] = spl(t_new)
    return out


def fit_bspline(points,n_points=100,k=3):
    """
    Fit a B-spline to 2D data points.
    
    Parameters:
    -----------
    points : ndarray of shape (n, 2)
        The x, y coordinates of the data points to fit
        
    Returns:
    --------
    tuple: (tck, fitted_points)
        tck is the B-spline representation (knots, coefficients, degree)
        fitted_points are the points on the fitted curve
    """
    # Extract x and y coordinates

    spl , u = interpolate.splprep(points.T, k = k,s = len(points)*1e-4)
    
    # Generate points along the B-spline
    u_fine = np.linspace(0, 1, n_points)
    
    fitted_points = np.array(interpolate.splev(u_fine, spl)).T
    
    return spl, fitted_points

def plot_results(points, fitted_points, control_points=None):
    """
    Plot the original points, fitted curve, and optionally control points.
    
    Parameters:
    -----------
    points : ndarray of shape (n, 2)
        The original data points
    fitted_points : ndarray of shape (m, 2)
        The points along the fitted B-spline curve
    control_points : ndarray of shape (p, 2) or None
        The control points of the B-spline curve (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Plot original points
    plt.plot(points[:, 0], points[:, 1], 'b.', markersize=3, label='Original Points')
    
    # Plot fitted B-spline curve
    plt.plot(fitted_points[:, 0], fitted_points[:, 1], 'r-', linewidth=2, label='B-spline Fit')
    
    # Plot control points if provided
    if control_points is not None:
        plt.plot(control_points[:, 0], control_points[:, 1], 'go', markersize=5, label='Control Points')
        plt.plot(control_points[:, 0], control_points[:, 1], 'g--', linewidth=1)
    
    plt.title('B-spline Curve Fitting')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis to match the image coordinate system
    # plt.axis('equal')
    plt.show()

def extract_control_points(tck):
    """
    Extract control points from a B-spline representation.
    
    Parameters:
    -----------
    tck : tuple
        B-spline representation returned by scipy.interpolate.splprep
    
    Returns:
    --------
    ndarray of shape (n, 2)
        The control points of the B-spline
    """
    # tck contains (knots, coefficients, degree)
    # coefficients is a list of arrays, one for each dimension
    x_coeffs = tck[1][0]
    y_coeffs = tck[1][1]
    
    return np.column_stack((x_coeffs, y_coeffs))

# Demo with sample data
def generate_sample_data():
    """Generate sample data similar to the provided image."""
    # Create a circular path with some additional shapes
    t = np.linspace(0, 2*np.pi, 100)
    
    # Main circular path
    x1 = 3 * np.cos(t) + 5
    y1 = 3 * np.sin(t) + 5
    
    # Connected extension upward
    t2 = np.linspace(0, np.pi, 100)
    x2 = 0.5 * np.cos(t2) + 8
    y2 = 3 * np.sin(t2) + 10
    
    # Another shape to the right
    t3 = np.linspace(0, 2*np.pi, 100)
    x3 = 2 * np.cos(t3) + 12
    y3 = 1 * np.sin(t3) + 8
    
    # Combine all paths
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    
    # Add some noise
    x += np.random.normal(0, 0.05, len(x))
    y += np.random.normal(0, 0.05, len(y))
    
    return np.column_stack((x, y))

def main():
    # Generate or load your points
    # For demo, we'll generate points
    points = generate_sample_data()
    
    # You can also load points from a file
    # points = np.loadtxt('your_data_file.txt')
    
    # Fit B-spline
    n_control_points = 30  # Adjust based on complexity
    degree = 3  # Cubic spline
    smoothing = 10  # Adjust based on noise level (0 for perfect interpolation)
    
    tck, fitted_points = fit_bspline(points)
    
    # Extract control points
    control_points = extract_control_points(tck)
    
    # Plot results
    plot_results(points, fitted_points, control_points)
    
    # If you want to save the fitted curve
    # np.savetxt('fitted_curve.txt', fitted_points)
    
    return tck, fitted_points, control_points

if __name__ == "__main__":
    tck, fitted_points, control_points = main()
