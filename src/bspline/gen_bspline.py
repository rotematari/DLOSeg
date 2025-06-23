# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev ,make_splprep ,BSpline
# from PIL import Image, ImageDraw
# # Generate random control points
# num_ctrl = 8
# # ctrl_pts = np.random.rand(num_ctrl, 3)
# ctrl_pts = np.random.random((num_ctrl, 3))   # Scale to [0, 1] range

# # ctrl_pts =np.array([[0.7335527 , 0.22488279, 0.72843749],
# #        [0.12793268, 0.50282796, 0.18708931],
# #        [0.44890546, 0.19913811, 0.72486173],
# #        [0.90407324, 0.28287865, 0.93314284],
# #        [0.84255725, 0.5467317 , 0.40076334],
# #        [0.82735912, 0.99791616, 0.14797252],
# #        [0.28848458, 0.5255787 , 0.52975955],
# #        [0.33078376, 0.99407352, 0.47422276]])
# # Compute B-spline representation of the control points
# # tck, u = splprep(ctrl_pts.T, s=0.01, k=5)
# u_fine = np.linspace(0, 1, 200)
# # spline = splev(u_fine, tck)
# spl, u = make_splprep(ctrl_pts.T, s=0.1, k=3)

# spline = spl(u_fine)


# # assume `spl` is your original BSpline
# t, c, k = spl.t, spl.c, spl.k
# # c has shape (ndim, n_ctrl)

# # 1) normalize each coefficient array to [0,1]
# c_norm = []
# for dim_vals in c:
#     mn, mx = dim_vals.min(), dim_vals.max()
#     if mx > mn:
#         c_norm.append((dim_vals - mn) / (mx - mn))
#     else:
#         # all values equal → leave them at zero (or whatever you prefer)
#         c_norm.append(np.zeros_like(dim_vals))
# c_norm = np.stack(c_norm, axis=0)

# # 2) rebuild a new, normalized spline
# spl_norm = BSpline(t, c_norm, k)

# spl_norm = spl_norm(u_fine)

# # # normalize the spline to [0, 1] range
# # spline_norm = np.array(spline)
# # spline_norm[0] = (spline_norm[0] - spline_norm[0].min()) / (spline_norm[0].max() - spline_norm[0].min())
# # spline_norm[1] = (spline_norm[1] - spline_norm[1].min()) / (spline_norm[1].max() - spline_norm[1].min())
# # spline_norm[2] = (spline_norm[2] - spline_norm[2].min()) / (spline_norm[2].max() - spline_norm[2].min())


# # check if the normelized spline has the same coeficients as the original spline




# # Plot the control points and the resulting B-spline curve
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2], marker='o')
# ax.plot(spline[0], spline[1], spline[2])
# ax.set_title('Random 3D B-spline')
# plt.pause(0.001)




# # 2. Project the spline onto the XY plane
# x, y = spline[0], spline[1]

# # 3. Prepare the output binary image
# W, H = 512, 512  # image dimensions
# border_frac = 0.1  # 10% border on each side
# px_margin = int(W * border_frac)
# py_margin = int(H * border_frac)

# # 3. Map to pixel coordinates with margins
# px = (x * (W - 1 - 2 * px_margin) + px_margin).astype(np.int32)
# py = ((1 - y) * (H - 1 - 2 * py_margin) + py_margin).astype(np.int32)

# #  Ensure indices are within image bounds
# px = np.clip(px, 0, W - 1)
# py = np.clip(py, 0, H - 1)


# # 4. Draw the projected curve into a PIL image
# img = Image.new('L', (W, H), 0)
# draw = ImageDraw.Draw(img)
# points = list(zip(px, py))
# draw.line(points, fill=255, width=5)

# # 5. Convert to a binary (0/1) numpy array
# binary = (np.array(img) > 0).astype(np.uint8)


# # 
# # 6. Display the binary image
# plt.figure(figsize=(5,5))
# plt.imshow(binary, cmap='gray')
# plt.title('2D Binary Projection of 3D B-spline')
# plt.axis('off')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, BSpline
from PIL import Image, ImageDraw
def gen_random_bspline():
    """
    Generate a random 3D B-spline, normalize its coefficients,
    sample it, and project onto the XY plane to create a binary mask.
    """
    # 0. Set random seed for reproducibility
    # np.random.seed(42)
    # 1. Generate random control points in normalized [0,1] range
    num_ctrl = np.random.randint(6, 10)  # Random number of control points
    size = np.random.uniform(1.0, 1.0)
    ctrl_pts = np.random.rand(num_ctrl, 3) 

    # 2. Fit a cubic B-spline (k=3) with smoothing factor s=0.1
    tck, u = splprep(ctrl_pts.T, s=0.2, k=5)

    # 3. Normalize the control-point coefficients to [0,1]
    t, c, k = tck
    c_norm = []
    for dim_vals in c:
        mn, mx = dim_vals.min(), dim_vals.max()
        if mx > mn:
            c_norm.append((dim_vals - mn) * size/ (mx - mn))
        else:
            c_norm.append(np.zeros_like(dim_vals))
    c_norm = np.stack(c_norm, axis=0)
    tck_norm_3d = (t, c_norm, k)
    tck_norm_2d = (t, c_norm[:2], k)  # 2D projection for rasterization

    # 4. Sample both original and normalized splines
    u_fine = np.linspace(0, 1, 200)
    spline_orig = np.vstack(splev(u_fine, tck)).T  # shape (200,3)
    spline_norm3d = np.vstack(splev(u_fine, tck_norm_3d)).T
    spline_norm2d = np.vstack(splev(u_fine, tck_norm_2d)).T

    # 5. Print coefficient ranges before and after normalization
    print("Original coeffs ranges per dim:", [(dim.min(), dim.max()) for dim in c])
    print("Normalized coeffs ranges per dim:", [(dim.min(), dim.max()) for dim in c_norm])

    return ctrl_pts, spline_orig, spline_norm3d, spline_norm2d

for _ in range(3):
    plt.close('all')  # Close any existing plots
    _, spline_orig, spline_norm3d, spline_norm2d = gen_random_bspline()
    # 6. 3D plot: control points, original spline, normalized spline
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(ctrl_pts[:,0], ctrl_pts[:,1], ctrl_pts[:,2], c='red', marker='o', label='Control Points')
    # ax.plot(spline_orig[:,0], spline_orig[:,1], spline_orig[:,2], label='Original Spline')
    ax.plot(spline_norm3d[:,0], spline_norm3d[:,1], spline_norm3d[:,2], linestyle='--', label='Normalized Spline')
    ax.set_title('3D B-Spline and Normalized Version')
    ax.legend()
    plt.pause(0.001)
    # 6. 2D plot: original spline projection
    plt.figure(figsize=(6, 6))
    plt.plot(spline_norm2d[:, 0], spline_norm2d[:, 1], label='Normalized Spline Projection')
    plt.title('2D Projection of Normalized B-Spline')
    plt.axis('equal')
    plt.legend()
    plt.pause(0.001)
    # 7. Project the original spline onto the XY plane and rasterize to binary mask
    W, H = 256, 256
    border_frac = 0.01
    px_margin = int(W * border_frac)
    py_margin = int(H * border_frac)

    # Map normalized [0,1] to pixel coords with margin
    xy = spline_norm3d[:, :2]
    # print min max xy
    print("XY min/max:", xy.min(axis=0), xy.max(axis=0))
    px = (xy[:,0] * (W - 1 - 2 * px_margin) + px_margin).astype(np.int32)
    py = ((1 - xy[:,1]) * (H - 1 - 2 * py_margin) + py_margin).astype(np.int32)
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    # Draw curve into a binary image
    img = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(img)
    draw.line(list(zip(px, py)), fill=255, width=4)
    binary_mask = (np.array(img) > 0).astype(np.uint8)

    # 8. Display the binary mask
    plt.figure(figsize=(5, 5))
    plt.imshow(binary_mask, cmap='gray')
    plt.title('2D Binary Projection of 3D B-Spline')
    plt.axis('off')
    plt.show()
