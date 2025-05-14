########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import open3d as o3d
import matplotlib.pyplot as plt



def main():
    
    svo_input_path = "/home/admina/Documents/ZED/HD2K_03.svo2"
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS  # Use NEURAL_PLUS depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use HD720 resolution
    # init_params.set_from_svo_file(svo_input_path)
    
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    
    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()



    while i < 500:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            point_cloud_np = point_cloud.get_data()
           
                            
            pc_mat = point_cloud                             # ← your sl.Mat
            pc_np  = pc_mat.get_data()                       # (H, W, 4) float32

            # keep only finite XYZ rows and flatten to N×3 / N×3 uint8 colour
            mask   = np.isfinite(pc_np[..., 2])
            xyz    = pc_np[..., :3][mask]
            rgba   = pc_np[..., 3][mask].view(np.uint32)
            rgb8   = np.column_stack(((rgba >>  0) & 255,
                                    (rgba >>  8) & 255,
                                    (rgba >> 16) & 255)).astype(np.uint8)

            cloud  = o3d.geometry.PointCloud()
            cloud.points  = o3d.utility.Vector3dVector(xyz)
            cloud.colors  = o3d.utility.Vector3dVector(rgb8.astype(np.float32) / 255.0)

            o3d.visualization.draw_geometries([cloud],
                window_name="ZED point cloud",
                width=960, height=540)
            
            
            print("Point Cloud Shape: ", point_cloud_np.shape)

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
