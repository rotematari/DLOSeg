import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import enum
import argparse
import os 

class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3
opt = {}
opt["input_svo_file"] = "src/zed/record/recordings/test.svo2"
opt["output_path_dir"] = "record/recordings"
opt["mode"] = 2
# Get input parameters
svo_input_path = "src/zed/record/recordings/HD2K_SN23135249_17-38-45.svo2"
output_dir = "record/recordings"
output_as_video = False    
app_type = AppType.LEFT_AND_RIGHT
if opt["mode"] == 1 or opt["mode"] == 3:
    app_type = AppType.LEFT_AND_DEPTH
if opt["mode"] == 4:
    app_type = AppType.LEFT_AND_DEPTH_16
    
# Check if exporting to AVI or SEQUENCE
if opt["mode"] !=0 and opt["mode"] !=1:
    output_as_video = False


# Specify SVO path parameter
init_params = sl.InitParameters()
init_params.set_from_svo_file(svo_input_path)
init_params.svo_real_time_mode = False  # Don't convert in realtime
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

# Create ZED objects
zed = sl.Camera()

# Open the SVO file specified as a parameter
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    sys.stdout.write(repr(err))
    zed.close()
    exit()

# Get image size
image_size = zed.get_camera_information().camera_configuration.resolution
width = image_size.width
height = image_size.height
width_sbs = width * 2

# Prepare side by side image container equivalent to CV_8UC4
svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

# Prepare single image containers
left_image = sl.Mat()
right_image = sl.Mat()
depth_image = sl.Mat()

video_writer = None

rt_param = sl.RuntimeParameters()

# Start SVO conversion to AVI/SEQUENCE
sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

nb_frames = zed.get_svo_number_of_frames()

while True:
    err = zed.grab(rt_param)
    
    if err == sl.ERROR_CODE.SUCCESS:
        svo_position = zed.get_svo_position()
        # Retrieve SVO images
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)

        # Generate file names
        filename1 = output_dir +"/"+ ("left%s.png" % str(svo_position).zfill(6))
        filename2 = output_dir +"/"+ (("right%s.png" if app_type == AppType.LEFT_AND_RIGHT
                                    else "depth%s.png") % str(svo_position).zfill(6))
        
        left_data = left_image.get_data()
        # Correct the dimensions if necessary:
        print("Original shape:", left_data.shape)
        if left_data.shape[0] != height or left_data.shape[1] != width:
            # Try transpose dimensions if they're reversed
            left_data = np.transpose(left_data, (1, 0, 2))
        
        # Check if left_data is a NumPy array
        if not isinstance(left_data, np.ndarray):
            raise TypeError("Input image is not a NumPy array.")
        # Ensure the array is writable
        if not left_data.flags.writeable:
            left_data = np.copy(left_data)
        # Confirm the data type is uint8
        if left_data.dtype != np.uint8:
            left_data = left_data.astype(np.uint8)
        # left_data = cv2.cvtColor(left_data, cv2.COLOR_RGBA2RGB)
        # Add debugging here
        print("Type of left_data:", type(left_data))
        if isinstance(left_data, np.ndarray):
            print("Shape of left_data:", left_data.shape)
            # Convert RGBA to RGB explicitly
            # left_data = np.ascontiguousarray(left_data[:, :, :4], dtype=np.uint8)
            # left_data = cv2.cvtColor(left_data, cv2.COLOR_RGBA2RGB)
            
        else:
            print("left_data is not a numpy array. It's value:", left_data)
            raise ValueError("left_data is not a numpy array")
        left_data = np.array(left_data, copy=True, dtype=np.uint8)
        print("Data type:", type(left_data))
        print("Shape:", left_data.shape)
        print("dtype:", left_data.dtype)
        print("Contiguous:", left_data.flags['C_CONTIGUOUS'])
        # Convert RGBA to RGB explicitly
        left_data_rgb = cv2.cvtColor(left_data, cv2.COLOR_RGBA2RGB)
        cv2.imshow("left_data", left_data_rgb)
        # cv2.imwrite(str(filename1), img=left_data)



        # Save right images
        cv2.imwrite(str(filename2), right_image.get_data())
        
    else:
        print("Error:", err)