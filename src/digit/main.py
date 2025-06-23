# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
import pprint
import time

import cv2

from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler

# logging.basicConfig(level=logging.DEBUG)

# Print a list of connected DIGIT's
digits = DigitHandler.list_digits()
print("Connected DIGIT's to Host:")
pprint.pprint(digits)

# Connect to a Digit device with serial number with friendly name
digit_1 = Digit("D21123", "Left Gripper")
digit_2 = Digit("D21124", "Right Gripper")

digit_1.connect()
digit_2.connect()

# Print device info
# print(digit.info())

# Change LED illumination intensity
digit_1.set_intensity(Digit.LIGHTING_MIN)
# time.sleep(1)
digit_2.set_intensity(Digit.LIGHTING_MAX)

# Change DIGIT resolution to QVGA
vga_res = Digit.STREAMS["VGA"]
digit_1.set_resolution(vga_res)

# Change DIGIT FPS to 15fps
fps_30 = Digit.STREAMS["VGA"]["fps"]["30fps"]
digit_1.set_fps(fps_30)

# Grab single frame from DIGIT
frame = digit_1.get_frame()
print(f"Frame WxH: {frame.shape[0]}{frame.shape[1]}")

# Display stream obtained from DIGIT
digit_1.show_view()

# Disconnect DIGIT stream
digit_1.disconnect()

