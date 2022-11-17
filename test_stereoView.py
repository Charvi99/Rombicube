## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

colorizer = rs.colorizer()
pipeline.start(config)

profile = pipeline.get_active_profile()
infrared_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 2))
infrared_intrinsics = infrared_profile.get_intrinsics()
# depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
# depth_intrinsics = depth_profile.get_intrinsics()

print(infrared_profile)
print(infrared_intrinsics)
# print(depth_profile)
# print(depth_intrinsics)

while True:

    frames = pipeline.wait_for_frames()

    infrared_frame_zero = frames.get_infrared_frame(1)
    infrared_frame_one  = frames.get_infrared_frame(2)
    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()

    infrared_colormap_zero = np.asanyarray(colorizer.colorize(infrared_frame_zero).get_data())
    infrared_colormap_one = np.asanyarray(colorizer.colorize(infrared_frame_one).get_data())
    # depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    # colormap = np.asanyarray(colorizer.colorize(color_frame).get_data())

    # images = np.hstack((depth_colormap, colormap))
    images2 = np.hstack((infrared_colormap_zero, infrared_colormap_one))

    # cv2.imshow('RealSense', images)
    cv2.imshow('RealSense2', images2)

    if cv2.waitKey(25) == ord('q'):
        break
pipeline.stop()
cv2.destroyAllWindows()