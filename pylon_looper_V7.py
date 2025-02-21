# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:00:05 2024

@author: revel
"""

import cv2
import time
from pypylon import pylon
import datetime
import os 


               ##############################
               #                            #
               #         Set values         #
               #                            #
               ##############################

# Set desired recording duration and output path base
duration = 2*60*60 # Duration in seconds (2 hours x 60 min x 60 seconds)
experiment_name = "pylon_LAM_Dhyd_Females_250220"  # Change the name
start_time = datetime.datetime.strptime("16:00:00", "%H:%M:%S").time()  # Change the time


               ##############################
               #                            #
               #     Functions are here     #
               #                            #
               ##############################


def record_video(duration, output_path_base, video_counter=1):

    frame_width = 1920
    frame_height = 1080
    fps = 5.0 # Frames per second
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Get the first Basler camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    
    converter = pylon.ImageFormatConverter()

    # Converting to OpenCV BGR format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


    while True:
        
        # Set grabbing strategy and frame rate
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        camera.AcquisitionFrameRateEnable.Value = True
        camera.AcquisitionFrameRate.Value = (fps)
        
        # Build full output path with counter
        output_path = f"{output_path_base}_{video_counter}.mp4"
        
        video_writer = cv2.VideoWriter(output_path, video_codec, fps, (frame_width, frame_height))

        # Start recording logic
        end_time = time.time() + duration
        frame_counter = 0
        while time.time() < end_time:
            try:
                # Get next frame and timestamp
                grabResult = camera.RetrieveResult(5000)
                if grabResult.GrabSucceeded:
                    
                    image = converter.Convert(grabResult)
                    frame = image.GetArray()
                    
                    # Get current time with microseconds
                    current_time, microseconds = time.localtime(), time.time() % 1

                    # Format time string with milliseconds
                    formatted_time = time.strftime("%d/%m/%Y %H:%M:%S", current_time) + f".{int(microseconds * 1000)}"

                    # Add current time with milliseconds to frame
                    cv2.putText(frame, f"Time:{formatted_time}",
                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Write frame to video
                    video_writer.write(frame)
                    frame_counter += 1
                    

            except Exception as e:
                print(f"Error capturing frame: {e}")

        video_writer.release()
        grabResult.Release()
        
        # Stop grabbing and release resources
        camera.StopGrabbing()

        print(f"Recorded {frame_counter} frames for {duration} seconds. Video #{video_counter} saved.")

        # Increment counter for next recording
        video_counter += 1


               ##############################
               #                            #
               #    Program starts here     #
               #                            #
               ##############################


output_path_base = f"/home/monitor/Desktop/{experiment_name}"

# Check if the directory exists, if not, create it
if not os.path.exists(output_path_base):
    os.makedirs(output_path_base)
    print(f"Directory '{output_path_base}' created.")

output_path_base = f"{output_path_base}/video"

while datetime.datetime.now().time() < start_time:
    time.sleep(1)

# Start recording loop
record_video(duration, output_path_base)