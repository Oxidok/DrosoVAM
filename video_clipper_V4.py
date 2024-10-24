# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:09:01 2024

@author: revel
"""


import ffmpeg
import os
import time
import configparser

               ##############################
               #                            #
               #         Set values         #
               #                            #
               ##############################

expname = "DPA_Dmel_Mix_V1"

# Define paths
base_folder = "C:/Users/revel/Desktop/Video_work/"
video_folder = f"{base_folder}vw_{expname}/"
output_folder = f"{base_folder}vw_{expname}_output_V9/"
cmd_path = "C:/Users/revel/Downloads/ffmpeg-6.1.1-full_build/bin/ffmpeg.exe"

# Define number of rows and columns in the grid (2 and 8 for food / 4 and 8 for activity)
rows = 2
cols = 8

# Define arena region coordinates (manual definition)
arena_x1, arena_y1, arena_x2, arena_y2 = 150, 53, 1500, 1010

# Define the video numbers, "video_x_y_z", the x parameter here.
first_video = 1
last_video = 121


               ##############################
               #                            #
               #    Program starts here     #
               #                            #
               ##############################


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize variables for time estimation
start_time = time.time()
last_video_processing_time = 0

# Loop through each video
for video_number in range(first_video, last_video + 1):
    video_start_time = time.time()
    input_video = os.path.join(video_folder, f"video_{video_number}.mp4")
    print(input_video)
    
    # Calculate width and height of each chamber based on the arena dimensions
    chamber_width = int((arena_x2 - arena_x1) / cols)
    chamber_height = int((arena_y2 - arena_y1) / rows)
    
    
    # Loop through each chamber and create output video
    for i in range(rows):
        for j in range(cols):
            # Define coordinates for cropping the video relative to the arena region
            x1 = arena_x1 + j * chamber_width
            y1 = arena_y1 + i * chamber_height

            # Define output video name
            output_video = os.path.join(output_folder, f"video_{video_number}_{i+1}_{j+1}.mp4")
            print(output_video)

            # Use ffmpeg-python to crop and save the video
            process = ffmpeg.input(input_video)
            process = ffmpeg.crop(process, x1, y1, chamber_width, chamber_height)
            process = ffmpeg.output(process, output_video)
            ffmpeg.run(process, cmd=cmd_path)
            
            
    # Calculate and print time estimations
    last_video_processing_time = time.time() - video_start_time
    remaining_videos = last_video - video_number
    estimation_time = (last_video_processing_time * remaining_videos) / 60  # in minutes
    finish_time = time.strftime('%H:%M:%S', time.localtime(time.time() + estimation_time * 60))

    print(f"Video processed in {last_video_processing_time:.2f} seconds. Time remaining: {estimation_time:.2f} minutes. Estimated time of completion: {finish_time}")

# Create a configparser object
config = configparser.ConfigParser()

# Write the config file
with open(os.path.join(output_folder, 'config.ini'), 'w') as configfile:
    configfile.write("[CHAMBER_DESIGN]\n")
    configfile.write("# Number of rows and columns in the monitoring chamber:\n")
    configfile.write(f"row_number = {rows}\n")
    configfile.write(f"col_number = {cols}\n")
    configfile.write("# Video used for the analysis:\n")
    configfile.write(f"first_video_number = {first_video}\n")
    configfile.write(f"last_video_number = {last_video}\n")
    configfile.write("# Food type used in the chamber:\n")
    configfile.write("# If no food, leave blank with ''\n")
    configfile.write("top_food = 'Food'\n")
    configfile.write("bottom_food = 'Food'\n\n")
    
    configfile.write("[FLIES_DETAILS]\n")
    configfile.write("# What are your flies? How are they grouped?\n")
    configfile.write("species = 'Dmel'\n")
    configfile.write("# Which rows are forming group 1?\n")
    configfile.write("group1 = [1, 2]\n")
    configfile.write("group1_name = 'Virgin'\n")
    configfile.write("group1_color = 'mediumseagreen'\n")
    configfile.write("# Same for group 2\n#Leave empty if there is only one group\n")
    configfile.write("group2 = [3, 4]\n")
    configfile.write("group2_name = 'Mated'\n")
    configfile.write("group2_color = 'mediumorchid'\n\n")
    
    configfile.write("# Personal color memo:\n")
    configfile.write("# mediumseagreen -> Virgin\n")
    configfile.write("# mediumorchid -> Mated\n")
    configfile.write("# tomato -> Males\n")
    configfile.write("# dodgerblue -> Mixed\n\n")             

    configfile.write("[EXPERIMENT]\n")
    configfile.write("# Were the videos analyzed in DLC with DrosoVAM_V3?\n")
    configfile.write("new_model = True\n")
    configfile.write("# How long are the videos in minutes?\n")
    configfile.write("duration = 10\n")
    configfile.write("# For how many hours the light was on before switching off?\n")
    configfile.write("delay = 4.5\n")
    configfile.write("# Was the experiment done in LD condition? (circadian Light/Dark cycles)\n")
    configfile.write("classic_ld = True\n")
    configfile.write("# If constant Darkness, after how many cycle did it started?\n")
    configfile.write("# Example: if 1 -> Light(delay) > 1*{Dark(12h) > Light(12h)} > Dark(constant)\n")
    configfile.write("# Ignore if classic_ld is True\n")
    configfile.write("cycles_before_dd = 1\n")
    configfile.write("# Is the experiment a Darkness Preference Assay?\n")
    configfile.write("is_dpa = True\n")
    configfile.write("# When plotting flies position, in how many bin to split the chamber?\n")
    configfile.write("position_bins = 20\n\n")
    
    configfile.write("[ANALYSIS]\n")
    configfile.write("fly_length = None\n")
    configfile.write("skip_tracking = False\n")
    configfile.write("skip_concat = False\n")
    configfile.write("skip_indiv_analysis = False # Changing to True will override the next 9 steps\n")
    configfile.write("skip_threshold_check = True\n")
    configfile.write("skip_data_curation = False\n")
    configfile.write("skip_offlimit_plots = True\n")
    configfile.write("skip_hist = False\n")
    configfile.write("skip_indiv_pos_plot = True\n")
    configfile.write("skip_indiv_distance_plot = False\n")
    configfile.write("skip_virtual_beam = False\n")
    configfile.write("skip_beams_indiv_plots = True\n")
    configfile.write("skip_group_virtual_beam = False\n")
    configfile.write("skip_position_plot = False\n")
    configfile.write("skip_fpi_plot = False\n")
    configfile.write("skip_distance_group_plot = False\n")
    configfile.write("skip_plot_check = True\n")
    configfile.write("plot_to_remove = []\n")
    configfile.write("skip_position_time_plot = False\n")

    
print("Videos successfully cropped and saved! Config file created!")



























