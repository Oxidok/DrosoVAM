# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:33:29 2024

@author: revel
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import ffmpeg
import cv2
import os
import time
import configparser


cmd_path = "C:/Users/revel/Downloads/ffmpeg-2024-09-26-git-f43916e217-full_build/bin/ffmpeg.exe"
click_count, point1, point2 = 0, 0, 0
col_number, row_number, vid_number = 0, 0, 0
frame_image = None
folder_path, output_path = "", ""
ratio = 0

positions = []

def browse_folder():
    """Opens file dialog to select folder and counts video files."""

    global folder_path

    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_entry.delete(0, tk.END)  # Clear existing text
        folder_entry.insert(0, folder_path)
        
        # Count video files in the selected folder
        video_count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more extensions if needed
                video_count += 1
        
        # Update the "Number of Videos" entry field
        num_vids_entry.delete(0, tk.END)  # Clear existing content
        num_vids_entry.insert(0, str(video_count))
        
    return folder_path


def open_folder():
    """
    Opens the selected folder, displays the first frame, and updates button states.
    """
    global col_number, row_number, frame_image, vid_number, output_path, folder_path
    
    folder_path = folder_entry.get()
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    video_path = os.path.join(folder_path, os.listdir(folder_path)[0])

    output_path = f"{folder_path}_output_V9/"
    folder_path = f"{folder_path}/"

    col_number = int(num_cols_entry.get())
    row_number = int(num_rows_entry.get())
    vid_number = int(num_vids_entry.get())


    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read video frame")
            return

        # Process and display the first frame
        processed_image = edit_image(frame, canvas_width, canvas_height)
        canvas.delete("all")
        frame_image = ImageTk.PhotoImage(processed_image)
        canvas.create_image(0, 0, image=frame_image, anchor="nw")

        # Update button states
        validate_button.config(state=tk.NORMAL)
        reset_button.config(state=tk.NORMAL)

    except FileNotFoundError:
        print("Error: Folder path not found or invalid video format")
    
    canvas.bind("<Button-1>", on_click)    
    
    return folder_path, output_path


def edit_image(frame, max_width, max_height):
    """
    Performs all image editing operations, including resizing and potential user-defined actions.

    Args:
        frame: The frame to be edited.
        max_width: Maximum allowed width for the resized image.
        max_height: Maximum allowed height for the resized image.

    Returns:
        A PIL Image object with all edits applied.
    """
    global ratio
    
    image = Image.fromarray(frame)
    width, height = image.size
    ratio = min(max_width / width, max_height / height)

    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    resized_image = image.resize((new_width, new_height))
    
    return resized_image


def reset_points():
    """
    Functionality for the "Redefine" button.
    Clears the overlay (red circles and yellow grid lines) from the canvas.
    """
    global point1, point2, click_count

    point1, point2 = None, None  # Reset points
    click_count = 0  # Reset click count

    # Remove red circles and yellow lines (overlay)
    for item in canvas.find_all():
        if canvas.type(item) in ("oval", "line"):  # Check if the item is an oval (circle) or a line
            canvas.delete(item)

    # Re-enable the on_click event
    canvas.bind("<Button-1>", on_click)
    
    
def on_click(event):
    """
    Handles mouse clicks on the canvas.

    - Records first click coordinates (point1).
    - Records second click coordinates (point2) and updates canvas with grid (optional).
    """
    global click_count, point1, point2, col_number, row_number
    
    x = event.x
    y = event.y

    if click_count == 0:
        point1 = (x, y)
        positions.append(x)
        positions.append(y)
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")
        click_count += 1
    elif click_count == 1:
        point2 = (x, y)
        positions.append(x)
        positions.append(y)
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")

        # Calculate grid parameters (optional, based on your needs)
        if point1 and point2:  # Check if both points are selected
            grid_width = abs(point2[0] - point1[0])
            grid_height = abs(point2[1] - point1[1])
            col_spacing = grid_width / col_number
            row_spacing = grid_height / row_number

            # Draw grid lines (optional, based on your needs)
            for i in range(0, col_number + 1):
                x = point1[0] + i * col_spacing
                canvas.create_line(x, point1[1], x, point2[1], fill="yellow")
            for i in range(0, row_number + 1):
                y = point1[1] + i * row_spacing
                canvas.create_line(point1[0], y, point2[0], y, fill="yellow")

        canvas.update()  # Update canvas content after drawing
        click_count = 0  # Reset click count
        
        canvas.unbind("<Button-1>")


def validate_points():
    
    global folder_path, output_path, vid_number, positions, col_number, row_number, cmd_path, ratio
    
    positions = [int(x / ratio) for x in positions]
    
    first_video = 1
    last_video = vid_number
    
    arena_x1, arena_y1, arena_x2, arena_y2 = positions[0], positions[1], positions[2], positions[3]

    rows = row_number
    cols = col_number
    
    video_folder = folder_path
    output_folder = output_path
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Close the tkinter window
    root.destroy()

               ##############################
               #                            #
               #    Program starts here     #
               #                            #
               ##############################
    
    # Initialize variable for time estimation
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
        
    print("Videos cropped successfully!")
    
    config_maker(output_folder)


def config_maker(output_folder):
    
    global row_number, col_number, vid_number, config
    
    # Create a configparser object
    config = configparser.ConfigParser()
    
    with open(os.path.join(output_folder, 'config.ini'), 'w') as configfile:
        configfile.write("[CHAMBER_DESIGN]\n")
        configfile.write("# Number of rows and columns in the monitoring chamber:\n")
        configfile.write(f"row_number = {row_number}\n")
        configfile.write(f"col_number = {col_number}\n")
        configfile.write("# Video used for the analysis:\n")
        configfile.write("first_video_number = 1\n")
        configfile.write(f"last_video_number = {vid_number}\n")
        configfile.write("# Food type used in the chamber:\n")
        configfile.write("# If no food, leave blank with ''\n")
        configfile.write("top_food = ''\n")
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
        configfile.write("duration = 120\n")
        configfile.write("# For how many hours the light was on before switching off?\n")
        configfile.write("delay = 6\n")
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
        configfile.write("skip_concat = False\n\n")
        configfile.write("skip_indiv_analysis = False # Changing to True will override the next 9 steps\n")
        configfile.write("skip_threshold_check = True\n")
        configfile.write("skip_data_curation = False\n")
        configfile.write("skip_offlimit_plots = True\n")
        configfile.write("skip_hist = False\n")
        configfile.write("skip_indiv_pos_plot = True\n")
        configfile.write("skip_indiv_distance_plot = False\n")
        configfile.write("skip_virtual_beam = False\n")
        configfile.write("skip_beams_indiv_plots = True\n")
        configfile.write("skip_group_virtual_beam = False\n\n")
        configfile.write("skip_position_plot = False\n")
        configfile.write("skip_fpi_plot = False\n")
        configfile.write("skip_distance_group_plot = False\n")
        configfile.write("skip_plot_check = True\n")
        configfile.write("plot_to_remove = []\n")
        configfile.write("skip_position_time_plot = False\n")
        
    print("Config file created!")

# Initialize GUI window
root = tk.Tk()
root.geometry("800x700")
root.title("Video Processing Tool")

# Folder entry and browse button
folder_label = tk.Label(root, text="Enter Folder Path:")
folder_label.grid(row=0, column=0, sticky="w")  # Use grid and align left

folder_entry = tk.Entry(root)
folder_entry.grid(row=0, column=1, sticky="ew")  # Use grid and expand horizontally

# Browse and Open buttons (created immediately after folder entry)
browse_button = tk.Button(root, text="Browse", command=browse_folder)

browse_button.grid(row=0, column=2)  # Place Browse next to folder_entry

# Labels and entry fields for rows and columns (added)
num_rows_label = tk.Label(root, text="Number of Rows:")
num_rows_label.grid(row=1, column=0, sticky="w")

num_rows_entry = tk.Entry(root)
num_rows_entry.grid(row=1, column=1, sticky="w")

num_cols_label = tk.Label(root, text="Number of Columns:")
num_cols_label.grid(row=2, column=0, sticky="w")

num_cols_entry = tk.Entry(root)
num_cols_entry.grid(row=2, column=1, sticky="w")

num_vids_label = tk.Label(root, text="Number of Videos:")
num_vids_label.grid(row=3, column=0, sticky="w")

num_vids_entry = tk.Entry(root)
num_vids_entry.grid(row=3, column=1, sticky="w")

# Remaining buttons (Redefine and Crop videos!)
open_folder_button = tk.Button(root, text="Open", command=open_folder)
validate_button = tk.Button(root, text="Redefine", command=reset_points, state=tk.DISABLED)
reset_button = tk.Button(root, text="Crop videos!", command=validate_points, state=tk.DISABLED)

open_folder_button.grid(row=4, column=0, columnspan=4)
validate_button.grid(row=5, column=0, columnspan=4)  # Span across all columns
reset_button.grid(row=6, column=0, columnspan=4)  # Span across all columns

canvas = tk.Canvas(root, width=800, height=600)
canvas.grid(row=7, column=0, columnspan=4, sticky="nsew") # Span and expand

canvas.bind("<Button-1>", on_click)  # Bind click event to canvas

# Configure column weights to make folder_entry expand
root.columnconfigure(1, weight=1)

root.mainloop()