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



click_count, point1, point2 = 0, 0, 0
col_number, row_number, vid_number = 0, 0, 0
frame_image = None
folder_path, output_path = "", ""
ratio = 0

positions = []

cmd_path = "C:/Users/revel/Downloads/ffmpeg-6.1.1-full_build/bin/ffmpeg.exe"

def browse_folder():
    """Opens file dialog to select folder."""
    
    global folder_path
    
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_entry.delete(0, tk.END)  # Clear existing text
        folder_entry.insert(0, folder_path)

        
    return folder_path


def open_folder():
    """
    Opens the selected folder, displays the first frame, and updates button states.
    """
    global col_number, row_number, frame_image, vid_number, output_path, folder_path
    
    folder_path = folder_entry.get()
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    video_path = os.path.join(folder_path, os.listdir(folder_path)[0])

    output_path = f"{folder_path}_output/"
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
    print(ratio)
    print(width, height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    print(new_width, new_height)
    
    resized_image = image.resize((new_width, new_height))
    
    return resized_image





def create_buttons():
    """Creates the four buttons (already done in the main code)."""
    global validate_button, reset_button

    validate_button = tk.Button(root, text="Redefine", command=reset_points, state=tk.DISABLED)
    reset_button = tk.Button(root, text="Crop videos!", command=validate_points, state=tk.DISABLED)
    browse_button = tk.Button(root, text="Browse", command=browse_folder)
    open_folder_button = tk.Button(root, text="Open", command=open_folder)

    # Button placement (modify as needed)
    browse_button.pack()
    open_folder_button.pack()
    validate_button.pack()
    reset_button.pack()





def reset_points():
    """Functionality for the "Redefine" button (e.g., clear selection)."""
    global point1, point2
    point1, point2 = None, None
    canvas.delete("all")  # Clear canvas


def validate_points():
    
    """Insert cropping from video_clipper_V2.py here"""
    
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

               ##############################
               #                            #
               #    Program starts here     #
               #                            #
               ##############################

    # Loop through each video
    for video_number in range(first_video, last_video + 1): # Change depending on the videos you have to treat
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

    print("Videos cropped successfully!")
        
    
    
    
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

        
        

# Initialize GUI window
root = tk.Tk()
root.geometry("800x900")
root.title("Video Processing Tool")

# Folder entry and browse button
folder_label = tk.Label(root, text="Enter Folder Path:")
folder_label.pack()

folder_entry = tk.Entry(root)
folder_entry.pack()

# Labels and entry fields for rows and columns (added)
num_rows_label = tk.Label(root, text="Number of Rows:")
num_rows_label.pack()

num_rows_entry = tk.Entry(root)
num_rows_entry.pack()

num_cols_label = tk.Label(root, text="Number of Columns:")
num_cols_label.pack()

num_cols_entry = tk.Entry(root)
num_cols_entry.pack()

num_vids_label = tk.Label(root, text="Number of Videos:")
num_vids_label.pack()

num_vids_entry = tk.Entry(root)
num_vids_entry.pack()


create_buttons()

canvas = tk.Canvas(root, width=320, height=240)
canvas.pack(fill=tk.BOTH, expand=True)


canvas.bind("<Button-1>", on_click)  # Bind click event to canvas

root.mainloop()