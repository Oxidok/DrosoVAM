# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:42:28 2024

@author: revel

Version V9 created on Mon Oct 23 2024

"""

import os
import csv
import glob
import cv2
import random
import ast
import configparser
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


               ##############################
               #                            #
               #         Set values         #
               #                            #
               ##############################

expname = "DPA_Dmel_Mix_V1"

# Define folder paths
base_folder = f"C:/Users/maxim/Desktop/Video_work/vw_{expname}_output_V9/"
tracking_folder = os.path.join(base_folder, "tracking_V7/")
plot_folder = os.path.join(base_folder, "plots_V7/")
data_folder = os.path.join(base_folder, "data_V7/")
indiv_plot_folder = os.path.join(plot_folder, "individual_plots/")
position_folder = os.path.join(tracking_folder, "combined_tracking_data/")
all_pos_filename = f"{position_folder}{expname}.csv"
all_pos_norm_filename = f"{position_folder}{expname}_norm.csv"
all_pos_norm_bins_filename = f"{position_folder}{expname}_norm_bins.csv"
beam_crossings_filename = f"{position_folder}{expname}_beam.csv"
pos_bins_plotname = f"{plot_folder}y_position_plot.png"
fpi_plotname = f"{plot_folder}food_preference_index.png"
virtual_beam_plotname = f"{plot_folder}virtual_beam_30min.png"

# Read the config file
config = configparser.ConfigParser()
config.read(os.path.join(base_folder, 'config.ini')) 

zt12, zt0, ct12, ct0 = "ZT12", "ZT0", "CT12", "CT0"
tick_vals, tick_labs = [], []
rejected_list = []

row_number = int(config['CHAMBER_DESIGN']['row_number'])
col_number = int(config['CHAMBER_DESIGN']['col_number'])
first_video_number = int(config['CHAMBER_DESIGN']['first_video_number'])
last_video_number = int(config['CHAMBER_DESIGN']['last_video_number'])
minbins = [1, 10, 30] # number of minutes per bins 

top_food = ast.literal_eval(config['CHAMBER_DESIGN']['top_food'])
bot_food = ast.literal_eval(config['CHAMBER_DESIGN']['bottom_food'])
group_foods = [top_food, bot_food]

group1 = eval(config['FLIES_DETAILS']['group1'])
group2 = eval(config['FLIES_DETAILS']['group2'])
if len(group2) > 1:
    groups = [group1, group2]
    group_names = [ast.literal_eval(config['FLIES_DETAILS']['group1_name']), 
                   ast.literal_eval(config['FLIES_DETAILS']['group2_name'])]
    group_color = [ast.literal_eval(config['FLIES_DETAILS']['group1_color']), 
                   ast.literal_eval(config['FLIES_DETAILS']['group2_color'])]
else:
    groups = [group1]
    group_names = [ast.literal_eval(config['FLIES_DETAILS']['group1_name'])]
    group_color = [ast.literal_eval(config['FLIES_DETAILS']['group1_color'])]

# Use for thresholding, normalization and binning
threshold_frameCount = 1000
num_position_bins = int(config['EXPERIMENT']['position_bins'])
i = 0
dictio, dictionorm, dictiobins, dictiobeam = [], [], [], []

video_length = int(config['EXPERIMENT']['duration'])
delay = float(config['EXPERIMENT']['delay'])
light_stat = 1 # 1 if the first light switch turns it off.
fps = 5
fph = fps * 60 * 60
fpv = video_length * 60 * fps 
tot_frame = last_video_number * fpv

new_model = eval(config['EXPERIMENT']['new_model'])

classic_LD = eval(config['EXPERIMENT']['classic_ld'])
number_of_cycles_before_DD = int(config['EXPERIMENT']['cycles_before_dd'])

is_DPA = eval(config['EXPERIMENT']['is_dpa'])

# Access and interpret the values from the ANALYSIS section
fly_length = ast.literal_eval(config['ANALYSIS']['fly_length']) 
skip_tracking = eval(config['ANALYSIS']['skip_tracking'])
skip_concat = eval(config['ANALYSIS']['skip_concat'])
skip_indiv_analysis = eval(config['ANALYSIS']['skip_indiv_analysis'])
skip_threshold_check = eval(config['ANALYSIS']['skip_threshold_check'])
skip_data_curation = eval(config['ANALYSIS']['skip_data_curation'])
skip_offlimit_plots = eval(config['ANALYSIS']['skip_offlimit_plots'])
skip_hist = eval(config['ANALYSIS']['skip_hist'])
skip_indiv_pos_plot = eval(config['ANALYSIS']['skip_indiv_pos_plot'])
skip_indiv_distance_plot = eval(config['ANALYSIS']['skip_indiv_distance_plot'])
skip_virtual_beam = eval(config['ANALYSIS']['skip_virtual_beam'])
skip_beams_indiv_plots = eval(config['ANALYSIS']['skip_beams_indiv_plots'])
skip_group_virtual_beam = eval(config['ANALYSIS']['skip_group_virtual_beam'])
skip_position_plot = eval(config['ANALYSIS']['skip_position_plot'])
skip_FPI_plot = eval(config['ANALYSIS']['skip_FPI_plot'])
skip_distance_group_plot = eval(config['ANALYSIS']['skip_distance_group_plot'])
skip_plot_check = eval(config['ANALYSIS']['skip_plot_check'])
plot_to_remove = ast.literal_eval(config['ANALYSIS']['plot_to_remove'])
skip_position_time_plot = eval(config['ANALYSIS']['skip_position_time_plot'])


default_settings = False # if 1, all settings will be reset to default before running

               ##############################
               #                            #
               #     Functions are here     #
               #                            #
               ##############################


def concatenate_data(chamber_y, chamber_z):

    """
    Concatenates data from all tracking files for a given chamber.

    Args:
        chamber_y (int): The chamber row number (Y-axis).
        chamber_z (int): The chamber column number (Z-axis).
    """

    specimen_id = f"{chamber_y}_{chamber_z}"
    df_combined = pd.DataFrame()
    output_filename = f"specimen_{specimen_id}.csv"

    for x in range(first_video_number, (last_video_number + 1)):
        filename = f"video_{x}_{specimen_id}DLC_tracking.csv"
        filepath = os.path.join(tracking_folder, filename)

        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df = df.fillna(0)
            df_combined = pd.concat([df_combined, df])

    df_combined.to_csv(os.path.join(position_folder, output_filename), index=False)
    print(f"Number of data points for specimen{specimen_id}: {len(df_combined)}")




def replace_off_limits(specimen_id, series, min_val, max_val):
    """
    Replaces off-limit values in a series with the average of the nearest in-limit values.
    
    Args:
      series: A pandas Series containing the data.
      min_val: The minimum acceptable value.
      max_val: The maximum acceptable value.
    
    Returns:
      A new pandas Series with off-limit values replaced.
    """
    
    # Create a boolean mask indicating in-limit values
    in_limits_mask = (series >= min_val) & (series <= max_val)
    
    # Find indices of off-limit values
    off_limit_indices = series[~in_limits_mask].index
    
    cnter = 0
    ttl = len(off_limit_indices)
    looper = 0
    
    # Replace off-limit values
    for idx in off_limit_indices:
                
        # Find the nearest in-limit values before and after the off-limit value
        try:
            prev_in_limit_idx = series.loc[:idx][in_limits_mask].index[-1]
        except IndexError:
            prev_in_limit_idx = None
        try:
            next_in_limit_idx = series.loc[idx:][in_limits_mask].index[0]
        except IndexError:
            next_in_limit_idx = None


        if prev_in_limit_idx is None:
            replacement_value = series.loc[next_in_limit_idx]
        elif next_in_limit_idx is None:
            replacement_value = series.loc[prev_in_limit_idx]
        else: # Calculate the average of the nearest in-limit values
            replacement_value = (series.loc[prev_in_limit_idx] + series.loc[next_in_limit_idx]) / 2

    
        # Replace the off-limit value
        series.loc[idx] = replacement_value
        
        cnter += 1
        lfter = ttl - cnter
        if looper % 100 == 0 or lfter == 0:
            print(f"Species: {specimen_id}, Replacing: {cnter}, left to do: {lfter}")
        looper += 1

    print("Done replacing values")
    return series


def activity_plot_maker(minbins, distances, chamber_y, chamber_z, group1):
    
    for bins in minbins:
        
        prodist_xmin, start_days, end_days, end_nights = [], [], [], []
        
        binframe = bins * 60 * 5
        
        # Create x-minute data file path
        data_file_xmin = f"chamber_{chamber_y}_{chamber_z}_distance_{bins}min.csv"
        data_filepath_xmin = os.path.join(data_folder, data_file_xmin)  
        
        # Process x-minute distances
        for i in range(0, len(distances), binframe):
            segment = distances[i:i + binframe]
            prodist_xmin.append(sum(segment))

        try:
            # Write x-minute data to CSV
            with open(data_filepath_xmin, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Frame Range", "Average Distance (pixels)"])  # Add header row
                for i, distance in enumerate(prodist_xmin):
                    frame_range = f"{(i * binframe) + 1} - {(i + 1) * binframe}"  # Calculate frame range
                    writer.writerow([frame_range, distance])
        except OSError as e:
            print(f"Error writing data to CSV files: {e}")
        
        
        if chamber_y in group1:
            linecolor = group_color[0]
            linename = group_names[0]
        else:
            linecolor = group_color[1]
            linename = group_names[1]
            
        # Generate plot
        plt.figure(figsize=(20, 5))

        # Plot distance per 10 minutes
        plt.plot(range(0, len(prodist_xmin) * binframe, binframe), prodist_xmin, color=linecolor, label=f"Distance per {bins} minutes")
        
        plt.xticks(tick_vals, tick_labs)  # Set the ticks and labels
        
        for i in range(0, len(tick_vals), 2):
            start_days.append(tick_vals[i])  
        for i in range(1, len(tick_vals), 2):
            end_days.append(tick_vals[i])
        end_days.append(len(prodist_xmin) * binframe)
        for i in range(2, len(tick_vals), 2):
            end_nights.append(tick_vals[i])
            
            
        if classic_LD is True:
            for start, end in zip(start_days, end_days):
              plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
            for start, end in zip(end_days, end_nights):
              plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
        else:
            LD_cycles = number_of_cycles_before_DD
            for i, (start, end) in enumerate(zip(start_days, end_days)):
                if i < LD_cycles + 1:
                    plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                else:
                    plt.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
            for start, end in zip(end_days, end_nights):
              plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
            
        # Label and title the plot
        plt.xlabel("Time (ZT)")
        plt.ylabel("Distance (pixels)")
        plt.title(f"Distance Covered in Chamber {chamber_y}-{chamber_z} - {linename}")
        plt.legend(loc='upper right')
        plot_filename = f"{indiv_plot_folder}chamber_{chamber_y}_{chamber_z}_distance_{bins}min_plot.png"
        plt.savefig(plot_filename)
        plt.close()
        
    print(f"Plots saved for chamber {chamber_y}_{chamber_z}")


# The reset setting thing

if default_settings is True:
    
    settings = [skip_tracking, skip_concat, skip_threshold_check, 
    skip_data_curation, skip_offlimit_plots, skip_indiv_analysis, 
    skip_indiv_distance_plot, skip_hist, skip_indiv_pos_plot, 
    skip_virtual_beam, skip_beams_indiv_plots, skip_group_virtual_beam, 
    skip_position_plot, skip_FPI_plot, skip_distance_group_plot, skip_plot_check]
    param = [False, False, True, False, True, False, False, False,
             False, False, True, False, False, False, False, False]
    
    fly_length = None
    for setting, value in zip(settings, param):
        globals()[setting] = value 
    plot_to_remove = []
    print("Using default settings")



               ##############################
               #                            #
               #    Program starts here     #
               #                            #
               ##############################

# Create necessary folders if they don't exist
folders = [tracking_folder, plot_folder, data_folder, position_folder, indiv_plot_folder]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Ensure group_names and group_color are not longer than groups
group_names = group_names[:len(groups)]
group_color = group_color[:len(groups)]

# Defines the tick and tick labels for LD or DD
if classic_LD is True:
    
    tick_vals.append(i)
    i = i + (delay * fph)
    tick_labs.append("")
    
    while i < tot_frame:
        tick_vals.append(i)
        i += (12 * fph) # 12h * 60 min * 60 sec * 5 fps
        if light_stat == 1:
            tick_labs.append(zt12) 
            light_stat = 0
        else:
            tick_labs.append(zt0)
            light_stat = 1
    tick_vals.append(tot_frame)
    tick_labs.append("")

else:
    tick_vals.append(i)
    i = i + (delay * fph)
    tick_labs.append("")
    last_label = 12
    LD_cycles = number_of_cycles_before_DD
    while i < tot_frame:
        tick_vals.append(i)
        i += (12 * fph) # 12h * 60 min * 60 sec * 5 fps
        if light_stat == 1:
            tick_labs.append(zt12) 
            light_stat = 0            
        else:
            if LD_cycles > 0:
                tick_labs.append(zt0)
                light_stat = 1
                LD_cycles -= 1
            else:
                if last_label == 12:
                    tick_labs.append(ct0)
                    last_label = 0
                else:
                    tick_labs.append(ct12)
                    last_label = 12
    tick_vals.append(tot_frame)
    tick_labs.append("")

print("Ticks and labels defined.") 
 
    
# Loop to define fly length    
if fly_length is None:
    
    fly_lengths = {}
    
    print("Defining fly length...")
    
    for filename in os.listdir(base_folder):
        if filename.endswith(".csv"):
            video_info = filename.split("_")[0:4]
            video_name = "_".join(video_info)
            total_distance = 0
            row_count = 0
            
            with open(os.path.join(base_folder, filename), "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header rows
                next(reader)
                next(reader)
    
                for row in reader:
                    if new_model is True:
                        _, head_x, head_y, _, _, _, _, tail_x, tail_y, _ = row
                    else:
                        _, head_x, head_y, _, tail_x, tail_y, _ = row
                    head_x, head_y, tail_x, tail_y = map(float, [head_x, head_y, tail_x, tail_y])
                    distance = ((head_x - tail_x) ** 2 + (head_y - tail_y) ** 2) ** 0.5
                    total_distance += distance
                    row_count += 1
            fly_lengths[video_name] = total_distance / row_count
    
    total_fly_length = sum(fly_lengths.values())
    fly_length = total_fly_length / len(fly_lengths)
    
    print(f"Average fly size: {fly_length} pixels")
        




# Loop to correct head/tail absurd movements and define fly center

if skip_tracking is False:
    
    print("Going through tracking data...")
    
    for filename in os.listdir(base_folder):
        if filename.endswith(".csv"):
            video_info = filename.split("_")[0:4]
            video_name = "_".join(video_info)
            tracking_filename = f"{video_name}_tracking.csv"
    
            with open(os.path.join(base_folder, filename), "r") as csvfile, \
                    open(os.path.join(tracking_folder, tracking_filename), "w", newline="") as output_file:
    
                reader = csv.reader(csvfile)
                writer = csv.writer(output_file)
                writer.writerow(["Frame", "Center_X", "Center_Y"])
    
                prev_head, prev_center, prev_tail = None, None, None
                next(reader)  # Skip header rows
                next(reader)  # Skip header rows
                next(reader)  # Skip header rows
    
                # Control for random head/tail movements
                for row in reader:
                    if new_model is True:
                        frame_number, head_x, head_y, _, _, _, _, tail_x, tail_y, _ = row
                    else:
                        frame_number, head_x, head_y, _, tail_x, tail_y, _ = row
                        
                    head_x, head_y, tail_x, tail_y = map(float, [head_x, head_y, tail_x, tail_y])
    
                    if prev_head is not None:
                        head_distance = ((head_x - prev_head[0]) ** 2 + (head_y - prev_head[1]) ** 2) ** 0.5
                        tail_distance = ((tail_x - prev_tail[0]) ** 2 + (tail_y - prev_tail[1]) ** 2) ** 0.5
    
                        if head_distance > (fly_length * 2) and tail_distance <= (fly_length * 2):
                            head_x = (prev_head[0] + tail_x) / 2
                            head_y = (prev_head[1] + tail_y) / 2
                        elif tail_distance > (fly_length * 2) and head_distance <= (fly_length * 2):
                            tail_x = (prev_tail[0] + head_x) / 2
                            tail_y = (prev_tail[1] + head_y) / 2
    
                    # Define center of the fly
                    center_x = (head_x + tail_x) / 2
                    center_y = (head_y + tail_y) / 2                
    
                    writer.writerow([frame_number, center_x, center_y])
    
                    prev_center, prev_head, prev_tail = (center_x, center_y), (head_x, head_y), (tail_x, tail_y) 
    
    print("Data filtered (heads and tails extracted from original csv file), ready for concatenation.")


if skip_concat is False:

    # Loop through all chambers to concatenate
    for chamber_y in range(1, (row_number + 1)): # Number of rows
        for chamber_z in range(1, col_number + 1): # Number of columns
    
            # Concatenate data for the current chamber
            concatenate_data(chamber_y, chamber_z)
            print(f"Data concatenated for chamber {chamber_y}_{chamber_z}")
            
    print("Data for all chambers concatenated.")        
   

if skip_indiv_analysis is False:
    
    # Loop through all chambers to filter, normalize and bin
    for chamber_y in range(1, (row_number + 1)): # Number of rows
        for chamber_z in range(1, col_number + 1): # Number of columns
    
            specimen_id = f"_{chamber_y}_{chamber_z}"
            video_name = f"{base_folder}video_{last_video_number}{specimen_id}.mp4" # Uses the last video to define threshold
            file_name = f"{position_folder}specimen{specimen_id}.csv"
            
            print(f"Curation of chamber {specimen_id} started")
            
            # Read the video and create x and y limits
            cap = cv2.VideoCapture(video_name)
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            buf = np.empty((threshold_frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
            fc = 0
            ret = True
            while (fc < threshold_frameCount and ret):
                ret, buf[fc] = cap.read()
                fc += 1
            ret, first_frame = cap.read()
            cap.release()
            
            avg = np.sum(buf, axis= 0)
            superavg = np.sum(avg, axis= 2)        
            threshold = np.mean(superavg) * 0.99 # Choose a threshold value that separates chamber and background
            binary_image = np.where(superavg > threshold, 255, 0).astype(np.uint8)        
            non_zero_indices = np.where(binary_image != 0)
                    
            # Define x and y limits
            min_x = (np.min(non_zero_indices[1]) - 20)
            max_x = (np.max(non_zero_indices[1]) + 20)
            min_y = (np.min(non_zero_indices[0]) - 5)
            max_y = (np.max(non_zero_indices[0]) + 5)
            ratio = (max_y - min_y) / (max_x - min_x)
    
            
            # Add a threshold confirmation step for the first iteration
            if skip_threshold_check is False:     
                # Display the first frame and binary image side-by-side
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(first_frame)
                ax2.imshow(binary_image)
                # Add horizontal lines at min_y and max_y
                ax1.hlines(y=[min_y, max_y], xmin=min_x, xmax=max_x, color="red")
                ax2.hlines(y=[min_y, max_y], xmin=min_x, xmax=max_x, color="red")
                ax1.set_title("First Frame")
                ax2.set_title("Binary Image")
                plt.suptitle(f"Threshold Verification (specimen {specimen_id})")
                plt.show()
                # Get user input
                key = input("Is the threshold appropriate (y/n)? ")
                if key.lower() != "y":
                    print("Script stopped by user.")
                    exit()
                skip_threshold_check = True        
                
                
            # Read the csv file and correct values that are off limits
            data = pd.read_csv(file_name)
            
            # Create a new filename for the output CSV
            filtered_filename = f"{position_folder}specimen{specimen_id}_filtered.csv"
            
            if skip_data_curation is False:
            
                fly_data = data.iloc[:, 1:]
                selected_columns = fly_data[["Center_X", "Center_Y"]] # Use column names in square brackets
                filtered_data = selected_columns.copy()
                filtered_data['Center_X'] = replace_off_limits(specimen_id, filtered_data['Center_X'], min_x, max_x)
                filtered_data['Center_Y'] = replace_off_limits(specimen_id, filtered_data['Center_Y'], min_y, max_y)
        
                if skip_offlimit_plots is False:
                    # Filter the selected_columns DataFrame to only include off-limits points
                    off_limits_points = selected_columns[
                        ~((filtered_data["Center_X"] >= min_x) & (filtered_data["Center_X"] <= max_x) &
                          (filtered_data["Center_Y"] >= min_y) & (filtered_data["Center_Y"] <= max_y))]
                    # Plot the off-limits points
                    plt.figure()
                    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
                    plt.scatter(off_limits_points["Center_X"], off_limits_points["Center_Y"], c='red', alpha=0.5, s=1)  # Red color for off-limits points
                    # Add horizontal lines at min_y and max_y
                    plt.hlines(y=[min_y, max_y], xmin=min_x, xmax=max_x, color="green")
                    plt.hlines(y=[min_y, max_y], xmin=min_x, xmax=max_x, color="green")
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.title(f"Off-Limits Points for {specimen_id}")
                    plt.show()
        
        
                # Convert the relevant columns to NumPy arrays
                center_x = filtered_data['Center_X'].values
                center_y = filtered_data['Center_Y'].values
                
                # Calculate the differences using NumPy's array slicing and vectorized operations
                x_diff = center_x[1:] - center_x[:-1] 
                y_diff = center_y[1:] - center_y[:-1]
                
                # Calculate the distances and prepend a 0 for the first row 
                distances = np.sqrt(x_diff**2 + y_diff**2)
                distances = np.insert(distances, 0, 0)  
                
                # Assign the calculated distances back to the DataFrame
                filtered_data['Distance'] = distances         
                
                
                # Fill the first row's distance with 0 (since there's no previous frame)
                filtered_data['Distance'] = filtered_data['Distance'].fillna(0)
                
                # Save the filtered DataFrame with the new column to a new CSV file
                filtered_data.to_csv(filtered_filename, index=False)
        
                print(f"Curated file created for specimen{specimen_id}")

            
            if skip_data_curation is True:
                
                filtered_data = pd.read_csv(filtered_filename)
            
            
            
            if skip_hist is False:
    
                # Plot the heatmap
                h, x, y, im = plt.hist2d(filtered_data["Center_X"], filtered_data["Center_Y"], bins=[30, int(30*ratio)], range=[[min_x, max_x], [min_y, max_y]])
                
                plt.figure()
                plt.imshow(h)
                # Label the axes
                plt.xlabel("Y")
                plt.ylabel("X")
                # Colorbar for intensity
                plt.colorbar(label="Frequency")
                # Title the plot
                plt.title("Fly Position Heatmap")
                plt.savefig(f"{indiv_plot_folder}specimen{specimen_id}_heatmap.png")
                plt.close()    
            
            if skip_indiv_pos_plot is False:
                
                # Plot y-position frequency
                fly_y = filtered_data["Center_Y"]
                average_y = np.mean(fly_y)
                median_y = np.median(fly_y)
                
                for i, group in enumerate(groups):
                    if chamber_y in group:
                        plotcolor = group_color[i]
                        plotname = group_names[i]



                plt.figure()
                
                plt.hist(fly_y, bins=200, color=plotcolor)  # Adjust bins as needed
                
                data_min, data_max = np.min(fly_y), np.max(fly_y)
                data_med = data_min + ((data_max - data_min)/2)
                
                xtick_values = [data_min, data_med, data_max]
                plt.xticks(xtick_values, [group_foods[0], "", group_foods[1]])
                
                # Label the axes (swapped due to extent order)
                plt.axvline(x=average_y, color='red', linestyle='dashed', linewidth=2, label='Average Y')
                plt.axvline(x=median_y, color='forestgreen', linestyle='dashed', linewidth=2, label='Median Y')
                plt.xlabel("Food preference")
                plt.ylabel("Frequency")
                plt.legend()
                plt.title(f"Fly Y-Position frequency - {plotname}")
                plt.savefig(f"{indiv_plot_folder}specimen{specimen_id}_histo_x.png")
                plt.close()
            
            
            
            data = pd.read_csv(filtered_filename)
            fly_data = data.iloc[:, 1:]
            distances = fly_data["Distance"]
            
            if skip_indiv_distance_plot is False:
                
                activity_plot_maker(minbins, distances, chamber_y, chamber_z, group1)
            
            
            # Create a dictionary with all the Y-coordinates
            
            fly = filtered_data["Center_Y"].tolist()
            dictio.append(pd.DataFrame({specimen_id:fly}))
            
            # Create a dictionary with all the Y-coordinates normalized
            
            scaler = (fly - min_y) / (max_y - min_y)
            fly_norm = scaler * 2 - 1
            
            dictionorm.append(pd.DataFrame({specimen_id:fly_norm}))
    
            # Create a dictionary with all the Y-coordinates normalized and binned
            
            bin_counts, bin_edges = np.histogram(fly_norm, bins=num_position_bins)
            
            dictiobins.append(pd.DataFrame({specimen_id:bin_counts}))
            
            
            
            if skip_virtual_beam is False:
                
                specimen_id = f"{chamber_y}_{chamber_z}" #/// Warning, the virtual beam names are y_z, not _y_z
                frame_delay = int(delay * fph)
                fly_norm_temp = fly_norm[frame_delay:]
                frame_left = len(fly_norm_temp) % (fps * 60 * 30)
                fly_beam = fly_norm_temp[:-frame_left]
            
                # Initialize a list to store beam crossings per minute
                beam_crossings_per_minute = []
                
                # Calculate the number of frames per minute
                frames_per_minute = fps * 60
                
                # Iterate through the fly_beam list in chunks corresponding to one minute
                for i in range(0, len(fly_beam), frames_per_minute):
                    minute_data = fly_beam[i:i + frames_per_minute] 
                    crossings = 0
                
                    # Check for crossings within this minute's data
                    for j in range(1, len(minute_data)):
                        if minute_data[j - 1] > 0 and minute_data[j] < 0:  # Positive to negative transition
                            crossings += 1
                
                    beam_crossings_per_minute.append(crossings)
                
                beam_crossings_per_30minutes = [sum(beam_crossings_per_minute[i:i+30]) for i in range(0, len(beam_crossings_per_minute), 30)] 
                
                if skip_beams_indiv_plots is False:

                    for i, group in enumerate(groups):
                        if chamber_y in group:
                            plotcolor = group_color[i]
                            plotname = group_names[i]

                    # Plotting
                    plt.figure(figsize=(20, 5))  # Adjust figure size if needed
                    
                    # Plot beam crossings per 30 minutes
                    plt.plot(beam_crossings_per_30minutes, marker='o', linestyle='-', color=plotcolor)
                    plt.title(f"Beam Crossings Per 30 Minutes - Specimen {specimen_id}")
                    plt.xlabel('30-Minute Interval')
                    plt.ylabel('Number of Crossings')
                    plt.savefig(f"{indiv_plot_folder}specimen{specimen_id}_virtual_beam.png")
                    plt.close()
                
                dictiobeam.append(pd.DataFrame({specimen_id:beam_crossings_per_30minutes}))                 

    print("Saving Y-position dictionaries...")    
    pd.concat(dictio, axis=1).to_csv(all_pos_filename, index = False)
    
    pd.concat(dictionorm, axis=1).to_csv(all_pos_norm_filename, index=False)
    
    pd.concat(dictiobins, axis=1).to_csv(all_pos_norm_bins_filename, index=False)
    
    if skip_virtual_beam is False:
        pd.concat(dictiobeam, axis=1).to_csv(beam_crossings_filename, index=False)

    print("Y-position dictionaries saved!")



if skip_position_plot is False:

    # Generate the binned position boxplot
    df = np.genfromtxt(all_pos_norm_bins_filename, delimiter=",", skip_header=1, 
                         usecols=range(0, col_number * row_number - 1), filling_values=0)
    
    plt.figure(figsize=(12, 7))
    
    positions = []
    for i in range(num_position_bins):
        for j, group in enumerate(groups):
            start_idx = sum(len(g) for g in groups[:j]) * col_number
            end_idx = start_idx + len(group) * col_number
            positions.append(df[i, start_idx:end_idx])

    bp1 = plt.boxplot(positions,
                      patch_artist=True,
                      whis=[0, 100],
                      positions=range(len(positions)),
                      widths=0.9)

    # Color boxes based on group
    color_plot_list = [color for color in group_color for _ in range(num_position_bins)]
    for patch, color in zip(bp1["boxes"], color_plot_list):
        patch.set_facecolor(color)

    plt.legend(bp1["boxes"], group_names)

    # Set x-ticks, with labels based on group_foods
    xtick_values = [0, len(positions) // 2 - 0.5, len(positions) - 1]
    xtick_labels = [group_foods[0], "", group_foods[1]]
    plt.xticks(xtick_values, xtick_labels)

    plt.ylabel("Frequency")
    plt.title("Fly Y-Position frequency")
    plt.savefig(pos_bins_plotname)
    plt.close()



if skip_FPI_plot is False:
    
    # Generate the FPI scatterplot
    df = np.genfromtxt(all_pos_norm_bins_filename, delimiter=",", skip_header=1, 
                         usecols=range(0, col_number * row_number - 1), filling_values=0)

    FPI_total = []
    for j, group in enumerate(groups):
        start_idx = sum(len(g) for g in groups[:j]) * col_number
        end_idx = start_idx + len(group) * col_number
        TopFoodTime = df[0, start_idx:end_idx]
        BotFoodTime = df[num_position_bins-1, start_idx:end_idx]
        FPI = (BotFoodTime - TopFoodTime) / (BotFoodTime + TopFoodTime)
        FPI_total.append(FPI)

    fig, ax = plt.subplots(figsize=(3, 4))

    # Define the range of random noise (e.g., +/- 0.02)
    noise_range = 0.03

    # Scatter plot for each group
    for j, group in enumerate(groups):
        x = [j + 1 + random.uniform(-noise_range, noise_range) for _ in FPI_total[j]]
        ax.scatter(x, FPI_total[j], marker='.', color=group_color[j])

    # Calculate and plot average FPI for each group
    x_avg = []
    avglist = []
    for j, group in enumerate(groups):
        FPIavg = sum(FPI_total[j]) / len(FPI_total[j])
        x_avg.extend([j + 0.95, j + 1, j + 1.05])  # Three x-values for the average line
        avglist.extend([FPIavg, FPIavg, FPIavg])
    ax.scatter(x_avg, avglist, marker='_', color="green")

    ax.set_xlim(0.5, len(groups) + 0.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks(range(1, len(groups) + 1), group_names)
    ax.set_yticks([-1, 0, 1], [group_foods[0], "", group_foods[1]])
    ax.set_ylabel("Food Preference Index")
    fig.tight_layout()
    fig.savefig(fpi_plotname)
    plt.close()




if skip_distance_group_plot is False:
    
    user_input = 0
    if len(plot_to_remove) != 0:
        
        rejected_list = plot_to_remove
        
    elif skip_plot_check is False:
            
        # Stop and wait for user input before generating average plots
        input_text = """Look at individual plots to identify unsatisfying flies.
        Enter coordinates of chambers to reject (eg: 1_8, 2_5, 4_1), and/or press Enter to skip:"""
        
        user_input = input(input_text)
        
        
    # Process user input
    if user_input:
        
        rejected_list = []
        
        # Split the input string into a list, handling commas and spaces
        rejected_input = user_input.split(",")  # Split by comma first
        
        # Further process each item to remove spaces (if needed)
        for item in rejected_input:
            rejected_list.append(item.strip())  # Remove leading/trailing spaces
    
        for value in rejected_list:
            for filename in glob.glob(f"{base_folder}data/chamber_{value}_*min.csv"):    
                new_filename = os.path.join(base_folder, f"data/rejected_{os.path.basename(filename)}")
                os.rename(filename, new_filename)
                
    # Average plots per group
    for i, bins in enumerate(minbins):
        for j, group in enumerate(groups):
        
            # Define the path to your CSV files
            input_files = f"{data_folder}chamber_{group}_*_{bins}min.csv"
            output_file = f"{data_folder}{bins}min_{group_names[j]}_combined.csv"
            
            # Get all CSV files in the path
            csv_files = glob.glob(input_files)
            
            # Read the first CSV file to get the base columns
            df = pd.read_csv(csv_files[0])
            
            # Iterate through remaining files, adding data as new columns
            for csv_file in csv_files[1:]:
                # Read the CSV file
                data = pd.read_csv(csv_file)
                # Get the filename without the extension as the new column name
                new_column_name = csv_file.split(".")[0]
                # Add data from the file as a new column with the filename as the column name
                df[new_column_name] = data.iloc[:, 1]  # Assuming the second column has the data
            
            coi_num = len(csv_files) + 1 # Number of columns of interest
    
            average_col = np.mean(df.iloc[:, 1:coi_num], axis = 1) # Ignore the first column, and take the next 16
            
            df["AVG"] = average_col 
            
            # Save the combined DataFrame to a new CSV file
            df.to_csv(output_file, index=False)
            
            start_days, end_days, end_nights = [], [], []
            binframe = bins * 60 * 5
            
            if group == group1:
                linecolor = group_color[0]
                linename = group_names[0]
            else:
                linecolor = group_color[1]
                linename = group_names[1]
            
            # Generate plot
            plt.figure(figsize=(20, 5))
    
            # Plot distance per 10 minutes
            plt.plot(range(0, len(average_col) * binframe, binframe), average_col, color=linecolor, label=f"Distance per {bins} minutes")
            
            plt.xticks(tick_vals, tick_labs)  # Set the ticks and labels
            
            
            for i in range(0, len(tick_vals), 2):
                start_days.append(tick_vals[i])  
            for i in range(1, len(tick_vals), 2):
                end_days.append(tick_vals[i])
            end_days.append(len(average_col) * binframe)
            for i in range(2, len(tick_vals), 2):
                end_nights.append(tick_vals[i])
            
            if classic_LD is True:                
                for start, end in zip(start_days, end_days):
                  plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                for start, end in zip(end_days, end_nights):
                  plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)              
            else:  
                LD_cycles = number_of_cycles_before_DD
                for i, (start, end) in enumerate(zip(start_days, end_days)):
                    if i < LD_cycles + 1:
                        plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                    else:
                        plt.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
                for start, end in zip(end_days, end_nights):
                  plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
            
                
            # Label and title the plot
            plt.xlabel("Time (ZT)")
            plt.ylabel("Distance (pixels)")
            plt.title(f"Average locomotor activity - {linename}")
            plt.legend(loc='upper right')
            plot_filename = f"{plot_folder}{group_names[j]}_average_plot_{bins}.png"
            plt.savefig(plot_filename)
            plt.close()
        
        
        # Generate plot
        plt.figure(figsize=(20, 5))
        
        for g, group in enumerate(groups):
            
            # Define the path to your CSV files
            input_files = f"{data_folder}chamber_{group}_*_{bins}min.csv"
            output_file = f"{data_folder}{bins}min_{group_names[g]}_combined.csv"
            
            # Get all CSV files in the path
            csv_files = glob.glob(input_files)
            
            # Read the first CSV file to get the base columns
            df = pd.read_csv(csv_files[0])
            
            # Iterate through remaining files, adding data as new columns
            for csv_file in csv_files[1:]:
                # Read the CSV file
                data = pd.read_csv(csv_file)
                # Get the filename without the extension as the new column name
                new_column_name = csv_file.split(".")[0]
                # Add data from the file as a new column with the filename as the column name
                df[new_column_name] = data.iloc[:, 1]  # Assuming the second column has the data
            
            coi_num = len(csv_files) + 1 # Number of columns of interest
    
            average_col = np.mean(df.iloc[:, 1:coi_num], axis = 1) # Ignore the first column, and take the next 16
            
            df["AVG"] = average_col 
            
            # Save the combined DataFrame to a new CSV file
            df.to_csv(output_file, index=False)
            
            start_days, end_days, end_nights = [], [], []
            binframe = bins * 60 * 5
            
            if group == group1:
                linecolor = group_color[0]
                linename = group_names[0]
            else:
                linecolor = group_color[1]
                linename = group_names[1]
    
            plt.plot(range(0, len(average_col) * binframe, binframe), average_col, color=linecolor, label=f"{linename}")
    
    
        plt.xticks(tick_vals, tick_labs)  # Set the ticks and labels
        

        for i in range(0, len(tick_vals), 2):
            start_days.append(tick_vals[i])  
        for i in range(1, len(tick_vals), 2):
            end_days.append(tick_vals[i])
        end_days.append(len(average_col) * binframe)
        for i in range(2, len(tick_vals), 2):
            end_nights.append(tick_vals[i])

        if classic_LD is True:                
            for start, end in zip(start_days, end_days):
              plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
            for start, end in zip(end_days, end_nights):
              plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)              
        else:  
            LD_cycles = number_of_cycles_before_DD
            for i, (start, end) in enumerate(zip(start_days, end_days)):
                if i < LD_cycles + 1:
                    plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                else:
                    plt.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
            for start, end in zip(end_days, end_nights):
              plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
            
        # Label and title the plot
        plt.xlabel("Time (ZT)")
        plt.ylabel("Distance (pixels)")
        plt.title(f"Average locomotor activity - {bins} minute bins")
        plt.legend(loc='upper right')
        plot_filename = f"{plot_folder}combined_average_plot_{bins}.png"
        plt.savefig(plot_filename)
        plt.close()
    
    
    
    
if skip_virtual_beam is False:
    if skip_group_virtual_beam is False:
    
        plt.figure(figsize=(20, 5))
        
        # Read the CSV file
        df = pd.read_csv(beam_crossings_filename, header=0)
    
            
        if len(plot_to_remove) != 0 or skip_plot_check is False:
            if len(plot_to_remove) != 0:
                rejected_list = plot_to_remove
            if len(rejected_list) != 0:
                df = df.drop(columns=[col for col in df.columns if col in rejected_list])
            
        # Get the remaining column names for each group using the specific pattern
        group1_cols = [col for col in df.columns if any(f"{str(g)}_" in col for g in group1)]
        group2_cols = [col for col in df.columns if any(f"{str(g)}_" in col for g in group2)]
        
        # Filter data for each group based on the updated column names
        group1_data = df[group1_cols]
        group2_data = df[group2_cols]

        
        # Calculate the mean for each group
        group1_median = group1_data.median(axis=1)
        group2_median = group2_data.median(axis=1)
        
        # Plotting (remains the same as before)
        plt.plot(group1_median, label=group_names[0], color=group_color[0])
        plt.plot(group2_median, label=group_names[1], color=group_color[1])
        plt.xlabel("Time (ZT)")  # Adjust label based on your data
        plt.ylabel("Number of Crossings")
        plt.title("Virtual Laser Crossings - Median - 30 minutes bins")
        
        
        beam_tick_vals, beam_tick_labs = [], []
        ld = 1

            
        for s in range(0, len(group1_median), 24):
            beam_tick_vals.append(s)
        beam_tick_vals.append(len(group1_median))
        
        for t in range(0, len(group1_median), 24):
            if ld == 1:
                beam_tick_labs.append("ZT12")
                ld = 0
            elif ld == 0:
                beam_tick_labs.append("ZT0")
                ld = 1                
        beam_tick_labs.append("")        

        plt.xticks(beam_tick_vals, beam_tick_labs)  # Set the ticks and labels
        
        beam_start_days, beam_end_days, beam_end_nights = [], [], []
        
            
        for i in range(0, len(beam_tick_vals), 2):
            beam_end_days.append(beam_tick_vals[i])  
        for i in range(1, len(beam_tick_vals)-1, 2):
            beam_start_days.append(beam_tick_vals[i])
        for i in range(2, len(beam_tick_vals), 2):
            beam_end_nights.append(beam_tick_vals[i])
        beam_start_days.append(beam_tick_vals[len(beam_tick_vals)-1])
        
        if classic_LD is True:                
            for start, end in zip(beam_start_days, beam_end_nights):
              plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
            for start, end in zip(beam_end_days, beam_start_days):
              plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)              
        else:  
            LD_cycles = number_of_cycles_before_DD
            for i, (start, end) in enumerate(zip(beam_start_days, beam_end_nights)):
                if i < LD_cycles:
                    plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                else:
                    plt.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
            for start, end in zip(beam_end_days, beam_start_days):
              plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)     
        
        plt.legend(loc='upper right')
        plt.savefig(virtual_beam_plotname)
        plt.close()
    
user_input = 0
if len(plot_to_remove) != 0:
    
    rejected_list = plot_to_remove
    
elif skip_plot_check is False:
        
    # Stop and wait for user input before generating average plots
    input_text = """Look at individual plots to identify unsatisfying flies.
    Enter coordinates of chambers to reject (eg: 1_8, 2_5, 4_1), and/or press Enter to skip:"""
    
    user_input = input(input_text)
    
    
# Process user input
if user_input:
    # Split the input string into a list, handling commas and spaces
    rejected_input = user_input.split(",")  # Split by comma first
    # Further process each item to remove spaces (if needed)
    rejected_list = []
    for item in rejected_input:
        rejected_list.append(item.strip())  # Remove leading/trailing spaces

    for value in rejected_list:
        
        for filename in glob.glob(f"{base_folder}data/chamber_{value}_*min.csv"):
            
            new_filename = os.path.join(base_folder, f"data/rejected_{os.path.basename(filename)}")
            os.rename(filename, new_filename)
            


# Average y_position/time plots per group with individual fly lines

if skip_position_time_plot is False:

    df = pd.read_csv(all_pos_norm_filename)
    group1_columns = [col for col in df.columns if int(col.split('_')[1]) in group1]
    group2_columns = [col for col in df.columns if int(col.split('_')[1]) in group2]

    # Create DataFrames for each group
    df_group1 = df[group1_columns]
    df_group2 = df[group2_columns]

    average_pos_gr1 = np.mean(df_group1, axis=1).tolist()
    average_pos_gr2 = np.mean(df_group2, axis=1).tolist()

    average_pos_grs = [average_pos_gr1, average_pos_gr2]

    
    for i, bins in enumerate(minbins):
        
        start_days, end_days, end_nights = [], [], []
        
        # Generate plot
        plt.figure(figsize=(20, 5))

        binframe = bins * 60 * 5

        for j, group in enumerate(groups):
            # Get the DataFrame for the current group
            df_group = df_group1 if group == group1 else df_group2 

            # Plot individual fly positions
            for col in df_group.columns:
                position_binned = []
                fly_positions = df_group[col].tolist()

                for t in range(0, len(fly_positions), binframe):
                    pos_bin = fly_positions[t:t + binframe]
                    position_binned.append((sum(pos_bin)) / binframe)

                plt.plot(range(0, len(position_binned) * binframe, binframe),
                         position_binned, color=group_color[j], alpha=0.1)  # Transparent lines

            # Plot average position
            position_binned = []
            average_pos_active = average_pos_grs[j]

            for t in range(0, len(average_pos_active), binframe):
                pos_bin = average_pos_active[t:t + binframe]
                position_binned.append((sum(pos_bin)) / binframe)

            plt.plot(range(0, len(position_binned) * binframe, binframe),
                     position_binned, color=group_color[j], label=f"{group_names[j]}")

        for i in range(0, len(tick_vals), 2):
            start_days.append(tick_vals[i])
        for i in range(1, len(tick_vals), 2):
            end_days.append(tick_vals[i])
        end_days.append(len(average_pos_gr1))
        for i in range(2, len(tick_vals), 2):
            end_nights.append(tick_vals[i])

        if classic_LD is True:                
            for start, end in zip(start_days, end_days):
              plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
            for start, end in zip(end_days, end_nights):
              plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0)              
        else:  
            LD_cycles = number_of_cycles_before_DD
            for i, (start, end) in enumerate(zip(start_days, end_days)):
                if i < LD_cycles + 1:
                    plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                else:
                    plt.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
            for start, end in zip(end_days, end_nights):
              plt.axvspan(start, end, color="grey", alpha=0.3, ymin=0) 

        plt.xticks(tick_vals, tick_labs)

        # Label and title the plot
        plt.ylim(1, -1)
        plt.xlabel("Time (ZT)")
        plt.ylabel("Position (normalized)")
        plt.title(f"Average position - {bins} minute bins")
        plt.legend(loc='upper right')
        plot_filename = f"{plot_folder}average_position_{bins}.png"
        plt.savefig(plot_filename)
        plt.close()
