# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:42:28 2024

@author: revel
"""

import os
import csv
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


               ##############################
               #                            #
               #         Set values         #
               #                            #
               ##############################

expname = "Exp5_LAM_Dmel"

# Define folder paths
base_folder = f"C:/Users/revel/Desktop/Video_work/vw_{expname}_output/"
tracking_folder = os.path.join(base_folder, "tracking_V2/")
plot_folder = os.path.join(base_folder, "plots_V2/")
data_folder = os.path.join(base_folder, "data_V2/")
position_folder = os.path.join(tracking_folder, "combined_tracking_data/")
all_pos_filename = f"{position_folder}{expname}.csv"
all_pos_norm_filename = f"{position_folder}{expname}_norm.csv"
all_pos_norm_bins_filename = f"{position_folder}{expname}_norm_bins.csv"
beam_crossings_filename = f"{position_folder}{expname}_beam.csv"
pos_bins_plotname = f"{position_folder}{expname}_plot.png"
fpi_plotname = f"{position_folder}{expname}_plot_FPI.png"
virtual_beam_plotname = f"{position_folder}{expname}_plot_virtual_beam.png"

zt12, zt0 = "ZT12", "ZT0"
tick_vals, tick_labs = [], []
rejected_list = []

row_number = 4 # Number of rows in the monitoring chamber
col_number = 8 # Number of columns in the monitoring chamber
first_video_number = 1 
last_video_number = 36
minbins = [1, 10, 30] # number of minutes per bins 


# Rows considered as part of group 1 (can be 1, or 1 and 2, or 1 to 4, etc...)
group1 = [1, 2] # Will be colored in group_color[0]!
group2 = [3, 4] # Will be colored in group_color[1]! (if not already in [0])
groups = [group1, group2]
group_names = ["virgin", "mated"]
group_color = ["mediumseagreen", "mediumorchid"]
group_foods = ["None", "Strd food"]

# # Alternative blocks
# group1 = [1, 2, 3, 4]
# groups = [group1]
# group_names = ["males"]
# group_color = ["mediumvioletred"]


# Use for thresholding, normalization and binning
threshold_frameCount = 1000
num_position_bins = 10
i = 0
dictio, dictionorm, dictiobins, dictiobeam = [], [], [], []

delay = 8 # Delay in hours until the first light switch in the incubator
light_stat = 1 # 1 if the first ligth switch turns it off.
fps = 5
fph = fps * 60 * 60
tot_frame = last_video_number * 2 * fph # 2h * 60 min * 60 sec * 5 fps
classic_LD = True # True if 12:12 LD experiment



LD_cycle = [12, 12]
switch_to_DD = False # True if experiment goes from LD to DD
number_of_cycles_before_DD = 0 # If 1: Light(delay) - 1 x { Night(12) - Light(12) } - Night(forever)


# if not LD experiment:
alt_tick_labels = ["", "ZT12", "ZT0", "ZT12", "ZT0", "ZT12", "ZT0", "ZT12", "CT0", "CT12", "CT0", "CT12", "CT0", ""]
alt_start_days = [0, (delay + 12*1)*fph, (delay + 12*3)*fph, (delay + 12*5)*fph]
alt_end_days = [delay*fph, (delay + 12*2)*fph, (delay + 12*4)*fph, (delay + 12*6)*fph]
alt_end_nights = [(delay + 12*1)*fph, (delay + 12*3)*fph, (delay + 12*5)*fph, tot_frame] 



# To skip steps // If you don't know, keep everything as default
fly_length = 18.705 # Put the fly length value here to skip, or None to not skip
skip_tracking = 1 # 0 to do it, 1 to skip / Valid only if tracking has already been done
skip_concat = 1 # 0 to do it, 1 to skip / Valid only if it has already been done
skip_threshold_check = 1 # 0 to do it, 1 to skip / default is 1
skip_data_curation = 1 # 0 to do it, 1 to skip /// Valid only if it has already been done
skip_offlimit_plots = 1 # 0 to show the off-limit plots // Useless due to curation, default is 1
skip_indiv_analysis = 0 # 0 to do it, 1 to skip / Valid only if it has already been done
skip_activity_plot = 0 # 0 to do it, 1 to skip
skip_hist = 1 # 0 to do it, 1 to skip
skip_indiv_pos_plot = 0 # 0 to do it, 1 to skip
skip_virtual_beam = 0 # 0 to do it, 1 to skip / This overrides the next setting
skip_beams_indiv_plots = 1 # 0 to do it, 1 to skip
skip_group_virtual_beam = 0 # 0 to do it, 1 to skip
skip_position_plot = 1 # 0 to do it, 1 to skip
skip_FPI_plot = 1 # 0 to do it, 1 to skip
skip_group_plot = 0 # 0 to do it, 1 to skip  / This overrides the two next settings
skip_plot_check = 1 # 0 to do it, 1 to skip
plot_to_remove = [] # Put the specimen IDs to drop if you know them already, or leave empty

default_settings = 0 # if 1, all settings will be reset to default before running

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
        
        if classic_LD is True:
            for i in range(0, len(tick_vals), 2):
                start_days.append(tick_vals[i])  
            for i in range(1, len(tick_vals), 2):
                end_days.append(tick_vals[i])
            end_days.append(len(prodist_xmin) * binframe)
            for i in range(2, len(tick_vals), 2):
                end_nights.append(tick_vals[i])
            
        else:
            start_days = alt_start_days
            end_days = alt_end_days
            end_nights = alt_end_nights
            

        for start, end in zip(start_days, end_days):
          plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
        for start, end in zip(end_days, end_nights):
          plt.axvspan(start, end, color="grey", alpha=0.2, ymin=0)            
            
        # Label and title the plot
        plt.xlabel("Time (ZT)")
        plt.ylabel("Distance (pixels)")
        plt.title(f"Distance Covered in Chamber {chamber_y}-{chamber_z} - {linename}")
        plt.legend(loc='upper right')
        plot_filename = f"{plot_folder}chamber_{chamber_y}_{chamber_z}_distance_{bins}min_plot.png"
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"Plot saved for bins of {bins} minutes, for chamber {chamber_y}_{chamber_z}")


# The reset setting thing

if default_settings == 1:
    
    settings = [skip_tracking, skip_concat, skip_threshold_check, 
    skip_data_curation, skip_offlimit_plots, skip_indiv_analysis, 
    skip_activity_plot, skip_hist, skip_indiv_pos_plot, 
    skip_virtual_beam, skip_beams_indiv_plots, skip_group_virtual_beam, 
    skip_position_plot, skip_FPI_plot, skip_group_plot, skip_plot_check]
    param = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    
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
folders = [tracking_folder, plot_folder, data_folder, position_folder]
for folder in folders:
    os.makedirs(folder, exist_ok=True)


# Defines the tick and tick labels for LD or DD

if classic_LD is True:
    
    tick_vals.append(i)
    i = i + (delay * fph)
    tick_labs.append("")

    while i < tot_frame:
        tick_vals.append(i)
        i += (12 * fph) # 12h * 60 min * 60 sec * 5 fps
        if  light_stat == 1:
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

    while i < tot_frame:
        tick_vals.append(i)
        i += (12 * fph) # 12h * 60 min * 60 sec * 5 fps
    tick_vals.append(tot_frame)

    # Use this only for labels of not LD experiments
    tick_labs = alt_tick_labels
        


    
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
                    _, head_x, head_y, _, tail_x, tail_y, _ = row
                    head_x, head_y, tail_x, tail_y = map(float, [head_x, head_y, tail_x, tail_y])
                    distance = ((head_x - tail_x) ** 2 + (head_y - tail_y) ** 2) ** 0.5
                    total_distance += distance
                    row_count += 1
            fly_lengths[video_name] = total_distance / row_count
    
    total_fly_length = sum(fly_lengths.values())
    fly_length = total_fly_length / len(fly_lengths)
    
    print(f"Average fly size: {fly_length}")
        




# Loop to correct head/tail absurd movements and define fly center

if skip_tracking == 0:
    
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
    
    print("Data filtered, ready for concatenation.")


if skip_concat == 0:

    # Loop through all chambers to concatenate
    for chamber_y in range(1, (row_number + 1)): # Number of rows
        for chamber_z in range(1, col_number + 1): # Number of columns
    
            # Concatenate data for the current chamber
            concatenate_data(chamber_y, chamber_z)
            print(f"Data concatenated for chamber {chamber_y}_{chamber_z}")
            
    print("Data for all chambers concatenated.")        
   

if skip_indiv_analysis == 0:
    
    # Loop through all chambers tofilter, normalize and bin
    for chamber_y in range(1, (row_number + 1)): # Number of rows
        for chamber_z in range(1, col_number + 1): # Number of columns # Put back the 1 where there is a w
    
            specimen_id = f"_{chamber_y}_{chamber_z}"
            video_name = f"{base_folder}video_1{specimen_id}.mp4"
            file_name = f"{position_folder}specimen{specimen_id}.csv"
            
            print(f"Working on chamber: {specimen_id}")
            
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
            if skip_threshold_check == 0:     
                # Display the first frame and binary image side-by-side
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(first_frame)
                ax2.imshow(binary_image)
                # Add horizontal lines at min_y and max_y
                ax1.hlines(y=[min_y, max_y], xmin=min_x, xmax=max_x, color="red")
                ax2.hlines(y=[min_y, max_y], xmin=min_x, xmax=max_x, color="red")
                # Set labels and title
                ax1.set_title("First Frame")
                ax2.set_title("Binary Image")
                plt.suptitle(f"Threshold Verification (specimen {specimen_id})")
                plt.show()
                # Get user input
                key = input("Is the threshold appropriate (y/n)? ")
                if key.lower() != "y":
                    print("Script stopped by user.")
                    exit()
                skip_threshold_check += 1        
                
                
            # Read the csv file and remove values that are out limits
            data = pd.read_csv(file_name)
            
            # Create a new filename for the output CSV
            filtered_filename = f"{position_folder}specimen{specimen_id}_filtered.csv"
            
            if skip_data_curation == 0:
            
                fly_data = data.iloc[:, 1:]
                selected_columns = fly_data[["Center_X", "Center_Y"]] # Use column names in square brackets
                filtered_data = selected_columns.copy()
                filtered_data['Center_X'] = replace_off_limits(specimen_id, filtered_data['Center_X'], min_x, max_x)
                filtered_data['Center_Y'] = replace_off_limits(specimen_id, filtered_data['Center_Y'], min_y, max_y)
        
                if skip_offlimit_plots == 0:
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
                filtered_data['Distance'].fillna(0, inplace=True)
                
                # Save the filtered DataFrame with the new column to a new CSV file
                filtered_data.to_csv(filtered_filename, index=False)
        
                print(f"Filtered file created for specimen{specimen_id}")

            
            if skip_data_curation != 0:
                
                filtered_data = pd.read_csv(filtered_filename)
            
            
            
            if skip_hist == 0:
    
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
                plt.savefig(f"{position_folder}specimen{specimen_id}_heatmap.png")
                plt.close()    
            
            if skip_indiv_pos_plot == 0:
                
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
                plt.savefig(f"{position_folder}specimen{specimen_id}_histo_x.png")
                plt.close()
            
            
            
            data = pd.read_csv(filtered_filename)
            fly_data = data.iloc[:, 1:]
            distances = fly_data["Distance"]
            
            if skip_activity_plot == 0:
                
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
            
            
            
            if skip_virtual_beam == 0:
                
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
                
                if skip_beams_indiv_plots == 0:

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
                    plt.savefig(f"{position_folder}specimen{specimen_id}_virtual_beam.png")
                    plt.close()
                
                dictiobeam.append(pd.DataFrame({specimen_id:beam_crossings_per_30minutes}))                 

    print("Saving Y-position dictionaries...")    
    pd.concat(dictio, axis=1).to_csv(all_pos_filename, index = False)
    
    pd.concat(dictionorm, axis=1).to_csv(all_pos_norm_filename, index=False)
    
    pd.concat(dictiobins, axis=1).to_csv(all_pos_norm_bins_filename, index=False)
    
    if skip_virtual_beam == 0:
        pd.concat(dictiobeam, axis=1).to_csv(beam_crossings_filename, index=False)

    print("Y-position dictionaries saved!")          



if skip_position_plot == 0:
    
    # Plotting of total positions here
    
    group_split = int((row_number * col_number)/2)
    
    # Generate the binned boxplot
    
    df = np.genfromtxt(all_pos_norm_bins_filename, delimiter=",", skip_header=1, usecols=range(0, col_number * row_number - 1), filling_values=0)
    
    list_pos = [i+j for i in range(0, num_position_bins*3, 3) for j in [0,1]]
    
    plt.figure(figsize=(12, 7))
    
    bp1 = plt.boxplot([df[i,j:(j+group_split)] for i in range(0,num_position_bins) for j in [0,group_split]],
                      patch_artist=True, 
                      whis=[0,100],
                      positions=list_pos, 
                      widths=0.9)
    color_plot_list = ["dodgerblue" if j == 0 else "coral" for i in range(0,num_position_bins) for j in [0,col_number]]
    for patch, color in zip(bp1["boxes"], color_plot_list):
        patch.set_facecolor(color)
    
    bp1 = plt.legend(bp1["boxes"], group_names)
    xtick_values = [0, num_position_bins*1.5-1, num_position_bins*3-2]
    bp1= plt.xticks(xtick_values, [group_foods[0], "", group_foods[1]])
    plt.ylabel("Frequency")
    plt.title("Fly Y-Position frequency")
    plt.savefig(pos_bins_plotname)



if skip_FPI_plot == 0:
    
    # Generate the FPI scatterplot
    TopFoodTime1 = df[0, :group_split]
    TopFoodTime2 = df[0, group_split:]
    BotFoodTime1 = df[num_position_bins-1, :group_split]
    BotFoodTime2 = df[num_position_bins-1, group_split:]
    
    FPIgr1 = (BotFoodTime1 - TopFoodTime1)/(BotFoodTime1 + TopFoodTime1)
    FPIgr2 = (BotFoodTime2 - TopFoodTime2)/(BotFoodTime2 + TopFoodTime2) 
    FPI_total = [FPIgr1, FPIgr2]
    
    FPIavg1 = sum(FPIgr1)/len(FPIgr1)
    FPIavg2 = sum(FPIgr2)/len(FPIgr2)
    
    # Define the range of random noise (e.g., +/- 0.02)
    noise_range = 0.03
    x1 = [1 + random.uniform(-noise_range, noise_range) for _ in FPIgr1]
    x2 = [2 + random.uniform(-noise_range, noise_range) for _ in FPIgr2]
    
    x_avg = [0.95, 1, 1.05, 1.95, 2, 2.05]
    avglist = [FPIavg1, FPIavg1, FPIavg1, FPIavg2, FPIavg2, FPIavg2]
    
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.scatter(x1, FPIgr1, marker='.', color="dodgerblue")
    ax.scatter(x2, FPIgr2, marker='.', color="coral")
    ax.scatter(x_avg, avglist, marker='_', color="green")
    
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks([1, 2], ["Virgin", "Mated"])
    ax.set_yticks([-1, 0, 1], [group_foods[0], "", group_foods[1]])
    ax.set_ylabel("Food Preference Index")
    fig.tight_layout()
    fig.savefig(fpi_plotname)




if skip_group_plot == 0:
    
    user_input = 0
    if len(plot_to_remove) != 0:
        
        rejected_list = plot_to_remove
        
    elif skip_plot_check == 0:
            
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
            
            if classic_LD is True:
                for i in range(0, len(tick_vals), 2):
                    start_days.append(tick_vals[i])  
                for i in range(1, len(tick_vals), 2):
                    end_days.append(tick_vals[i])
                end_days.append(len(average_col) * binframe)
                for i in range(2, len(tick_vals), 2):
                    end_nights.append(tick_vals[i])
            else:
                start_days = alt_start_days
                end_days = alt_end_days
                end_nights = alt_end_nights
            
            
            for start, end in zip(start_days, end_days):
              plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
            for start, end in zip(end_days, end_nights):
              plt.axvspan(start, end, color="grey", alpha=0.2, ymin=0)            
                
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
        
        if classic_LD is True:
            for i in range(0, len(tick_vals), 2):
                start_days.append(tick_vals[i])  
            for i in range(1, len(tick_vals), 2):
                end_days.append(tick_vals[i])
            end_days.append(len(average_col) * binframe)
            for i in range(2, len(tick_vals), 2):
                end_nights.append(tick_vals[i])
        else:
            start_days = alt_start_days
            end_days = alt_end_days
            end_nights = alt_end_nights
        
        
        for start, end in zip(start_days, end_days):
          plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
        for start, end in zip(end_days, end_nights):
          plt.axvspan(start, end, color="grey", alpha=0.2, ymin=0)            
            
        # Label and title the plot
        plt.xlabel("Time (ZT)")
        plt.ylabel("Distance (pixels)")
        plt.title(f"Average locomotor activity - {bins} minute bins")
        plt.legend(loc='upper right')
        plot_filename = f"{plot_folder}combined_average_plot_{bins}.png"
        plt.savefig(plot_filename)
        plt.close()
    
    
    
    
if skip_virtual_beam == 0:
    if skip_group_virtual_beam == 0:
    
        plt.figure(figsize=(20, 5))
        
        # Read the CSV file
        df = pd.read_csv(beam_crossings_filename, header=0)
    
            
        if len(plot_to_remove) != 0 or skip_plot_check == 0:
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

        if classic_LD is True:
            
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
        
        if classic_LD is True:
            
            for i in range(0, len(beam_tick_vals), 2):
                beam_end_days.append(beam_tick_vals[i])  
            for i in range(1, len(beam_tick_vals)-1, 2):
                beam_start_days.append(beam_tick_vals[i])
            for i in range(2, len(beam_tick_vals), 2):
                beam_end_nights.append(beam_tick_vals[i])
        
        for start, end in zip(beam_end_days, beam_start_days):
          plt.axvspan(start, end, color="grey", alpha=0.2, ymin=0)
        for start, end in zip(beam_start_days, beam_end_nights):
          plt.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)     
        
        plt.legend(loc='upper right')
        plt.savefig(virtual_beam_plotname)
        plt.close()
    
    
    
