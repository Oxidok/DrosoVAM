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
from scipy.signal import savgol_filter


               ##############################
               #                            #
               #         Set values         #
               #                            #
               ##############################

expname = "LAM_Dhyd_Females_FCA_V1"

base_folder = f"C:/Users/revel/Desktop/Video_work/vw_{expname}_output_V9/"


skip_averageday_distance = False
skip_averageday_position = False
skip_averageday_virtualbeam = False


# Read the config file
config = configparser.ConfigParser()
config_file = os.path.join(base_folder, 'config.ini')
print(config_file)
config.read(config_file)

# Define folder paths
tracking_folder = os.path.join(base_folder, "tracking_V12/") # Version 12 created on 09.01.2025
plot_folder = os.path.join(base_folder, "plots_V12/")
data_folder = os.path.join(base_folder, "data_V12/")
indiv_plot_folder = os.path.join(plot_folder, "individual_plots/")
position_folder = os.path.join(tracking_folder, "combined_tracking_data/")
average_days_folder = os.path.join(tracking_folder, "average_days_analysis/")  # Newly added on 06.01.2025
all_pos_filename = f"{position_folder}{expname}.csv"
all_pos_norm_filename = f"{position_folder}{expname}_norm.csv"
all_pos_norm_bins_filename = f"{position_folder}{expname}_norm_bins.csv"
beam_crossings_filename = f"{position_folder}{expname}_beam.csv"
pos_bins_plotname = f"{plot_folder}y_position_plot.png"
fpi_plotname = f"{plot_folder}food_preference_index.png"

zt12, zt0, ct12, ct0 = "ZT12", "ZT0", "CT12", "CT0"
tick_vals, tick_labs = [], []
rejected_list = []

row_number = int(config['CHAMBER_DESIGN']['row_number'])
col_number = int(config['CHAMBER_DESIGN']['col_number'])
first_video_number = int(config['CHAMBER_DESIGN']['first_video_number'])
last_video_number = int(config['CHAMBER_DESIGN']['last_video_number'])
minbins = [1, 10, 30, 60] # number of minutes per bins

top_food = ast.literal_eval(config['CHAMBER_DESIGN']['top_food'])
bot_food = ast.literal_eval(config['CHAMBER_DESIGN']['bottom_food'])
group_foods = [top_food, bot_food]


group1 = eval(config['FLIES_DETAILS']['group1'])
group2 = eval(config['FLIES_DETAILS']['group2'])
group3 = eval(config['FLIES_DETAILS']['group3'])

groups = [group1]
group_names = [ast.literal_eval(config['FLIES_DETAILS']['group1_name'])]
group_colors = [ast.literal_eval(config['FLIES_DETAILS']['group1_color'])]
if len(group2) > 0:
    groups.append(group2)    
    group_names.append(ast.literal_eval(config['FLIES_DETAILS']['group2_name']))
    group_colors.append(ast.literal_eval(config['FLIES_DETAILS']['group2_color']))
    if len(group3) > 0:
        groups.append(group3)
        group_names.append(ast.literal_eval(config['FLIES_DETAILS']['group3_name']))
        group_colors.append(ast.literal_eval(config['FLIES_DETAILS']['group3_color']))
        

# Use for thresholding, normalization and binning
num_position_bins = int(config['EXPERIMENT']['position_bins'])
threshold_frameCount = 1000
i = 0
dictio, dictionorm, dictiobins, dictiobeam = [], [], [], []

video_length = int(config['EXPERIMENT']['duration'])
delay = float(config['EXPERIMENT']['delay'])
light_stat = 1 # 1 if the first light switch turns it off.
fps = 5
fph = fps * 60 * 60
fpv = video_length * 60 * fps
tot_frame = (last_video_number - (first_video_number - 1)) * fpv

new_model = eval(config['EXPERIMENT']['new_model'])

if new_model is True:
    model_name = "resnet50_DrosoVAM_v3Sep18shuffle1_750000"
else:
    model_name = "mobnet_100_DLC_Test_1Mar1shuffle1_100000"

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

smoothened_position_plot = False


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
    
    filecount = 0

    for x in range(first_video_number, (last_video_number + 1)):
        filename = f"video_{x}_{specimen_id}DLC_tracking.csv"
        filepath = os.path.join(tracking_folder, filename)
        if os.path.exists(filepath):
            filecount += 1
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
    if ttl > 1000:
        if ttl > 10000:
            loop_size = 5000
        else:
            loop_size = 1000
    else:
        loop_size = 100



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
        if looper % loop_size == 0 or lfter == 0:
            print(f"Species: {specimen_id}, Replacing: {cnter}, left to do: {lfter}")
        looper += 1

    print("Done replacing values")
    return series

def define_yellow_grey_areas(end_val, period, delay):
    """
    Creates lists of yellow and grey areas for light/dark cycle visualization.

    Args:
        end_val: The value to use as the "end" of the plot.
        period: The number of values in a 12h time period.
        delay: The delay value (needs to be adapted for each plot).

    Returns:
        Lists: yellow_areas and grey_areas.
    """
  
    yellow_areas = []
    grey_areas = []
    runner = 0 # Start at 0, beginning of the plots, will progress day by day
    yellow_area = [runner, delay] # Defines the first light period, the delay
    grey_area = [delay, delay + period] # Defines the first dark period

    yellow_areas.append(yellow_area)
    grey_areas.append(grey_area)

    runner += delay + period # runner changes to the beginning of day 2 ahead of the loop

    while runner < end_val:
        yellow_area = [runner]
        if runner + period < end_val:
            yellow_area.append(runner + period)
            grey_area = [runner + period]
        else:
            yellow_area.append(end_val)
            yellow_areas.append(yellow_area)
            break
        if runner + 2 * period < end_val:
            grey_area.append(runner + 2 * period)
        else:
            grey_area.append(end_val)
            grey_areas.append(grey_area)
            yellow_areas.append(yellow_area)
            break
        yellow_areas.append(yellow_area)
        grey_areas.append(grey_area)
        runner += 2 * period # runner goes to next day

    return yellow_areas, grey_areas


def activity_plot_maker(minbins, distances, chamber_y, chamber_z):

    for bins in minbins:

        prodist_xmin = []

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

        for i, group in enumerate(groups):
            if chamber_y in group:
                linecolor = group_colors[i]
                linename = group_names[i]
                
        # Generate plot
        fig, ax = plt.subplots(figsize=(20, 5))

        # Plot distance per 10 minutes
        ax.plot(range(0, len(prodist_xmin) * binframe, binframe), prodist_xmin, color=linecolor, label=f"Distance per {bins} minutes")

        ax.set_xticks(tick_vals)  # Set the ticks
        ax.set_xticklabels(tick_labs)  # Set the labels

        yellow_areas, grey_areas = define_yellow_grey_areas(len(prodist_xmin) * binframe, 12 * fph, delay * fph)
                
        if classic_LD is True:
            if is_DPA is True:
                for start, end in yellow_areas:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0, ymax=0.5)
                    ax.axvspan(start, end, color="grey", alpha=0.2, ymin=0.5, ymax=1)
                for start, end in grey_areas:
                    ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
            else:
                for start, end in yellow_areas:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                for start, end in grey_areas:
                    ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
        
        else:
            LD_cycles = number_of_cycles_before_DD
            for i, (start, end) in enumerate(yellow_areas):
                if i < LD_cycles + 1:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                else:
                    ax.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
            for start, end in grey_areas:
                ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)

        # Label and title the plot
        ax.set_xlabel("Time (ZT)")
        ax.set_ylabel("Distance (pixels)")
        ax.set_title(f"Distance Covered in Chamber {chamber_y}-{chamber_z} - {linename}")
        ax.legend(loc='upper right')
        plt.tight_layout()
        plot_filename = f"{indiv_plot_folder}chamber_{chamber_y}_{chamber_z}_distance_{bins}min_plot.png"
        plt.savefig(plot_filename)
        plt.close()

    print(f"Plots saved for chamber {chamber_y}_{chamber_z}")


def rearrange_and_save_data(binned_df, bin_label, average_days_folder, method, group_name):
    """
    Rearranges the binned data by days, inverts day and night periods,
    and saves the results to CSV files.

    Args:
        binned_df: The DataFrame containing binned distance data.
        bin_label: The bin size label (e.g., 10 or 60).
        average_days_folder: The folder where results will be saved.
        method (str): The type of analysis, distance or position.
        group_name (str): The label of the group.
    """

    rows_per_day = int(24  * (60 / bin_label))
    inversion_point = int(12 * (60 / bin_label))

    num_days = len(binned_df) // rows_per_day

    # Use a list to collect DataFrames, then concat at the end
    all_days_data = []
    for day in range(num_days):
        start_index = day * rows_per_day
        end_index = start_index + rows_per_day
        day_data = binned_df.iloc[start_index:end_index].copy()
        day_data.reset_index(drop=True, inplace=True)
        day_data.columns = [f"{col}_day_{day+1}" for col in day_data.columns]
        all_days_data.append(day_data)

    # Concatenate all day data at once
    rearranged_df = pd.concat(all_days_data, axis=1)

    # Save the rearranged dataframe
    rearranged_df_filename = f"{average_days_folder}all_{method}s_binned_all_days_{bin_label}min_{group_name}.csv"
    rearranged_df.to_csv(rearranged_df_filename, index=False)

    # Invert day and night
    first_half = rearranged_df.iloc[:inversion_point].copy()
    rearranged_df = rearranged_df.iloc[inversion_point:]
    rearranged_df = pd.concat([rearranged_df, first_half], ignore_index=True)

    # Save the inverted dataframe
    inverted_df_filename = f"{average_days_folder}all_{method}s_binned_day_night_{bin_label}min_{group_name}.csv"
    rearranged_df.to_csv(inverted_df_filename, index=False)


def plot_averageday_data(bin_labels, method):
    """
    Reads the averageday CSV files for all groups, for a given method, and plot the average day.

    Args:
        bin_labels (list): The bin size labels used.
        method: The type of analysis, usually "distance" or "position".
    """
    for bin_label in bin_labels:

        # Create the plot
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        for group_name, group_color in zip(group_names, group_colors):

            averageday_filename = f"{average_days_folder}all_{method}s_binned_day_night_{bin_label}min_{group_name}.csv"
            df = pd.read_csv(averageday_filename)

            plt.plot(df, color=group_color, alpha=0.1)

            df_average = df.mean(axis=1)

            plt.plot(df_average, label=group_name, color=group_color)


        num_data_points = len(df_average)

        x_ticks = [0, num_data_points //2 - 0.5, num_data_points - 1]
        x_labels = ["ZT0", "ZT12", "ZT24"]
        plt.xticks(x_ticks, x_labels)
        plt.xlabel("Time of Day")
        plt.title(f"Average Day - {method} - ({bin_label} min bins)")

        if method == "position":
            plt.ylabel("Average Position")
            plt.ylim(1, -1)
            if is_DPA is True:
                ax.axvspan(0, num_data_points // 2 - 0.5, color="yellow", alpha=0.2,  ymin=0, ymax=0.5)  # Use ax.axvspan to put color in the background
                ax.axvspan(0, num_data_points // 2 - 0.5, color="grey", alpha=0.2,  ymin=0.5, ymax=1)
            else:
                ax.axvspan(0, num_data_points // 2 - 0.5, color="yellow", alpha=0.2, ymin=0)
        elif method == "distance":
            plt.ylabel("Average Distance (pixels)")
            ax.axvspan(0, num_data_points // 2 - 0.5, color="yellow", alpha=0.2, ymin=0)

        elif method == "virtualbeam":
            plt.ylabel("Detections per hour")
            ax.axvspan(0, num_data_points // 2 - 0.5, color="yellow", alpha=0.2, ymin=0)

        ax.axvspan(num_data_points // 2 - 0.5, num_data_points - 1, color="grey", alpha=0.3, ymin=0)

        plt.legend(loc='upper right')
        plt.tight_layout()

        plot_filename = f"{average_days_folder}averageday_plot_{method}_{bin_label}min.png"
        plt.savefig(plot_filename)
        plt.close()




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
folders = [tracking_folder, plot_folder, data_folder, position_folder, indiv_plot_folder, average_days_folder]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Ensure group_names and group_colors are not longer than groups
group_names = group_names[:len(groups)]
group_colors = group_colors[:len(groups)]

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
        if filename.endswith(f"{model_name}.csv"):
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

    config['ANALYSIS']['fly_length'] = str(fly_length)
    with open(config_file, 'w') as configfile:
        config.write(configfile)

    print(f"Average fly size: {fly_length} pixels")


if skip_tracking is False:

    print("Going through tracking data...")

    for filename in os.listdir(base_folder):
        if filename.endswith(f"{model_name}.csv"):
            # Extract the video number
            video_number = int(filename.split("_")[1])  # Assuming video number is after the first underscore
            # Check if the video number is within the range
            if first_video_number <= video_number <= last_video_number:
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
            normalized_filename = f"{position_folder}norm_specimen{specimen_id}.csv"

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
                        plotcolor = group_colors[i]
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

                activity_plot_maker(minbins, distances, chamber_y, chamber_z)


            # Create a dictionary with all the Y-coordinates

            fly_y = filtered_data["Center_Y"].tolist()
            dictio.append(pd.DataFrame({specimen_id:fly_y}))

            # Create a dictionary with all the Y-coordinates normalized

            scaler = (fly_y - min_y) / (max_y - min_y)
            fly_norm_y = scaler * 2 - 1

            dictionorm.append(pd.DataFrame({specimen_id:fly_norm_y}))

            # Create a dictionary with all the Y-coordinates normalized and binned

            bin_counts, bin_edges = np.histogram(fly_norm_y, bins=num_position_bins)

            dictiobins.append(pd.DataFrame({specimen_id:bin_counts}))





            # To create all normalized file

            fly_y = filtered_data["Center_Y"].tolist()
            fly_x = filtered_data["Center_X"].tolist()

            scaler_y = (fly_y - min_y) / (max_y - min_y) # Normalize: 0 -> 1
            scaler_x = (fly_x - min_x) / (max_x - min_x)

            fly_norm_y = scaler_y * 2 - 1 # Normalize: -1 -> +1
            fly_norm_x = scaler_x * 2 - 1



            # Generate frame numbers
            frame_numbers = range(1, len(fly_norm_y) + 1)

            # Create a new DataFrame with the normalized X and Y coordinates
            normalized_data = pd.DataFrame({
                'Frame': frame_numbers,
                'Center_X': fly_norm_x,
                'Center_Y': fly_norm_y
            })

            # Save the normalized data to the CSV file
            normalized_data.to_csv(normalized_filename, index=False)


            if skip_virtual_beam is False:

                specimen_id = f"{chamber_y}_{chamber_z}" #/// Warning, the virtual beam names are y_z, not _y_z

                crossings_per_frame = np.zeros_like(fly_norm_y)
                for i in range(1, len(fly_norm_y)):
                    if (fly_norm_y[i] >= 0 and fly_norm_y[i-1] < 0) or \
                       (fly_norm_y[i] < 0 and fly_norm_y[i-1] >= 0):
                        crossings_per_frame[i] = 1

                if skip_beams_indiv_plots is False:

                    for i, group in enumerate(groups):
                        if chamber_y in group:
                            plotcolor = group_colors[i]
                            plotname = group_names[i]

                    # Plotting
                    plt.figure(figsize=(20, 5))  # Adjust figure size if needed
                    plt.plot(crossings_per_frame, marker='o', linestyle='-', color=plotcolor)
                    plt.title(f"Beam Crossings Per Minute - Specimen {specimen_id}")
                    plt.xlabel('Time')
                    plt.ylabel('Number of Crossings')
                    plt.savefig(f"{indiv_plot_folder}specimen{specimen_id}_virtual_beam.png")
                    plt.close()

                dictiobeam.append(pd.DataFrame({specimen_id:crossings_per_frame}))

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

    color_plot_list = []
    for i in range(num_position_bins):
        for color in group_colors:
            color_plot_list.append(color)

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
        ax.scatter(x, FPI_total[j], marker='.', color=group_colors[j])

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

        plot_to_remove = rejected_list

        # Update the config file with the rejected list
        config['ANALYSIS']['plot_to_remove'] = str(plot_to_remove)
        with open(config_file, 'w') as configfile:
            config.write(configfile)

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

            linecolor = group_colors[j]
            linename = group_names[j]

            # Generate plot
            fig, ax = plt.subplots(figsize=(20, 5))  # Use subplots to get axes object

            # Plot distance per 10 minutes
            ax.plot(range(0, len(average_col) * binframe, binframe), average_col, color=linecolor, label=f"Distance per {bins} minutes")

            ax.set_xticks(tick_vals)  # Set the ticks using ax.set_xticks
            ax.set_xticklabels(tick_labs)  # Set the labels using ax.set_xticklabels

            yellow_areas, grey_areas = define_yellow_grey_areas(len(average_col) * binframe, 12 * fph, delay * fph)
    
            if classic_LD is True:
                if is_DPA is True:
                    for start, end in yellow_areas:
                        ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0, ymax=0.5)
                        ax.axvspan(start, end, color="grey", alpha=0.2, ymin=0.5, ymax=1)
                    for start, end in grey_areas:
                        ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
                else:
                    for start, end in yellow_areas:
                        ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                    for start, end in grey_areas:
                        ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
            
            else:
                LD_cycles = number_of_cycles_before_DD
                for i, (start, end) in enumerate(yellow_areas):
                    if i < LD_cycles + 1:
                        ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                    else:
                        ax.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
                for start, end in grey_areas:
                    ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)

            # Label and title the plot
            ax.set_xlabel("Time (ZT)")  # Use ax.set_xlabel
            ax.set_ylabel("Distance (pixels)")  # Use ax.set_ylabel
            ax.set_title(f"Average locomotor activity - {linename}")  # Use ax.set_title
            ax.legend(loc='upper right')

            # Adjust layout to prevent clipping of labels
            plt.tight_layout()  # This is key to preventing clipping

            plot_filename = f"{plot_folder}{group_names[j]}_average_plot_{bins}.png"
            plt.savefig(plot_filename)
            plt.close()


        # Generate plot
        fig, ax = plt.subplots(figsize=(20, 5))  # Use subplots to get the axes object

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

            coi_num = len(csv_files) + 1  # Number of columns of interest

            average_col = np.mean(df.iloc[:, 1:coi_num], axis=1)  # Ignore the first column, and take the next 16

            df["AVG"] = average_col

            # Save the combined DataFrame to a new CSV file
            df.to_csv(output_file, index=False)

            start_days, end_days, end_nights = [], [], []
            binframe = bins * 60 * 5

            linecolor = group_colors[g]
            linename = group_names[g]

            ax.plot(range(0, len(average_col) * binframe, binframe), average_col, color=linecolor, label=f"{linename}")

        ax.set_xticks(tick_vals)  # Set the ticks using ax.set_xticks
        ax.set_xticklabels(tick_labs)  # Set the labels using ax.set_xticklabels

        yellow_areas, grey_areas = define_yellow_grey_areas(len(average_col) * binframe, 12 * fph, delay * fph) 

        if classic_LD is True:
            if is_DPA is True:
                for start, end in yellow_areas:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0, ymax=0.5)
                    ax.axvspan(start, end, color="grey", alpha=0.2, ymin=0.5, ymax=1)
                for start, end in grey_areas:
                    ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
            else:
                for start, end in yellow_areas:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                for start, end in grey_areas:
                    ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
        
        else:
            LD_cycles = number_of_cycles_before_DD
            for i, (start, end) in enumerate(yellow_areas):
                if i < LD_cycles + 1:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                else:
                    ax.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
            for start, end in grey_areas:
                ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)

        # Label and title the plot
        ax.set_xlabel("Time (ZT)")  # Use ax.set_xlabel
        ax.set_ylabel("Distance (pixels)")  # Use ax.set_ylabel
        ax.set_title(f"Average locomotor activity - {bins} minute bins")
        ax.legend(loc='upper right')
        plt.tight_layout()

        plot_filename = f"{plot_folder}combined_average_plot_{bins}.png"
        plt.savefig(plot_filename)
        plt.close()




if skip_virtual_beam is False:
    if skip_group_virtual_beam is False:

        print("Virtual Beam analysis ongoing, it may take time...")

        df = pd.read_csv(beam_crossings_filename, header=0)

        if len(plot_to_remove) != 0 or skip_plot_check is False:
            if len(plot_to_remove) != 0:
                rejected_list = plot_to_remove
            if len(rejected_list) != 0:
                df = df.drop(columns=[col for col in df.columns if col in rejected_list])

        df_groups = []  # List to store columns for each group
        for group in groups:
            group_cols = [col for col in df.columns if any(f"{str(g)}_" in col for g in group)]
            df_groups.append(group_cols)

        for min_bin in minbins:

            if min_bin > 30:
                continue
            if min_bin < 10:
                continue

            fig, ax = plt.subplots(figsize=(20, 5))

            window_size = int(min_bin * 60  * 5) # Define a window size for the binning

            group_medians = [] # List to store median values for each group

            for j, group_cols in enumerate(df_groups): # Iterate through groups
                group_vals = []
                for i in range(0, len(df), window_size):
                    start_index = i
                    end_index = min(i + window_size, len(df))

                    group_sum = df[group_cols].iloc[start_index:end_index].sum()
                    group_vals.append(group_sum)

                df_group_sums = pd.DataFrame(group_vals)
                group_median = df_group_sums.median(axis=1)
                group_medians.append(group_median)

                ax.plot(group_median, label=group_names[j], color=group_colors[j]) # Plot each group median



            yellow_start, yellow_end, grey_end = [], [], []
            beam_tick_vals, beam_tick_labs = [0], [""]
            ld = 1

            tick_step = int(12 * ( 60 // int(min_bin)))
            delay_ticks = int(delay * (60 // int(min_bin)))

            for s in range(delay_ticks, len(group_medians[0]), tick_step):
                beam_tick_vals.append(s)
                if ld == 1:
                    beam_tick_labs.append("ZT12")
                    ld = 0
                elif ld == 0:
                    beam_tick_labs.append("ZT0")
                    ld = 1
            beam_tick_vals.append(len(group_medians[0]))
            beam_tick_labs.append("")
            
            yellow_areas, grey_areas = define_yellow_grey_areas(len(group_medians[0]), 12 * (60 // int(min_bin)), delay * (60 // int(min_bin)))

            if classic_LD is True:
                if is_DPA is True:
                    for start, end in yellow_areas:
                        ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0, ymax=0.5)
                        ax.axvspan(start, end, color="grey", alpha=0.2, ymin=0.5, ymax=1)
                    for start, end in grey_areas:
                        ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
                else:
                    for start, end in yellow_areas:
                        ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                    for start, end in grey_areas:
                        ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
            
            else:
                LD_cycles = number_of_cycles_before_DD
                for i, (start, end) in enumerate(yellow_areas):
                    if i < LD_cycles + 1:
                        ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                    else:
                        ax.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
                for start, end in grey_areas:
                    ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
                    
            ax.set_xlabel("Time (ZT)")
            ax.set_ylabel(f"Number of Crossings per {min_bin} Minutes")
            ax.set_title(f"Virtual Laser Crossings - {min_bin} minutes bins - median")

            virtual_beam_plotname = f"{plot_folder}virtual_beam_{min_bin}min.png"

            ax.set_xticks(beam_tick_vals)
            ax.set_xticklabels(beam_tick_labs)

            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(virtual_beam_plotname)
            plt.close()




if len(plot_to_remove) != 0:

    rejected_list = plot_to_remove

elif skip_plot_check is False:

    user_input = input(input_text)


# Process user input
if user_input:
    # Split the input string into a list, handling commas and spaces
    rejected_input = user_input.split(",")  # Split by comma first
    # Further process each item to remove spaces (if needed)
    rejected_list = []
    for item in rejected_input:
        rejected_list.append(item.strip())  # Remove leading/trailing spaces

    plot_to_remove = rejected_list

    for value in rejected_list:

        for filename in glob.glob(f"{base_folder}data/chamber_{value}_*min.csv"):

            new_filename = os.path.join(base_folder, f"data/rejected_{os.path.basename(filename)}")
            os.rename(filename, new_filename)



# Average y_position/time plots per group with individual fly lines

if skip_position_time_plot is False:

    df = pd.read_csv(all_pos_norm_filename)

    columns_to_remove = ['_' + col for col in plot_to_remove]
    df = df.drop(columns=columns_to_remove, errors='ignore')
    
    df_groups = []  # List to store DataFrames for each group
    average_pos_grs = [] # List to store average positions for each group
    
    for group in groups:
        group_columns = [col for col in df.columns if int(col.split('_')[1]) in group]
        df_group = df[group_columns]
        df_groups.append(df_group)
        average_pos_grs.append(np.mean(df_group, axis=1).tolist())


    for i, bins in enumerate(minbins):

        start_days, end_days, end_nights = [], [], []

        # Generate plot
        fig, ax = plt.subplots(figsize=(20, 5))

        binframe = bins * 60 * 5

        for j, group in enumerate(groups):
            # Get the DataFrame for the current group
            df_group = df_groups[j]

            # Plot individual fly positions
            for col in df_group.columns:
                position_binned = []
                fly_positions = df_group[col].tolist()

                for t in range(0, len(fly_positions), binframe):
                    pos_bin = fly_positions[t:t + binframe]
                    position_binned.append((sum(pos_bin)) / binframe)

                plt.plot(range(0, len(position_binned) * binframe, binframe),
                         position_binned, color=group_colors[j], alpha=0.1)  # Transparent lines

            # Plot average position
            position_binned = []
            average_pos_active = average_pos_grs[j]

            for t in range(0, len(average_pos_active), binframe):
                pos_bin = average_pos_active[t:t + binframe]
                position_binned.append((sum(pos_bin)) / binframe)

            if smoothened_position_plot is True:
                # Smooth the average position data
                window_length = 10  # Adjust this value for desired smoothing level
                polyorder = 5
                position_binned =  savgol_filter(position_binned, window_length, polyorder)
                position_plot_filename = f"{plot_folder}average_position_{bins}_smooth.png"

            else:
                position_plot_filename = f"{plot_folder}average_position_{bins}.png"

            plt.plot(range(0, len(position_binned) * binframe, binframe),
                     position_binned, color=group_colors[j], label=f"{group_names[j]}")
            
        yellow_areas, grey_areas = define_yellow_grey_areas(len(average_pos_grs[j]), 12 * fph, delay * fph)
        
        if classic_LD is True:
            if is_DPA is True:
                for start, end in yellow_areas:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0, ymax=0.5)
                    ax.axvspan(start, end, color="grey", alpha=0.2, ymin=0.5, ymax=1)
                for start, end in grey_areas:
                    ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
            else:
                for start, end in yellow_areas:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                for start, end in grey_areas:
                    ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)
        
        else:
            LD_cycles = number_of_cycles_before_DD
            for i, (start, end) in enumerate(yellow_areas):
                if i < LD_cycles + 1:
                    ax.axvspan(start, end, color="yellow", alpha=0.2, ymin=0)
                else:
                    ax.axvspan(start, end, color="lightgrey", alpha=0.3, ymin=0)
            for start, end in grey_areas:
                ax.axvspan(start, end, color="grey", alpha=0.3, ymin=0)

        plt.xticks(tick_vals, tick_labs)

        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_labs)
        ax.set_ylim(1, -1)  # Use ax.set_ylim
        ax.set_xlabel("Time (ZT)")  # Use ax.set_xlabel
        ax.set_ylabel("Position (normalized)")  # Use ax.set_ylabel
        ax.set_title(f"Average position - {bins} minute bins")  # Use ax.set_title
        ax.legend(loc='upper right')

        plt.tight_layout()  # Add tight_layout
        plt.savefig(position_plot_filename)
        plt.close()

if skip_averageday_distance is True or skip_averageday_position is True or skip_averageday_virtualbeam is True:
    print("Generating Average-day data...")

if skip_averageday_distance is False:

    for group, group_name in zip(groups, group_names):

        # Create a dataframe to store the distance values of several specimens.
        distance_df = pd.DataFrame()

        for filename in glob.glob(f"{position_folder}specimen_{group}_*filtered.csv"):
            specimen_number = filename.split("specimen_")[1].split("_filtered.csv")[0]

            if specimen_number not in plot_to_remove:

                df = pd.read_csv(filename)
                distance_df[specimen_number] = df.iloc[:, 2]  # 3rd column (index 2)

        # Save the dataframe as a csv file
        distance_df_filename = f"{average_days_folder}all_distances_{group_name}.csv"
        distance_df.to_csv(distance_df_filename, index=False)

        # Remove the first x rows
        delay_rows = int(delay * fph)
        distance_df = distance_df.iloc[delay_rows:]

        # Apply binning of 10 or 60 minutes
        bin_sizes = [3000, 18000]  # 3000 = 10 min * 60 sec * 5 fps
        bin_labels = [10, 60]

        for bin_size, bin_label in zip(bin_sizes, bin_labels):
            binned_df = pd.DataFrame()
            for col in distance_df.columns:
                binned_values = [
                    distance_df[col].iloc[i : i + bin_size].sum()
                    for i in range(0, len(distance_df), bin_size)
                ]
                binned_df[col] = binned_values

            # Save the binned files
            binned_df_filename = f"{average_days_folder}all_distances_binned_{bin_label}min_{group_name}.csv"
            binned_df.to_csv(binned_df_filename, index=False)

            method = "distance"

            # Process for 60-minute and 10-minutes bins
            rearrange_and_save_data(binned_df, bin_label, average_days_folder, method, group_name)

    plot_averageday_data(bin_labels, method)



if skip_averageday_position is False:


    if not os.path.exists(all_pos_norm_filename):
        print(f"File not found: {all_pos_norm_filename}")
    else:
        df = pd.read_csv(all_pos_norm_filename)

        for group, group_name in zip(groups, group_names):

            # Create a dataframe to store the position values of several specimens.
            position_df = pd.DataFrame()

            for col in df.columns:
              if col.startswith('Unnamed') or col == "Frame":
                continue
              specimen_number = col[1:]

              if specimen_number not in plot_to_remove:
                  specimen_group = int(specimen_number[0])
                  if specimen_group in group:
                      position_df[specimen_number] = df[col]

            # Save the dataframe as a csv file
            position_df_filename = f"{average_days_folder}all_positions_{group_name}.csv"
            position_df.to_csv(position_df_filename, index=False)

            # Remove the first x rows
            delay_rows = int(delay * fph)
            position_df = position_df.iloc[delay_rows:]

            # Apply binning of 10 or 60 minutes
            bin_sizes = [3000, 18000]  # 10 minutes and 60 minutes
            bin_labels = [10, 60]

            for bin_size, bin_label in zip(bin_sizes, bin_labels):
                binned_df = pd.DataFrame()
                for col in position_df.columns:
                    binned_values = [position_df[col].iloc[i:i+bin_size].mean() for i in range(0, len(position_df), bin_size)]
                    binned_df[col] = binned_values

                # Save the binned files
                binned_df_filename = f"{average_days_folder}all_positions_binned_{bin_label}min_{group_name}.csv"
                binned_df.to_csv(binned_df_filename, index=False)

                method = "position"

                # Process for 60-minute and 10-minutes bins
                rearrange_and_save_data(binned_df, bin_label, average_days_folder, method, group_name)

        plot_averageday_data(bin_labels, method)


if skip_averageday_virtualbeam is False:

    if not os.path.exists(beam_crossings_filename):
        print(f"File not found: {beam_crossings_filename}")
    else:
        df = pd.read_csv(beam_crossings_filename, header=0)

        for group, group_name in zip(groups, group_names):

            # Create a dataframe to store the position values of several specimens.
            virtualbeam_df = pd.DataFrame()

            for col in df.columns:
              if col.startswith('Unnamed') or col == "Frame":
                continue
              specimen_number = col

              if specimen_number not in plot_to_remove:
                  specimen_group = int(specimen_number[0])
                  if specimen_group in group:
                      virtualbeam_df[specimen_number] = df[col]

            # Save the dataframe as a csv file
            virtualbeam_df_filename = f"{average_days_folder}all_virtualbeams_{group_name}.csv"
            virtualbeam_df.to_csv(virtualbeam_df_filename, index=False)

            # Remove the first x rows
            delay_rows = int(delay * fph)
            virtualbeam_df = virtualbeam_df.iloc[delay_rows:]

            # Apply binning of 60 minutes
            bin_sizes = [int(60*60*5)]  # 60 minutes  * 60 seconds * 5fps
            bin_labels = [60]

            for bin_size, bin_label in zip(bin_sizes, bin_labels):
                binned_df = pd.DataFrame()
                for col in virtualbeam_df.columns:
                    binned_values = [virtualbeam_df[col].iloc[i:i+bin_size].sum() for i in range(0, len(virtualbeam_df), bin_size)]
                    binned_df[col] = binned_values

                # Save the binned files
                binned_df_filename = f"{average_days_folder}all_virtualbeams_binned_{bin_label}min_{group_name}.csv"
                binned_df.to_csv(binned_df_filename, index=False)

                method = "virtualbeam"

                # Process for 60-minute and 10-minutes bins
                rearrange_and_save_data(binned_df, bin_label, average_days_folder, method, group_name)

        plot_averageday_data(bin_labels, method)






















